import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from core.models.base_model import BaseModel


class DecisionTreeClassifierModel(BaseModel):
    def __init__(self):
        self.model = DecisionTreeClassifier()

    def train(self, data):
        X, y = data
        self.model.fit(X, y)

    def predict(self, input_data):
        return self.model.predict(input_data)

    def evaluate(self, test_data):
        X_test, y_test = test_data

        # 預測
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]  # 取得正類別的機率

        # 計算各項指標
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }

        return metrics

    def save_model(self, file_path):
        joblib.dump(self.model, file_path)

    def load_model(self, file_path):
        self.model = joblib.load(file_path)
