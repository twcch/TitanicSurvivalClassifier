import joblib
from sklearn.tree import DecisionTreeClassifier
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
        return self.model.score(X_test, y_test)

    def save(self, file_path):
        joblib.dump(self.model, file_path)

    def load(self, file_path):
        self.model = joblib.load(file_path)
