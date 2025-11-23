import joblib
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from core.models.base_model import BaseModel


class DecisionTreeClassifierModel(BaseModel):
    def __init__(self):
        self.model = DecisionTreeClassifier(
            criterion="gini",
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=10,
            random_state=42,
        )
        self.feature_names = None

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

    def visualize_tree(self, output_path: str = "outputs/decision_tree.png", figsize=(20, 10)):
        """
        視覺化決策樹並儲存為圖片
        
        Args:
            output_path: 輸出圖片路徑
            figsize: 圖片大小 (寬, 高)
        """
        plt.figure(figsize=figsize)
        
        plot_tree(
            self.model,
            feature_names=self.feature_names,
            class_names=['Died', 'Survived'],
            filled=True,
            rounded=True,
            fontsize=10
        )
        
        plt.title("Decision Tree for Titanic Survival Prediction", fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 決策樹視覺化已儲存至 {output_path}")
    
    def get_feature_importance(self, top_n: int = 10):
        """
        取得特徵重要性
        
        Args:
            top_n: 顯示前 N 個重要特徵
        
        Returns:
            dict: 特徵名稱與重要性分數
        """
        if self.feature_names is None:
            return None
        
        importances = self.model.feature_importances_
        feature_importance = dict(zip(self.feature_names, importances))
        
        # 排序並取前 N 個
        sorted_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        )
        
        return sorted_importance
    
    def visualize_feature_importance(self, output_path: str = "outputs/feature_importance.png", top_n: int = 10):
        """
        視覺化特徵重要性
        
        Args:
            output_path: 輸出圖片路徑
            top_n: 顯示前 N 個重要特徵
        """
        importance_dict = self.get_feature_importance(top_n)
        
        if importance_dict is None:
            print("⚠️  無法取得特徵重要性")
            return
        
        features = list(importance_dict.keys())
        importances = list(importance_dict.values())
        
        plt.figure(figsize=(10, 6))
        plt.barh(features, importances, color='steelblue')
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance', fontsize=14)
        plt.gca().invert_yaxis()  # 最重要的在上面
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 特徵重要性圖已儲存至 {output_path}")

    def save_model(self, file_path):
        joblib.dump(self.model, file_path)

    def load_model(self, file_path):
        self.model = joblib.load(file_path)
