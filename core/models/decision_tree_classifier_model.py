import joblib
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from core.models.base_model import BaseModel


class DecisionTreeClassifierModel(BaseModel):
    def __init__(
        self, use_tuning: bool = True, tuning_method: str = "grid", cv: int = 5
    ):
        """
        初始化決策樹模型

        Args:
            use_tuning: 是否使用超參數調優
            tuning_method: 調優方法 ("grid" 或 "random")
            cv: 交叉驗證折數
        """
        self.use_tuning = use_tuning
        self.tuning_method = tuning_method
        self.cv = cv
        self.feature_names = None
        self.best_params = None
        self.tuning_results = None

        # 預設參數
        self.default_params = {
            "criterion": "gini",
            "max_depth": 5,
            "min_samples_split": 10,
            "min_samples_leaf": 10,
            "random_state": 42,
        }

        # 超參數搜尋空間
        self.param_grid = {
            "criterion": ["gini", "entropy"],
            "max_depth": [3, 5, 7, 10, 15, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 5, 10, 20],
            "max_features": [None, "sqrt", "log2"],
            "min_impurity_decrease": [0.0, 0.01, 0.05],
        }

        # 初始化模型
        self.model = DecisionTreeClassifier(**self.default_params)

    def train(self, data):
        X, y = data

        # 儲存特徵名稱
        if hasattr(X, "columns"):
            self.feature_names = X.columns.tolist()

        if self.use_tuning:
            print("\n" + "=" * 50)
            print(f"開始超參數調優 ({self.tuning_method.upper()} Search)...")
            print("=" * 50)

            # 選擇調優方法
            if self.tuning_method == "grid":
                search = GridSearchCV(
                    estimator=DecisionTreeClassifier(random_state=42),
                    param_grid=self.param_grid,
                    cv=self.cv,
                    scoring="accuracy",
                    n_jobs=-1,
                    verbose=1,
                )
            elif self.tuning_method == "random":
                search = RandomizedSearchCV(
                    estimator=DecisionTreeClassifier(random_state=42),
                    param_distributions=self.param_grid,
                    n_iter=50,  # 隨機嘗試 50 種組合
                    cv=self.cv,
                    scoring="accuracy",
                    n_jobs=-1,
                    verbose=1,
                    random_state=42,
                )
            else:
                raise ValueError(f"Unknown tuning method: {self.tuning_method}")

            # 執行超參數搜尋
            search.fit(X, y)

            # 儲存最佳參數和結果
            self.best_params = search.best_params_
            self.tuning_results = {
                "best_score": search.best_score_,
                "best_params": search.best_params_,
                "cv_results": search.cv_results_,
            }

            # 使用最佳參數的模型
            self.model = search.best_estimator_

            # 印出結果
            print("\n" + "=" * 50)
            print("超參數調優完成！")
            print("=" * 50)
            print(f"最佳交叉驗證分數: {search.best_score_:.4f}")
            print("\n最佳參數:")
            print("-" * 50)
            for param, value in self.best_params.items():
                print(f"  {param:25s}: {value}")
            print("=" * 50 + "\n")

        else:
            # 使用預設參數直接訓練
            self.model.fit(X, y)
            print("使用預設參數訓練模型")

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

    def get_tuning_summary(self):
        """
        取得超參數調優的詳細結果

        Returns:
            dict: 調優結果摘要
        """
        if self.tuning_results is None:
            return None

        import pandas as pd

        # 取得前 10 個最佳結果
        cv_results = self.tuning_results["cv_results"]
        results_df = pd.DataFrame(
            {
                "params": cv_results["params"],
                "mean_score": cv_results["mean_test_score"],
                "std_score": cv_results["std_test_score"],
                "rank": cv_results["rank_test_score"],
            }
        )

        top_10 = results_df.nsmallest(10, "rank")

        return {
            "best_score": self.tuning_results["best_score"],
            "best_params": self.tuning_results["best_params"],
            "top_10_results": top_10,
        }

    def visualize_tree(
        self, output_path: str = "outputs/decision_tree.png", figsize=(20, 10)
    ):
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
            class_names=["Died", "Survived"],
            filled=True,
            rounded=True,
            fontsize=10,
        )

        plt.title("Decision Tree for Titanic Survival Prediction", fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
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

    def visualize_feature_importance(
        self, output_path: str = "outputs/feature_importance.png", top_n: int = 10
    ):
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
        plt.barh(features, importances, color="steelblue")
        plt.xlabel("Importance", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        plt.title(f"Top {top_n} Feature Importance", fontsize=14)
        plt.gca().invert_yaxis()  # 最重要的在上面
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✅ 特徵重要性圖已儲存至 {output_path}")

    def save_model(self, file_path):
        # 同時儲存模型、特徵名稱和最佳參數
        model_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "best_params": self.best_params,
            "tuning_results": self.tuning_results,
        }
        joblib.dump(model_data, file_path)

    def load_model(self, file_path):
        model_data = joblib.load(file_path)
        if isinstance(model_data, dict):
            self.model = model_data["model"]
            self.feature_names = model_data.get("feature_names", None)
            self.best_params = model_data.get("best_params", None)
            self.tuning_results = model_data.get("tuning_results", None)
        else:
            # 向後兼容舊版本的模型檔案
            self.model = model_data
            self.feature_names = None
            self.best_params = None
