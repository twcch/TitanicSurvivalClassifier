import joblib
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from core.models.base_model import BaseModel


class XGBoostModel(BaseModel):
    def __init__(
        self, use_tuning: bool = True, tuning_method: str = "grid", cv: int = 5
    ):
        """
        初始化 XGBoost 模型

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
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "gamma": 0,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss",
        }

        # 超參數搜尋空間
        self.param_grid = {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 3, 5],
            "gamma": [0, 0.1, 0.2],
            "reg_alpha": [0, 0.01, 0.1],
            "reg_lambda": [1, 1.5, 2],
        }

        # 初始化模型
        self.model = xgb.XGBClassifier(**self.default_params)

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
                    estimator=xgb.XGBClassifier(
                        random_state=42,
                        n_jobs=-1,
                        eval_metric="logloss"
                    ),
                    param_grid=self.param_grid,
                    cv=self.cv,
                    scoring="accuracy",
                    n_jobs=-1,
                    verbose=1,
                )
            elif self.tuning_method == "random":
                search = RandomizedSearchCV(
                    estimator=xgb.XGBClassifier(
                        random_state=42,
                        n_jobs=-1,
                        eval_metric="logloss"
                    ),
                    param_distributions=self.param_grid,
                    n_iter=50,
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
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # 計算各項指標
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }

        return metrics

    def get_feature_importance(self, top_n: int = 10):
        """取得特徵重要性"""
        if self.feature_names is None:
            return None

        # XGBoost 使用 feature_importances_ 屬性
        importances = self.model.feature_importances_
        feature_importance = dict(zip(self.feature_names, importances))

        sorted_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[
                :top_n
            ]
        )

        return sorted_importance

    def visualize_feature_importance(
        self, output_path: str = "outputs/feature_importance.png", top_n: int = 10
    ):
        """視覺化特徵重要性"""
        importance_dict = self.get_feature_importance(top_n)

        if importance_dict is None:
            print("⚠️  無法取得特徵重要性")
            return

        features = list(importance_dict.keys())
        importances = list(importance_dict.values())

        plt.figure(figsize=(10, 6))
        plt.barh(features, importances, color="darkorange")
        plt.xlabel("Importance", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        plt.title(f"Top {top_n} Feature Importance (XGBoost)", fontsize=14)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✅ 特徵重要性圖已儲存至 {output_path}")

    def visualize_tree(
        self, output_path: str = "outputs/xgboost_tree.png", figsize=(20, 10)
    ):
        """視覺化 XGBoost 的第一棵樹"""
        plt.figure(figsize=figsize)

        # 視覺化第一棵樹
        xgb.plot_tree(
            self.model,
            num_trees=0,  # 第一棵樹
            rankdir="LR",  # 左到右的布局
        )

        plt.title(
            "First Tree in XGBoost for Titanic Survival Prediction",
            fontsize=16,
            pad=20,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✅ XGBoost 第一棵樹視覺化已儲存至 {output_path}")

    def plot_importance(
        self, output_path: str = "outputs/xgboost_importance_plot.png"
    ):
        """使用 XGBoost 內建的特徵重要性圖"""
        plt.figure(figsize=(10, 8))
        xgb.plot_importance(
            self.model,
            max_num_features=10,
            importance_type="weight",  # 可選: 'weight', 'gain', 'cover'
            title="XGBoost Feature Importance",
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✅ XGBoost 特徵重要性圖已儲存至 {output_path}")

    def save_model(self, file_path):
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
            self.model = model_data
            self.feature_names = None
            self.best_params = None