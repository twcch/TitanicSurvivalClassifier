from core.models.decision_tree_classifier_model import DecisionTreeClassifierModel
from core.models.random_forest_classifier_model import RandomForestClassifierModel
from core.models.base_model import BaseModel


class ModelFactory:
    """模型工廠類別，用於創建不同類型的模型"""

    @staticmethod
    def create_model(
        model_type: str,
        use_tuning: bool = True,
        tuning_method: str = "grid",
        cv: int = 5,
    ) -> BaseModel:
        """
        創建指定類型的模型

        Args:
            model_type: 模型類型 ("decision_tree", "random_forest", "xgboost", 等)
            use_tuning: 是否使用超參數調優
            tuning_method: 調優方法 ("grid" 或 "random")
            cv: 交叉驗證折數

        Returns:
            BaseModel: 指定類型的模型實例

        Raises:
            ValueError: 當模型類型不支援時
        """
        models = {
            "decision_tree": DecisionTreeClassifierModel,
            "random_forest": RandomForestClassifierModel,
            # 未來可以繼續添加
            # "xgboost": XGBoostModel,
            # "logistic_regression": LogisticRegressionModel,
            # "svm": SVMModel,
        }

        if model_type not in models:
            raise ValueError(
                f"不支援的模型類型: {model_type}. "
                f"支援的模型: {list(models.keys())}"
            )

        return models[model_type](
            use_tuning=use_tuning, tuning_method=tuning_method, cv=cv
        )

    @staticmethod
    def get_available_models():
        """取得所有可用的模型類型"""
        return ["decision_tree", "random_forest"]