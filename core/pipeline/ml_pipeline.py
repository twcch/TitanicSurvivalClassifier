from core.data.data_loader import DataLoader
from core.preprocessing.preprocessor import (
    PreprocessingPipeline,
    MissingValueHandler,
    DropColumnsPreprocessor,
)
from core.features.feature_engineer import (
    FeatureEngineerPipeline,
    OneHotEncoder,
)
from core.models.decision_tree_classifier_model import DecisionTreeClassifierModel
from sklearn.model_selection import train_test_split


class MLPipeline:
    def __init__(self):
        self.data_loader = DataLoader()

        preprocessing_steps = [
            # 先移除不需要的欄位
            DropColumnsPreprocessor(columns_to_drop=["PassengerId", "Name", "Ticket", "Cabin"]),
            MissingValueHandler(strategy="mean"),
        ]

        self.preprocessing_pipeline = PreprocessingPipeline(steps=preprocessing_steps)

        feature_engineering_steps = [
            OneHotEncoder(columns=["Sex", "Embarked", "Pclass"]),
        ]

        self.feature_engineer_pipeline = FeatureEngineerPipeline(
            steps=feature_engineering_steps
        )

        self.model = DecisionTreeClassifierModel()

    def run_training_pipeline(self, train_path: str):
        df = self.data_loader.load_data(train_path)

        y = df["Survived"]
        X = df.drop(columns=["Survived"])

        X = self.preprocessing_pipeline.fit_transform(X)
        X = self.feature_engineer_pipeline.fit_transform(X)

        # 4. 分割訓練集與驗證集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 5. 訓練模型
        self.model.train((X_train, y_train))

        # 6. 評估模型
        metrics = self.model.evaluate((X_val, y_val))

        # 7. 印出評估結果
        print("\n" + "=" * 50)
        print("模型評估結果 (Model Evaluation Results)")
        print("=" * 50)
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name:12s}: {metric_value:.4f}")
        print("=" * 50 + "\n")

        self.model.train((X, y))

        self.model.save_model("decision_tree_model.pkl")

    def run_inference_pipeline(self, model_path: str, test_path: str):
        df = self.data_loader.load_data(test_path)
        df = self.preprocessing_pipeline.transform(df)
        df = self.feature_engineer_pipeline.transform(df)

        self.model.load_model(model_path)
        predictions = self.model.predict(df)

