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


class MLPipeline:
    def __init__(self):
        self.data_loader = DataLoader()

        preprocessing_steps = [
            MissingValueHandler(strategy="mean"),
            DropColumnsPreprocessor(columns_to_drop=["Cabin"]),
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

        self.model.train(X, y)
        self.model.save_model("decision_tree_model.pkl")

    def run_inference_pipeline(self, model_path: str, test_path: str):
        df = self.data_loader.load_data(test_path)
        df = self.preprocessing_pipeline.transform(df)
        df = self.feature_engineer_pipeline.transform(df)

        self.model.load_model(model_path)
        predictions = self.model.predict(df)

        print(predictions)
