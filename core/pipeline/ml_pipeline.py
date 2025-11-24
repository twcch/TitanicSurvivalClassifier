import pandas as pd
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


class MLPipeline:
    def __init__(self, use_tuning: bool = True, tuning_method: str = "grid"):
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

        self.model = DecisionTreeClassifierModel(
            use_tuning=use_tuning,
            tuning_method=tuning_method,
            cv=5
        )

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

        print("使用最佳參數在全部資料上重新訓練...")
        # 建立新模型使用最佳參數
        if self.model.best_params:
            best_model = DecisionTreeClassifier(**self.model.best_params)
            best_model.fit(X, y)
            self.model.model = best_model
            if hasattr(X, 'columns'):
                self.model.feature_names = X.columns.tolist()
        else:
            self.model.train((X, y))

        self.model.save_model("outputs/decision_tree_model.pkl")
        
        # 視覺化決策樹
        print("\n" + "=" * 50)
        print("生成視覺化圖表...")
        print("=" * 50)
        self.model.visualize_tree("outputs/decision_tree.png")
        self.model.visualize_feature_importance("outputs/feature_importance.png")
        
        # 11. 印出特徵重要性
        importance = self.model.get_feature_importance(top_n=10)
        if importance:
            print("\n特徵重要性 (Top 10):")
            print("-" * 50)
            for i, (feature, score) in enumerate(importance.items(), 1):
                print(f"{i:2d}. {feature:30s}: {score:.4f}")
            print("=" * 50 + "\n")

    def run_inference_pipeline(self, model_path: str, test_path: str, output_path: str = "outputs/submission.csv"):
        # 載入測試資料
        df = self.data_loader.load_data(test_path)
        
        # 保存 PassengerId
        passenger_ids = df["PassengerId"].copy()
        
        # 移除 PassengerId 進行預測
        if "PassengerId" in df.columns:
            df = df.drop(columns=["PassengerId"])
        
        # 前處理和特徵工程
        df = self.preprocessing_pipeline.transform(df)
        df = self.feature_engineer_pipeline.transform(df)

        # 載入模型並預測
        self.model.load_model(model_path)
        predictions = self.model.predict(df)
        
        # 建立提交檔案
        submission = pd.DataFrame({
            'PassengerId': passenger_ids,
            'Survived': predictions
        })
        
        # 儲存結果
        submission.to_csv(output_path, index=False)
        print(f"\n✅ 預測完成！結果已儲存至 {output_path}")
        print(f"\n預測結果前 10 筆：")
        print(submission.head(10))
        print(f"\n總共預測 {len(submission)} 筆資料")
        print(f"預測存活人數: {submission['Survived'].sum()}")
        print(f"預測死亡人數: {(submission['Survived'] == 0).sum()}")
        
        return submission
