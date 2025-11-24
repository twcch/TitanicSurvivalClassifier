import pandas as pd
import os
from datetime import datetime
from pathlib import Path
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
from core.models.model_factory import ModelFactory
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split


class MLPipeline:
    def __init__(
        self,
        model_type: str = "decision_tree",
        use_tuning: bool = True,
        tuning_method: str = "grid",
    ):
        """
        初始化 ML Pipeline

        Args:
            model_type: 模型類型 ("decision_tree", "random_forest", "xgboost", 等)
            use_tuning: 是否使用超參數調優
            tuning_method: 調優方法 ("grid" 或 "random")
        """
        self.model_type = model_type
        self.data_loader = DataLoader()

        preprocessing_steps = [
            DropColumnsPreprocessor(
                columns_to_drop=["PassengerId", "Name", "Ticket", "Cabin"]
            ),
            MissingValueHandler(strategy="mean"),
        ]

        self.preprocessing_pipeline = PreprocessingPipeline(steps=preprocessing_steps)

        feature_engineering_steps = [
            OneHotEncoder(columns=["Sex", "Embarked", "Pclass"]),
        ]

        self.feature_engineer_pipeline = FeatureEngineerPipeline(
            steps=feature_engineering_steps
        )

        # 使用工廠模式創建模型
        self.model = ModelFactory.create_model(
            model_type=model_type,
            use_tuning=use_tuning,
            tuning_method=tuning_method,
            cv=5,
        )

        # 生成唯一的實驗資料夾名稱
        self.experiment_dir = self._create_experiment_dir()

    def _create_experiment_dir(self) -> str:
        """
        建立唯一的實驗資料夾
        格式: results_MODELTYPE_YYYYMMDDHHMMSS_xxx_N
        """
        base_dir = Path("outputs")
        base_dir.mkdir(exist_ok=True)

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        milliseconds = now.strftime("%f")[:3]

        sequence = 0
        while True:
            dir_name = f"results_{self.model_type}_{timestamp}{milliseconds}{sequence}"
            experiment_path = base_dir / dir_name

            if not experiment_path.exists():
                experiment_path.mkdir(parents=True)
                print(f"\n📁 建立實驗資料夾: {experiment_path}")
                print(f"📊 使用模型: {self.model_type.upper()}")
                return str(experiment_path)

            sequence += 1

    def run_training_pipeline(self, train_path: str):
        """執行訓練流程"""
        df = self.data_loader.load_data(train_path)

        y = df["Survived"]
        X = df.drop(columns=["Survived"])

        X = self.preprocessing_pipeline.fit_transform(X)
        X = self.feature_engineer_pipeline.fit_transform(X)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 訓練模型
        self.model.train((X_train, y_train))

        # 評估模型
        metrics = self.model.evaluate((X_val, y_val))

        print("\n" + "=" * 50)
        print("模型評估結果 (Model Evaluation Results)")
        print("=" * 50)
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name:12s}: {metric_value:.4f}")
        print("=" * 50 + "\n")

        print("使用最佳參數在全部資料上重新訓練...")

        # 根據模型類型選擇對應的類別
        if self.model.best_params:
            if self.model_type == "decision_tree":
                best_model = DecisionTreeClassifier(**self.model.best_params)
            elif self.model_type == "random_forest":
                best_model = RandomForestClassifier(**self.model.best_params)
            elif self.model_type == "xgboost":
                best_model = xgb.XGBClassifier(**self.model.best_params)
            else:
                best_model = self.model.model.__class__(**self.model.best_params)

            best_model.fit(X, y)
            self.model.model = best_model
            if hasattr(X, "columns"):
                self.model.feature_names = X.columns.tolist()
        else:
            self.model.train((X, y))

        # 儲存模型
        model_filename = f"{self.model_type}_model.pkl"
        model_path = os.path.join(self.experiment_dir, model_filename)
        self.model.save_model(model_path)
        print(f"✅ 模型已儲存至 {model_path}")

        # 視覺化
        print("\n" + "=" * 50)
        print("生成視覺化圖表...")
        print("=" * 50)

        tree_filename = f"{self.model_type}_visualization.png"
        tree_path = os.path.join(self.experiment_dir, tree_filename)
        importance_path = os.path.join(self.experiment_dir, "feature_importance.png")

        self.model.visualize_tree(tree_path)
        self.model.visualize_feature_importance(importance_path)
        
        # XGBoost 額外的視覺化
        if self.model_type == "xgboost":
            xgb_importance_path = os.path.join(
                self.experiment_dir, "xgboost_importance_plot.png"
            )
            self.model.plot_importance(xgb_importance_path)

        # 特徵重要性
        importance = self.model.get_feature_importance(top_n=10)
        if importance:
            print("\n特徵重要性 (Top 10):")
            print("-" * 50)
            for i, (feature, score) in enumerate(importance.items(), 1):
                print(f"{i:2d}. {feature:30s}: {score:.4f}")
            print("=" * 50 + "\n")

        # 儲存實驗報告
        self._save_experiment_report(metrics, importance)

        return metrics

    def run_inference_pipeline(
        self, model_path: str, test_path: str, output_path: str = None
    ):
        """執行推論流程"""
        if output_path is None:
            output_path = os.path.join(self.experiment_dir, "submission.csv")

        df = self.data_loader.load_data(test_path)
        passenger_ids = df["PassengerId"].copy()

        if "PassengerId" in df.columns:
            df = df.drop(columns=["PassengerId"])

        df = self.preprocessing_pipeline.transform(df)
        df = self.feature_engineer_pipeline.transform(df)

        self.model.load_model(model_path)
        predictions = self.model.predict(df)

        submission = pd.DataFrame(
            {"PassengerId": passenger_ids, "Survived": predictions}
        )

        submission.to_csv(output_path, index=False)
        print(f"\n✅ 預測完成！結果已儲存至 {output_path}")
        print(f"\n預測結果前 10 筆：")
        print(submission.head(10))
        print(f"\n總共預測 {len(submission)} 筆資料")
        print(f"預測存活人數: {submission['Survived'].sum()}")
        print(f"預測死亡人數: {(submission['Survived'] == 0).sum()}")

        return submission

    def _save_experiment_report(self, metrics: dict, feature_importance: dict):
        """儲存實驗報告"""
        report_path = os.path.join(self.experiment_dir, "experiment_report.txt")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("實驗報告 (Experiment Report)\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"實驗時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"實驗資料夾: {self.experiment_dir}\n")
            f.write(f"模型類型: {self.model_type.upper()}\n\n")

            f.write("-" * 60 + "\n")
            f.write("模型設定\n")
            f.write("-" * 60 + "\n")
            f.write(f"使用超參數調優: {self.model.use_tuning}\n")
            if self.model.use_tuning:
                f.write(f"調優方法: {self.model.tuning_method}\n")
                f.write(f"交叉驗證折數: {self.model.cv}\n")
            f.write("\n")

            if self.model.best_params:
                f.write("-" * 60 + "\n")
                f.write("最佳超參數\n")
                f.write("-" * 60 + "\n")
                for param, value in self.model.best_params.items():
                    f.write(f"  {param:30s}: {value}\n")
                f.write("\n")

            f.write("-" * 60 + "\n")
            f.write("模型評估結果\n")
            f.write("-" * 60 + "\n")
            for metric_name, metric_value in metrics.items():
                f.write(f"  {metric_name:15s}: {metric_value:.4f}\n")
            f.write("\n")

            if feature_importance:
                f.write("-" * 60 + "\n")
                f.write("特徵重要性 (Top 10)\n")
                f.write("-" * 60 + "\n")
                for i, (feature, score) in enumerate(feature_importance.items(), 1):
                    f.write(f"  {i:2d}. {feature:35s}: {score:.4f}\n")
                f.write("\n")

            f.write("=" * 60 + "\n")

        print(f"✅ 實驗報告已儲存至 {report_path}")