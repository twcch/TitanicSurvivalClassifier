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
from core.models.decision_tree_classifier_model import DecisionTreeClassifierModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


class MLPipeline:
    def __init__(self, use_tuning: bool = True, tuning_method: str = "grid"):
        self.data_loader = DataLoader()

        preprocessing_steps = [
            # 先移除不需要的欄位
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

        self.model = DecisionTreeClassifierModel(
            use_tuning=use_tuning, tuning_method=tuning_method, cv=5
        )

        # 生成唯一的實驗資料夾名稱
        self.experiment_dir = self._create_experiment_dir()

    def _create_experiment_dir(self) -> str:
        """
        建立唯一的實驗資料夾
        格式: results_YYYYMMDDHHMMSS_xxx_N
        xxx: 毫秒
        N: 流水序號

        Returns:
            str: 實驗資料夾路徑
        """
        base_dir = Path("outputs")
        base_dir.mkdir(exist_ok=True)

        # 生成時間戳記 (含毫秒)
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        milliseconds = now.strftime("%f")[:3]  # 取前三位毫秒

        # 找出同一秒內的流水序號
        sequence = 0
        while True:
            dir_name = f"results_{timestamp}{milliseconds}{sequence}"
            experiment_path = base_dir / dir_name

            if not experiment_path.exists():
                experiment_path.mkdir(parents=True)
                print(f"\n📁 建立實驗資料夾: {experiment_path}")
                return str(experiment_path)

            sequence += 1

    def run_training_pipeline(self, train_path: str):
        """
        執行訓練流程

        Args:
            train_path: 訓練資料路徑
        """
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
            if hasattr(X, "columns"):
                self.model.feature_names = X.columns.tolist()
        else:
            self.model.train((X, y))

        # 8. 儲存模型到實驗資料夾
        model_path = os.path.join(self.experiment_dir, "decision_tree_model.pkl")
        self.model.save_model(model_path)
        print(f"✅ 模型已儲存至 {model_path}")

        # 9. 視覺化決策樹
        print("\n" + "=" * 50)
        print("生成視覺化圖表...")
        print("=" * 50)
        tree_path = os.path.join(self.experiment_dir, "decision_tree.png")
        importance_path = os.path.join(self.experiment_dir, "feature_importance.png")

        self.model.visualize_tree(tree_path)
        self.model.visualize_feature_importance(importance_path)

        # 10. 印出特徵重要性
        importance = self.model.get_feature_importance(top_n=10)
        if importance:
            print("\n特徵重要性 (Top 10):")
            print("-" * 50)
            for i, (feature, score) in enumerate(importance.items(), 1):
                print(f"{i:2d}. {feature:30s}: {score:.4f}")
            print("=" * 50 + "\n")

        # 11. 儲存實驗報告
        self._save_experiment_report(metrics, importance)

        return metrics

    def run_inference_pipeline(
        self, model_path: str, test_path: str, output_path: str = None
    ):
        """
        執行推論流程

        Args:
            model_path: 模型路徑（如果是相對路徑，會在實驗資料夾中尋找）
            test_path: 測試資料路徑
            output_path: 輸出路徑（如果為 None，會自動在實驗資料夾中生成）
        """
        # 如果沒有指定輸出路徑，使用實驗資料夾
        if output_path is None:
            output_path = os.path.join(self.experiment_dir, "submission.csv")

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
        submission = pd.DataFrame(
            {"PassengerId": passenger_ids, "Survived": predictions}
        )

        # 儲存結果
        submission.to_csv(output_path, index=False)
        print(f"\n✅ 預測完成！結果已儲存至 {output_path}")
        print(f"\n預測結果前 10 筆：")
        print(submission.head(10))
        print(f"\n總共預測 {len(submission)} 筆資料")
        print(f"預測存活人數: {submission['Survived'].sum()}")
        print(f"預測死亡人數: {(submission['Survived'] == 0).sum()}")

        return submission

    def _save_experiment_report(self, metrics: dict, feature_importance: dict):
        """
        儲存實驗報告

        Args:
            metrics: 評估指標
            feature_importance: 特徵重要性
        """
        report_path = os.path.join(self.experiment_dir, "experiment_report.txt")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("實驗報告 (Experiment Report)\n")
            f.write("=" * 60 + "\n\n")

            # 實驗資訊
            f.write(f"實驗時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"實驗資料夾: {self.experiment_dir}\n\n")

            # 模型設定
            f.write("-" * 60 + "\n")
            f.write("模型設定\n")
            f.write("-" * 60 + "\n")
            f.write(f"使用超參數調優: {self.model.use_tuning}\n")
            if self.model.use_tuning:
                f.write(f"調優方法: {self.model.tuning_method}\n")
                f.write(f"交叉驗證折數: {self.model.cv}\n")
            f.write("\n")

            # 最佳參數
            if self.model.best_params:
                f.write("-" * 60 + "\n")
                f.write("最佳超參數\n")
                f.write("-" * 60 + "\n")
                for param, value in self.model.best_params.items():
                    f.write(f"  {param:30s}: {value}\n")
                f.write("\n")

            # 評估指標
            f.write("-" * 60 + "\n")
            f.write("模型評估結果\n")
            f.write("-" * 60 + "\n")
            for metric_name, metric_value in metrics.items():
                f.write(f"  {metric_name:15s}: {metric_value:.4f}\n")
            f.write("\n")

            # 特徵重要性
            if feature_importance:
                f.write("-" * 60 + "\n")
                f.write("特徵重要性 (Top 10)\n")
                f.write("-" * 60 + "\n")
                for i, (feature, score) in enumerate(feature_importance.items(), 1):
                    f.write(f"  {i:2d}. {feature:35s}: {score:.4f}\n")
                f.write("\n")

            f.write("=" * 60 + "\n")

        print(f"✅ 實驗報告已儲存至 {report_path}")
