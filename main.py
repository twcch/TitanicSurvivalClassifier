import os
from core.pipeline.ml_pipeline import MLPipeline


def main():
    # 使用超參數調優 (Grid Search)
    # 若要使用 Random Search (更快但可能略不準確)
    # ml = MLPipeline(use_tuning=True, tuning_method="random")
    # 若要關閉超參數調優
    # ml = MLPipeline(use_tuning=False)
    ml = MLPipeline(use_tuning=True, tuning_method="grid")

    # 訓練模型
    print("=" * 50)
    print("開始訓練模型...")
    print("=" * 50)
    ml.run_training_pipeline("data/raw/train.csv")

    # 執行推論並輸出結果
    print("\n" + "=" * 50)
    print("開始推論...")
    print("=" * 50)

    # 模型路徑會自動從實驗資料夾讀取
    model_path = os.path.join(ml.experiment_dir, "decision_tree_model.pkl")

    submission = ml.run_inference_pipeline(
        model_path=model_path,
        test_path="data/raw/test.csv",
        # output_path 會自動設定到實驗資料夾中
    )


if __name__ == "__main__":
    main()
