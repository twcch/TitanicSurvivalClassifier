from core.pipeline.ml_pipeline import MLPipeline


def main():
    ml = MLPipeline()
    
    # 訓練模型
    print("=" * 50)
    print("開始訓練模型...")
    print("=" * 50)
    ml.run_training_pipeline("data/raw/train.csv")
    
    # 執行推論並輸出結果
    print("\n" + "=" * 50)
    print("開始推論...")
    print("=" * 50)
    submission = ml.run_inference_pipeline(
        model_path="outputs/decision_tree_model.pkl",
        test_path="data/raw/test.csv",
        output_path="outputs/submission.csv"
    )


if __name__ == "__main__":
    main()
