import os
from core.pipeline.ml_pipeline import MLPipeline
from core.models.model_factory import ModelFactory


def main():
    # ========== 選擇模型 ==========
    # 可用模型: "decision_tree", "random_forest"
    MODEL_TYPE = "decision_tree"  # 🔧 在這裡切換模型

    # ========== 超參數調優設定 ==========
    USE_TUNING = True
    TUNING_METHOD = "grid"  # "grid" 或 "random"

    print("=" * 60)
    print(f"🚀 開始訓練 {MODEL_TYPE.upper()} 模型")
    print("=" * 60)
    print(f"📊 可用模型: {ModelFactory.get_available_models()}")
    print(f"⚙️  超參數調優: {'開啟' if USE_TUNING else '關閉'}")
    if USE_TUNING:
        print(f"🔍 調優方法: {TUNING_METHOD.upper()}")
    print("=" * 60)

    # 建立 Pipeline
    ml = MLPipeline(
        model_type=MODEL_TYPE, use_tuning=USE_TUNING, tuning_method=TUNING_METHOD
    )

    # 訓練模型
    print("\n" + "=" * 50)
    print("開始訓練模型...")
    print("=" * 50)
    ml.run_training_pipeline("data/raw/train.csv")

    # 執行推論
    print("\n" + "=" * 50)
    print("開始推論...")
    print("=" * 50)

    model_filename = f"{MODEL_TYPE}_model.pkl"
    model_path = os.path.join(ml.experiment_dir, model_filename)

    submission = ml.run_inference_pipeline(
        model_path=model_path,
        test_path="data/raw/test.csv",
    )


if __name__ == "__main__":
    main()
