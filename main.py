from core.pipeline.ml_pipeline import MLPipeline


def main():
    ml = MLPipeline()
    ml.run_training_pipeline("data/raw/train.csv")
    ml.run_inference_pipeline(
        model_path="outputs/decision_tree_model.pkl", test_path="data/raw/test.csv"
    )


if __name__ == "__main__":
    main()
