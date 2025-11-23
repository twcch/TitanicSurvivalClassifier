from core.pipeline.ml_pipeline import MLPipeline


def main():
    ml = MLPipeline()
    ml.run_training_pipeline("data/raw/train.csv")


if __name__ == "__main__":
    main()
