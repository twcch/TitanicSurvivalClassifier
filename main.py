from core.pipeline.ml_pipeline import MLPipeline


ml = MLPipeline()
ml.run_training_pipeline("data/raw/train.csv")

