import pandas as pd
from abc import ABC, abstractmethod


class BasePreprocessor(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame) -> "BasePreprocessor":
        """Fit the preprocessor to the data."""
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data using the fitted preprocessor."""
        pass

    @abstractmethod
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit the preprocessor and transform the data."""
        return self.fit(X).transform(X)


class MissingValueHandler(BasePreprocessor):
    def __init__(self, strategy: str = "mean"):
        self.strategy = strategy
        self.fill_values = {}

    def fit(self, X: pd.DataFrame) -> "MissingValueHandler":
        for column in X.columns:
            if self.strategy == "mean":
                self.fill_values[column] = X[column].mean()
            elif self.strategy == "median":
                self.fill_values[column] = X[column].median()
            elif self.strategy == "mode":
                self.fill_values[column] = X[column].mode()[0]
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_filled = X.copy()
        for column, value in self.fill_values.items():
            X_filled[column].fillna(value, inplace=True)
        return X_filled

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


class OutilerHandler(BasePreprocessor):
    def __init__(self, method: str = "zscore", threshold: float = 3.0):
        self.method = method
        self.threshold = threshold

    def fit(self, X: pd.DataFrame) -> "OutlierHandler":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_cleaned = X.copy()
        for column in X.columns:
            if self.method == "zscore":
                mean = X[column].mean()
                std = X[column].std()
                z_scores = (X[column] - mean) / std
                X_cleaned = X_cleaned[(z_scores.abs() <= self.threshold)]
            elif self.method == "iqr":
                Q1 = X[column].quantile(0.25)
                Q3 = X[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.threshold * IQR
                upper_bound = Q3 + self.threshold * IQR
                X_cleaned = X_cleaned[
                    (X[column] >= lower_bound) & (X[column] <= upper_bound)
                ]
            else:
                raise ValueError(f"Unknown method: {self.method}")
        return X_cleaned

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


class PreprocessingPipeline(BasePreprocessor):
    def __init__(self, steps: list[BasePreprocessor]):
        self.steps = steps

    def fit(self, X: pd.DataFrame) -> "PreprocessingPipeline":
        for step in self.steps:
            step.fit(X)
            X = step.transform(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for step in self.steps:
            X = step.transform(X)
        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for step in self.steps:
            X = step.fit_transform(X)
        return X
