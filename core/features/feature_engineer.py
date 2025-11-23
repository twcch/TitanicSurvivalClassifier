import pandas as pd
from abc import ABC, abstractmethod


class BaseFeatureEngineer(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame) -> "BaseFeatureEngineer":
        """Fit the feature engineer to the data."""
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the input features."""
        pass

    @abstractmethod
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit the feature engineer to the data and transform the input features."""
        pass


class OneHotEncoder(BaseFeatureEngineer):
    def __init__(self, columns: list):
        self.columns = columns
        self.fitted = False

    def fit(self, X: pd.DataFrame) -> "OneHotEncoder":
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError(
                "The feature engineer must be fitted before calling transform."
            )
        return pd.get_dummies(X, columns=self.columns)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)


class FeatureEngineerPipeline(BaseFeatureEngineer):
    def __init__(self, steps: list):
        self.steps = steps

    def fit(self, X: pd.DataFrame) -> "FeatureEngineerPipeline":
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
            step.fit(X)
            X = step.transform(X)
        return X
