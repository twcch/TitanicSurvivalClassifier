from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def train(self, data):
        """Train the model with the provided data."""
        pass

    @abstractmethod
    def predict(self, input_data):
        """Make predictions based on the input data."""
        pass

    @abstractmethod
    def evaluate(self, test_data):
        """Evaluate the model's performance on test data."""
        pass

    @abstractmethod
    def save_model(self, file_path):
        """Save the model to the specified file path."""
        pass

    @abstractmethod
    def load_model(self, file_path):
        """Load the model from the specified file path."""
        pass
