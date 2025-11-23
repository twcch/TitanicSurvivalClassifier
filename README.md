# Titanic Survival Classifier

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready machine learning pipeline for predicting Titanic passenger survival using Object-Oriented Programming (OOP) design patterns.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Pipeline Components](#pipeline-components)
- [Model Performance](#model-performance)
- [Advanced Usage](#advanced-usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

This project implements a complete machine learning pipeline for the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic) using **Object-Oriented Design Principles**. The architecture is designed for:

- ✅ **Modularity**: Each component is independent and reusable
- ✅ **Extensibility**: Easy to add new preprocessing steps, features, or models
- ✅ **Maintainability**: Clear separation of concerns with SOLID principles
- ✅ **Production-Ready**: Includes model persistence, evaluation metrics, and logging

## ✨ Features

- 🏗️ **OOP Architecture**: Clean, modular design with abstract base classes
- 🔄 **Pipeline Pattern**: Composable preprocessing and feature engineering steps
- 📊 **Multiple Models**: Support for Decision Tree, Random Forest, XGBoost, etc.
- 📈 **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- 💾 **Model Persistence**: Save and load trained models
- 🎨 **One-Hot Encoding**: Automatic categorical feature encoding
- 🔧 **Missing Value Handling**: Multiple strategies (mean, median, mode)
- 📝 **Logging Support**: Track experiments and model performance

## 🏛️ Project Architecture

The project follows **Clean Architecture** principles with clear separation between layers:

```
┌─────────────────────────────────────────┐
│         Pipeline Orchestration          │
│         (ml_pipeline.py)                │
└─────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│   Data   │  │  Model   │  │Evaluation│
│  Layer   │  │  Layer   │  │  Layer   │
└──────────┘  └──────────┘  └──────────┘
        │           │           │
        ▼           ▼           ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│Preprocess│  │ Feature  │  │ Encoding │
│  Layer   │  │Engineer  │  │  Layer   │
└──────────┘  └──────────┘  └──────────┘
```

### Design Patterns Used

- **Strategy Pattern**: Interchangeable preprocessing and feature engineering strategies
- **Pipeline Pattern**: Sequential data transformations
- **Template Method Pattern**: Base classes define workflow, subclasses implement specifics
- **Factory Pattern**: Model creation and instantiation

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/TitanicSurvivalClassifier.git
cd TitanicSurvivalClassifier
```

2. **Create virtual environment** (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

### Required Packages

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
joblib>=1.1.0
```

## 🎬 Quick Start

### Training a Model

```python
from core.pipeline.ml_pipeline import MLPipeline

# Initialize pipeline
pipeline = MLPipeline()

# Train model
pipeline.run_training_pipeline("data/raw/train.csv")
```

### Making Predictions

```python
# Run inference on test data
submission = pipeline.run_inference_pipeline(
    model_path="outputs/decision_tree_model.pkl",
    test_path="data/raw/test.csv",
    output_path="outputs/submission.csv"
)
```

### Complete Example

```bash
python main.py
```

**Expected Output:**

```
==================================================
開始訓練模型...
==================================================

==================================================
模型評估結果 (Model Evaluation Results)
==================================================
accuracy    : 0.8324
precision   : 0.8156
recall      : 0.7234
f1_score    : 0.7667
roc_auc     : 0.8891
==================================================

✅ 預測完成！結果已儲存至 outputs/submission.csv
```

## 📁 Project Structure

```
TitanicSurvivalClassifier/
├── core/
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py              # Data loading utilities
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── preprocessor.py             # Preprocessing strategies
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineer.py         # Feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py               # Abstract model interface
│   │   └── decision_tree_classifier_model.py
│   └── pipeline/
│       ├── __init__.py
│       └── ml_pipeline.py              # Pipeline orchestration
├── data/
│   └── raw/
│       ├── train.csv                    # Training data
│       └── test.csv                     # Test data
├── outputs/
│   ├── decision_tree_model.pkl         # Saved model
│   └── submission.csv                  # Predictions
├── main.py                              # Entry point
├── requirements.txt
└── README.md
```

## 🔧 Pipeline Components

### 1. Data Loading ([`DataLoader`](core/data/data_loader.py))

```python
from core.data.data_loader import DataLoader

loader = DataLoader()
df = loader.load_data("data/raw/train.csv")
```

### 2. Preprocessing ([`PreprocessingPipeline`](core/preprocessing/preprocessor.py))

```python
from core.preprocessing.preprocessor import (
    PreprocessingPipeline,
    MissingValueHandler,
    DropColumnsPreprocessor
)

preprocessing_steps = [
    DropColumnsPreprocessor(columns_to_drop=["PassengerId", "Name", "Ticket", "Cabin"]),
    MissingValueHandler(strategy="mean")
]

preprocessor = PreprocessingPipeline(steps=preprocessing_steps)
X_processed = preprocessor.fit_transform(X)
```

**Available Preprocessing Strategies:**
- `MissingValueHandler`: Handle missing values (mean, median, mode, drop)
- `OutlierHandler`: Detect and handle outliers (z-score, IQR)
- `DropColumnsPreprocessor`: Remove unnecessary columns

### 3. Feature Engineering ([`FeatureEngineerPipeline`](core/features/feature_engineer.py))

```python
from core.features.feature_engineer import (
    FeatureEngineerPipeline,
    OneHotEncoder
)

feature_steps = [
    OneHotEncoder(columns=["Sex", "Embarked", "Pclass"])
]

feature_engineer = FeatureEngineerPipeline(steps=feature_steps)
X_features = feature_engineer.fit_transform(X)
```

### 4. Model Training ([`DecisionTreeClassifierModel`](core/models/decision_tree_classifier_model.py))

```python
from core.models.decision_tree_classifier_model import DecisionTreeClassifierModel

model = DecisionTreeClassifierModel()
model.train((X_train, y_train))
metrics = model.evaluate((X_val, y_val))
model.save_model("outputs/model.pkl")
```

## 📊 Model Performance

### Evaluation Metrics

The model is evaluated using multiple metrics:

| Metric | Score | Description |
|--------|-------|-------------|
| **Accuracy** | 0.8324 | Overall prediction accuracy |
| **Precision** | 0.8156 | Positive prediction accuracy |
| **Recall** | 0.7234 | True positive detection rate |
| **F1-Score** | 0.7667 | Harmonic mean of precision/recall |
| **ROC-AUC** | 0.8891 | Area under ROC curve |

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model.model, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

## 🎓 Advanced Usage

### Adding Custom Preprocessing Step

```python
from core.preprocessing.preprocessor import BasePreprocessor

class CustomScaler(BasePreprocessor):
    def __init__(self):
        self.scaler = StandardScaler()
    
    def fit(self, X):
        self.scaler.fit(X)
        return self
    
    def transform(self, X):
        return pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
```

### Adding New Model

```python
from core.models.base_model import BaseModel
from sklearn.ensemble import RandomForestClassifier

class RandomForestModel(BaseModel):
    def __init__(self, n_estimators=100):
        self.model = RandomForestClassifier(n_estimators=n_estimators)
    
    def train(self, data):
        X, y = data
        self.model.fit(X, y)
    
    def predict(self, input_data):
        return self.model.predict(input_data)
    
    def evaluate(self, test_data):
        # Implementation similar to DecisionTreeClassifierModel
        pass
```

### Custom Feature Engineering

```python
from core.features.feature_engineer import BaseFeatureEngineer

class FamilySizeFeature(BaseFeatureEngineer):
    def fit(self, X):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        X['IsAlone'] = (X['FamilySize'] == 1).astype(int)
        return X
```

## 🔄 Continuous Integration

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black core/ tests/

# Lint code
pylint core/

# Type checking
mypy core/
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Write docstrings for all public methods
- Add unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Chih-Chien Hsieh**
- Email: twcch1218 [at] gmail.com
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

## 🙏 Acknowledgments

- [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic) for the dataset
- Scikit-learn for machine learning tools
- The open-source community for inspiration

## 📚 Additional Resources

- [Kaggle Competition Page](https://www.kaggle.com/competitions/titanic)
- [Project Documentation](docs/)
- [API Reference](docs/api/)
- [Tutorial Notebooks](notebooks/)

## 🗺️ Roadmap

- [ ] Add support for ensemble models
- [ ] Implement hyperparameter tuning with Optuna
- [ ] Add SHAP values for model interpretation
- [ ] Create web interface with Streamlit
- [ ] Add experiment tracking with MLflow
- [ ] Implement automated feature selection
- [ ] Add Docker support for deployment

## License

Auralytics is licensed under the Apache License 2.0. You are free to use, modify, and distribute the project, as long as you comply with the terms of the license, including proper attribution and inclusion of the license notice.

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Contact Us

If you have any questions or suggestions, feel free to reach out to us:

- Email: twcch1218 [at] gmail.com

Thank you for your interest in TitanicSurvivalClassifer! We look forward to your contributions and hope you enjoy using and improving this project.

## Notes

- Kaggle url: https://www.kaggle.com/competitions/titanic