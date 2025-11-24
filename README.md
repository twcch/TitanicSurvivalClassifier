# 🚢 Titanic Survival Prediction - OOP Machine Learning Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

一個採用 **物件導向程式設計 (OOP)** 原則建構的生產級機器學習流程，用於預測鐵達尼號乘客存活率。本專案展示了如何運用 SOLID 原則、設計模式和模組化架構，打造可維護、可擴展的機器學習系統。

> 📊 **Kaggle Competition**: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

---

## 📋 目錄

- [專案特色](#專案特色)
- [系統架構](#系統架構)
- [快速開始](#快速開始)
- [專案結構](#專案結構)
- [核心組件](#核心組件)
- [模型效能](#模型效能)
- [進階用法](#進階用法)
- [實驗管理](#實驗管理)
- [擴展指南](#擴展指南)
- [開發指南](#開發指南)
- [常見問題](#常見問題)
- [授權資訊](#授權資訊)
- [聯絡方式](#聯絡方式)

---

## ✨ 專案特色

### 🏗️ 架構設計
- **物件導向設計**: 完整的 OOP 架構，遵循 SOLID 原則
- **設計模式應用**: Factory、Strategy、Pipeline、Template Method
- **模組化結構**: 高內聚低耦合的組件設計
- **可擴展性**: 輕鬆新增模型、前處理器、特徵工程器

### 🔧 功能特性
- **多模型支援**: Decision Tree、Random Forest（可輕鬆擴展）
- **自動超參數調優**: Grid Search / Random Search
- **完整評估指標**: Accuracy、Precision、Recall、F1-Score、ROC-AUC
- **實驗追蹤**: 自動記錄每次訓練的參數、指標、產出
- **視覺化**: 決策樹圖、特徵重要性圖

### 📊 資料處理
- **智能前處理**: 缺失值處理、異常值偵測
- **自動特徵工程**: One-Hot Encoding、特徵選擇
- **Pipeline 機制**: 可組合的資料轉換流程

---

## 🏛️ 系統架構

### 整體架構圖

```
┌─────────────────────────────────────────────────────────┐
│                   Main Entry Point                       │
│                     (main.py)                            │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Pipeline Orchestration Layer                │
│                  (ml_pipeline.py)                        │
│  • 流程協調  • 實驗管理  • 結果輸出                       │
└─────────────────────────────────────────────────────────┘
            │                 │                 │
    ┌───────┴────────┐  ┌────┴────┐  ┌────────┴────────┐
    ▼                ▼  ▼         ▼  ▼                 ▼
┌─────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  Data   │   │Preprocess│   │ Feature  │   │  Model   │
│  Layer  │   │  Layer   │   │ Engineer │   │  Layer   │
└─────────┘   └──────────┘   └──────────┘   └──────────┘
    │              │              │               │
    ▼              ▼              ▼               ▼
• DataLoader  • MissingValue  • OneHotEncoder • BaseModel
              • Outlier       • FamilySize    • DecisionTree
              • DropColumns                   • RandomForest
```

### 設計模式應用

| 模式 | 應用位置 | 說明 |
|------|---------|------|
| **Factory Pattern** | [`ModelFactory`](core/models/model_factory.py) | 統一創建不同類型的模型 |
| **Strategy Pattern** | [`BasePreprocessor`](core/preprocessing/preprocessor.py) | 可切換的前處理策略 |
| **Pipeline Pattern** | [`MLPipeline`](core/pipeline/ml_pipeline.py) | 串聯資料處理流程 |
| **Template Method** | [`BaseModel`](core/models/base_model.py) | 定義訓練評估流程骨架 |

### SOLID 原則體現

- **S - 單一職責**: 每個類別只負責一項功能
- **O - 開放封閉**: 對擴展開放（新增模型），對修改封閉
- **L - 里氏替換**: 所有模型都可替換 BaseModel
- **I - 介面隔離**: 清晰的抽象介面定義
- **D - 依賴反轉**: 依賴抽象類別而非具體實作

---

## 🚀 快速開始

### 環境需求

```
Python 3.8+
pip 21.0+
```

### 安裝步驟

1. **克隆專案**

```bash
git clone https://github.com/yourusername/TitanicSurvivalClassifier.git
cd TitanicSurvivalClassifier
```

2. **建立虛擬環境** (推薦)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **安裝依賴套件**

```bash
pip install -r requirements.txt
```

### 基本使用

#### 方法一：使用 main.py

```python
# 修改 main.py 中的設定
MODEL_TYPE = "decision_tree"  # 或 "random_forest"
USE_TUNING = True
TUNING_METHOD = "grid"  # 或 "random"

# 執行訓練與預測
python main.py
```

#### 方法二：程式碼調用

```python
from core.pipeline.ml_pipeline import MLPipeline

# 建立 Pipeline
pipeline = MLPipeline(
    model_type="decision_tree",
    use_tuning=True,
    tuning_method="grid"
)

# 訓練模型
pipeline.run_training_pipeline("data/raw/train.csv")

# 執行推論
pipeline.run_inference_pipeline(
    model_path="outputs/results_xxx/decision_tree_model.pkl",
    test_path="data/raw/test.csv"
)
```

### 執行結果示例

```
============================================================
🚀 開始訓練 DECISION_TREE 模型
============================================================
📊 可用模型: ['decision_tree', 'random_forest']
⚙️  超參數調優: 開啟
🔍 調優方法: GRID
============================================================

📁 建立實驗資料夾: outputs/results_decision_tree_202511240913399760
📊 使用模型: DECISION_TREE

==================================================
開始超參數調優 (GRID Search)...
==================================================
Fitting 5 folds for each of 576 candidates, totalling 2880 fits

==================================================
超參數調優完成！
==================================================
最佳交叉驗證分數: 0.8426

最佳參數:
--------------------------------------------------
  criterion                : entropy
  max_depth                : 10
  max_features             : None
  min_impurity_decrease    : 0.0
  min_samples_leaf         : 5
  min_samples_split        : 2
==================================================

==================================================
模型評估結果 (Model Evaluation Results)
==================================================
accuracy    : 0.7709
precision   : 0.7333
recall      : 0.6377
f1_score    : 0.6822
roc_auc     : 0.7803
==================================================

✅ 決策樹視覺化已儲存至 outputs/results_xxx/decision_tree_visualization.png
✅ 特徵重要性圖已儲存至 outputs/results_xxx/feature_importance.png

特徵重要性 (Top 10):
--------------------------------------------------
 1. Sex_female                      : 0.3734
 2. Fare                            : 0.2574
 3. Age                             : 0.1780
 4. Pclass_3                        : 0.1229
 5. SibSp                           : 0.0251
==================================================

✅ 預測完成！結果已儲存至 outputs/results_xxx/submission.csv
```

---

## 📁 專案結構

```
TitanicSurvivalClassifier/
├── core/                           # 核心程式庫
│   ├── data/                       # 資料載入模組
│   │   ├── __init__.py
│   │   └── data_loader.py         # 資料讀取器
│   │
│   ├── preprocessing/              # 前處理模組
│   │   ├── __init__.py
│   │   └── preprocessor.py        # 前處理器（缺失值、異常值等）
│   │
│   ├── features/                   # 特徵工程模組
│   │   ├── __init__.py
│   │   └── feature_engineer.py    # 特徵轉換器
│   │
│   ├── models/                     # 模型模組
│   │   ├── __init__.py
│   │   ├── base_model.py          # 模型抽象基底類別
│   │   ├── decision_tree_classifier_model.py
│   │   ├── random_forest_model.py
│   │   └── model_factory.py       # 模型工廠
│   │
│   └── pipeline/                   # 流程協調模組
│       ├── __init__.py
│       └── ml_pipeline.py         # 主要 Pipeline
│
├── data/                           # 資料目錄
│   └── raw/
│       ├── train.csv              # 訓練資料
│       └── test.csv               # 測試資料
│
├── outputs/                        # 輸出目錄
│   └── results_MODELTYPE_YYYYMMDDHHMMSS_xxx_N/
│       ├── decision_tree_model.pkl         # 訓練好的模型
│       ├── decision_tree_visualization.png # 決策樹圖
│       ├── feature_importance.png          # 特徵重要性圖
│       ├── experiment_report.txt           # 實驗報告
│       └── submission.csv                  # 預測結果
│
├── main.py                         # 程式進入點
├── requirements.txt                # 依賴套件清單
├── README.md                       # 專案說明文件
├── LICENSE                         # MIT 授權
└── .gitignore                      # Git 忽略清單
```

---

## 🔧 核心組件

### 1. 資料載入器 - [`DataLoader`](core/data/data_loader.py)

負責讀取和驗證資料。

```python
from core.data.data_loader import DataLoader

loader = DataLoader()
df = loader.load_data("data/raw/train.csv")
```

### 2. 前處理器 - [`PreprocessingPipeline`](core/preprocessing/preprocessor.py)

提供可組合的前處理步驟。

```python
from core.preprocessing.preprocessor import (
    PreprocessingPipeline,
    MissingValueHandler,
    DropColumnsPreprocessor
)

# 建立前處理流程
steps = [
    DropColumnsPreprocessor(
        columns_to_drop=["PassengerId", "Name", "Ticket", "Cabin"]
    ),
    MissingValueHandler(strategy="mean")
]

preprocessor = PreprocessingPipeline(steps=steps)
X_processed = preprocessor.fit_transform(X)
```

**可用的前處理器：**

| 類別 | 功能 | 參數 |
|------|------|------|
| `MissingValueHandler` | 處理缺失值 | `strategy`: "mean", "median", "mode", "drop" |
| `OutlierHandler` | 處理異常值 | `method`: "zscore", "iqr" |
| `DropColumnsPreprocessor` | 移除欄位 | `columns_to_drop`: List[str] |

### 3. 特徵工程器 - [`FeatureEngineerPipeline`](core/features/feature_engineer.py)

自動化特徵轉換。

```python
from core.features.feature_engineer import (
    FeatureEngineerPipeline,
    OneHotEncoder
)

steps = [
    OneHotEncoder(columns=["Sex", "Embarked", "Pclass"])
]

feature_engineer = FeatureEngineerPipeline(steps=steps)
X_features = feature_engineer.fit_transform(X)
```

### 4. 模型層 - [`BaseModel`](core/models/base_model.py)

所有模型的抽象基底類別。

```python
from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def train(self, data):
        """訓練模型"""
        pass
    
    @abstractmethod
    def predict(self, input_data):
        """進行預測"""
        pass
    
    @abstractmethod
    def evaluate(self, test_data):
        """評估模型"""
        pass
    
    @abstractmethod
    def save_model(self, file_path):
        """儲存模型"""
        pass
    
    @abstractmethod
    def load_model(self, file_path):
        """載入模型"""
        pass
```

### 5. 模型工廠 - [`ModelFactory`](core/models/model_factory.py)

統一的模型創建介面。

```python
from core.models.model_factory import ModelFactory

# 創建決策樹模型
model = ModelFactory.create_model(
    model_type="decision_tree",
    use_tuning=True,
    tuning_method="grid",
    cv=5
)

# 創建隨機森林模型
model = ModelFactory.create_model(
    model_type="random_forest",
    use_tuning=True,
    tuning_method="random",
    cv=5
)
```

---

## 📊 模型效能

### Decision Tree (with Hyperparameter Tuning)

| 指標 | 驗證集分數 | 說明 |
|------|-----------|------|
| **Accuracy** | 0.7709 | 整體預測準確率 |
| **Precision** | 0.7333 | 正類別預測精確度 |
| **Recall** | 0.6377 | 正類別召回率 |
| **F1-Score** | 0.6822 | Precision 與 Recall 的調和平均 |
| **ROC-AUC** | 0.7803 | ROC 曲線下面積 |

### 最佳超參數

```python
{
    'criterion': 'entropy',
    'max_depth': 10,
    'max_features': None,
    'min_impurity_decrease': 0.0,
    'min_samples_leaf': 5,
    'min_samples_split': 2
}
```

### 特徵重要性 (Top 5)

1. **Sex_female** (0.3734) - 性別是最重要的存活預測因子
2. **Fare** (0.2574) - 船票價格反映社經地位
3. **Age** (0.1780) - 年齡影響存活率
4. **Pclass_3** (0.1229) - 三等艙乘客存活率較低
5. **SibSp** (0.0251) - 兄弟姊妹/配偶數量

---

## 🎓 進階用法

### 新增自訂前處理器

```python
from core.preprocessing.preprocessor import BasePreprocessor
from sklearn.preprocessing import StandardScaler
import pandas as pd

class CustomScaler(BasePreprocessor):
    """標準化數值特徵"""
    
    def __init__(self, columns=None):
        self.columns = columns
        self.scaler = StandardScaler()
    
    def fit(self, X: pd.DataFrame):
        cols = self.columns or X.select_dtypes(include=['float64', 'int64']).columns
        self.scaler.fit(X[cols])
        return self
    
    def transform(self, X: pd.DataFrame):
        X = X.copy()
        cols = self.columns or X.select_dtypes(include=['float64', 'int64']).columns
        X[cols] = self.scaler.transform(X[cols])
        return X

# 使用方式
from core.preprocessing.preprocessor import PreprocessingPipeline

steps = [
    DropColumnsPreprocessor(columns_to_drop=["PassengerId", "Name"]),
    CustomScaler(columns=["Age", "Fare"]),  # 新增的標準化器
    MissingValueHandler(strategy="mean")
]

preprocessor = PreprocessingPipeline(steps=steps)
```

### 新增自訂特徵工程器

```python
from core.features.feature_engineer import BaseFeatureEngineer
import pandas as pd

class FamilySizeFeature(BaseFeatureEngineer):
    """創建家庭人數相關特徵"""
    
    def fit(self, X: pd.DataFrame):
        return self
    
    def transform(self, X: pd.DataFrame):
        X = X.copy()
        # 家庭總人數 = 自己 + SibSp + Parch
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        # 是否獨自一人
        X['IsAlone'] = (X['FamilySize'] == 1).astype(int)
        # 家庭人數分類
        X['FamilyCategory'] = pd.cut(
            X['FamilySize'],
            bins=[0, 1, 4, 20],
            labels=['Alone', 'Small', 'Large']
        )
        return X

# 使用方式
from core.features.feature_engineer import FeatureEngineerPipeline

steps = [
    FamilySizeFeature(),  # 新增的特徵工程器
    OneHotEncoder(columns=["Sex", "Embarked", "FamilyCategory"])
]

feature_engineer = FeatureEngineerPipeline(steps=steps)
```

### 新增自訂模型

```python
from core.models.base_model import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

class LogisticRegressionModel(BaseModel):
    """邏輯回歸模型"""
    
    def __init__(self, use_tuning=True, tuning_method="grid", cv=5):
        self.use_tuning = use_tuning
        self.tuning_method = tuning_method
        self.cv = cv
        self.feature_names = None
        self.best_params = None
        
        self.default_params = {
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'lbfgs',
            'max_iter': 200,
            'random_state': 42
        }
        
        self.param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
        
        self.model = LogisticRegression(**self.default_params)
    
    def train(self, data):
        X, y = data
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        
        if self.use_tuning:
            search = GridSearchCV(
                estimator=LogisticRegression(random_state=42, max_iter=200),
                param_grid=self.param_grid,
                cv=self.cv,
                scoring='accuracy',
                n_jobs=-1
            )
            search.fit(X, y)
            self.model = search.best_estimator_
            self.best_params = search.best_params_
        else:
            self.model.fit(X, y)
    
    def predict(self, input_data):
        return self.model.predict(input_data)
    
    def evaluate(self, test_data):
        X_test, y_test = test_data
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
    
    def save_model(self, file_path):
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'best_params': self.best_params
        }
        joblib.dump(model_data, file_path)
    
    def load_model(self, file_path):
        model_data = joblib.load(file_path)
        self.model = model_data['model']
        self.feature_names = model_data.get('feature_names')
        self.best_params = model_data.get('best_params')

# 註冊到工廠
# 修改 core/models/model_factory.py
models = {
    "decision_tree": DecisionTreeClassifierModel,
    "random_forest": RandomForestClassifierModel,
    "logistic_regression": LogisticRegressionModel,  # 新增
}
```

### 批量實驗比較

```python
from core.pipeline.ml_pipeline import MLPipeline
import pandas as pd

def compare_models():
    """比較不同模型的效能"""
    models = ["decision_tree", "random_forest"]
    results = []
    
    for model_type in models:
        print(f"\n{'='*60}")
        print(f"訓練模型: {model_type.upper()}")
        print(f"{'='*60}")
        
        # 建立 Pipeline
        pipeline = MLPipeline(
            model_type=model_type,
            use_tuning=True,
            tuning_method="grid"
        )
        
        # 訓練並取得結果
        metrics = pipeline.run_training_pipeline("data/raw/train.csv")
        
        # 記錄結果
        result = {'model': model_type}
        result.update(metrics)
        results.append(result)
    
    # 建立比較表
    df_results = pd.DataFrame(results)
    print("\n" + "="*60)
    print("模型比較結果")
    print("="*60)
    print(df_results.to_string(index=False))
    
    # 儲存比較結果
    df_results.to_csv("outputs/model_comparison.csv", index=False)
    print(f"\n✅ 比較結果已儲存至 outputs/model_comparison.csv")
    
    return df_results

if __name__ == "__main__":
    compare_models()
```

---

## 🗂️ 實驗管理

### 實驗資料夾結構

每次執行都會自動建立一個唯一的實驗資料夾：

```
outputs/
└── results_MODELTYPE_YYYYMMDDHHMMSS_xxx_N/
    ├── decision_tree_model.pkl        # 訓練好的模型
    ├── decision_tree_visualization.png # 決策樹視覺化
    ├── feature_importance.png          # 特徵重要性圖
    ├── experiment_report.txt           # 完整的實驗報告
    └── submission.csv                  # Kaggle 提交檔案
```

**資料夾命名規則:**
- `MODELTYPE`: 模型類型（decision_tree, random_forest）
- `YYYYMMDDHHMMSS`: 時間戳記（年月日時分秒）
- `xxx`: 毫秒
- `N`: 流水序號（同一毫秒內的第 N 次執行）

### 實驗報告內容

[`experiment_report.txt`](outputs/results_decision_tree_202511240913399760/experiment_report.txt) 包含：

```
============================================================
實驗報告 (Experiment Report)
============================================================

實驗時間: 2025-11-24 09:13:45
實驗資料夾: outputs/results_decision_tree_202511240913399760
模型類型: DECISION_TREE

------------------------------------------------------------
模型設定
------------------------------------------------------------
使用超參數調優: True
調優方法: grid
交叉驗證折數: 5

------------------------------------------------------------
最佳超參數
------------------------------------------------------------
  criterion                     : entropy
  max_depth                     : 10
  max_features                  : None
  min_impurity_decrease         : 0.0
  min_samples_leaf              : 5
  min_samples_split             : 2

------------------------------------------------------------
模型評估結果
------------------------------------------------------------
  accuracy       : 0.7709
  precision      : 0.7333
  recall         : 0.6377
  f1_score       : 0.6822
  roc_auc        : 0.7803

------------------------------------------------------------
特徵重要性 (Top 10)
------------------------------------------------------------
   1. Sex_female                         : 0.3734
   2. Fare                               : 0.2574
   3. Age                                : 0.1780
   ...
============================================================
```

---

## 🔍 擴展指南

### 支援的模型類型

目前支援的模型：

| 模型類型 | 類別名稱 | 檔案位置 |
|---------|---------|---------|
| Decision Tree | [`DecisionTreeClassifierModel`](core/models/decision_tree_classifier_model.py) | `core/models/decision_tree_classifier_model.py` |
| Random Forest | [`RandomForestModel`](core/models/random_forest_model.py) | `core/models/random_forest_model.py` |

**新增模型的步驟：**

1. 繼承 [`BaseModel`](core/models/base_model.py)
2. 實作所有抽象方法
3. 在 [`ModelFactory`](core/models/model_factory.py) 註冊模型
4. 在 [`main.py`](main.py) 中即可使用

### 切換模型

```python
# 在 main.py 中修改
MODEL_TYPE = "random_forest"  # 改為隨機森林

# 或在程式碼中
pipeline = MLPipeline(model_type="random_forest")
```

### 超參數調優設定

```python
# Grid Search（窮舉搜尋，準確但慢）
pipeline = MLPipeline(
    model_type="decision_tree",
    use_tuning=True,
    tuning_method="grid"
)

# Random Search（隨機搜尋，快速）
pipeline = MLPipeline(
    model_type="decision_tree",
    use_tuning=True,
    tuning_method="random"
)

# 不使用調優（使用預設參數）
pipeline = MLPipeline(
    model_type="decision_tree",
    use_tuning=False
)
```

---

## 🛠️ 開發指南

### 依賴套件

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
joblib>=1.1.0
matplotlib>=3.5.0
```

### 程式碼風格

本專案遵循 PEP 8 規範：

```bash
# 格式化程式碼
black core/ tests/

# 檢查程式碼風格
pylint core/

# 型別檢查
mypy core/
```

### 測試

```bash
# 執行所有測試
pytest tests/

# 執行特定測試
pytest tests/test_models.py

# 查看測試覆蓋率
pytest --cov=core tests/
```

### 貢獻指南

歡迎貢獻！請遵循以下步驟：

1. Fork 本專案
2. 建立功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交變更 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

**程式碼要求：**
- 遵循 PEP 8 規範
- 所有 public 方法都需要 docstring
- 新增功能需包含單元測試
- 更新相關文件

---

## ❓ 常見問題

### Q1: 如何查看所有可用的模型？

```python
from core.models.model_factory import ModelFactory

available_models = ModelFactory.get_available_models()
print(available_models)  # ['decision_tree', 'random_forest']
```

### Q2: 如何調整 Decision Tree 的深度限制？

修改 [`DecisionTreeClassifierModel`](core/models/decision_tree_classifier_model.py) 中的 `param_grid`:

```python
self.param_grid = {
    "max_depth": [3, 5, 7, 10, 15, 20, None],  # 新增更多選項
    # ...
}
```

### Q3: 如何使用自己的資料集？

1. 準備 CSV 格式資料（需包含 `Survived` 欄位）
2. 修改 [`main.py`](main.py) 中的路徑
3. 確保資料格式與 Titanic 資料集相容

### Q4: 實驗資料夾太多，如何管理？

建議定期清理舊的實驗資料夾，或使用腳本自動管理：

```python
import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta

def cleanup_old_experiments(days=30):
    """刪除超過 N 天的實驗資料夾"""
    outputs_dir = Path("outputs")
    cutoff_date = datetime.now() - timedelta(days=days)
    
    for folder in outputs_dir.glob("results_*"):
        # 從資料夾名稱解析日期
        timestamp = folder.name.split("_")[2][:14]
        folder_date = datetime.strptime(timestamp, "%Y%m%d%H%M%S")
        
        if folder_date < cutoff_date:
            shutil.rmtree(folder)
            print(f"已刪除: {folder}")

cleanup_old_experiments(days=30)
```

### Q5: 如何提交到 Kaggle？

1. 找到最新的實驗資料夾
2. 上傳 `submission.csv` 到 Kaggle
3. 查看排行榜結果

---

## 📊 效能優化建議

### 1. 加速超參數搜尋

```python
# 使用 Random Search 代替 Grid Search
pipeline = MLPipeline(
    model_type="random_forest",
    use_tuning=True,
    tuning_method="random"  # 更快
)
```

### 2. 減少超參數搜尋空間

```python
# 在模型類別中調整
self.param_grid = {
    "n_estimators": [100, 200],  # 減少選項
    "max_depth": [10, 15, 20]    # 減少選項
}
```

### 3. 使用平行運算

大多數模型已預設啟用 `n_jobs=-1`，使用所有 CPU 核心。

---

## 🗺️ 未來規劃

- [ ] 新增更多模型（XGBoost、LightGBM、CatBoost）
- [ ] 實作 Ensemble 方法（Voting、Stacking）
- [ ] 整合 Optuna 進行更智能的超參數調優
- [ ] 加入 SHAP 值分析模型可解釋性
- [ ] 建立 Streamlit Web 介面
- [ ] 整合 MLflow 進行實驗追蹤
- [ ] 實作自動特徵選擇
- [ ] 新增交叉驗證視覺化
- [ ] Docker 容器化部署

---

## 📄 授權資訊

本專案採用 MIT License 授權。

```
MIT License

Copyright (c) 2025 Chih-Chien Hsieh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
...
```

完整授權內容請見 [LICENSE](LICENSE) 檔案。

---

## 📧 聯絡方式

**作者**: Chih-Chien Hsieh

- 📧 Email: twcch1218 [at] gmail.com
- 🐙 GitHub: [@twcch](https://github.com/twcch)

---

## 🙏 致謝

- [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic) - 提供資料集和競賽平台
- [Scikit-learn](https://scikit-learn.org/) - 強大的機器學習工具
- [Python Software Foundation](https://www.python.org/) - Python 程式語言
- 所有開源貢獻者

---

## 📚 參考資源

### 官方文件
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

### 相關文章
- [Design Patterns in Python](https://refactoring.guru/design-patterns/python)
- [Machine Learning Pipeline Best Practices](https://towardsdatascience.com/)
- [SOLID Principles in Python](https://realpython.com/solid-principles-python/)

### Kaggle 資源
- [Titanic Competition Overview](https://www.kaggle.com/competitions/titanic)
- [Top Solutions](https://www.kaggle.com/competitions/titanic/discussion)
- [Feature Engineering Ideas](https://www.kaggle.com/competitions/titanic/data)

---

<div align="center">

**⭐ 如果這個專案對你有幫助，請給個 Star！**

Made with ❤️ by Chih-Chien Hsieh

</div>