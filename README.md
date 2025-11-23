# Titanic Survival Classifier

> 機器學習實戰專案｜模組化設計 × 設定檔驅動 × 結構清晰  
> 本專案為一個以 Titanic 生存預測競賽為藍本的機器學習實戰練習，聚焦於模組化設計、工程化流程與設定檔驅動開發，展示我作為資料分析師轉型資料科學家所需之工程能力與架構設計思維

## 專案亮點 | Highlights

✅ 模組化架構: 依循業界慣例，分離 `data / features / models / utils` 等模組，利於擴充與維護  
✅ 設定檔驅動: 使用 `config.json` 管理模型參數、特徵欄位與前處理規則，一鍵切換實驗設定  
✅ 完整流程自動化: 從資料預處理、特徵建構、模型訓練到推論，全流程由 `main.py` 控制執行  
✅ 可擴充日誌紀錄系統: 每次訓練自動產生 `logs/run_yyyymmdd_HHMMSS/`，儲存 config、metrics、summary  
✅ 符合生產環境邏輯： 支援 `artifact` 儲存（如 encoder）、JSON 記錄模型設定與結果，便於部署與回溯

## 問題定義 | Problem Definition

鐵達尼號生存預測問題是經典的分類任務，其核心目標是根據乘客的個人資訊 (如性別、年齡、艙等、船票金額等) 預測其是否能在沈船事故中存活。此問題不僅可作為機器學習分類演算法的入門範例，更可延伸應用於風險預測、人群行為建模與生存分析等實務領域。

本專案旨在模擬真實專案開發流程，將資料前處理、特徵工程、模型訓練與推論流程進行模組化設計與設定檔驅動開發，並以此建立可重現、可維護、具備工程思維的機器學習專案範本。

## 專案結構 | Project Structure

```bash
TitanicSurvivalPrediction/
├── configs/                     # 設定檔 (包含模型、特徵、訓練方式)
│   └── ...                     
├── core/                        # 核心模組 (資料處理、模型、前處理、編碼)
│   ├── features/
│   │   └── one_hot_feature_encoder.py   # One-hot 編碼器封裝
│   ├── models/
│   │   └── xgboost_model.py             # XGBoost 模型封裝
│   ├── pipeline/
│   │   ├── encoding.py                 # 特徵編碼流程
│   │   ├── feature_engineering.py     # 特徵工程流程
│   │   └── preprocessing.py           # 前處理流程
│   ├── data.py                        # 資料存取與儲存 (封裝 I/O 操作)
│   ├── generate_summary.py           # 統計摘要報表產出
│   └── log_writer.py                 # 訓練與評估日誌紀錄器
├── data/                        # 資料夾
│   ├── raw/                     # 原始資料
│   ├── processed/               # 前處理後資料
│   └── features/                # 特徵工程後資料
├── notebooks/                   # Jupyter Notebook 開發草稿區
├── results/                     # 模型與輸出結果
│   ├── logs/                    # 訓練過程與評估結果紀錄
│   │   └── run_YYYYMMDD_HHMMSS/
│   ├── polts/                   # 可視化圖片輸出，如特徵分布、模型重要性圖等
│   └── v1_0_0/                  # 版本化輸出結果
│       ├── models/             # 儲存模型檔案 (*.pkl)
│       └── submission/         # 儲存提交檔案 (submission.csv)
├── scripts/                     # 主程序腳本 (可執行)
│   ├── preprocess_data.py
│   ├── build_features.py
│   ├── train_model.py
│   └── inference.py
├── main.py                      # 主控腳本 (依序執行整個 pipeline)
├── requirements.txt            # Python 套件需求清單
└── README.md                   # 專案說明文件
```

## 技術與套件 | Tech Stack

- Python 3.11
- pandas, numpy
- scikit-learn
- xgboost
- joblib (模型儲存)
- pathlib, json (設定與日誌處理)

## 執行方式 | How to Run

### 1. 安裝套件

```bash
pip install -r requirements.txt
```

### 2. 一鍵執行完整流程

```bash
python3 main.py
```

## 輸出結果 | Outputs

- 模型儲存於 `models/v1/model_xgb.pkl`
- 預測輸出於 `data/submission/submission.csv`
- 訓練紀錄自動寫入 `logs/run_YYYYMMDD_HHMMSS/`

## 訓練成果範例 | Training Results

| 指標             | 數值     |
|----------------|--------|
| Accuracy Score | 0.8379 |

## 延伸功能建議 | Extension Ideas

- 支援更多模型 (如 RandomForest、LogisticRegression)
- 加入交叉驗證、Grid Search、SHAP 模型解釋
- 加入標準化 (StandardScaler) 模組
- 將訓練與預測流程包成 CLI 工具或 API

## 延伸功能建議 | Possible Extensions

- 支援多模型訓練與結果比較 (RandomForest、Logistic Regression、LightGBM 等)
- 整合超參數搜尋 (Grid Search / Optuna / Cross Validation)
- 加入 SHAP 或 LIME 模型解釋，提升模型可解釋性與商業應用可信度
- 輸出統一報表與版本紀錄 (支援實驗管理)
- 將 pipeline 封裝為 Python Package 或 CLI 工具，提高跨專案重用性

## License

Auralytics is licensed under the Apache License 2.0. You are free to use, modify, and distribute the project, as long as you comply with the terms of the license, including proper attribution and inclusion of the license notice.

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Contact Us

If you have any questions or suggestions, feel free to reach out to us:

- Email: twcch1218 [at] gmail.com

Thank you for your interest in TitanicSurvivalClassifer! We look forward to your contributions and hope you enjoy using and improving this project.

## Notes

- Kaggle url: https://www.kaggle.com/competitions/titanic