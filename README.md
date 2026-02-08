# 🚀 NVIDIA Stock Price Analysis & Forecasting (2016-2026)

## 📊 A Production-Ready ML Pipeline for Time-Series Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🎯 Project Objectives

This project demonstrates **professional-grade machine learning engineering** with a focus on:

1. ✅ **Data Leakage Prevention**: Building robust pipelines that respect temporal dependencies
2. ✅ **Time-Series Best Practices**: Proper cross-validation, feature engineering, and evaluation
3. ✅ **Production-Ready Code**: Modular, testable, and maintainable codebase
4. ✅ **Reproducibility**: Version control, environment management, and experiment tracking
5. ✅ **Real-World Skills**: Solving problems that matter in 2026 ML production systems

---

## ⚠️ Critical Learning Points

### **What Makes This Different from Amateur Projects?**

| ❌ Amateur Approach | ✅ Professional Approach (This Project) |
|---------------------|----------------------------------------|
| One messy notebook | Modular, testable code structure |
| No train-test split awareness | Strict temporal ordering in splits |
| Feature engineering with leakage | Leakage detection and prevention |
| "Good" accuracy is enough | Understanding WHY metrics matter |
| Copy-paste from tutorials | First principles engineering |

### **The Data Leakage Problem (Our Main Focus)**

**Real-world scenario**: You build a model with 99.9% accuracy in development, deploy it to production, and it fails catastrophically. Why?

**Answer**: Data leakage. Your model "cheated" by seeing future information during training.

**This project teaches you to**:
- Detect subtle forms of leakage
- Build pipelines that prevent leakage by design
- Validate models using proper time-series techniques
- Think like a senior ML engineer, not just a Kaggle competitor

---

## 📁 Project Structure

```
nvidia-daily-stock-analysis-prices-(2016-2026)/
├── data/                   # Data directory (gitignored)
│   ├── raw/               # Original NVDA_yfinance_clean.csv
│   ├── processed/         # Cleaned, validated data
│   └── external/          # Additional data sources
│
├── notebooks/             # Jupyter notebooks for exploration
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_advanced_models.ipynb
│   └── 05_data_leakage_audit.ipynb  # 🔥 CRITICAL
│
├── src/                   # Production-ready source code
│   ├── config/           # Configuration management
│   ├── data/             # Data loading and validation
│   ├── features/         # Feature engineering (leakage-safe)
│   ├── models/           # Model definitions
│   ├── evaluation/       # Metrics and validation
│   └── utils/            # Helper functions
│
├── tests/                # Unit tests (pytest)
│   └── test_leakage_detection.py  # 🔥 CRITICAL
│
├── scripts/              # Executable scripts
│   ├── train.py
│   ├── evaluate.py
│   └── detect_leakage.py  # 🔥 CRITICAL
│
└── requirements.txt      # Python dependencies
```

---

## 🛠️ Setup Instructions

### **1. Clone the Repository**

```bash
git clone <your-repo-url>
cd nvidia-daily-stock-analysis-prices-(2016-2026)
```

### **2. Create Virtual Environment**

```bash
# Using venv (recommended)
python3.10 -m venv venv
source venv/bin/activate  # On Ubuntu

# Or using conda
conda create -n nvidia-stock python=3.10
conda activate nvidia-stock
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4. Download Dataset**

Place `NVDA_yfinance_clean.csv` in `data/raw/`

```bash
# If using Kaggle API
kaggle datasets download -d <dataset-id> -p data/raw/
unzip data/raw/*.zip -d data/raw/
```

### **5. Run Initial Setup**

```bash
# Initialize git (if not cloned)
git init
git add .
git commit -m "Initial commit: Project structure"

# Run tests to verify setup
pytest tests/ -v
```

---

## 🚀 Quick Start

### **Run the Full Pipeline**

```bash
# 1. Data validation and preprocessing
python scripts/train.py --mode validate

# 2. Detect data leakage (IMPORTANT!)
python scripts/detect_leakage.py --data data/raw/NVDA_yfinance_clean.csv

# 3. Train models
python scripts/train.py --model xgboost --cv-splits 5

# 4. Evaluate
python scripts/evaluate.py --model-path models/xgboost_final.pkl
```

### **Interactive Exploration**

```bash
# Launch Jupyter
jupyter lab

# Open notebooks in order:
# 01 → 02 → 03 → 04 → 05 (Data Leakage Audit)
```

---

## 📚 Learning Path

### **Phase 1: Understanding the Data (Week 1)**
- [ ] Run `01_eda.ipynb` - Understand NVIDIA stock patterns
- [ ] Learn about OHLCV data structure
- [ ] Identify trends, seasonality, volatility

### **Phase 2: Feature Engineering (Week 2)**
- [ ] Study `02_feature_engineering.ipynb`
- [ ] Understand lag features and why they're dangerous
- [ ] Learn to create leakage-safe features
- [ ] **KEY**: Understand what "future information" means

### **Phase 3: Baseline Models (Week 3)**
- [ ] Run `03_baseline_models.ipynb`
- [ ] Implement naive forecasting
- [ ] Understand baseline metrics
- [ ] Learn proper time-series train-test splits

### **Phase 4: Advanced Models (Week 4)**
- [ ] Study `04_advanced_models.ipynb`
- [ ] Implement XGBoost with proper validation
- [ ] Understand walk-forward validation
- [ ] Compare against baselines

### **Phase 5: Data Leakage Audit (Week 5)** 🔥 **MOST IMPORTANT**
- [ ] Run `05_data_leakage_audit.ipynb`
- [ ] Deliberately introduce leakage
- [ ] Learn to detect it
- [ ] Understand why 99.9% accuracy can be a red flag

---

## 🧪 Critical Experiments

### **Experiment 1: The Leakage Detector**

```python
# This is what separates amateurs from professionals
from src.evaluation.validators import DataLeakageDetector

detector = DataLeakageDetector(df, target='Close')
report = detector.run_full_audit()

if report['leakage_detected']:
    print("🚨 LEAKAGE FOUND!")
    print(report['details'])
```

### **Experiment 2: Temporal Cross-Validation**

```python
from src.evaluation.validators import TimeSeriesCrossValidator

cv = TimeSeriesCrossValidator(n_splits=5, test_size=60)
scores = cv.validate(model, X, y)

# Compare against non-temporal CV (will show why it's wrong)
```

---

## 📊 Key Metrics & What They Mean

| Metric | Good Value | Warning Signs |
|--------|-----------|---------------|
| **R² (Test)** | 0.3 - 0.7 | > 0.95 (likely leakage), < 0 (worse than baseline) |
| **MAE** | Context-dependent | Should be >> 0 for volatile stocks |
| **RMSE** | Context-dependent | Should be similar to MAE for time-series |
| **Directional Accuracy** | > 55% | 100% (definitely leakage) |

**Pro Tip**: In time-series, a model with R² = 0.5 that generalizes is better than R² = 0.99 that fails in production.

---

## 🔍 Data Leakage Examples (What to Watch For)

### **Example 1: Feature Scaling Before Split**
```python
# ❌ WRONG (leakage)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Uses info from test set
X_train, X_test = train_test_split(X_scaled)

# ✅ CORRECT
X_train, X_test = train_test_split(X)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Only transform
```

### **Example 2: Lag Features (Most Common in Time-Series)**
```python
# ❌ WRONG (leakage)
df['price_lag_1'] = df['Close'].shift(1)  # If done after split
# At test time, lag_1 might contain "future" info

# ✅ CORRECT
# Create lags BEFORE split, ensuring proper temporal order
```

### **Example 3: Target Encoding**
```python
# ❌ WRONG
df['Close_mean_by_month'] = df.groupby('Month')['Close'].transform('mean')
# This uses the mean of ALL months, including future ones

# ✅ CORRECT
# Use expanding window or ensure no future info leaks
```

---

## 🎓 Advanced Topics

### **1. Walk-Forward Validation**
- Instead of single train-test split, use rolling windows
- Simulates real production scenario
- More realistic performance estimates

### **2. Feature Importance Analysis**
- Understand which features actually matter
- Detect features that might be proxies for the target
- Build trust in your model

### **3. Production Deployment**
- Save models with versioning
- Create inference pipelines
- Monitor for data drift

---

## 🤝 Contributing

This project is a learning resource. Contributions that improve clarity, add better examples, or fix bugs are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📖 Resources

- [Time Series Forecasting Best Practices](https://www.kaggle.com/learn/time-series)
- [Data Leakage in ML](https://machinelearningmastery.com/data-leakage-machine-learning/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [MLflow for Experiment Tracking](https://mlflow.org/)

---

## 📝 License

MIT License - See LICENSE file for details

---

## 👨‍💻 Author

**Your Name**
- LinkedIn: [Your Profile]
- GitHub: [Your Profile]
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- NVIDIA stock data from Yahoo Finance
- Inspired by real-world ML engineering challenges
- Built to help aspiring ML engineers avoid common pitfalls

---

## 🔥 Final Note

**If you learn ONE thing from this project**, let it be this:

> "A model that performs poorly but generalizes well is infinitely more valuable than a model that performs perfectly but fails in production."

Good luck on your journey to becoming a 100x engineer! 🚀
