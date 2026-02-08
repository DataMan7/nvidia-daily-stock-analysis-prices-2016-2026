# 🚀 QUICK REFERENCE - Command Cheat Sheet

## 📦 **SETUP COMMANDS**

```bash
# Extract project
cd ~/Desktop
tar -xzf nvidia-stock-project.tar.gz
cd nvidia-daily-stock-analysis-prices-\(2016-2026\)

# Run setup
python3 scripts/setup_project.py

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🔍 **DATA LEAKAGE DETECTION**

```bash
# Basic leakage audit
python scripts/detect_leakage.py --data data/raw/NVDA_yfinance_clean.csv

# Educational mode: Introduce scaling leakage
python scripts/detect_leakage.py \
    --data data/raw/NVDA_yfinance_clean.csv \
    --introduce-leakage scaling

# Educational mode: Introduce future lag leakage  
python scripts/detect_leakage.py \
    --data data/raw/NVDA_yfinance_clean.csv \
    --introduce-leakage future_lag

# Save report to JSON
python scripts/detect_leakage.py \
    --data data/raw/NVDA_yfinance_clean.csv \
    --output reports/leakage_report.json
```

---

## 📊 **JUPYTER NOTEBOOKS**

```bash
# Start Jupyter Lab
jupyter lab

# Start Jupyter Notebook
jupyter notebook

# Convert notebook to HTML
jupyter nbconvert --to html notebooks/01_eda.ipynb

# Execute notebook from command line
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb
```

---

## 🧪 **TESTING**

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_leakage_detection.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

---

## 📁 **GIT COMMANDS**

```bash
# Initialize repository (if not done)
git init
git add .
git commit -m "Initial commit"

# Create and switch to new branch
git checkout -b feature/data-leakage-detector

# Check status
git status

# Add files
git add src/evaluation/validators.py
git commit -m "Add data leakage detector"

# Push to remote
git remote add origin <your-github-url>
git push -u origin main

# Pull latest changes
git pull origin main
```

---

## 🐍 **PYTHON SNIPPETS**

### **Load Data**

```python
from src.data.loader import NVDADataLoader

loader = NVDADataLoader('data/raw/NVDA_yfinance_clean.csv')
df, report = loader.load_and_validate()

print(f"Loaded {len(df):,} rows")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
```

### **Detect Leakage**

```python
from src.evaluation.validators import DataLeakageDetector

detector = DataLeakageDetector(df, target_col='Close')
report = detector.run_full_audit()

if report.leakage_detected:
    print(f"🚨 Leakage found: {report.leakage_types}")
    print(f"Severity: {report.severity}")
else:
    print("✅ No leakage detected!")
```

### **Introduce Leakage (Educational)**

```python
from src.evaluation.validators import deliberately_introduce_leakage

# ❌ WRONG: For learning purposes only!
leaky_df = deliberately_introduce_leakage(df, 'Close', 'scaling')

# Now detect it
detector = DataLeakageDetector(leaky_df, target_col='Close')
report = detector.run_full_audit()
```

---

## 📈 **COMMON DATA OPERATIONS**

```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/raw/NVDA_yfinance_clean.csv', parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# ✅ CORRECT: Create lag features
df['close_lag_1'] = df['Close'].shift(1)  # Previous day

# ✅ CORRECT: Rolling window
df['ma_7'] = df['Close'].rolling(window=7).mean()

# ✅ CORRECT: Returns
df['daily_return'] = df['Close'].pct_change()

# ❌ WRONG: Don't do this! (future information)
df['close_lead_1'] = df['Close'].shift(-1)  # Next day - LEAKAGE!

# ✅ CORRECT: Train-test split (time-series)
train_size = int(len(df) * 0.8)
train = df.iloc[:train_size]
test = df.iloc[train_size:]

# ❌ WRONG: Don't do this! (random split breaks temporal order)
from sklearn.model_selection import train_test_split
train, test = train_test_split(df)  # BAD for time-series!
```

---

## 🎯 **FEATURE ENGINEERING CHECKLIST**

Before creating any feature, ask:

```
□ Does this feature use future information?
□ Would this feature be available at prediction time?
□ Did I shift/lag correctly (positive shifts only for past)?
□ Did I handle NaN values from rolling windows?
□ Did I create this feature AFTER train-test split or ensure no leakage?
□ Can I explain this feature to a 10-year-old?
```

---

## ⚠️ **COMMON MISTAKES TO AVOID**

### **❌ Scaling Before Split**

```python
# ❌ WRONG
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Uses test data!
X_train, X_test = train_test_split(X_scaled)

# ✅ CORRECT
X_train, X_test = train_test_split(X)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Only transform
```

### **❌ Negative Shifts (Future Info)**

```python
# ❌ WRONG
df['tomorrow_price'] = df['Close'].shift(-1)  # LEAKAGE!

# ✅ CORRECT  
df['yesterday_price'] = df['Close'].shift(1)  # Past info
```

### **❌ Using Target Mean**

```python
# ❌ WRONG
df['target_mean'] = df['Close'].mean()  # Uses all data!

# ✅ CORRECT
df['target_ma'] = df['Close'].rolling(7).mean()  # Only past
```

---

## 🔧 **DEBUGGING TIPS**

```python
# Check for NaN values
print(df.isnull().sum())

# Check data types
print(df.dtypes)

# Check date ordering
print(df['Date'].is_monotonic_increasing)

# Visualize feature
import matplotlib.pyplot as plt
df['Close'].plot()
plt.show()

# Check correlation
print(df[['Close', 'close_lag_1']].corr())
```

---

## 📊 **MODEL EVALUATION**

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Calculate metrics
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# Directional accuracy (for time-series)
direction_true = (y_true.diff() > 0).astype(int)
direction_pred = (pd.Series(y_pred).diff() > 0).astype(int)
directional_accuracy = (direction_true == direction_pred).mean()
print(f"Directional Accuracy: {directional_accuracy:.4f}")
```

---

## 🎓 **LEARNING CHECKLIST**

### **Week 1: Data Understanding**
```
□ Loaded data successfully
□ Understood OHLCV format
□ Created basic visualizations
□ Identified patterns
□ No missing values confirmed
```

### **Week 2: Feature Engineering**
```
□ Created lag features correctly
□ Implemented moving averages
□ Added technical indicators (RSI, Bollinger Bands)
□ Validated no future information leakage
□ Documented feature creation logic
```

### **Week 3-4: Modeling**
```
□ Implemented naive baseline
□ Built XGBoost model
□ Used walk-forward validation
□ Compared models properly
□ Feature importance analysis
```

### **Week 5: Leakage Audit** 🔥
```
□ Ran leakage detector on clean data
□ Introduced scaling leakage (educational)
□ Introduced future lag leakage (educational)
□ Documented detection process
□ Written report on findings
```

---

## 🚀 **FINAL PROJECT SUBMISSION**

```bash
# 1. Clean up code
black src/ tests/
isort src/ tests/

# 2. Run tests
pytest tests/ -v

# 3. Generate documentation
# (Add your documentation here)

# 4. Commit everything
git add .
git commit -m "Final project submission"

# 5. Push to GitHub
git push origin main

# 6. Create release
git tag -a v1.0 -m "NVIDIA Stock Analysis - Complete"
git push origin v1.0
```

---

## 📞 **QUICK HELP**

| Issue | Solution |
|-------|----------|
| Can't import module | Check if in virtual environment: `which python` |
| File not found | Check path: `ls data/raw/` |
| Leakage detected | Review feature engineering logic |
| Low accuracy | Compare against baseline first |
| Out of memory | Reduce data size or use sampling |

---

**💡 Pro Tip**: Keep this file open in a separate terminal/tab for quick reference!

---

**Last Updated**: February 2026  
**Author**: Senior ML Engineering Team  
**Purpose**: Learning aid for production ML pipelines
