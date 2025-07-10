# SVM-Based Trading Strategies 🧠📉📈

This project implements machine learning models using **Support Vector Machines (SVM)** to predict the direction of financial instruments. The scripts explore two SVM-based strategies:

- ✅ **Lagged Price Features** (`SVM_strategy_implementation.py`)
- ✅ **Technical Indicators: RSI & SMA** (`SVM_SMA_RSI_implementation.py`)

Both models use historical market data and are trained using `scikit-learn` and `yfinance`.

---

## 📂 Contents

| File                             | Description |
|----------------------------------|-------------|
| `SVM_strategy_implementation.py` | SVM using percentage change of previous 2 days' returns (Lag1, Lag2) on the S&P 500. |
| `SVM_SMA_RSI_implementation.py`  | SVM using **trend** (price vs. SMA) and **RSI** indicators for EUR/USD currency pair. |

---

## 📊 Features Used

| Script | Features |
|--------|----------|
| `SVM_strategy_implementation.py` | Lag1 %, Lag2 % |
| `SVM_SMA_RSI_implementation.py`  | RSI (normalized), Trend (Open - SMA) |

---

## 🛠️ Installation

```bash
pip install -r requirements.txt
```
## Run files

Each script will:

  Download historical price data using yfinance

  Generate features

  Train and test an SVM classifier with hyperparameter tuning via grid search

  Print accuracy and confusion matrix

## 🧠 SVM Concepts Used

  Kernel-based classification using SVC from scikit-learn

  Grid search over C and gamma parameters

  Feature engineering using returns, RSI, and moving averages

## ⚠️ Disclaimer

This project is for educational purposes only and is not intended for live trading. Always consult a financial professional before using any trading model in real markets.
