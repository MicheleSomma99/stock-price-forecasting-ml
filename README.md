# Stock Price Trend Forecasting — ML-Driven Trading Strategy Evaluation

> Machine Learning pipeline for short-term equity return forecasting (21-day horizon) across a 43-ticker universe (2013–2025), with binary and multi-class classification, model calibration, SHAP interpretability, and financial backtesting against a Buy & Hold benchmark.

Academic project — Corso di Machine Learning, Laurea Magistrale in Finanza.

---

## 🎯 Project Objective

This project builds and rigorously evaluates a Machine Learning pipeline for forecasting 21-day forward stock returns across a heterogeneous equity universe. Rather than chasing a single "best model," the project compares **three model families in parity** — Random Forest, XGBoost on the full feature set, and XGBoost on a reduced feature set — across two prediction tasks (binary UP/DOWN and 4-class intensity), and translates their outputs into trading strategies benchmarked against a passive Buy & Hold portfolio.

The core methodological commitments of the project are:

- **Strict anti-leakage discipline**: every statistic used for imputation, scaling, target quantiles, or model calibration is computed exclusively on the training set and applied unchanged to the test set.
- **Survivorship-bias mitigation**: the investable universe deliberately includes financially distressed and highly volatile names, not just consolidated blue chips.
- **Model-dependent feature selection**: the project documents empirically why a Random-Forest-native feature selection method (Lopez de Prado, 2018) underperforms when applied to a boosting model, motivating an XGBoost-native selection pipeline (Gain + out-of-sample SHAP).
- **Probability calibration**: boosting models are shown to be overconfident and are recalibrated via isotonic regression before being used in threshold-based trading strategies.
- **Financial, not just statistical, evaluation**: the final judgment on model usefulness is made via Sharpe Ratio, Max Drawdown, and Calmar Ratio in a realistic backtest with transaction costs and minimum holding periods — not via accuracy alone.

---

## 🗂️ Repository Structure

```
.
├── data_pipeline.ipynb          # STEP 1 — builds the dataset from raw sources
├── ml_pipeline.ipynb            # STEP 2 — modeling, calibration, backtest
├── dataset_train.csv            # Output of Step 1 — training set (2013-2023)
├── dataset_test.csv             # Output of Step 1 — test set (2024-2025)
├── requirements.txt             # Python dependencies
└── README.md
```

> **Note:** `dataset_train.csv` and `dataset_test.csv` are the *only* link between the two notebooks. If you already have these two files, you can skip Step 1 entirely and jump straight into `ml_pipeline.ipynb`.

---

## ⚙️ How to Run — Two Sequential Steps

This project is split into two independent notebooks that **must be run in order**, because the second depends on the CSV files produced by the first.

### Step 0 — Install dependencies

```bash
pip install -r requirements.txt
```

Minimum dependencies used across the two notebooks:

```
pandas
numpy
scikit-learn
xgboost
shap
yfinance
fredapi
simfin
matplotlib
joblib
```

> `fredapi` and `simfin` require free API keys (FRED and SimFin respectively). Register at [fred.stlouisfed.org](https://fred.stlouisfed.org) and [simfin.com](https://simfin.com), then set your keys in the configuration cell of `data_pipeline.ipynb`.

### Step 1 — Run the Data Pipeline

Open and run **`data_pipeline.ipynb`** top to bottom.

This notebook:
1. Downloads daily OHLCV prices (`yfinance`) for a 43-ticker universe split into Blue Chip / Volatile / Distressed categories
2. Downloads macroeconomic series from FRED (VIX, yield curve, CPI, unemployment, oil)
3. Downloads company fundamentals from SimFin, merged strictly on `publish_date` (never `report_date`) to avoid look-ahead bias
4. Engineers ~44 features across five thematic blocks: returns/momentum, technical indicators, volatility/volume, fundamentals & Altman Z-Score, macro
5. Builds two prediction targets: binary (UP/DOWN) and 4-class intensity (Strong Down / Weak Down / Weak Up / Strong Up), with quantile thresholds fit **only on the training period**
6. Performs a strict chronological split (train: 2013–2023, test: 2024–2025)

**Output:** `dataset_train.csv`, `dataset_test.csv`

### Step 2 — Run the ML Pipeline

Open and run **`ml_pipeline.ipynb`** top to bottom, with the two CSVs from Step 1 in the working directory.

This notebook:
1. **Pre-processing** — anti-leakage imputation and scaling, `TimeSeriesSplit` cross-validation
2. **Baseline models** — Logistic Regression vs. Random Forest, to justify the use of more expressive models
3. **Feature selection** — XGBoost-native pipeline (Gain importance + out-of-sample SHAP + Spearman collinearity filter) producing a Reduced Feature Set
4. **Binary classification** — three models compared in parity (Random Forest / XGBoost-Full / XGBoost-Reduced), with SHAP interpretability
5. **Multi-class classification** — same three models on the 4-class target, with adjacency-error analysis and **isotonic calibration** of the (overconfident) boosting models
6. **Financial backtest** — two parallel strategy families (binary long-only, multi-class long-only selective) evaluated against an equally-weighted Buy & Hold benchmark, with transaction costs and a minimum holding period
7. **Conclusions** — methodological synthesis, robustness checks, and explicit discussion of limitations

Total runtime: roughly 10–15 minutes on a standard machine (no GPU required).

---

## 🧠 Models & Results

*(Draft section — to be completed with final charts, screenshots, and numbers)*

### Models compared

| Model | Family | Feature set | Role |
|---|---|---|---|
| Random Forest | Bagging | Full (44) | Conservative baseline / defensive profile |
| XGBoost-Full | Boosting | Full (44) | Balanced profile |
| XGBoost-Reduced | Boosting | Reduced (~28, XGBoost-native selection) | Aggressive / parsimonious profile |

### Classification performance (test set, 2024–2025)

*[Insert final metrics table here: Accuracy, Precision, Recall, F1, ROC-AUC for binary; Accuracy, F1-macro, Cohen's Kappa for multi-class]*

*[Insert SHAP summary plot / top feature drivers screenshot here]*

### Financial backtest (test set, 2024–2025)

*[Insert final results table here: Total Return, CAGR, Sharpe Ratio, Max Drawdown, Calmar Ratio for each strategy vs. Buy & Hold]*

*[Insert equity curves chart here]*

**Key takeaway:** *[Summarize here whether/which model(s) beat the Buy & Hold benchmark, and on which metric — total return, Sharpe, or drawdown control]*

---

## 📚 Key References

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. *The Review of Financial Studies*, 33(5), 2223–2273.
- Niculescu-Mizil, A., & Caruana, R. (2005). Predicting Good Probabilities with Supervised Learning. *ICML*.
- Hasbrouck, J. (2009). Trading Costs and Returns for U.S. Equities. *The Journal of Finance*, 64(3).
- Altman, E. I. (1968). Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy. *The Journal of Finance*.

---

## ⚠️ Disclaimer

This project is for academic and educational purposes only. It does not constitute financial advice. Past performance in a backtest does not guarantee future results. The financial backtest uses synthetic assumptions (e.g., synthetic shorting, simplified transaction costs) that do not fully reflect real trading conditions.

---

## 👥 Author

*Michele Somma*
https://www.linkedin.com/in/michele-somma/

Project developed for the course "AI in Banking and Finance" — Laurea Magistrale.
