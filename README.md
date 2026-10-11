# Stock Price Trend Forecasting — ML-Driven Trading Strategy Evaluation

> Machine Learning pipeline for short-term equity return forecasting (21-day horizon) across a 43-ticker universe (2013–2025), with binary and multi-class classification, model calibration, SHAP interpretability, and financial backtesting against a Buy & Hold benchmark.

Academic Project — Machine Learning Course | M.Sc. in Financial Institutions, International Finance and Risk Management | Sapienza University of Rome

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

> **Note:** `dataset_train.csv` and `dataset_test.csv` are the *only* link between the two notebooks. If you have already uploaded these two files, you can skip Step 1 entirely and jump straight into `ml_pipeline.ipynb`.

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

### Models compared

| Model | Family | Feature set |
|---|---|---|
| Random Forest | Bagging | Full (44) |
| XGBoost-Full | Boosting | Full (44) |
| XGBoost-Reduced | Boosting | Reduced (~28, XGBoost-native selection) |
### Classification performance (test set, 2024–2025)

<img width="1501" height="1107" alt="image" src="https://github.com/user-attachments/assets/1e552a0d-7796-44f8-8319-f8c27efb842b" />

<img width="1674" height="1127" alt="image" src="https://github.com/user-attachments/assets/b6b9b208-078a-4bf8-bc36-a976b383db1a" />


<img width="1786" height="788" alt="image" src="https://github.com/user-attachments/assets/2c45e130-15c7-4f20-9b71-2e83128b3c95" />


<img width="768" height="940" alt="image" src="https://github.com/user-attachments/assets/362bdd04-7d2d-480a-af97-1d5d6ece7df2" />


### Financial backtest (test set, 2024–2025)

<img width="1612" height="1281" alt="image" src="https://github.com/user-attachments/assets/7d0da6f7-875c-449d-985f-a6fe0611436f" />

<img width="1679" height="1281" alt="image" src="https://github.com/user-attachments/assets/de7f57e3-c228-4166-9d8d-ac7f7792f9df" />


| Strategy | Total Return | CAGR | Volatility | Sharpe | Max DD | Calmar | Win Rate | N. Trades |
|---|---|---|---|---|---|---|---|---|
| Buy & Hold | 0.4430 | 0.2071 | 0.2149 | 0.9826 | -0.2252 | 0.9196 | 0.5580 | 43 |
| RF (binary) | -0.1494 | -0.0797 | 0.1028 | -0.7550 | -0.1894 | -0.4207 | 0.4375 | 6 |
| XGBoost-44 (binary) | 0.3570 | 0.1696 | 0.2059 | 0.8639 | -0.2049 | 0.8279 | 0.5507 | 738 |
| XGBoost-Reduced (binary) | 0.3762 | 0.1781 | 0.2223 | 0.8477 | -0.2274 | 0.7832 | 0.5340 | 823 |
| RF (multi LS) | 0.2841 | 0.1369 | 0.2603 | 0.6232 | -0.3135 | 0.4367 | 0.5193 | 427 |
| XGBoost-44 (multi LS) | 0.6225 | 0.2819 | 0.3612 | 0.8558 | -0.2393 | 1.1781 | 0.5316 | 597 |
| XGBoost-Reduced (multi LS) | 0.0204 | 0.0104 | 0.2958 | 0.1820 | -0.3123 | 0.0334 | 0.5084 | 598 |

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

Project developed for the "Machine Learning" Course, M.Sc. in Financial Institutions, International Finance and Risk Management | Sapienza University of Rome.
