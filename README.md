# DL4AI Stock Market Project — Student ID: 230156

> **Final Project: Time-Series Data and Application to Stock Markets**
> Familiarises students with time-series analysis, deep learning (LSTM), trading signal identification, portfolio construction, and production-grade model deployment.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Datasets](#3-datasets)
4. [Task Summary](#4-task-summary)
5. [Setup Instructions](#5-setup-instructions)
6. [Running Instructions](#6-running-instructions)
7. [Reproducibility Note (Task 5)](#7-reproducibility-note-task-5)
8. [Tech Stack](#8-tech-stack)

---

## 1. Project Overview

This project applies Long Short-Term Memory (LSTM) neural networks to two real-world stock market datasets — **Nasdaq** (US market) and **Vietnam stock exchange** — covering the following areas:

- Exploratory Data Analysis (EDA) and data quality screening
- Multi-feature and multi-horizon price forecasting
- Buy/Sell signal classification with threshold optimisation
- Profitable stock selection, risk management, and portfolio construction
- REST API deployment (FastAPI) and a SaaS-style web interface (Gradio)

---

## 2. Repository Structure

```
DL4AI-230156-project/
│
├── 230156_project_notebook.ipynb   # Main Jupyter notebook (all tasks)
├── 230156_project_report.pdf       # Written report (≥ 2 000 words)
├── README.md                       # This file
│
└── models/                         # Saved Keras models (generated at runtime)
    ├── aapl_lstm_multi_feature.keras
    ├── buy_signal_model.keras
    ├── sell_signal_model.keras
    ├── buy_scaler.pkl
    └── sell_scaler.pkl
```

> **Note:** The `models/` directory is generated when the notebook cells are executed. It is not committed to the repository to keep the repo lightweight.

---

## 3. Datasets

All data is pulled automatically from the public GitHub repository below during the EDA cell:

```
https://github.com/trucle0209/stockmarketLLMprojects
```

| Market   | Path inside repo                              | Date column    | Key features                       |
|----------|-----------------------------------------------|----------------|------------------------------------|
| Nasdaq   | `nasdaq/csv/<TICKER>.csv`                     | `Date`         | Open, High, Low, Close, Adj Close, Volume |
| Vietnam  | `vietnam/stock-historical-data/<TICKER>.csv`  | `TradingDate`  | Open, High, Low, Close, Volume     |
| VN Div.  | `vietnam/dividend-history/`                   | —              | Dividend records (EDA only)        |
| VN Fin.  | `vietnam/financial-ratio/`                    | —              | Financial ratios (EDA only)        |

Only tickers with **≥ 120 trading days** and **zero unparseable dates** are retained for modelling.

---

## 4. Task Summary

### EDA
Scans all CSV files, builds a quality inventory (row counts, date ranges, missing values, duplicates), and produces six diagnostic plots (row-count histograms, coverage-in-years, sample price histories for AAPL and HPG, and a bucket-count bar chart).

### Task 1 — Nasdaq Price Prediction
| Sub-task | Description |
|----------|-------------|
| **1.1** | Multi-feature (OHLCV) next-day close price forecast using a 2-layer LSTM |
| **1.2** | k-th day ahead forecast (demonstrated at k = 3; any k supported) |
| **1.3** | Multi-step forecast — predicts the next k consecutive days simultaneously |

Benchmark companies: AAPL, MSFT, AMZN, NVDA, INTC, CSCO, AMD.

### Task 2 — Vietnam Price Prediction
Mirrors Tasks 1.1–1.3 for the Vietnam market using HPG (Hoà Phát) as the reference ticker. Feature set: Open, High, Low, Close, Volume (no Adjusted Close in VN data).

### Task 3 — Trading Signal Identification (Vietnam)
| Sub-task | Description |
|----------|-------------|
| **3.1** | **Buy signal** — LSTM classifier with 13 features (OHLCV + SMA10/20 + RSI + MACD + Signal Line + Price_vs_High20 + Volatility_10 + Volume_Spike). Target: price rises > 2 % within 3 days. Threshold tuned to maximise F1 (high Recall). |
| **3.2** | **Sell signal** — Same 13-feature architecture. Target: price drops > 2 % within 3 days. |

Both models use 5-fold time-series cross-validation before final evaluation.

### Task 4 — Portfolio Construction (Vietnam)
| Sub-task | Description |
|----------|-------------|
| **4.1** | **Profitable stock selection** — Two-stage pipeline: (i) fast statistical screening (Sharpe, recent return, trend slope, low-volatility bonus) → top 50 candidates; (ii) LSTM regression on each candidate to predict the 10-day forward return. Final Profit Score = 60 % predicted return + 40 % Sharpe. |
| **4.2** | **Risk management** — Computes Volatility, Max Drawdown, VaR (95 %), Beta (vs. VN market proxy), and Downside Deviation for every stock from Task 4.1. Produces a composite Risk Score. |
| **4.3** | **Portfolio composition** — Two profiles over the top-10 selected stocks: *Prudent* (50/50 profit/risk) and *Risk-Taking* (70/30 profit/risk). Equal-weight allocation with portfolio-level Sharpe, Volatility, and Drawdown reported. |

### Task 5 — Deployment
| Sub-task | Description |
|----------|-------------|
| **5.1** | **FastAPI REST backend** running on `localhost:8000` inside the Colab runtime. Endpoints: `GET /health`, `POST /predict/buy`, `POST /predict/sell`, interactive Swagger UI at `/docs`. Input: 20 × 13 feature sequence; output: signal (0/1), probability, confidence level. |
| **5.2** | **Gradio web app (SaaS)** — user selects a VN ticker, the app automatically loads the last 20 trading days, computes the 13 features, calls the FastAPI backend, and displays the Buy/Sell prediction with a feature table. A public share link is generated by Gradio. |

---

## 5. Setup Instructions

### Prerequisites

| Tool | Version |
|------|---------|
| Python | ≥ 3.9 |
| Google Colab (recommended) | — |
| GPU runtime | Optional but speeds up LSTM training |

### Environment (Google Colab — recommended)

No local installation is required. Open the notebook in Google Colab and enable a GPU runtime:

```
Runtime → Change runtime type → T4 GPU
```

The notebook installs all required packages inline:

```python
!pip install fastapi uvicorn nest_asyncio gradio --quiet
```

Core packages used (pre-installed on Colab):

```
tensorflow >= 2.12
pandas
numpy
scikit-learn
matplotlib
joblib
requests
pathlib
```

### Local Installation (optional)

```bash
git clone <your-repo-url>
cd DL4AI-230156-project

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install tensorflow pandas numpy scikit-learn matplotlib \
            fastapi uvicorn nest_asyncio gradio joblib requests
```

> Local execution is not required for grading. Task 5 is designed for the Colab environment.

---

## 6. Running Instructions

Open `230156_project_notebook.ipynb` in Google Colab and run cells **in top-to-bottom order**.

### Step-by-step

| Step | Cell / Section | What it does |
|------|---------------|--------------|
| 1 | **EDA** | Clones the dataset repo, builds the quality inventory (`clean_nasdaq`, `clean_vn`), and renders 6 diagnostic plots. **Must be the first cell executed.** |
| 2 | **Task 1.1** | Trains and evaluates the multi-feature Nasdaq LSTM. Saves `models/aapl_lstm_multi_feature.keras`. |
| 3 | **Task 1.2** | Trains the k-th day forecast model (default k = 3). Change the `k` variable to try other horizons. |
| 4 | **Task 1.3** | Trains the multi-step model (default 3 consecutive days). Change `forecast_steps` to adjust. |
| 5 | **Task 2.1–2.3** | Repeats Tasks 1.1–1.3 for the Vietnam market (HPG ticker). |
| 6 | **Task 3.1** | Trains the buy-signal classifier. Saves `buy_signal_model.keras` and `buy_scaler.pkl`. |
| 7 | **Task 3.2** | Trains the sell-signal classifier. Saves `sell_signal_model.keras` and `sell_scaler.pkl`. |
| 8 | **Task 4.1** | Statistical screening + LSTM profitability scoring across VN tickers. |
| 9 | **Task 4.2** | Risk metric computation for Task 4.1 results. |
| 10 | **Task 4.3** | Portfolio construction (Prudent and Risk-Taking profiles). |
| 11 | **Task 5.1** | Starts the FastAPI server on port 8000, runs self-tests, and opens a Colab iframe. |
| 12 | **Task 5.2** | Launches the Gradio interface and prints a public share link. |

### Key parameters (easy to adjust)

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| `window_size` | Task 1 / 2 data pipeline | `60` | Lookback window in trading days |
| `k` | Task 1.2 / 2.2 | `3` | Which future day to predict |
| `forecast_steps` | Task 1.3 / 2.3 | `3` | How many consecutive days to predict |
| `TOP_CANDIDATES` | Task 4.1 | `50` | Candidates passed to LSTM screening |
| `TOP_N` | Task 4.3 | `10` | Portfolio size |
| `HORIZON` | Task 4 | `10` | Forward return horizon (days) |

---

## 7. Reproducibility Note (Task 5)

> **The Gradio share link is ephemeral and session-bound.**

To reproduce the live demo, notebook cells must be executed in the following order:

1. **EDA cell** — required to load the dataset into memory.
2. **Task 5.1 cell** — initialises and launches the FastAPI backend on `localhost:8000`.
3. **Task 5.2 cell** — launches the Gradio interface and generates a new share link.

Each new Colab session produces a different share link. The link expires when the session ends. Running only Task 5.2 without Task 5.1 will cause API call failures inside the Gradio app.

### Testing the API manually

```bash
# Health check
curl http://localhost:8000/health

# Buy signal prediction (replace sequence with real feature values)
curl -X POST http://localhost:8000/predict/buy \
     -H "Content-Type: application/json" \
     -d '{"ticker": "HPG", "sequence": [[...20 rows × 13 features...]]}'

# Sell signal prediction
curl -X POST http://localhost:8000/predict/sell \
     -H "Content-Type: application/json" \
     -d '{"ticker": "HPG", "sequence": [[...20 rows × 13 features...]]}'

# Swagger UI (browser)
open http://localhost:8000/docs
```

**Input format for `/predict/buy` and `/predict/sell`:**

```json
{
  "ticker": "HPG",
  "sequence": [
    [open, high, low, close, volume, sma_10, sma_20, rsi, macd,
     signal_line, volatility, price_change, vol_change],
    ...
  ]
}
```

The `sequence` field is a list of **20 rows × 13 features**.

---

## 8. Tech Stack

| Category | Library / Tool |
|----------|---------------|
| Deep Learning | TensorFlow / Keras (LSTM) |
| Data Processing | pandas, NumPy |
| Machine Learning | scikit-learn (StandardScaler, MinMaxScaler, TimeSeriesSplit, metrics) |
| Visualisation | Matplotlib |
| API Backend | FastAPI, Uvicorn, nest_asyncio |
| Web Interface | Gradio |
| Model Persistence | Keras `.keras` format, joblib (`.pkl` scalers) |
| Data Source | GitHub (`trucle0209/stockmarketLLMprojects`) |
| Execution Environment | Google Colab (recommended), Python ≥ 3.9 locally |

---

*For questions or issues, refer to the project report (`230156_project_report.pdf`) or open an issue in this repository.*
