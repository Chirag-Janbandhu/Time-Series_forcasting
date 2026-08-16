# Time Series Forecasting for Portfolio Management

A data-driven framework that applies time series forecasting to historical financial data to predict stock price movements, optimize portfolio allocation, and validate the strategy through backtesting.

Built as part of a financial analysis exercise for **Guide Me in Finance (GMF) Investments**, a fictional financial advisory firm.

---

## Table of Contents

- [Business Objective](#business-objective)
- [Features](#features)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Installation and Usage](#installation-and-usage)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

---

## Business Objective

GMF Investments aims to leverage advanced time series forecasting to predict market trends, optimize asset allocation, and enhance portfolio performance. The goal is to help clients achieve their financial objectives by minimizing risk and capitalizing on market opportunities.

This project analyzes three key assets:
- **TSLA** — Tesla Inc. (high-volatility growth equity)
- **SPY** — S&P 500 ETF (broad market benchmark)
- **BND** — Vanguard Total Bond Market ETF (low-risk fixed income)

---

## Features

- **Data Extraction & Cleaning:** Fetches historical financial data from the YFinance API and cleans it by handling non-trading days (weekends, holidays) using the NYSE market calendar.
- **Exploratory Data Analysis (EDA):** In-depth visual and statistical analysis of price trends, daily returns, and volatility for TSLA, SPY, and BND.
- **Risk Metric Calculation:** Computes foundational risk metrics including Value at Risk (VaR) and the Sharpe Ratio.
- **Comparative Modeling:** Implements, trains, and evaluates two forecasting models for TSLA:
  - A classical statistical model (**ARIMA** with `auto_arima` order selection)
  - A deep learning model (**LSTM** with Dropout regularization)
- **Future Forecasting with Uncertainty Quantification:** Generates a 6-month price forecast using the trained LSTM model with **Monte Carlo Dropout** to produce 95% confidence intervals.
- **Portfolio Optimization via Modern Portfolio Theory (MPT):** Combines the LSTM forecast with historical data to calculate an optimal portfolio and generates an interactive **Efficient Frontier** to identify the Maximum Sharpe Ratio and Minimum Volatility portfolios.
- **Strategy Backtesting:** Simulates the optimized portfolio's performance over a one-year period and compares cumulative returns and Sharpe Ratio against a traditional 60/40 benchmark.
- **Modular Codebase:** The entire workflow is organized into reusable Python modules and narrative Jupyter Notebooks.

---

## Project Structure

```
Time-Series_forecasting/
├── models/
│   └── tsla_lstm_model.keras       # Saved trained LSTM model
├── notebooks/
│   ├── extraction.ipynb            # Step 1: Data extraction
│   ├── clean_and_EDA.ipynb         # Step 2: Cleaning, EDA & risk metrics
│   └── modeling.ipynb              # Step 3: ARIMA, LSTM, forecasting, optimization & backtesting
├── src/
│   ├── yf_extract.py               # Data extraction via YFinance
│   ├── yf_clean.py                 # Cleaning, EDA, ADF tests, risk metrics
│   ├── yf_tesla_modeling.py        # ARIMA and LSTM modeling functions
│   ├── future_forecaster.py        # Monte Carlo forecasting with CI
│   ├── portfolio_optimizer.py      # MPT-based portfolio optimization
│   └── backtester.py               # Strategy backtesting vs benchmark
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── requirements-dev.txt
```

---

## Technologies Used

| Category | Libraries |
|---|---|
| **Language** | Python 3.11 |
| **Data Handling** | Pandas, NumPy |
| **Data Extraction** | YFinance |
| **Statistical Modeling** | Statsmodels, Pmdarima |
| **Deep Learning** | TensorFlow / Keras |
| **Data Scaling** | Scikit-learn |
| **Portfolio Optimization** | PyPortfolioOpt |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Market Calendar** | pandas-market-calendars |

---

## Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/Chirag-Janbandhu/Time-Series_forcasting
cd Time-Series_forcasting
```

### 2. Create and Activate a Virtual Environment

```bash
# Create the environment
python -m venv .venv

# Activate on Windows
.\.venv\Scripts\Activate.ps1

# Activate on Mac/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Notebooks in Order

The project is organized into Jupyter Notebooks. Run them sequentially for the full pipeline:

| Step | Notebook | Description |
|------|----------|-------------|
| 1 | `notebooks/extraction.ipynb` | Downloads and saves raw TSLA, SPY, BND data |
| 2 | `notebooks/clean_and_EDA.ipynb` | Cleans data, runs EDA, computes risk metrics |
| 3 | `notebooks/modeling.ipynb` | ARIMA & LSTM modeling, forecasting, portfolio optimization, backtesting |

> **Note:** Run `extraction.ipynb` first — it generates the `data/` folder that subsequent notebooks depend on.

---

## Results

### Model Performance Comparison

The LSTM model significantly outperformed the classical ARIMA model on the TSLA test set.

| Metric | ARIMA | LSTM |
|--------|-------|------|
| MAE    | 63.71 | **12.58** |
| RMSE   | 78.99 | **17.57** |
| MAPE   | 24.15% | **4.46%** |

The LSTM's 4.46% MAPE confirms its strong ability to capture the non-linear patterns in TSLA's price history.

### Portfolio Optimization

Using the LSTM forecast as expected return for TSLA within a Modern Portfolio Theory (MPT) framework, an optimal portfolio was constructed. The **Maximum Sharpe Ratio Portfolio** was selected for its efficiency in maximizing return per unit of risk.

### Backtesting vs. Benchmark

The model-driven strategy was backtested against a simple 60/40 (SPY/BND) benchmark over the final year of the dataset.

| Metric | My Strategy | Benchmark (60/40) |
|--------|-------------|-------------------|
| Total Return | 10.20% | 12.22% |
| Annualized Sharpe Ratio | **0.84** | **0.84** |

While the benchmark achieved a higher absolute return, both strategies achieved **identical risk-adjusted efficiency** (Sharpe = 0.84), validating the model-driven approach as a viable portfolio construction tool.

---

## Future Work

- Extend forecasting to **SPY and BND** using separate LSTM models
- Incorporate **sentiment analysis** (news headlines, earnings calls) as additional features
- Explore **Transformer-based** time series models (e.g., Temporal Fusion Transformer)
- Build a **Streamlit dashboard** for interactive forecasting and portfolio exploration
- Add **transaction cost modeling** to the backtester for more realistic simulations

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
