

# Time Series Forecasting for Portfolio Management Optimization

This project develops a data-driven framework to enhance portfolio management strategies by applying time series forecasting to historical financial data. It uses statistical and deep learning models to predict stock price movements, which inform portfolio optimization and risk management decisions.

## Table of Contents

  - [Business Objective](https://www.google.com/search?q=%23business-objective)
  - [Features](https://www.google.com/search?q=%23features)
  - [Project Structure](https://www.google.com/search?q=%23project-structure)
  - [Technologies Used](https://www.google.com/search?q=%23technologies-used)
  - [Installation and Usage](https://www.google.com/search?q=%23installation-and-usage)
  - [Results](https://www.google.com/search?q=%23results)
  - [Future Work](https://www.google.com/search?q=%23future-work)
  - [License](https://www.google.com/search?q=%23license)
  - [Contact](https://www.google.com/search?q=%23contact)

-----

## Business Objective

The project is framed from the perspective of a Financial Analyst at **Guide Me in Finance (GMF) Investments**, a financial advisory firm. GMF aims to leverage advanced time series forecasting models to predict market trends, optimize asset allocation, and enhance portfolio performance.The primary goal is to help clients achieve their financial objectives by minimizing risks and capitalizing on market opportunities.This involves analyzing key assets like TSLA, SPY, and BND to provide data-driven investment recommendations


---
## ## Features

* **Data Extraction & Cleaning:** Fetches historical financial data from the YFinance API and cleans it by handling non-trading days (weekends, holidays).
* **Exploratory Data Analysis (EDA):** Provides in-depth visual and statistical analysis of price trends, daily returns, and volatility for TSLA, SPY, and BND.
* **Risk Metric Calculation:** Computes foundational risk metrics like Value at Risk (VaR) and the Sharpe Ratio.
* **Comparative Modeling:** Implements, trains, and evaluates two different forecasting models for TSLA's stock price:
    * A classical statistical model (**ARIMA**)
    * A deep learning model (**LSTM**)
* **Future Forecasting with Uncertainty Quantification:** Generates a 6-12 month price forecast for TSLA using the trained LSTM model and employs Monte Carlo Dropout to calculate and visualize 95% confidence intervals.
* **Portfolio Optimization via Modern Portfolio Theory (MPT):** Combines the LSTM forecast with historical data to calculate an optimal portfolio, generating an interactive Efficient Frontier to identify the Maximum Sharpe Ratio and Minimum Volatility portfolios.
* **Strategy Backtesting:** Conducts a historical simulation of the optimized portfolio's performance over a one-year period, comparing its cumulative returns and Sharpe Ratio against a traditional 60/40 benchmark.
* **Modular Codebase:** The entire workflow is organized into reusable functions and clear, narrative notebooks for maximum readability and scalability.
-----
Here is an edited and refined version of your `README.md` section with improved formatting and clarity.

-----

### \#\# Project Structure

The project is organized into a modular structure to separate concerns, making the codebase clean, scalable, and easy to navigate.

```
TIME-SERIES-FORECASTING/
├── .github/
├── .venv/
├── data/
├── models/
│   └── tsla_lstm_model.keras
├── notebooks/
│   ├── 01_data_extraction_and_cleaning.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_comparative_modeling.ipynb
│   └── 04_portfolio_optimization_and_backtesting.ipynb
├── src/
│   ├── data_handler.py         # Combined extraction and cleaning functions
│   ├── time_series_models.py   # Combined ARIMA and LSTM model functions
│   ├── portfolio_optimizer.py
│   └── backtester.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt

-----

### \#\# Technologies Used

  * **Programming Language:** Python 3.11
  * **Data Handling:** Pandas, NumPy
  * **Data Extraction:** YFinance
  * **Statistical Modeling:** Statsmodels, Pmdarima
  * **Deep Learning:** TensorFlow, Keras
  * **Data Scaling:** Scikit-learn
  * **Visualization:** Matplotlib, Plotly

-----

### \#\# Installation and Usage

Follow these steps to set up and run the project locally.

#### 1\. Clone the Repository

```bash
git clone https://github.com/sumeyaaaa/Time-Series-Forecasting
cd Time-Series-Forecasting
```

#### 2\. Create and Activate a Virtual Environment

```bash
# Create the environment
python -m venv .venv

# Activate on Windows
.\.venv\Scripts\Activate.ps1

# Activate on Mac/Linux
source .venv/bin/activate
```

#### 3\. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4\. Running the Analysis

The project is organized into Jupyter Notebooks. For the best experience, run them in the following order:

1.  `notebooks/extraction.ipynb`
2.  `notebooks/clean_and_EDA.ipynb`
3.  `notebooks/modeling.ipynb`

-----

Of course. Here is a more comprehensive "Results" section that incorporates the key findings from all the major tasks of your project, including the portfolio optimization and backtesting outcomes.

---
## ## Results and Key Findings

The project successfully moved from raw data to a fully validated, model-driven portfolio strategy. The primary outcomes from each phase of the analysis are summarized below.

### Model Performance Comparison
The initial analysis involved developing and comparing ARIMA and LSTM models to forecast TSLA's stock price. The LSTM model demonstrated a significantly higher accuracy, proving its superior ability to capture the complex, non-linear patterns within the financial data.

| Metric | ARIMA Model | **LSTM Model** |
| :--- | :--- | :--- |
| MAE | 63.71 | **12.58** |
| RMSE | 78.99 | **17.57** |
| MAPE | 24.15% | **4.46%** |

This result confirms that the deep learning approach provides a more reliable forecast for the subsequent portfolio optimization tasks.

### Optimal Portfolio Allocation
Using the LSTM forecast for Tesla's expected returns in a Modern Portfolio Theory (MPT) framework, an optimal portfolio was calculated. The **Maximum Sharpe Ratio Portfolio** was selected for its efficiency in maximizing return for each unit of risk.


### Backtesting Validation
The final step was to backtest the model-driven strategy against a simple 60/40 SPY/BND benchmark over the last year of the dataset. The results showed that while the benchmark achieved a higher total return, the model-driven strategy was **equally efficient on a risk-adjusted basis**.

| Metric | My Strategy | Benchmark (60/40) |
| :--- | :--- | :--- |
| **Total Return** | 10.20% | **12.22%** |
| **Annualized Sharpe Ratio** | **0.84** | **0.84** |

This nuanced result indicates that the strategy successfully generated a unique risk-return profile that was just as efficient as the benchmark, validating the model-driven approach as a viable tool for portfolio construction.orecast for the subsequent portfolio optimization tasks.

-----

## ## Features

* **Data Extraction & Cleaning:** Fetches historical financial data from the YFinance API and cleans it by handling non-trading days (weekends, holidays).
* **Exploratory Data Analysis (EDA):** Provides in-depth visual and statistical analysis of price trends, daily returns, and volatility for TSLA, SPY, and BND.
* **Risk Metric Calculation:** Computes foundational risk metrics like Value at Risk (VaR) and the Sharpe Ratio.
* **Comparative Modeling:** Implements, trains, and evaluates two different forecasting models for TSLA's stock price:
    * A classical statistical model (**ARIMA**)
    * A deep learning model (**LSTM**)
* **Future Forecasting with Uncertainty Quantification:** Generates a 6-12 month price forecast for TSLA using the trained LSTM model and employs Monte Carlo Dropout to calculate and visualize 95% confidence intervals.
* **Portfolio Optimization via Modern Portfolio Theory (MPT):** Combines the LSTM forecast with historical data to calculate an optimal portfolio, generating an interactive Efficient Frontier to identify the Maximum Sharpe Ratio and Minimum Volatility portfolios.
* **Strategy Backtesting:** Conducts a historical simulation of the optimized portfolio's performance over a one-year period, comparing its cumulative returns and Sharpe Ratio against a traditional 60/40 benchmark.
* **Modular Codebase:** The entire workflow is organized into reusable functions and clear, narrative notebooks for maximum readability and scalability.
-----
Here is an edited and refined version of your `README.md` section with improved formatting and clarity.
-----

### \#\# License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

-----
