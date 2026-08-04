import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
def clean_and_understand_data(df):
    """
    Cleans and provides an initial understanding of Yahoo Finance multi-ticker data.
    - Prints basic info and statistics
    - Converts 'Date' to datetime
    - Converts all numeric columns to numeric dtype
    - Handles missing values by forward and backward fill

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns like 'TSLA.Open', 'SPY.Close', etc.

    Returns
    -------
    pd.DataFrame : cleaned DataFrame
    """
    import pandas as pd

    print("\n--- Dataset Info ---")
    print(df.info())

    print("\n--- Missing Values Per Column ---")
    print(df.isna().sum())

    print("\n--- Basic Statistics ---")
    print(df.describe(include='all'))

    # Convert Date column
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Identify numeric columns (all except 'Date' and 'index' for our case)
    non_numeric_cols = ['Date', 'index']
    numeric_cols = [col for col in df.columns if col not in non_numeric_cols]

    # Convert all numeric columns to numeric dtype (float or int)
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle missing values with forward fill then backward fill
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    return df



def eda_volatility_analysis(df, ticker='TSLA'):
    """
    Perform EDA and volatility analysis on a given ticker closing price.
    Args:
        df (pd.DataFrame): Cleaned dataframe with columns like 'TSLA.Close', 'Date', etc.
        ticker (str): Ticker symbol to analyze (default 'TSLA').
    """

    close_col = f"{ticker}.Close"

    # 1. Plot Closing Price Over Time
    plt.figure(figsize=(14,6))
    plt.plot(df['Date'], df[close_col], label=f"{ticker} Close Price")
    plt.title(f"{ticker} Closing Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    # 2. Calculate daily % change (returns)
    df[f'{ticker}_Daily_Return'] = df[close_col].pct_change() * 100

    # Plot daily returns
    plt.figure(figsize=(14,6))
    plt.plot(df['Date'], df[f'{ticker}_Daily_Return'], color='orange')
    plt.title(f"{ticker} Daily Percentage Change (%)")
    plt.xlabel("Date")
    plt.ylabel("Daily % Change")
    plt.show()

    # 3. Rolling mean and std (volatility) - e.g., 20 days window
    rolling_window = 20
    df[f'{ticker}_Rolling_Mean'] = df[close_col].rolling(window=rolling_window).mean()
    df[f'{ticker}_Rolling_Std'] = df[close_col].rolling(window=rolling_window).std()

    plt.figure(figsize=(14,6))
    plt.plot(df['Date'], df[close_col], label='Close Price')
    plt.plot(df['Date'], df[f'{ticker}_Rolling_Mean'], label=f'{rolling_window}-Day Rolling Mean')
    plt.fill_between(df['Date'], 
                     df[f'{ticker}_Rolling_Mean'] - 2 * df[f'{ticker}_Rolling_Std'], 
                     df[f'{ticker}_Rolling_Mean'] + 2 * df[f'{ticker}_Rolling_Std'], 
                     color='gray', alpha=0.2, label='Rolling Mean ± 2 Std Dev')
    plt.title(f"{ticker} Price with Rolling Mean & Volatility Bands")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    # 4. Outlier Detection - Identify days with unusually high/low returns (e.g., beyond 3 std dev)
    mean_return = df[f'{ticker}_Daily_Return'].mean()
    std_return = df[f'{ticker}_Daily_Return'].std()
    outlier_thresh = 3 * std_return

    outliers = df[(df[f'{ticker}_Daily_Return'] > mean_return + outlier_thresh) | 
                  (df[f'{ticker}_Daily_Return'] < mean_return - outlier_thresh)]

    print(f"\nDays with unusually high or low {ticker} returns (beyond ±3 std dev):")
    print(outliers[['Date', close_col, f'{ticker}_Daily_Return']])

    # 5. Seasonality and Stationarity - Augmented Dickey-Fuller Test on Close Price
    print(f"\nADF Test on {ticker} Closing Price:")
    adf_result = adfuller(df[close_col].dropna())
    print(f"ADF Statistic: {adf_result[0]:.4f}")
    print(f"p-value: {adf_result[1]:.4f}")
    print("Critical Values:")
    for key, value in adf_result[4].items():
        print(f"   {key}: {value:.4f}")
    if adf_result[1] < 0.05:
        print("=> Reject null hypothesis: Series is stationary.")
    else:
        print("=> Fail to reject null hypothesis: Series is non-stationary.")

    # ADF test on daily returns (often returns are stationary)
    print(f"\nADF Test on {ticker} Daily Returns:")
    adf_ret_result = adfuller(df[f'{ticker}_Daily_Return'].dropna())
    print(f"ADF Statistic: {adf_ret_result[0]:.4f}")
    print(f"p-value: {adf_ret_result[1]:.4f}")
    if adf_ret_result[1] < 0.05:
        print("=> Reject null hypothesis: Daily returns are stationary.")
    else:
        print("=> Fail to reject null hypothesis: Daily returns are non-stationary.")

from statsmodels.tsa.stattools import adfuller

def adf_test(series, series_name='Series'):
    """
    Perform Augmented Dickey-Fuller test and print detailed results.

    Args:
        series (pd.Series): Time series data (e.g., closing price).
        series_name (str): Name of the series for printing purposes.

    Returns:
        dict: Dictionary with ADF statistic, p-value, and critical values.
    """
    result = adfuller(series.dropna(), autolag='AIC')
    adf_stat = result[0]
    p_value = result[1]
    used_lag = result[2]
    n_obs = result[3]
    crit_values = result[4]
    icbest = result[5]

    print(f"\nADF Test on {series_name}:")
    print(f"ADF Statistic: {adf_stat:.6f}")
    print(f"p-value: {p_value:.6f}")
    print(f"# Lags Used: {used_lag}")
    print(f"Number of Observations Used: {n_obs}")
    print("Critical Values:")
    for key, value in crit_values.items():
        print(f"   {key}: {value:.6f}")
    
    if p_value < 0.05:
        print("=> Reject null hypothesis: The series is stationary.")
    else:
        print("=> Fail to reject null hypothesis: The series is non-stationary.")

    return {
        'adf_statistic': adf_stat,
        'p_value': p_value,
        'used_lag': used_lag,
        'n_obs': n_obs,
        'critical_values': crit_values,
        'icbest': icbest
    }


def difference_series(series, periods=1):
    """
    Returns the differenced series.
    """
    return series.diff(periods=periods)

def plot_series(series, title='', figsize=(12,6)):
    """
    Plot a time series.
    """
    plt.figure(figsize=figsize)
    plt.plot(series)
    plt.title(title)
    plt.show()

def plot_acf_pacf(series, lags=40):
    """
    Plot ACF and PACF for a given time series.
    """
    plot_acf(series, lags=lags)
    plot_pacf(series, lags=lags)
    plt.show()

import numpy as np
import pandas as pd

# Note: The get_stock_data function has been removed as we are using a pre-existing DataFrame.

def calculate_returns(prices_series):
    """
    Calculates logarithmic daily returns on a given price series.
    
    Args:
        prices_series (pandas.Series): A Series containing asset prices.
        
    Returns:
        pandas.Series: A Series containing the calculated log returns.
    """
    log_returns = np.log(prices_series / prices_series.shift(1))
    return log_returns.dropna()

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Calculates the annualized Sharpe Ratio.
    """
    daily_risk_free_rate = (1 + risk_free_rate)**(1/252) - 1
    excess_returns = returns - daily_risk_free_rate
    sharpe_ratio = (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252))
    return sharpe_ratio

def calculate_var(returns, confidence_level=0.95):
    """
    Calculates the Value at Risk (VaR) using the historical method.
    """
    var_value = returns.quantile(1 - confidence_level)
    return var_value

def analyze_and_report_metrics(df, ticker):
    """
    Analyzes columns from a combined DataFrame to report risk metrics for a specific ticker.
    
    Args:
        df (pandas.DataFrame): The DataFrame containing combined data for all tickers.
        ticker (str): The ticker symbol to analyze (e.g., 'TSLA').
    """
    # --- Parameters for analysis ---
    RISK_FREE_RATE = 0.02   # Assumed 2% annual risk-free rate
    CONFIDENCE_LEVEL = 0.95 # 95% confidence for VaR

    # Dynamically select the correct 'Adj Close' column based on the ticker
    price_column = f'{ticker}.Adj Close'
    
    if price_column not in df.columns:
        print(f"\nError: Column '{price_column}' not found in the DataFrame.")
        return

    # 1. Calculate Returns for the specific ticker
    log_returns = calculate_returns(df[price_column])
    
    # 2. Calculate Sharpe Ratio
    sharpe = calculate_sharpe_ratio(log_returns, RISK_FREE_RATE)
    
    # 3. Calculate VaR
    var = calculate_var(log_returns, CONFIDENCE_LEVEL)
    
    # --- Print Results ---
    print("\n--- Risk Metrics Analysis ---")
    print(f"Ticker: {ticker}")
    print("-" * 30)
    print(f"Annualized Sharpe Ratio: {sharpe:.2f}")
    print(f"Value at Risk (VaR) at {CONFIDENCE_LEVEL:.0%} confidence:")
    print(f"  {abs(var):.2%} - This means on a typical day, we are {CONFIDENCE_LEVEL:.0%} confident that the stock will not lose more than {abs(var):.2%} of its value.")
    print("-" * 30)


import pandas as pd
import pandas_market_calendars as mcal

# If you don't have this library, install it first:
# pip install pandas_market_calendars

def clean_trading_days(df):
    """
    Filters a DataFrame to include only valid NYSE trading days, removing weekends and holidays.
    
    Args:
        df (pandas.DataFrame): The input DataFrame. Must contain a 'Date' column.
        
    Returns:
        pandas.DataFrame: A new DataFrame containing only rows from valid trading days.
    """
    # 1. Ensure 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 2. Get the official NYSE trading calendar
    nyse = mcal.get_calendar('NYSE')
    
    # 3. Determine the date range from the data
    start_date = df['Date'].min()
    end_date = df['Date'].max()
    
    # 4. Generate the schedule of all valid trading days in that range
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    
    # Extract just the dates from the schedule's index
    valid_trading_days = schedule.index.date
    
    # 5. Filter the original DataFrame
    # Keep only the rows where the date is in our list of valid days
    clean_df = df[df['Date'].dt.date.isin(valid_trading_days)].copy()
    
    return clean_df

# --- Example of how to use the function ---
