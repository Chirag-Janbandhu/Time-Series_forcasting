# src/backtester.py

import pandas as pd
import numpy as np
import plotly.graph_objects as go

def run_backtest(prices, strategy_weights, benchmark_weights, start_date, end_date, initial_investment=100000):
    """
    Simulates the performance of a strategy portfolio against a benchmark.
    """
    backtest_prices = prices.loc[start_date:end_date]
    daily_returns = backtest_prices.pct_change().dropna()

    # Simulate Strategy Portfolio
    strategy_returns = daily_returns.dot(pd.Series(strategy_weights))
    strategy_cumulative_returns = (1 + strategy_returns).cumprod()
    strategy_portfolio_value = initial_investment * strategy_cumulative_returns

    # Simulate Benchmark Portfolio
    benchmark_returns = daily_returns.dot(pd.Series(benchmark_weights))
    benchmark_cumulative_returns = (1 + benchmark_returns).cumprod()
    benchmark_portfolio_value = initial_investment * benchmark_cumulative_returns
    
    return strategy_portfolio_value, benchmark_portfolio_value, strategy_returns, benchmark_returns

def calculate_performance_metrics(returns_series, risk_free_rate=0.02):
    """
    Calculates total return and Sharpe ratio for a returns series.
    """
    total_return = (1 + returns_series).prod() - 1
    sharpe_ratio = (returns_series.mean() * 252 - risk_free_rate) / (returns_series.std() * np.sqrt(252))
    
    metrics = {
        "Total Return": f"{total_return:.2%}",
        "Annualized Sharpe Ratio": f"{sharpe_ratio:.2f}"
    }
    return metrics

def plot_backtest_results(strategy_value_df, benchmark_value_df):
    """
    [cite_start]Generates an interactive Plotly chart of the backtest performance. [cite: 231]
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strategy_value_df.index, y=strategy_value_df, mode='lines', name='My Strategy Portfolio'))
    fig.add_trace(go.Scatter(x=benchmark_value_df.index, y=benchmark_value_df, mode='lines', name='Benchmark (60/40)'))
    
    fig.update_layout(
        title='Backtest: Strategy vs. Benchmark Performance',
        xaxis_title='Date', yaxis_title='Portfolio Value ($)',
        template='plotly_white', legend=dict(x=0.01, y=0.99)
    )
    fig.show()