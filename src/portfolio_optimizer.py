# src/portfolio_optimizer.py

import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns
import plotly.graph_objects as go

def prepare_optimization_inputs(prices_df, forecast_df):
    """
    Prepares and PRINTS the expected returns vector (mu) and covariance matrix (S).
    """
    # Calculate expected return for the forecasted asset (TSLA)
    last_actual_price = prices_df['TSLA'].iloc[-1]
    last_forecast_price = forecast_df['Forecast'].iloc[-1]
    forecast_days = len(forecast_df)
    tsla_expected_return = ((last_forecast_price / last_actual_price) ** (252 / forecast_days)) - 1

    # Calculate historical returns for other assets
    mu_historical = expected_returns.mean_historical_return(prices_df[['SPY', 'BND']])

    # Combine into a single expected returns vector
    mu = pd.Series({
        'TSLA': tsla_expected_return,
        'SPY': mu_historical['SPY'],
        'BND': mu_historical['BND']
    })

    # Calculate the annualized sample covariance matrix of returns
    S = risk_models.sample_cov(prices_df)
    
    # --- This section now prints all required inputs ---
    print("--- Inputs for Optimization ---")
    print("Expected Annual Returns (mu):")
    print(mu)
    print("\nCovariance Matrix (S):")
    print(S)  # This line prints the covariance matrix
    
    return mu, S

def find_optimal_portfolios(mu, S):
    """
    Calculates and PRINTS the Maximum Sharpe Ratio and Minimum Volatility portfolios.
    """
    # Max Sharpe Ratio Portfolio
    ef_sharpe = EfficientFrontier(mu, S)
    weights_sharpe = ef_sharpe.max_sharpe()
    ef_sharpe.portfolio_performance(verbose=True) # verbose=True prints the performance
    
    # Minimum Volatility Portfolio
    ef_min_vol = EfficientFrontier(mu, S)
    weights_min_vol = ef_min_vol.min_volatility()
    ef_min_vol.portfolio_performance(verbose=True) # verbose=True prints the performance

    print("\n--- Max Sharpe Ratio Portfolio Weights ---")
    print(ef_sharpe.clean_weights())
    
    print("\n--- Min Volatility Portfolio Weights ---")
    print(ef_min_vol.clean_weights())

    return ef_sharpe.portfolio_performance(), ef_sharpe.clean_weights(), ef_min_vol.portfolio_performance(), ef_min_vol.clean_weights()

def plot_efficient_frontier(mu, S, perf_sharpe, perf_min_vol):
    """
    Generates an interactive Plotly chart of the Efficient Frontier.
    """
    ef = EfficientFrontier(mu, S)
    
    risk_range = np.linspace(perf_min_vol[1], perf_sharpe[1] * 1.5, 100)
    returns = []
    for r in risk_range:
        try:
            returns.append(ef.efficient_risk(r))
        except:
            continue
            
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=risk_range[:len(returns)], y=returns, mode='lines', name='Efficient Frontier'))
    fig.add_trace(go.Scatter(x=[perf_sharpe[1]], y=[perf_sharpe[0]], mode='markers', marker_size=12, name='Max Sharpe Ratio'))
    fig.add_trace(go.Scatter(x=[perf_min_vol[1]], y=[perf_min_vol[0]], mode='markers', marker_size=12, name='Min Volatility'))

    fig.update_layout(
        title='Efficient Frontier with Optimal Portfolios',
        xaxis_title='Annual Volatility (Risk)',
        yaxis_title='Expected Annual Return',
        template='plotly_white',
        legend=dict(x=0.01, y=0.99)
    )
    fig.show()