# src/future_forecaster.py

import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
from tqdm import tqdm

def run_monte_carlo_forecast(trained_model, scaler, last_sequence, time_step, forecast_days=180, n_iterations=50):
    """
    Runs an iterative Monte Carlo forecast using a trained LSTM model.

    Args:
        trained_model (keras.Model): The trained LSTM model.
        scaler (MinMaxScaler): The scaler fitted on the training data.
        last_sequence (np.array): The last sequence of historical data to start the forecast.
        time_step (int): The look-back period of the model.
        forecast_days (int): The number of days to forecast into the future.
        n_iterations (int): The number of Monte Carlo simulations to run.

    Returns:
        np.array: A NumPy array containing all the different forecast paths.
    """
    # Create a new model instance that enables dropout during prediction
    inputs = tf.keras.Input(shape=(time_step, 1))
    outputs = trained_model(inputs, training=True)
    mc_model = tf.keras.Model(inputs, outputs)
    print("Model configured for Monte Carlo forecasting.")

    # Scale the initial input sequence
    scaled_last_sequence = scaler.transform(last_sequence)
    all_forecast_paths = []

    print(f"\nRunning {n_iterations} Monte Carlo simulations for a {forecast_days}-day forecast...")
    for _ in tqdm(range(n_iterations), desc="Forecasting Progress"):
        current_input = scaled_last_sequence.reshape(1, time_step, 1)
        future_predictions_single_run = []
        for _ in range(forecast_days):
            next_prediction_scaled = mc_model.predict(current_input, verbose=0)
            future_predictions_single_run.append(next_prediction_scaled[0, 0])
            current_input = np.append(current_input[:, 1:, :], next_prediction_scaled.reshape(1, 1, 1), axis=1)
        all_forecast_paths.append(future_predictions_single_run)
        
    return np.array(all_forecast_paths)

def process_forecasts(all_forecast_paths, scaler, last_date, forecast_days):
    """
    Processes the raw forecast paths to calculate the mean and confidence intervals.

    Returns:
        pd.DataFrame: A DataFrame with the mean forecast and CI bounds.
    """
    # Inverse transform all paths to get real dollar values
    all_forecast_paths_unscaled = scaler.inverse_transform(all_forecast_paths.T).T

    # Calculate the mean forecast and the 95% confidence interval
    mean_forecast = np.mean(all_forecast_paths_unscaled, axis=0)
    lower_bound = np.percentile(all_forecast_paths_unscaled, 2.5, axis=0)
    upper_bound = np.percentile(all_forecast_paths_unscaled, 97.5, axis=0)

    # Create a future date index
    future_date_index = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

    future_forecast_df = pd.DataFrame({
        'Forecast': mean_forecast,
        'Lower_CI': lower_bound,
        'Upper_CI': upper_bound
    }, index=future_date_index)
    
    print("Forecast and confidence intervals generated successfully.")
    return future_forecast_df

def plot_future_forecast(historical_df, forecast_df):
    """
    Visualizes the historical data and the future forecast with confidence intervals.
    """
    historical_data_to_plot = historical_df.loc[historical_df.index > (historical_df.index.max() - pd.DateOffset(years=1))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_data_to_plot.index, y=historical_data_to_plot['Adj Close'], mode='lines', name='Historical Price'))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Upper_CI'], mode='lines', name='Upper 95% CI', line=dict(width=0)))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Lower_CI'], mode='lines', name='Lower 95% CI', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 165, 0, 0.2)'))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines', name='Mean Forecast', line=dict(color='orange', width=3)))

    fig.update_layout(
        title='TSLA 6-Month Future Forecast with 95% Confidence Interval',
        xaxis_title='Date', yaxis_title='Adjusted Close Price',
        template='plotly_white', legend=dict(x=0.01, y=0.99)
    )
    fig.show()