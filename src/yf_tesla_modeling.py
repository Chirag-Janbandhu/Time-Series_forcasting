# arima_modeler.py

import pandas as pd
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np



def find_best_arima_params(train_series):
    """
    Uses auto_arima to find the best (p,d,q) for a given training set.
    
    Args:
        train_series (pd.Series): The time series data for training.
        
    Returns:
        pmdarima.ARIMA: The fitted auto_arima model object.
    """
    print("Running auto_arima to find the best model parameters...")
    
    # The function now correctly uses the 'train_series' argument passed to it
    auto_arima_model = pm.auto_arima(
        train_series, 
        start_p=1, start_q=1,
        max_p=3, max_q=3,
        m=1, d=None, seasonal=False,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )
    return auto_arima_model

# ... (your other functions like train_arima_model, etc.)

def train_arima_model(train_data, order):
    """
    Trains a final ARIMA model with a specific order.
    
    Args:
        train_data (pd.Series): The time series data for training.
        order (tuple): The (p,d,q) order for the ARIMA model.
        
    Returns:
        statsmodels.tsa.arima.model.ARIMAResultsWrapper: The results object of the fitted model.
    """
    print(f"\nTraining final ARIMA{order} model...")
    model = ARIMA(train_data, order=order)
    fitted_model = model.fit()
    print(fitted_model.summary())
    return fitted_model

def forecast_with_arima(fitted_model, steps):
    """
    Generates forecasts from a fitted ARIMA model.
    
    Args:
        fitted_model: The fitted model results object from statsmodels.
        steps (int): The number of future steps to forecast.
        
    Returns:
        pd.Series: A series containing the forecast values.
    """
    print(f"Generating forecast for the next {steps} steps...")
    forecast = fitted_model.forecast(steps=steps)
    return forecast

def evaluate_forecast(y_true, y_pred):
    """
    Calculates key performance metrics for a forecast.
    
    Args:
        y_true (pd.Series): The actual, true values.
        y_pred (pd.Series): The predicted values from the model.
        
    Returns:
        dict: A dictionary containing MAE, RMSE, and MAPE scores.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    metrics = {
        'Mean Absolute Error (MAE)': mae,
        'Root Mean Squared Error (RMSE)': rmse,
        'Mean Absolute Percentage Error (MAPE)': mape
    }
    
    print("\n--- Forecast Evaluation Metrics ---")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
        
    return metrics


from sklearn.preprocessing import MinMaxScaler
import numpy as np


# --- 2. Create Sequences ---
# This function converts our data into input sequences (X) and an output value (y).
# This is your create_dataset function
def create_dataset(dataset, time_step=60):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)
# In a file like lstm_modeler.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

def build_and_train_lstm(X_train, y_train, epochs=25, batch_size=32):
    """
    Builds, compiles, and trains an LSTM model.
    
    Args:
        X_train (np.array): The 3D array of training input sequences.
        y_train (np.array): The 1D array of training target values.
        epochs (int): The number of training epochs.
        batch_size (int): The batch size for training.
        
    Returns:
        tensorflow.keras.Model: The trained LSTM model.
    """
    # 1. Build the LSTM Model
    model_lstm = Sequential()
    
    # Define the input shape from the training data
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model_lstm.add(Dropout(0.2))
    
    model_lstm.add(LSTM(units=50, return_sequences=False))
    model_lstm.add(Dropout(0.2))
    
    model_lstm.add(Dense(units=25))
    model_lstm.add(Dense(units=1))
    
    # 2. Compile the Model
    model_lstm.compile(optimizer='adam', loss='mean_squared_error')
    
    # Print a summary of the model's architecture
    print("--- LSTM Model Architecture ---")
    model_lstm.summary()
    
    # 3. Train the Model
    print("\n--- Training the LSTM Model ---")
    model_lstm.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )
    
    print("\nModel training complete.")
    return model_lstm

# In a file like lstm_modeler.py

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

def predict_and_evaluate_lstm(trained_model, X_test, y_test, scaler, test_index, time_step):
    """
    Uses a trained LSTM model to make predictions and evaluates their performance.

    Args:
        trained_model (keras.Model): The trained LSTM model.
        X_test (np.array): The 3D array of scaled test input sequences.
        y_test (np.array): The 1D array of scaled test target values.
        scaler (MinMaxScaler): The scaler fitted on the training data.
        test_index (pd.DatetimeIndex): The original index of the test set.
        time_step (int): The number of time steps used in the sequences.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The final forecast with a correct datetime index.
            - dict: A dictionary of performance metrics (MAE, RMSE, MAPE).
    """
    print("Making predictions on the test data...")
    # 1. Predict on scaled data
    lstm_predictions_scaled = trained_model.predict(X_test)

    # 2. Inverse transform predictions and actual values
    lstm_forecast = scaler.inverse_transform(lstm_predictions_scaled)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # 3. Align the forecast with the correct datetime index
    # THIS IS THE LINE TO FIX:
    # Use the length of the predictions to create an index of the exact same size.
    forecast_index = test_index[time_step : len(lstm_forecast) + time_step]
    forecast_df = pd.DataFrame(lstm_forecast, index=forecast_index, columns=['Forecast'])

    # 4. Calculate performance metrics
    mae = mean_absolute_error(y_test_actual, lstm_forecast)
    rmse = np.sqrt(mean_squared_error(y_test_actual, lstm_forecast))
    mape = mean_absolute_percentage_error(y_test_actual, lstm_forecast)

    metrics = {
        'Mean Absolute Error (MAE)': mae,
        'Root Mean Squared Error (RMSE)': rmse,
        'Mean Absolute Percentage Error (MAPE)': mape
    }

    # 5. Print the results
    print("\n--- LSTM Forecast Evaluation Metrics ---")
    for name, value in metrics.items():
        if "MAPE" in name:
            print(f"{name}: {value:.2%}")
        else:
            print(f"{name}: {value:.2f}")
            
    return forecast_df, metrics