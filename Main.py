import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

# Load your dataset
data = pd.read_csv("C:/Users/aarav/OneDrive/Desktop/computer science/EE/new_database.csv")
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Remove duplicate dates
data = data[~data.index.duplicated(keep='first')]  # Keep the first occurrence of each duplicate

# Ensure the Date index has a frequency
data = data.asfreq('D')  # Set frequency to daily ('D'); adjust as needed (e.g., 'M' for monthly)

# Split data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data['Sales'][:train_size], data['Sales'][train_size:]

# Track time taken for model training
start_time = time.time()
model = ARIMA(train, order=(5, 0, 7))  # Adjust order as needed
arima_model = model.fit()
training_time = time.time() - start_time
print(f"Training Time: {training_time:.2f} seconds")

# Forecast with the fitted model
start_time = time.time()
forecast = arima_model.forecast(steps=len(test))
forecasting_time = time.time() - start_time
print(f"Forecasting Time: {forecasting_time:.2f} seconds")

# Rolling Forecast (for robustness)
rolling_forecast = []
rolling_rmse_list = []

for i in range(len(test)):
    # Fit the model on the initial training + test subset up to point i
    train_subset = data['Sales'][:train_size + i]
    model = ARIMA(train_subset, order=(5, 0, 7))
    rolling_arima_model = model.fit()

    # Forecast one step ahead and extract the value
    forecasted_value = rolling_arima_model.forecast(steps=1).iloc[0]
    rolling_forecast.append(forecasted_value)

    # Calculate RMSE for each step
    if i > 0:
        rmse_step = np.sqrt(mean_squared_error(test[:i], rolling_forecast[:i]))
        rolling_rmse_list.append(rmse_step)

# Convert rolling forecast to a Series for plotting
rolling_forecast_series = pd.Series(rolling_forecast, index=test.index)

# Residual Analysis for Robustness
residuals = test - rolling_forecast_series
mean_residual = residuals.mean()
std_residual = residuals.std()

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Training Sales', color='blue')
plt.plot(test.index, test, label='Actual Sales', color='green')
plt.plot(rolling_forecast_series.index, rolling_forecast_series, label='Rolling Forecasted Sales (ARIMA)',
         color='orange', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Actual vs Rolling Forecasted Sales')
plt.legend()
plt.show()

# Plot Residuals
plt.figure(figsize=(10, 4))
plt.plot(residuals.index, residuals, label='Residuals', color='purple')
plt.axhline(y=mean_residual, color='gray', linestyle='--', label=f'Mean Residual ({mean_residual:.2f})')
plt.fill_between(residuals.index, mean_residual - std_residual, mean_residual + std_residual, color='gray', alpha=0.3,
                 label='1 Std Dev')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.title('Residual Analysis for Robustness')
plt.legend()
plt.show()

# Calculate accuracy metrics
rmse = np.sqrt(mean_squared_error(test, forecast))
mae = mean_absolute_error(test, forecast)
mean_rolling_rmse = np.mean(rolling_rmse_list) if rolling_rmse_list else None
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(
    f'Average Rolling RMSE (Robustness Indicator): {mean_rolling_rmse:.2f}' if mean_rolling_rmse else "Not enough data for rolling RMSE")
print(f"Mean Residual (Robustness Indicator): {mean_residual:.2f}")
print(f"Standard Deviation of Residuals (Stability Indicator): {std_residual:.2f}")
