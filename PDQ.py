import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from itertools import product
import time

# Load your dataset
data = pd.read_csv("C:/Users/aarav/Desktop/computer science/EE/new_database.csv")
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Remove duplicate dates
data = data[~data.index.duplicated(keep='first')]

# Ensure the Date index has a frequency
data = data.asfreq('D')  # Set frequency to daily ('D')

# Split data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data['Sales'][:train_size], data['Sales'][train_size:]

# Define ranges for p, d, q to search for the best parameters
p = range(0, 10)  # Adjust these ranges as needed
d = range(0, 10)
q = range(0, 10)
pdq_combinations = list(product(p, d, q))

# Initialize variables to store the best parameters and metrics
best_pdq = None
lowest_rmse = float("inf")

# Grid search over pdq combinations
for pdq in pdq_combinations:
    try:
        # Track time taken for model training
        start_time = time.time()


        # Fit ARIMA model with the current (p, d, q)
        model = ARIMA(train, order=pdq)
        arima_model = model.fit()

        # Forecast on the test set
        forecast = arima_model.forecast(steps=len(test))

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(test, forecast))

        print(f"ARIMA{pdq} - RMSE: {rmse:.4f}, Time Taken: {time.time() - start_time:.2f} seconds")

        # Update the best pdq if this combination has the lowest RMSE
        if rmse < lowest_rmse:
            best_pdq = pdq
            lowest_rmse = rmse

    except Exception as e:
        print(f"ARIMA{pdq} failed to fit: {e}")

# Output the best pdq parameters and corresponding RMSE
print("\nBest Model:")
print(f"Order (p, d, q): {best_pdq}")
print(f"Lowest RMSE: {lowest_rmse:.4f}")

# Fit the best model on the entire training data
best_model = ARIMA(train, order=best_pdq).fit()

# Forecast with the best model
forecast = best_model.forecast(steps=len(test))

# Plotting actual vs forecasted values
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Training Sales', color='blue')
plt.plot(test.index, test, label='Actual Sales', color='green')
plt.plot(test.index, forecast, label='Forecasted Sales', color='orange', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title(f'Actual vs Forecasted Sales with ARIMA{best_pdq}')
plt.legend()
plt.show()
