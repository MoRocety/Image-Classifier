import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from flatten import *
from task1 import *
# Extract features and labels for training
X_train = np.vstack(strat_train_set['Image_Features'].values)
y_train = strat_train_set['Age'] 

# Extract features and labels for testing
X_test = np.vstack(strat_test_set['Image_Features'].values)
y_test = strat_test_set['Age'] 

print(X_train)
print(y_train)
print(X_test)
print(y_test)
# Train SGD Linear Regressor
sgd_model = SGDRegressor()
sgd_model.fit(X_train, y_train)

# Predict using SGD Linear Regressor
y_pred_sgd = sgd_model.predict(X_test)

# Calculate R-squared and MSE for SGD Linear Regressor
r2_sgd = r2_score(y_test, y_pred_sgd)
mse_sgd = mean_squared_error(y_test, y_pred_sgd)

# Print metrics for SGD Linear Regressor
print("SGD Linear Regressor:")
print("R-squared:", r2_sgd)
print("Mean Squared Error:", mse_sgd)

# Compare with OLS Linear Regression
y_pred_ols = model.predict(X_test)

# Calculate R-squared and MSE for OLS Linear Regression
r2_ols = r2_score(y_test, y_pred_ols)
mse_ols = mean_squared_error(y_test, y_pred_ols)

# Print metrics for OLS Linear Regression
print("\nOLS Linear Regression:")
print("R-squared:", r2_ols)
print("Mean Squared Error:", mse_ols)

# Compare the metrics
print("\nComparison of Metrics:")
print("R-squared (SGD vs OLS):", r2_sgd, "vs", r2_ols)
print("Mean Squared Error (SGD vs OLS):", mse_sgd, "vs", mse_ols)
