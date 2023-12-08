import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from flatten import *

# Load the saved OLS Linear Regression model from task1
with open('Assignment2\\linear_regression_model.pkl', 'rb') as model_file:
    ols_model = pickle.load(model_file)

# Extract features and labels for training and testing
X_train = np.vstack(strat_train_set['Image_Features'].values)
y_train = strat_train_set['Age'] 

X_test = np.vstack(strat_test_set['Image_Features'].values)
y_test = strat_test_set['Age'] 

# Print metrics for OLS Linear Regression
y_pred_ols = ols_model.predict(X_test)

# Calculate R-squared and MSE for OLS Linear Regression
r2_ols = r2_score(y_test, y_pred_ols)
mse_ols = mean_squared_error(y_test, y_pred_ols)

print("OLS Linear Regression (from task1):")
print("R-squared:", r2_ols)
print("Mean Squared Error:", mse_ols)

# Train the SGD Linear Regressor with a fixed random state
sgd_model = SGDRegressor(random_state=300) 
sgd_model.fit(X_train, y_train)

# Save the trained SGD model to a file
with open('Assignment2\\sgd_regression_model.pkl', 'wb') as model_file:
    pickle.dump(sgd_model, model_file)

# Get predictions using the trained model for testing split
y_pred_sgd = sgd_model.predict(X_test)

# Calculate R-squared and MSE for SGD Linear Regression
r2_sgd = r2_score(y_test, y_pred_sgd)
mse_sgd = mean_squared_error(y_test, y_pred_sgd)

# Print learned coefficients for the SGD model
print("Learned coefficients (thetas):")
print("Intercept:", sgd_model.intercept_)
print("Coefficients (Weights):", sgd_model.coef_)

print("\nSGD Linear Regressor:")
print("R-squared:", r2_sgd)
print("Mean Squared Error:", mse_sgd)

print("\nComparison of Metrics:")
print("R-squared (SGD vs OLS):", r2_sgd, "vs", r2_ols)
print("Mean Squared Error (SGD vs OLS):", mse_sgd, "vs", mse_ols)
