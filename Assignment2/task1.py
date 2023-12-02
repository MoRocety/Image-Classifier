import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import pickle
from flatten import *

# Extract features and labels for training
X_train = np.vstack(strat_train_set['Image_Features'].values)
y_train = strat_train_set['Age'] 

# Extract features and labels for testing
X_test = np.vstack(strat_test_set['Image_Features'].values)
y_test = strat_test_set['Age'] 

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Print the learned regression coefficients
print("Learned coefficients (weights):", model.coef_)
print("Intercept term:", model.intercept_)

# Save the trained model to a file
with open('linear_regression_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Get predictions using the trained model for the testing split
y_pred = model.predict(X_test)

# Compute R-squared and Mean Squared Error (MSE) for predictions
r_squared = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Print the R-squared and MSE
print("R-squared on the test set:", r_squared)
print("Mean Squared Error on the test set:", mse)
