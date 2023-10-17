import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

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

# Predict ages on the test set
y_pred = model.predict(X_test)

# Compute R-squared and MSE
r_squared = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Print the R-squared and MSE
print("R-squared on test set:", r_squared)
print("Mean Squared Error on test set:", mse)
