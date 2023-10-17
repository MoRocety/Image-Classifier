import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

from flatten import *
from task1 import *
# Extract features and labels for training
X_train = np.vstack(strat_train_set['Image_Features'].values)
y_train = strat_train_set['Age'] 

# Extract features and labels for testing
X_test = np.vstack(strat_test_set['Image_Features'].values)
y_test = strat_test_set['Age'] 

# Assume you have X_train, X_test, y_train, and y_test defined

# Initialize the SGD Regressor
sgd_model = SGDRegressor(max_iter=1000, tol=1e-3)

# Train the model and keep track of the loss
train_losses = []
test_losses = []

n_iterations = 1000  # Adjust this based on your needs

for i in range(n_iterations):
    sgd_model.partial_fit(X_train, y_train)  # Perform one epoch of training

    # Calculate training loss
    train_loss = mean_squared_error(y_train, sgd_model.predict(X_train))
    train_losses.append(train_loss)

    # Calculate test loss
    test_loss = mean_squared_error(y_test, sgd_model.predict(X_test))
    test_losses.append(test_loss)

# Plot the loss curve
plt.figure(figsize=(12, 6))
plt.plot(range(1, n_iterations + 1), train_losses, label='Training Loss')
plt.plot(range(1, n_iterations + 1), test_losses, label='Test Loss')
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()
