import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, precision_recall_fscore_support
import pickle
from flatten import *

# Extract features and labels for training and testing
X_train = np.vstack(strat_train_set['Image_Features'].values)
y_train = strat_train_set['Name'] 

X_test = np.vstack(strat_test_set['Image_Features'].values)
y_test = strat_test_set['Name'] 

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the trained scaler
with open('Assignment3\\task2\\scaler.pkl', 'wb') as model_file:
    pickle.dump(scaler, model_file)

# Train Logistic Regression Classifier
logistic_reg_model = LogisticRegression(random_state=42, max_iter=1000, penalty='l2')
logistic_reg_model.fit(X_train_scaled, y_train)

# Train Decision Tree Classifier
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)

# Make predictions
y_pred_logistic_reg = logistic_reg_model.predict(X_test_scaled)
y_pred_decision_tree = decision_tree_model.predict(X_test)

# Calculate F1-score for Logistic Regression and Decision Tree
f1_score_logistic_reg = f1_score(y_test, y_pred_logistic_reg, pos_label='Me')
f1_score_decision_tree = f1_score(y_test, y_pred_decision_tree, pos_label='Me')

# Calculate Log Loss for Logistic Regression (assuming binary classification)
# Use predict_proba to get probability estimates
y_proba_logistic_reg = logistic_reg_model.predict_proba(X_test_scaled)[:, 1]
log_loss_logistic_reg = log_loss(y_test, y_proba_logistic_reg)

# Print F1-scores and Log Loss
print("Me and Not me models:")
print("F1-Score for Logistic Regression:", f1_score_logistic_reg)
print("F1-Score for Decision Tree:", f1_score_decision_tree)
print("Log Loss for Logistic Regression:", log_loss_logistic_reg)

# Save the best model as pickle dump
with open('Assignment3\\task2\\logistic_reg_name_model.pkl', 'wb') as model_file:
    pickle.dump(logistic_reg_model, model_file)

# Assuming 'Expression' column contains labels like 'Smiling', 'Neutral', 'Angry'
X_train = np.vstack(strat_train_set['Image_Features'].values)
y_train = strat_train_set['Expression']

X_test = np.vstack(strat_test_set['Image_Features'].values)
y_test = strat_test_set['Expression']


# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Choose 'l1' or 'l2' for the penalty based on your preference
logistic_reg_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=0.001, penalty='l2', max_iter=1000, random_state=42)

# Assuming you have your features and labels as X_train, y_train
logistic_reg_model.fit(X_train_scaled, y_train)

# Now you can evaluate your model on the test set
y_pred_proba = logistic_reg_model.predict_proba(X_test_scaled)

# Compute log loss
logloss = log_loss(y_test, y_pred_proba)

# Compute macro-averaged precision, recall, and f1-score
y_pred = logistic_reg_model.predict(X_test_scaled)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

# Print the results
print("\nLog Loss for Expressions:", logloss)
print("Macro-Averaged Precision:", precision)
print("Macro-Averaged Recall:", recall)
print("Macro-Averaged F1-Score:", f1)

# Save the best model as pickle dump
with open('Assignment3\\task2\\logistic_reg_expression_model.pkl', 'wb') as model_file:
    pickle.dump(logistic_reg_model, model_file)