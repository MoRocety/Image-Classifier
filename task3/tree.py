import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image
import numpy as np
import sys
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
sys.path.append('C:\\Users\\shahb\\Documents\\Machine Learning\\task1')

from flatten import strat_train_set, strat_test_set

# Convert labels to binary format
strat_train_set['Name'] = strat_train_set['Name'].map({'Muhammad Bhatti': 1, 'unknown': 0})
strat_test_set['Name'] = strat_test_set['Name'].map({'Muhammad Bhatti': 1, 'unknown': 0})

X_train = np.vstack(strat_train_set['Image_Features'].values)
y_train = strat_train_set['Name']
X_test = np.vstack(strat_test_set['Image_Features'].values)
y_test = strat_test_set['Name']

# Calculate mutual information (information gain) for each feature
mi = mutual_info_classif(X_train, y_train, random_state=42)

# Get indices of top 10 features
indices = np.argsort(mi)[-10:]

# Print mutual information for each of the top 10 features
print("Top 10 features:")
for i in indices:
    print(f"Feature {i}: {mi[i]}")

from sklearn.feature_selection import SelectKBest

# Create a SelectKBest object to select features with the 10 highest mutual information
selector = SelectKBest(mutual_info_classif, k=15)

# Use the selector to retrieve the 10 best features
X_train_kbest = selector.fit_transform(X_train, y_train)
X_test_kbest = selector.transform(X_test)

def train_and_evaluate(X_train, y_train, X_test, y_test):
    # Train decision tree with entropy as criterion
    clf = DecisionTreeClassifier(criterion='gini', random_state=42)
    clf.fit(X_train, y_train)

    # Predict on test data
    y_pred = clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 score:", f1_score(y_test, y_pred))


#Image classification Dataset

print("\nMetrics for binary labels me/unknown without feature selection:")
train_and_evaluate(X_train, y_train, X_test, y_test)


print("\nMetrics for binary labels me/unknown with feature selection:")
train_and_evaluate(X_train_kbest, y_train, X_test_kbest, y_test)