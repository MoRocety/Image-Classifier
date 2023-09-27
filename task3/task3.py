import sys
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append('C:\\Users\\shahb\\Documents\\Machine Learning\\task1')
sys.path.append('C:\\Users\\shahb\\Documents\\Machine Learning\\task2')

from flatten import *
from features import *

# Select the top k features
k = 10
selector = SelectKBest(mutual_info_classif, k=k)
train_vectors_bin_selected = selector.fit_transform(train_vectors_bin, train_labels)
test_vectors_bin_selected = selector.transform(test_vectors_bin)

# Function to train a decision tree and print metrics
def train_and_evaluate(X_train, y_train, X_test, y_test):
    # Train decision tree with entropy as criterion
    clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
    clf.fit(X_train, y_train)

    # Predict on test data
    y_pred = clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 score:", f1_score(y_test, y_pred))

print("\nMetrics for train and test split without feature selection:")
train_and_evaluate(train_vectors_freq, train_labels, test_vectors_freq, test_labels)

print("\nMetrics for train and test with feature selection:")
train_and_evaluate(train_vectors_bin_selected, train_labels, test_vectors_bin_selected, test_labels)

#Image classification Dataset
strat_train_set['Name'] = strat_train_set['Name'].map({'Muhammad Bhatti': 1, 'unknown': 0})
strat_test_set['Name'] = strat_test_set['Name'].map({'Muhammad Bhatti': 1, 'unknown': 0})

X_train = np.vstack(strat_train_set['Image_Features'].values)
y_train = strat_train_set['Name']
X_test = np.vstack(strat_test_set['Image_Features'].values)
y_test = strat_test_set['Name']


print("\nMetrics for binary labels me/unknown:")
train_and_evaluate(X_train, y_train, X_test, y_test)
