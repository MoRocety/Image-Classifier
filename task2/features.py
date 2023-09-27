import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif

# a. Read both csv files
train = pd.read_csv('C:\\Users\\shahb\\Documents\\Machine Learning\\task2\\test.csv')
test = pd.read_csv('C:\\Users\\shahb\\Documents\\Machine Learning\\task2\\train.csv')

# b. remove all <br /> tokens
train['review'] = train['review'].apply(lambda x: re.sub('<br />', ' ', x))
test['review'] = test['review'].apply(lambda x: re.sub('<br />', ' ', x))

# c. Change labels from text to integers (0 for negative and 1 for positive)
# Map labels to integers
le = LabelEncoder()
train['sentiment'] = le.fit_transform(train['sentiment'])
test['sentiment'] = le.transform(test['sentiment'])

# d. Separate out labels and texts
train_texts = train['review']
train_labels = train['sentiment']
test_texts = test['review']
test_labels = test['sentiment']

# e. Convert them to vectors based on frequency (counts) using scikit-learn
vectorizer_freq = CountVectorizer()
# Each unique word is assigned a unique integer index,
# and this mapping from words to integers is stored in the vectorizer.

train_vectors_freq = vectorizer_freq.fit_transform(train_texts)

# Do not fit the vectorizer on test data, 
# because in a real-world scenario, no access to this data during training. 
test_vectors_freq = vectorizer_freq.transform(test_texts)


############################## 2. Feature Importance #############################

# 2a. Convert them to vectors based on binary occurrences of words

vectorizer_binary = CountVectorizer(binary=True)
# Fit the vectorizer and transform the training data
train_vectors_bin = vectorizer_binary.fit_transform(train_texts)
# Transform the test data
test_vectors_bin = vectorizer_binary.transform(test_texts)

# 2b. Calculate feature importance using information gain
info_gain = mutual_info_classif(train_vectors_bin, train_labels, discrete_features=True, random_state=42)
feature_names = vectorizer_binary.get_feature_names_out()


features_info_gain = pd.DataFrame({'feature': feature_names, 'info_gain': info_gain})
features_info_gain_sorted = features_info_gain.sort_values(by='info_gain', ascending=False)

print("\nWith feature selection")
print("Top 10 features:")
print(features_info_gain_sorted.head(10))

print("\nLeast 10 features:")
print(features_info_gain_sorted.tail(10))



