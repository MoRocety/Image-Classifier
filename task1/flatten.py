import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image
import numpy as np

# Load your data
data = pd.read_csv('C:\\Users\\shahb\\Documents\\Machine Learning\\task1\\dataset.csv')

# Create a StratifiedShuffleSplit object
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

print(sss)

# Perform the split
for train_index, test_index in sss.split(data, data['Name']):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

# Define a function to convert an image into a feature vector
def image_to_feature_vector(image_path):
    image = Image.open(f'C:\\Users\\shahb\\Documents\\Machine Learning\\Dataset\\train\\{image_path}')  # Open the image
    feature_vector = np.array(image).flatten()  # Flatten the pixel values
    return feature_vector

# Apply the function to each image path in the training and testing sets
strat_train_set['Image_Features'] = strat_train_set['Image Paths'].apply(image_to_feature_vector)
strat_test_set['Image_Features'] = strat_test_set['Image Paths'].apply(image_to_feature_vector)

for i in strat_test_set["Image_Features"]:
    print(i)

for i in strat_train_set["Image_Features"]:
    print(i)

# Save the train and test sets to CSV without image features
strat_train_set.to_csv('C:\\Users\\shahb\\Documents\\Machine Learning\\task1\\stratified_train_set.csv', index=False)
strat_test_set.to_csv('C:\\Users\\shahb\\Documents\\Machine Learning\\task1\\stratified_test_set.csv', index=False)


