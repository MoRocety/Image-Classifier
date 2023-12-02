import os
import shutil

# Set the directory where your original image files are located
original_directory = 'C:\\Users\\shahb\\Documents\\Machine Learning\\Dataset\\original'

# Set the directory where you want to store the renamed images
new_directory = 'C:\\Users\\shahb\\Documents\\Machine Learning\\Dataset\\new'

# Create the new directory if it doesn't exist
if not os.path.exists(new_directory):
    os.makedirs(new_directory)

# Get a list of all image files in the original directory
image_files = [f for f in os.listdir(original_directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Start renaming from 1
start_index = 1

for old_name in image_files:
    # Create a new file name with the '.jpg' extension
    new_name = f"{start_index}.jpg"

    # Construct the full paths for old and new names
    old_path = os.path.join(original_directory, old_name)
    new_path = os.path.join(new_directory, new_name)

    # Rename and move the file to the new directory
    shutil.move(old_path, new_path)

    # Increment the start index
    start_index += 1

print("Renaming and moving completed.")
