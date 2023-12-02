import os

# Set the directory where your image files are located
directory = 'C:\\Users\\shahb\\Documents\\Machine Learning\\Dataset\\original'

# Get a list of all image files in the directory
image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Start renaming from 200
start_index = 1

for old_name in image_files:
    # Create a new file name
    new_name = f"{start_index}.jpg"

    # Construct the full paths for old and new names
    old_path = os.path.join(directory, old_name)
    new_path = os.path.join(directory, new_name)

    # Rename the file
    os.rename(old_path, new_path)

    # Increment the start index
    start_index += 1

print("Renaming completed.")
