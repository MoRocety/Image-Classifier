import os
from PIL import Image
def resize_images(input_folder, target_width, target_height):
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # Check if the file is an image (e.g., JPEG, PNG)
        if os.path.isfile(input_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                # Open the image file
                with Image.open(input_path) as img:
                    # Resize the image
                    resized_img = img.resize((target_width, target_height))

                    # Replace the original image with the resized one
                    resized_img.save(input_path)
                    print(f'Successfully resized {filename} to {target_width}x{target_height} pixels and replaced the original.')

            except Exception as e:
                print(f'Error resizing {filename}: {str(e)}')

# Set the input folder and target dimensions
input_folder = 'C:\\Users\\shahb\\Documents\\Machine Learning\\Assignment3\\task2\\testimages'
target_width = 32
target_height = 32

# Call the function to resize images and replace them
resize_images(input_folder, target_width, target_height)
