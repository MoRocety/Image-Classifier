import os
from PIL import Image

def convert_images(directory):
    for filename in os.listdir(directory):
        
        if filename.endswith(".png") or filename.endswith(".jpeg") or filename.endswith(".JPG"):
            img = Image.open(os.path.join(directory, filename))
            rgb_img = img.convert('RGB')
            rgb_img.save(os.path.join(directory, filename.split('.')[0] + '.jpg'), 'JPEG')
            os.remove(os.path.join(directory, filename))

convert_images("C:\\Users\\shahb\\Documents\\Machine Learning\\Dataset\\train")