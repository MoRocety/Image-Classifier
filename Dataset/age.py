import cv2
import dlib
import math

# Load the pre-trained face detection model from dlib
face_detector = dlib.get_frontal_face_detector()

# Load the pre-trained facial landmarks model from dlib
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the pre-trained age estimation model from OpenCV
age_model = cv2.dnn.readNet("age_net.caffemodel", "age_deploy.prototxt")

# Load the pre-trained smile detection model from OpenCV
smile_model = cv2.dnn.readNet("smile_net.caffemodel", "smile_deploy.prototxt")

# Function to estimate age from a face image
def estimate_age(face):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    age_model.setInput(blob)
    age_preds = age_model.forward()
    age = int(age_preds[0].dot(list(range(0, 101))) )
    return age

# Function to detect smiles in a face image
def detect_smile(face):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    smile_model.setInput(blob)
    smile_preds = smile_model.forward()
    smile = smile_preds[0][0]
    return smile > 0.5

# Function to process an image and determine smile and age
def process_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_detector(gray)

    for face in faces:
        landmarks = landmark_predictor(gray, face)

        # Extract coordinates of mouth corners
        left_mouth_corner = (landmarks.part(48).x, landmarks.part(48).y)
        right_mouth_corner = (landmarks.part(54).x, landmarks.part(54).y)

        # Calculate the width of the mouth
        mouth_width = math.dist(left_mouth_corner, right_mouth_corner)

        # Extract the face region
        face = image[face.top():face.bottom(), face.left():face.right()]

        # Estimate age
        age = estimate_age(face)

        # Detect smile
        is_smiling = detect_smile(face)

        label = "Smiling" if is_smiling else "Not Smiling"
        print(f"Age: {age}, Smile: {label}")

# Directory containing the images to process
image_directory = 'C:\\Users\\shahb\\Documents\\Machine Learning\\Dataset\\original'

import os
for filename in os.listdir(image_directory):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        image_path = os.path.join(image_directory, filename)
        process_image(image_path)

