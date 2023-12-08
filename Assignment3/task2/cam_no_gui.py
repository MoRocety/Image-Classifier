import joblib
import cv2
import numpy as np

# Load the models
age_model = joblib.load('Assignment3\\task1\\best_model.pkl')
expression_model = joblib.load('Assignment3\\task2\\logistic_reg_expression_model.pkl')
name_model = joblib.load('Assignment3\\task2\\logistic_reg_name_model.pkl')
# Open the webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam index

while True:
    ret, frame = cap.read()  # Read a frame from the webcam

    if not ret:
        break

    # Preprocess the frame (e.g., resize to 32x32)
    frame = cv2.resize(frame, (32, 32))

    # Flatten and reshape the frame for prediction
    flattened_frame = frame.flatten().reshape(1, -1)

    # Perform predictions using all the models
    prediction1 = age_model.predict(flattened_frame)
    prediction2 = expression_model.predict(flattened_frame)
    prediction3 = name_model.predict(flattened_frame)

    # Resize the frame for display
    frame = cv2.resize(frame, (640, 480))  # Adjust the dimensions as needed

    # Display the predictions on the frame
    cv2.putText(frame, f'Age: {prediction1[0]:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Expression: {prediction2[0]}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'name: {prediction3[0]}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Age Prediction', frame)

    if cv2.waitKey(1) & 0xFF in (27, ord('q')):  # Press 'Esc' or 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
