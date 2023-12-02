import joblib
import cv2
import numpy as np

# Load the models
model1 = joblib.load('task1/best_model.pkl')
model2 = joblib.load('task1/sgd_regression_model.pkl')
model3 = joblib.load('task1/elastic_net_model.pkl')
model4 = joblib.load('task1/ridge_model.pkl')
model5 = joblib.load('task1/sgd_regression_optimal_model.pkl')

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
    prediction1 = model1.predict(flattened_frame)
    prediction2 = model2.predict(flattened_frame)
    prediction3 = model3.predict(flattened_frame)
    prediction4 = model4.predict(flattened_frame)
    prediction5 = model5.predict(flattened_frame)

    # Resize the frame for display
    frame = cv2.resize(frame, (640, 480))  # Adjust the dimensions as needed

    # Display the predictions on the frame
    cv2.putText(frame, f'Best Model: {prediction1[0]:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'SGD: {prediction2[0]:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Elastic Net: {prediction3[0]:.2f}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Ridge: {prediction4[0]:.2f}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'SGD Optimal: {prediction5[0]:.2f}', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Age Prediction', frame)

    if cv2.waitKey(1) & 0xFF in (27, ord('q')):  # Press 'Esc' or 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
