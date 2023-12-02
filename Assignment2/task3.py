import joblib
import cv2

# Load the models
model1 = joblib.load('Assignment2\linear_regression_model.pkl')
model2 = joblib.load('Assignment2\sgd_regression_model.pkl')

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam index

while True:
    ret, frame = cap.read()  # Read a frame from the webcam

    if not ret:
        break

    # Preprocess the frame (e.g., resize to 32x32)
    frame = cv2.resize(frame, (32, 32))

    # Perform predictions using your models
    prediction1 = model1.predict(frame.flatten().reshape(1, -1))
    prediction2 = model2.predict(frame.flatten().reshape(1, -1))

    # Resize the frame for display
    frame = cv2.resize(frame, (640, 480))  # Adjust the dimensions as needed

    # Display the predictions on the frame
    cv2.putText(frame, f'SGD: {prediction1[0]:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'OLS: {prediction2[0]:.2f}', (440, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Age Prediction', frame)

    if cv2.waitKey(1) & 0xFF in (27, ord('q')):  # Press 'Esc' or 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
