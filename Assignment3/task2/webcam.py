import joblib
import cv2
import numpy as np
import gradio as gr

# Load the models
age_model = joblib.load('Assignment3\\task1\\best_model.pkl')
expression_model = joblib.load('Assignment3\\task2\\logistic_reg_expression_model.pkl')
name_model = joblib.load('Assignment3\\task2\\logistic_reg_name_model.pkl')

# Define a function to predict all using the models
def predict_all(frame):
    # Check if the frame is not empty
    if frame is None or frame.size == 0:
        return frame

    # Flatten and reshape the frame for prediction
    flattened_frame = frame.flatten().reshape(1, -1)

    # Perform predictions using all three models
    age_prediction = age_model.predict(flattened_frame)

    # Debugging: Print intermediate results
    print("Age Prediction:", age_prediction)

    expression_prediction_proba = expression_model.predict_proba(flattened_frame)
    expression_predicted_class = expression_model.classes_[np.argmax(expression_prediction_proba)]

    # Debugging: Print intermediate results
    print("Expression Prediction Probabilities:", expression_prediction_proba)
    print("Expression Predicted Class:", expression_predicted_class)

    name_prediction_proba = name_model.predict_proba(flattened_frame)
    name_predicted_class = name_model.classes_[np.argmax(name_prediction_proba)]

    # Debugging: Print intermediate results
    print("Name Prediction Probabilities:", name_prediction_proba)
    print("Name Predicted Class:", name_predicted_class)

    return f'Age: {age_prediction[0]:.2f}\nExpression: {expression_predicted_class}\n' \
           f'Expression Probability: {expression_prediction_proba[0, np.argmax(expression_prediction_proba)]:.2f}\n' \
           f'Name: {name_predicted_class}\nName Probability: {name_prediction_proba[0, np.argmax(name_prediction_proba)]:.2f}'

# Use Gradio's Interface with Image input
iface = gr.Interface(fn=predict_all, inputs="image", outputs="text")
iface.launch()

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam index

while True:
    ret, frame = cap.read()  # Read a frame from the webcam

    # Update the Gradio interface with the new frame
    iface.update(frame)

    if cv2.waitKey(1) & 0xFF in (27, ord('q')):  # Press 'Esc' or 'q' to exit
        break

# Release the webcam and close the OpenCV window when the GUI is closed
cap.release()
cv2.destroyAllWindows()
