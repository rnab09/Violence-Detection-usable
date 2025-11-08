import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# Parameters
SEQUENCE_LENGTH = 16
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
CLASSES_LIST = ["NonFight", "Fight"]
SMOOTHING_WINDOW = 10  # Number of recent predictions to smooth output
PREDICTION_THRESHOLD = 0.6  # Confidence threshold for displaying predictions

# Load the trained model
model_path = "CNN_model.h5"  # Update with the path to your saved model
model = load_model(model_path)

# Initialize video capture (webcam)
cap = cv2.VideoCapture(0)

# Buffer to store recent predictions
recent_predictions = deque(maxlen=SMOOTHING_WINDOW)

# Helper function to process frames
def process_frame(frame):
    resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
    normalized_frame = resized_frame / 255.0
    return normalized_frame

# Read frames from the webcam
frame_buffer = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error accessing webcam.")
        break

    # Preprocess the frame and add to the buffer
    processed_frame = process_frame(frame)
    frame_buffer.append(processed_frame)

    # Ensure buffer contains the required number of frames
    if len(frame_buffer) == SEQUENCE_LENGTH:
        # Convert buffer to numpy array and predict
        input_frames = np.expand_dims(frame_buffer, axis=0)  # Shape: (1, 16, 64, 64, 3)
        predictions = model.predict(input_frames, verbose=0)[0]  # Get prediction scores
        predicted_class = np.argmax(predictions)  # Get class index
        confidence = predictions[predicted_class]  # Get confidence score

        # Add the current prediction to the recent predictions
        recent_predictions.append(predicted_class)

        # Smooth prediction using the recent predictions buffer
        smoothed_prediction = np.argmax(np.bincount(recent_predictions))

        # Display prediction with confidence
        if smoothed_prediction == 1 and confidence >= PREDICTION_THRESHOLD:
            label = "Fight"
            color = (0, 0, 255)  # Red
        else:
            label = "NonFight"
            color = (0, 255, 0)  # Green

        # Overlay prediction on the frame
        cv2.putText(frame, f"{label} ({confidence:.2f})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Remove the oldest frame from the buffer
        frame_buffer.pop(0)

    # Show the frame with predictions
    cv2.imshow("Violence Detection", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()


# from tensorflow.keras.models import load_model
# import cv2
# import numpy as np

# IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
# SEQUENCE_LENGTH = 16
# CLASSES_LIST = ["NonFight", "Fight"]

# # Load the pre-trained model
# model = load_model("CNN_model.h5")

# def predict_live_video(model):
#     """
#     Predict violence in real-time from a webcam feed.
    
#     Args:
#     - model (tf.keras.Model): Trained Keras model.
#     """
#     # Initialize the webcam
#     video_capture = cv2.VideoCapture(0)
    
#     if not video_capture.isOpened():
#         print("Error: Unable to access the webcam.")
#         return
    
#     print("Press 'q' to quit the webcam feed.")
    
#     while True:
#         # Read a frame from the webcam
#         ret, frame = video_capture.read()
#         if not ret:
#             print("Error: Unable to read frame from the webcam.")
#             break

#         # Resize and normalize the frame
#         resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
#         normalized_frame = resized_frame / 255.0

#         # Stack the same frame to simulate a sequence for prediction
#         # (Since we are processing live frames, we will replicate the current frame for SEQUENCE_LENGTH)
#         frames_list = [normalized_frame] * SEQUENCE_LENGTH
#         video_input = np.asarray(frames_list)
#         video_input = video_input.reshape(1, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)

#         # Predict the class probabilities
#         prediction_probs = model.predict(video_input)
#         predicted_class = np.argmax(prediction_probs)
#         prediction_label = CLASSES_LIST[predicted_class]
#         prediction_prob = prediction_probs[0][predicted_class]

#         # Display prediction on the video feed
#         cv2.putText(
#             frame,
#             f"Prediction: {prediction_label} ({prediction_prob:.2f})",
#             (10, 30),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             (0, 255, 0) if prediction_label == "NonFight" else (0, 0, 255),
#             2,
#         )
        
#         # Show the video feed
#         cv2.imshow("Live Violence Detection", frame)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the webcam and close the window
#     video_capture.release()
#     cv2.destroyAllWindows()

# # Call the function to start real-time prediction
# predict_live_video(model)


