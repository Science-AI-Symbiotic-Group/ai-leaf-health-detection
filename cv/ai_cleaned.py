import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import tensorflow as tf

def load_pretrained_model(model_path):
    """Load the pre-trained Keras model."""
    return load_model(model_path)

def preprocess_frame(frame):
    """Preprocess the captured frame for model prediction."""
    resized_frame = cv2.resize(frame, (256, 256))
    img_array = np.expand_dims(resized_frame, axis=0)
    return img_array

def predict_leaf(model, img_array, threshold=0.5):
    """Make predictions using the loaded model."""
    prediction = model.predict(img_array)
    return prediction[0][0] > threshold

def main():
    # Load the pre-trained model
    model_path = 'tensorflow_model.h5'
    model = load_pretrained_model(model_path)

    # Open the video capture device (default camera)
    video = cv2.VideoCapture(0)

    while True:
        # Read a frame from the video capture
        _, frame = video.read()

        # Preprocess the frame for prediction
        img_array = preprocess_frame(frame)

        # Make predictions using the loaded model
        is_leaf = predict_leaf(model, img_array)

        # Customize this part to your liking...
        label = "Leaf Detected" if is_leaf else "No Leaf"

        # Put the label on the frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame with the prediction label
        cv2.imshow("Leaf Detection", frame)

        # Check for 'q' key to exit the loop
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Release the video capture and close all windows
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
