# THIS IS A CODE FILE WHICH HAS SOME ERRORS AND NO LONGER WORKS, ITS EDITED VERSION ALSO DOESNT WORK.

import cv2
import numpy as np
from PIL import Image
import os
import time
import tensorflow as tf

# Load the leaf detection model (replace 'your_leaf_model.h5' with the actual model)
model = tf.keras.models.load_model('tensorflow_model.h5')

# Access the webcam
video = cv2.VideoCapture(0)

# Create a directory to store captured leaf photos and medical reports
if not os.path.exists("captured_data"):
    os.makedirs("captured_data")

leaf_detected = False

while not leaf_detected:
    ret, frame = video.read()

    # Convert the captured frame into RGB
    im = Image.fromarray(frame, 'RGB')

    # Resize the frame to match the dimensions used during training
    im = im.resize((256, 256))
    img_array = np.array(im)

    # Expand dimensions to match the 4D tensor shape
    img_array = np.expand_dims(img_array, axis=0)

    # Make predictions using the loaded model
    prediction = model.predict(img_array)

    # Display the frame
    cv2.imshow("Leaf Detection", frame)
    key = cv2.waitKey(1)

    # Check if the space bar is pressed
    if key == 32:  # 32 is the ASCII code for space
        leaf_detected = True

# Capture a photo when the space bar is pressed
photo_filename = f"captured_data/leaf_{int(time.time())}.jpg"
cv2.imwrite(photo_filename, frame)

# Process the captured leaf photo for medical condition analysis (you should add your analysis code)

# Determine the health status based on the prediction
if prediction <= 0.5:
    medical_report = "Healthy"
else:
    medical_report = "Unhealthy"

# Save the medical report
report_filename = f"captured_data/report_{int(time.time())}.txt"
with open(report_filename, 'w') as report_file:
    report_file.write(medical_report)

# Release the webcam and close the OpenCV windows
video.release()
cv2.destroyAllWindows()
