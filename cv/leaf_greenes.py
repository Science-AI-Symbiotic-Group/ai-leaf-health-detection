import cv2
import numpy as np

def calculate_greenness(frame):
    # Convert the frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of green color in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Threshold the image to get only green regions
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Calculate the percentage of green pixels
    total_pixels = frame.shape[0] * frame.shape[1]
    green_pixels = cv2.countNonZero(green_mask)
    greenness_percentage = (green_pixels / total_pixels) * 100

    return greenness_percentage

# Open a connection to the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Calculate greenness
    greenness = calculate_greenness(frame)

    # Display the frame and greenness value
    cv2.putText(frame, f'Greenness: {greenness:.2f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Leaf Greenness Detection', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
