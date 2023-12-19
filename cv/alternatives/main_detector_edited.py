# THIS IS A EDITED VERSION OF main_code.py WHICH NO LONGER WORKS.



import cv2
import numpy as np
from PIL import Image
from os import makedirs, path
from time import time
import tensorflow as tf


def generate_paths(is_image,file_time,image_format=".jpg",): # A simple utility function which returns the file path and name for a image/report file based on the current time.
    file_path = f"captured_data/report_{file_time}"
    if path.isdir(f'captured_data/report_{file_time}'):
            for i in range(500):
                if path.isdir(f'captured_data/report_{file_time}_{i}'):
                    file_path = f"captured_data/report_{file_time}_{i+1}" 
                    makedirs(file_path)

    else:
        makedirs(f"captured_data/report_{file_time}")

    if is_image == True:
        if image_format.startswith("."):
            return f"{file_path}/leaf_{file_time}{image_format}"
        else:
            return f"{file_path}/leaf_{file_time}.{image_format}"
    else:
        return f"captured_data/report_{file_time}/report_{file_time}.txt"




# Loads the leaf detection model in a try/except block
try:
    model = tf.keras.models.load_model('tensorflow_model.h5')
except FileNotFoundError as error: # Catches a "FileNotFoundError" if the file path for the model file is incorrect.
    print(error)
    print("Tensorflow Model was not found in the file system, Change the File Path in the code or the execute the file from the command prompt.")

print("Loaded Model")

# Access the webcam
video = cv2.VideoCapture(0)

# Create a directory to store captured leaf photos and medical reports
if not path.exists("captured_data"):
    makedirs("captured_data")
    

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

    
    # Display the frame
    cv2.imshow("Leaf Detection", frame)
    key = cv2.waitKey(1)

    # Cheking if the Spacebar is pressed, Once it is pressed the while loop is broken and the code outside of it is executed.
    if key == 32:  # 32 is the ASCII code for space
        # Make predictions using the loaded model
        prediction = model.predict(img_array)
        
        file_time = int(time())


        # Generates the file paths for the Image and the Report from the generate_paths function.

        ImagePath = generate_paths(is_image=True,image_format=".jpg",file_time=file_time)
        ReportPath = generate_paths(is_image=False,file_time=file_time)


        # Writes the Frame into a Image with the format specified with the given path in "ImagePath"
        cv2.imwrite(ImagePath, frame)

        # Determine the health status based on the prediction
        if prediction <= 0.5:
            medical_report = "Healthy"
        else:
            medical_report = "Unhealthy"



        # Save the medical report
            
        with open(ReportPath, 'w') as report_file:
            report_file.write(f"The Leaf in the image is {medical_report}.")
    
    elif key == 27:
        leaf_detected = True


# Release the webcam and close the OpenCV windows
video.release()
cv2.destroyAllWindows()
