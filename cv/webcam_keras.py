import cv2
import numpy as np
from PIL import Image
from keras import models
from os import path, makedirs, strerror
import tensorflow as tf
from time import time
import argparse
import errno


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

# creating argument parser variable to get arguments from command line
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--camera_number",help="a integer to get the camera number, starts from 0")
argument_parser.add_argument("--model_path",help="specify the .h5 model path")


arguments = argument_parser.parse_args()
camera_number = arguments.camera_number # argument for what camera to use for "cv2.VideoCapture"
model_path = arguments.model_path # argument for the .h5 model math

camera_number = int(camera_number)

if not path.exists("captured_data"): # If the "captured_data" report folder does not exist, create it
    makedirs("captured_data")

try:
    model = models.load_model(model_path)
except OSError:
    raise FileNotFoundError(errno.ENOENT, strerror(errno.ENOENT), model_path) # raise's a FileNotFoundError if the model_path argument does not point to a real file.
    
video = cv2.VideoCapture(camera_number)

leaf_detected = False

while not leaf_detected:
        _, frame = video.read()
        #Convert the captured frame into RGB
        try:
            im = Image.fromarray(frame, 'RGB')
        except AttributeError:
            raise AttributeError(f"Camera number {camera_number} does not exist.") # raise's a AttributeError if the camera_number argument does not point to a real camera.

        #Resizing into dimensions that were used during training
        im = im.resize((256,256))
        img_array = np.array(im)
        
        #Expanding dimensions to match the 4D Tensor shape.
        img_array = np.expand_dims(img_array, axis=0)
        
        #Calling the predict function using keras
        prediction_value = model.predict(img_array) 
        print(prediction_value)

        
        if(prediction_value == 1 ):
            prediction_string = "HEALTHY LEAF"
            print("Leaf is healthy.")
        elif(prediction_value<1):
            prediction_string = "UNHEALTHY LEAF"
            print('Leaf is unhealthy.')
        
        
        cv2.imshow("Prediction", frame)
        
        key=cv2.waitKey(1)
        if key == 32: #32 Key code is for the "Spacebar"
            file_time = int(time()) # Get current time which will be used for the report name.


            # Generates the file paths for the Image and the Report from the generate_paths function.

            ImagePath = generate_paths(is_image=True,image_format=".jpg",file_time=file_time)
            ReportPath = generate_paths(is_image=False,file_time=file_time)

            # Writes the Frame into a Image with the format specified with the given path in "ImagePath"
            cv2.imwrite(ImagePath, frame)

            with open(ReportPath, 'w') as report_file:
                report_file.write(f"The Leaf in the image is a  {prediction_string}.")

        elif key == ord('q'):
             break
video.release()
cv2.destroyAllWindows()
