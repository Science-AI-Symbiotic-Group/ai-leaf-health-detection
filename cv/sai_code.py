import cv2
import numpy as np
from PIL import Image
from keras import models
from os import path, makedirs
import tensorflow as tf
from time import time


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


if not path.exists("captured_data"):
    makedirs("captured_data")


model = models.load_model('tensorflow_model.h5')
video = cv2.VideoCapture(0)

leaf_detected = False

while not leaf_detected:
        _, frame = video.read()
        #Convert the captured frame into RGB
        im = Image.fromarray(frame, 'RGB')
        #Resizing into dimensions you used while training
        im = im.resize((256,256))
        img_array = np.array(im)
        #Expand dimensions to match the 4D Tensor shape.
        img_array = np.expand_dims(img_array, axis=0)
        #Calling the predict function using keras
        prediction_value = model.predict(img_array)#[0][0]
        print(prediction_value)
        
        #Customize this part to your liking...
        if(prediction_value == 1 or prediction_value == 0):
            prediction_string = "NO LEAF"
            print(prediction_string)
        elif(prediction_value < 0.5 and prediction_value != 0):
            prediction_string = "HEALTHY LEAF"
            print(prediction_string)
        elif(prediction_value > 0.5 and prediction_value != 1):
            prediction_string = "UNHEALTHY LEAF"
            print(prediction_string)
        
        cv2.imshow("Prediction", frame)
        
        key=cv2.waitKey(1)
        if key == 32:
            file_time = int(time())


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