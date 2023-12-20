import cv2
import numpy as np
from PIL import Image
from keras import models
import tensorflow as tf
import argparse
import errno
from os import strerror


argument_parser = argparse.ArgumentParser()

argument_parser.add_argument("image",help="Image path of the code to run")
argument_parser.add_argument("model_path",help="specify the .h5 models path")

arguments = argument_parser.parse_args()

image_path = arguments.image
model_path = arguments.model_path


frame = cv2.imread(image_path)

try:
    model = models.load_model(model_path)
except OSError:
    raise FileNotFoundError(errno.ENOENT, strerror(errno.ENOENT), model_path)
#Convert the captured frame into RGB

try:
    im = Image.fromarray(frame, 'RGB')
except AttributeError:
    raise FileNotFoundError(errno.ENOENT, strerror(errno.ENOENT), image_path)

#Resizing into dimensions you used while training
im = im.resize((256,256))
img_array = np.array(im)
#Expand dimensions to match the 4D Tensor shape.
img_array = np.expand_dims(img_array, axis=0)
#Calling the predict function using keras
prediction = model.predict(img_array)#[0][0]
print(prediction)

#Customize this part to your liking...
if(prediction == 1 ):
    print("Leaf is Healthy.")
elif(prediction<1):
    print('Leaf is Unhealthy.')
    

cv2.imshow("Prediction", frame)

key=cv2.waitKey(0)

if key == ord('q'):
    cv2.destroyAllWindows()
