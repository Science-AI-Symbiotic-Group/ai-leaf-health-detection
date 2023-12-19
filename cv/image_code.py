import cv2
import numpy as np
from PIL import Image
from keras import models
import os
import tensorflow as tf



frame = cv2.imread("images/unhealthy_leaf.jpg")
os.system('dir')
model = models.load_model('models/tensorflow_model_with_dense.h5')
#Convert the captured frame into RGB
im = Image.fromarray(frame, 'RGB')
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

key=cv2.waitKey(1)

if key == ord('q'):
    cv2.destroyAllWindows()