import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.python.keras import layers
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Rescaling, Dropout, Resizing



batch_size = 32
img_height = 256
img_width = 256

train_ds = tf.keras.utils.image_dataset_from_directory(
  "../../dataset/leaves_healthy_or_diseased",
  validation_split=0.2,
  subset="training",
  seed=98,
  image_size=(img_height, img_width),
  batch_size=batch_size)


val_ds = tf.keras.utils.image_dataset_from_directory(
  "../../dataset/leaves_healthy_or_diseased",
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


training_dataset = train_ds

AUTOTUNE = tf.data.AUTOTUNE

#train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
#val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(training_dataset)

#model = Sequential([
  #Resizing(img_height,img_width,interpolation="bilinear",crop_to_aspect_ratio=False),
  #Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  #Dense(128, activation='relu'),
  #Conv2D(16, 3, padding='same', activation='relu'),
  #MaxPooling2D(),
  #Conv2D(32, 3, padding='same', activation='relu'),
  #MaxPooling2D(),
  #Conv2D(64, 3, padding='same', activation='relu'),
  #MaxPooling2D(),
  #Flatten(),
  #Dense(128, activation='relu'),
  #Dropout(0.2),
 # Dense(1,activation='sigmoid')
#])
model = Sequential([
  #Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  Conv2D(16, 3, padding='same', activation='relu'),
  MaxPooling2D(),
  Conv2D(32, 3, padding='same', activation='relu'),
  MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(1,activation='sigmoid')
])
 

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

epochs=5
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

model.save("tensorflow_model_new.h5",save_format="h5")

print("Saved Model.")




acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
