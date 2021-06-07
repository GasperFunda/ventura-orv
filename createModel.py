import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from tensorflow.python.keras import layers
from keras import layers,datasets,models
import os
import glob
def resize_cv(im):
    return cv2.resize(im, (32, 32), interpolation = cv2.INTER_LINEAR)


image_list = []
image_labels = []
for folder in glob.glob("Train/*"):
    for filename in glob.glob(folder+"/*"):
        im = cv2.imread(filename)
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        resized = resize_cv(im)
        image_list.append(resized)
        tmp = folder[6:len(folder)]
        image_labels.append(int(tmp))


image_list = np.array(image_list)
image_labels = np.array(image_labels)
image_list = image_list

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation="relu"))
model.add(layers.Dense(44,activation="softmax"))

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

model.fit(image_list,image_labels,epochs=10,verbose=1)

model.save("imageClassifier.model")


