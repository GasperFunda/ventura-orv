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


# image_list = np.array([])
# image_labels = np.array([])

image_list = []
image_labels = []
for filename in glob.glob("data/1/*"):
    im = cv2.imread(filename)
    print(filename)
    resized = resize_cv(im)
    image_list.append(resized)
    image_labels.append(0)
    # np.concatenate(image_list,resized)
    # np.append(image_labels,"Bicycle")

for filename in glob.glob("data\stop/*"):
    im = cv2.imread(filename)
    resized = resize_cv(im)
    image_list.append(resized)
    image_labels.append(1)
    # np.concatenate(image_list,resized)
    # np.append(image_labels,"Stop")

image_list = np.array(image_list)
image_labels = np.array(image_labels)
image_list = image_list
print(image_labels)


class_names = ["Bicycle","Stop"]

# model = models.Sequential()
# model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(32,32,3)))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64,(3,3),activation="relu"))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64,(3,3),activation="relu"))
# model.add(layers.Flatten())
# model.add(layers.Dense(64,activation="relu"))
# model.add(layers.Dense(10,activation="softmax"))

# model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

# model.fit(image_list,image_labels,epochs=10,verbose=1)

# model.save("imageClassifier.model")

model = models.load_model("imageClassifier.model")

img = cv2.imread("data/1/16.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = resize_cv(img)

prediction = model.predict(np.array([img]))
print(prediction)
index = np.argmax(prediction)
print(index)
print(class_names[index])
