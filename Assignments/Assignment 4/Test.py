# Packages
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Input, Dense
from keras.preprocessing import image_dataset_from_directory
from keras.layers import Rescaling, Conv2D, Dense, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import EfficientNetV2S
from keras.models import Model

from matplotlib import pyplot as plt 
import os
import glob
import numpy as np

## Removed Corrupted JPEGs per A4 Slides ; Only needs to be run Once!
# files = glob.glob("Assignment 4/Monkey Species Data/*/*/*/*")
# for file in files :
#    f = open(file, "rb") # open to read binary file
#    if not b"JFIF" in f.peek(10) :
#           f.close()
#           print(file)
#           os.remove(file)
#    else :
#           f.close()

# Define Training and Test Sets
training_set = image_dataset_from_directory("Monkey Species Data/Training Data", label_mode="categorical", image_size=(100,100))

test_set = image_dataset_from_directory("Monkey Species Data/Prediction Data", label_mode="categorical", image_size=(100,100), shuffle=False)

## Iterates through Data for images (in byte array foramt)
# for e in training_set :
# 	print(e)
# 	a = input() # to pause

## Iterates through images
# for e in training_set :
#     print(e[1][0])   
#     plt.imshow(e[0][0]/255)
#     plt.show()

## Setup, Train, and Save Model 
# m = Sequential()
# m.add(Input((100,100,3)))
# m.add(Rescaling(1/255)) 
# m.add(Conv2D(32, kernel_size=(3, 3),activation="relu"))
# m.add(MaxPooling2D(pool_size=(2, 2)))
# m.add(Conv2D(64, (3, 3), activation="relu"))
# m.add(MaxPooling2D(pool_size=(2, 2)))
# m.add(Flatten())
# m.add(Dense(128, activation="relu"))
# m.add(Dropout(0.2))
# m.add(Dense(10, activation="softmax"))
# m.compile(loss="categorical_crossentropy", metrics=["accuracy"])
# m.summary()

# epochs = 10
# for i in range(epochs) :
#     history  = m.fit(training_set, epochs=1)
#     print("Epoch:",i+1,"Training Accuracy:",history.history["accuracy"])
# score = m.evaluate(test_set)
# print("Test accuracy:", score[1])
# m.save("my_cnn1.keras")

# m = load_model("my_cnn1.keras")

# image_file = "test.jpeg"
# img = load_img(image_file,target_size=(100,100))
# img_arr = img_to_array(img)

###### NOT WORKING ; BC PLT bugged? Restart!! ###########
#or  plt.imshow(img_arr.astype("uint8"))
#or  plt.imshow(img_arr/255)
# plt.show()

# img_cl = img_arr.reshape(1,100,100,3)
# score = m.predict(img_cl)
# print(score.round(3))

# corr = []
# for e in test_set :
#     corr = corr + list(np.argmax(e[1],axis=1))

# p = m.predict(test_set) 
# pr  = np.argmax(p, axis=1)
# cf = confusion_matrix(corr, pr)
# print(cf)

## Pre Trained Model
# Load Pre-Trained Model
base_model = EfficientNetV2S (include_top=False)

# Fine Tune Model
x = base_model.output 
x = GlobalAveragePooling2D()(x) 
x = Dense(1024, activation='relu')(x)
output_layer = Dense(10, activation="softmax")(x)

m = Model(inputs=base_model.input, outputs=output_layer)

for layer in base_model.layers:
        layer.trainable = False

m.compile(loss="categorical_crossentropy", metrics=["accuracy"])
history  = m.fit(training_set, epochs=10)
print("Training Accuracy:",history.history["accuracy"])
score = m.evaluate(test_set)

# m.summary()

image_file = "test.jpeg"
img = load_img(image_file,target_size=(100,100))
img_arr = img_to_array(img)
img_cl = img_arr.reshape(1,100,100,3)
score = m.predict(img_cl)
print(score.round(3))

