#############################################################################################################
#   File Name: JQ-CompSci711-Assign4.py   
#   Course: CompSci 711 - Intro to Machine Learing
#   Author: John Quinn
#   Purpose: 
#       Complete assignment 4 requiremnts 
#               		Task 1: Your own CNN architectures:
                        #     Try two different CNN architectures (one can be from the slides if you prefer). You can change the layers, change the number or size of filters, with and without dropout, etc. to get different architectures.
                        #     Train each model till the training accuracy does not seem to improve over epochs. Test each model on the test data.
                        #     Compare the accuracy of the two models on the test data. Obtain confusion matrix for the model with better accuracy on the test data.
                        #     Save the model which gives better accuracy on test data.
                            
	                    # Task 2: Fine-tuning a pre-trained CNN architecture:
                        #     Fine-tune a pre-trained architecture of your choice. The layer(s) you add on the top is your design decision.
                        #     Train the model till the training accuracy does not seem to improve over epochs. Test it on the test data.
                        #     Compare the accuracy of the fine-tuned model with the better model from Task 1 on the test data. Obtain the confusion matrix for the fine-tuned model on the test data.

#                         Task 3: Error Analysis:
#                             Take any 10 test images on which the better model from Task 1 makes incorrect predictions.
#                             Qualitatively see if there may be reason(s) why the model makes incorrect predictions.
#                             See if the fine-tuned model of Task 2 improves on those 10 images or not.

#            Dataset for assignemnt: https://www.kaggle.com/datasets/utkarshsaxenadn/10-species-of-monkey-multiclass-classification?resource=downloa
#                   10 Species of Monkey
#############################################################################################################
##      Package Imports     ##
    #   General Packages  #
import pandas as pd
import numpy as np
from tabulate import tabulate
from matplotlib import pyplot as plt
import os
import glob

    #   SK Learn Imports
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix

#   Keras Imports
from keras.models import Sequential, Model,load_model
from keras.preprocessing import image_dataset_from_directory
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Rescaling, Conv2D, Dense, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D, Input, Dense
from keras.applications import EfficientNetV2S

#############################################################################################################
## User Functions

# Returns Catagory label from index
def get_label_name(index):
    label_names = ["Bald Uakari", "Emperor Tamarin", "Golden Monkey", "Gray Langur", "Hamadryas Baboon", 
                   "Mandril", "Proboscis Monkey", "Red Howler","Vervet Monkey", "White Faced Saki"]
    return label_names[index]

label_names = ["Bald Uakari", "Emperor Tamarin", "Golden Monkey", "Gray Langur", "Hamadryas Baboon", 
                   "Mandril", "Proboscis Monkey", "Red Howler","Vervet Monkey", "White Faced Saki"]
#############################################################################################################
### Pre Processing 

# Data previusly downloaded from link in discription and placed into working directory

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

## Define Training and Test Sets
training_set = image_dataset_from_directory("Monkey Species Data/Training Data", label_mode="categorical", image_size=(100,100))

test_set = image_dataset_from_directory("Monkey Species Data/Prediction Data", label_mode="categorical", image_size=(100,100), shuffle=False)
test_images = [] # Setting up true labels for test images for later
test_labels = []

for images, labels in test_set:
    test_images.append(images.numpy())
    test_labels.append(labels.numpy())
test_images = np.concatenate(test_images)
test_labels = np.concatenate(test_labels)

true_lables = np.argmax(test_labels, axis=1)

#############################################################################################################
# ## Task 1 - Train 2 CNNs and compare perfromance 

# ## Model Setup 

# # Model 1
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

# # Model 2 
# m2 = Sequential()
# m2.add(Input((100,100,3)))
# m2.add(Rescaling(1/255)) 
# m2.add(Conv2D(64, kernel_size=(3, 3),activation="relu")) # Doubled # of Filters 
# m2.add(MaxPooling2D(pool_size=(2, 2)))
# m2.add(Conv2D(32, (3, 3), activation="relu")) # Halved # of Filters
# m2.add(MaxPooling2D(pool_size=(2, 2)))
# m2.add(Flatten())
# m2.add(Dense(64, activation="relu")) # Halved the # of dense layer neurons 
# m2.add(Dropout(0.2)) # Increased from .2 to .3
# m2.add(Dense(10, activation="softmax"))
# m2.compile(loss="categorical_crossentropy", metrics=["accuracy"])
# m2.summary()


# ## Training 
# epochs = 15

# # Model 1 
# print("\n \n Train Model 1")
# for i in range(epochs) :
#     history  = m.fit(training_set, epochs=1)
#     print("Epoch:",i+1,"Training Accuracy:",history.history["accuracy"])

# # Model 2 
# print("\n \n Train Model 2")
# for i in range(epochs) :
#     history  = m2.fit(training_set, epochs=1)
#     print("Epoch:",i+1,"Training Accuracy:",history.history["accuracy"])

# # ## Save Models
# # m.save("CNN_Model_1.keras")
# # m2.save("CNN_Model_2.keras")

# # ## If models aready trained can comment out above section and load
# # m = load_model('CNN_Model_1.keras')
# # m2 = load_model('CNN_Model_2.keras')

# ## Models Evaluation ; Decide which more accuracte and save that model and print its confustion matrix : and test on test image
# loss, mscore = m.evaluate(test_set)
# loss, m2score = m2.evaluate(test_set)

# print("Model 1 Test accuracy:", mscore)
# print("Model 2 Test accuracy:", m2score)

# if mscore > m2score:
#     init_score = mscore 
#     print("Model 1 more accurate on test data! Evaluating Model 1 and saving")
#     m.save("CNN_Model_1.keras")

#     corr = []
#     for e in test_set :
#         corr = corr + list(np.argmax(e[1],axis=1))
#     p = m.predict(test_set) 
#     pr  = np.argmax(p, axis=1)
#     cf = confusion_matrix(corr, pr)
#     print(cf)

# else:
#     init_score = m2score
#     print("Model 2 more accurate on test data! Evaluating Model 1 and saving")
#     m2.save("CNN_Model_2.keras")

#     corr = []
#     for e in test_set :
#         corr = corr + list(np.argmax(e[1],axis=1))
#     p = m2.predict(test_set) 
#     pr  = np.argmax(p, axis=1)
#     cf = confusion_matrix(corr, pr)
#     print(cf)

#############################################################################################################
# Task 2 - Fine-Tune a pre-trined CNN structure 

# Load Pre-Trained Model
base_model = EfficientNetV2S (include_top=False)
base_model.summary()

# Fine Tune Model and add layers for my data
x = base_model.output 
x = GlobalAveragePooling2D()(x) 
x = Dense(1024, activation='relu')(x)
output_layer = Dense(10, activation="softmax")(x)
pretrain = Model(inputs=base_model.input, outputs=output_layer)
for layer in base_model.layers:
        layer.trainable = False
    
# Complie and Train
pretrain.compile(loss="categorical_crossentropy", metrics=["accuracy"])
pretrain_history  = pretrain.fit(training_set, epochs=15)
print("Training Accuracy:",pretrain_history.history["accuracy"])

# ## If model aready trained can comment out above section and load
# pretrain = load_model('Pretrianed_CNN_Model.keras')

loss, pretrian_score = pretrain.evaluate(test_set)
print("Pre-Trained Model Test accuracy:",pretrian_score)

if pretrian_score > init_score:
    print("Pretriained Model MORE accurate ({pretrain_score}) on test data than my CNN ({init_score}) ; Saving pretrained model")
else:
    print("Pretriained Model LESS accurate ({pretrain_score}) on test data than my CNN ({init_score}) ; Saving pretrained model")
  
# pretrain.save("Pretrianed_CNN_Model.keras")

# Confusion Matrix 
pretrain_corr = []
for e in test_set :
    pretrain_corr = pretrain_corr + list(np.argmax(e[1],axis=1))
pretrain_p = pretrain.predict(test_set) 
pretrain_pr  = np.argmax(pretrain_p, axis=1)
pretrain_cf = confusion_matrix(pretrain_corr,pretrain_pr)
print(pretrain_cf)

#############################################################################################################
# Task 3 - Error Analysis

## Find incorrectly labeled images using predicted labeles and orignal "true lables"
incorrect_indices = np.where(pr != true_lables)[0]

# Etraxt 10 Random incorrect images and indices 
random_incorrect_indices= np.random.choice(incorrect_indices, size=10, replace=False)

for index in random_incorrect_indices:
    # Get the corresponding image, true label, and predicted label
    image = test_images[index]
    true_label = true_lables[index]
    my_model_predicted_label = pr[index]
    pretrain_predicted_label = pretrain_pr[index]

    # Convert label indices to label names
    true_label_name = get_label_name(true_label)
    my_model_predicted_label_name = get_label_name(my_model_predicted_label)
    pretrain_predicted_label_name = get_label_name(pretrain_predicted_label)

    # Plot the image and show true and predicted labels
    plt.imshow(image.astype('uint8'))
    plt.title(f"True Label: {true_label_name}, My Model Predicted Label: {my_model_predicted_label_name}, Finetuned Model Predicted Label: {pretrain_predicted_label_name} ")
    plt.axis('off')
    plt.show()


