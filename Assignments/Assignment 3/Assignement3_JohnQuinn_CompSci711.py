#############################################################################################################
#   File Name: JQ-CompSci711-Assign3.py   
#   Course: CompSci 711 - Intro to Machine Learing
#   Author: John Quinn
#   Purpose: 
#       Complete assignment 3 requiremnts 
#               	Task 1 (Regression):
                     	# 4 different regression models
                        #     Vary # of hidden nodes, hidden layers, and activation function
                        #     			Only use input and dense layers
                        #     			For each 
                        #     				Plot training and validation errors vs # of epochs
                        #     					If training accuracy has not platued increase epochs 
                        #     				1 graph for each model
                        #     Table of minimum validation error for each model

                    # Task 2 (Classification):
                  		# 4 different classification models
			            #     Vary # of hidden nodes, hidden layers, and activation function
			            #     Only use input and dense layers
			            #     For each 
				        #      Plot training accuracy vs # of epochs
					    #         If training accuracy has not platued increase epochs 
				        #     1 graph for each model
                        # Table of max validation accuracy for each model
#
#   Completed + Submitted : 4/21/2024
#############################################################################################################
##      Package Imports     ##
    #   General Packages  #
import pandas as pd
import numpy as np
from tabulate import tabulate
from matplotlib import pyplot as plt

    #   SK Learn Imports
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

    #   Keras Imports
from keras.models import Sequential
from keras.layers import Input, Dense


##  #     Task 1 - Regression      #    ##
#############################################################################################################
##     Load Data Set     ##
wine_quality = fetch_openml(data_id=287)

##     Split Data Set     ##
wine_quality_x_train, wine_quality_x_val, wine_quality_y_train, wine_quality_y_val = train_test_split(wine_quality.data, wine_quality.target, test_size=0.2, random_state=0)

##     Create 4 Sperate Models      ##
wine_quality_nn_1 = Sequential()
wine_quality_nn_2 = Sequential()
wine_quality_nn_3 = Sequential()
wine_quality_nn_4 = Sequential()

# Input Layers
wine_quality_nn_1.add(Input((11,)))
wine_quality_nn_2.add(Input((11,)))
wine_quality_nn_3.add(Input((11,)))
wine_quality_nn_4.add(Input((11,)))

# Hidden Layers 
wine_quality_nn_1.add(Dense(30,activation="sigmoid"))
wine_quality_nn_1.add(Dense(10,activation="sigmoid"))

wine_quality_nn_2.add(Dense(30,activation="relu"))
wine_quality_nn_2.add(Dense(10,activation="relu"))

wine_quality_nn_3.add(Dense(15,activation="sigmoid"))
wine_quality_nn_3.add(Dense(5,activation="sigmoid"))

wine_quality_nn_4.add(Dense(10,activation="relu"))

# Output Layers
wine_quality_nn_1.add(Dense(1))
wine_quality_nn_2.add(Dense(1))
wine_quality_nn_3.add(Dense(1))
wine_quality_nn_4.add(Dense(1))

# Complie Models
wine_quality_nn_1.compile(optimizer="adam", loss="mse", metrics=["mse"])
wine_quality_nn_2.compile(optimizer="adam", loss="mse", metrics=["mse"])
wine_quality_nn_3.compile(optimizer="adam", loss="mse", metrics=["mse"])
wine_quality_nn_4.compile(optimizer="adam", loss="mse", metrics=["mse"])

# Fit Models
wine_quality_history_1 = wine_quality_nn_1.fit(wine_quality_x_train, wine_quality_y_train, epochs=200,validation_data=(wine_quality_x_val,wine_quality_y_val),verbose=1)
wine_quality_history_2 = wine_quality_nn_2.fit(wine_quality_x_train, wine_quality_y_train, epochs=200,validation_data=(wine_quality_x_val,wine_quality_y_val),verbose=1)
wine_quality_history_3 = wine_quality_nn_3.fit(wine_quality_x_train, wine_quality_y_train, epochs=200,validation_data=(wine_quality_x_val,wine_quality_y_val),verbose=1)
wine_quality_history_4 = wine_quality_nn_4.fit(wine_quality_x_train, wine_quality_y_train, epochs=200,validation_data=(wine_quality_x_val,wine_quality_y_val),verbose=1)

# Plot Graphs
plt.title("Model 1 - 30 Node Layer ; 10 Node Layer ; ReLu Activation ")
plt.plot(wine_quality_history_1.history["mse"], label="Training")
plt.plot(wine_quality_history_1.history["val_mse"], label="Validation")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()
plt.show()

plt.title("Model 2 - 30 Node Layer ; 10 Node Layer ; Sigmoid Activation ")
plt.plot(wine_quality_history_2.history["mse"], label="Training")
plt.plot(wine_quality_history_2.history["val_mse"], label="Validation")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()
plt.show()

plt.title("Model 3 - 15 Node Layer ; 5 Node Layer ; ReLu Activation ")
plt.plot(wine_quality_history_3.history["mse"], label="Training")
plt.plot(wine_quality_history_3.history["val_mse"], label="Validation")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()
plt.show()

plt.title("Model 4 - 15 Node Layer ; 5 Node Layer ; Sigmoid Activation ")
plt.plot(wine_quality_history_4.history["mse"], label="Training")
plt.plot(wine_quality_history_4.history["val_mse"], label="Validation")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()
plt.show()

# Print Table of Min MSE for each Model
min_mse = [
    ("Model 1", min(wine_quality_history_1.history["val_mse"]), max(wine_quality_history_1.history["val_mse"])),
    ("Model 2", min(wine_quality_history_2.history["val_mse"]), max(wine_quality_history_2.history["val_mse"])),
    ("Model 3", min(wine_quality_history_3.history["val_mse"]), max(wine_quality_history_3.history["val_mse"])),
    ("Model 4", min(wine_quality_history_4.history["val_mse"]), max(wine_quality_history_4.history["val_mse"]))
]

# Display table
print(tabulate(min_mse, headers=["Model", "Minimum MSE", "Max MSE"]))


##  #     Task 2 - Classification      #    ##
#############################################################################################################
##     Load Data Set     ##
spambase = fetch_openml(data_id=44)

enc = OneHotEncoder(sparse_output=False)
tmp=[[x] for x in spambase.target]
ohe_target=enc.fit_transform(tmp)

##     Split Data Set     ## 
spambase_x_train, spambase_x_val, spambase_y_train, spambase_y_val = train_test_split(spambase.data, ohe_target, test_size=0.2, random_state=0)

##     Create 4 Sperate Models      ##
spambase_nn_1 = Sequential()
spambase_nn_2 = Sequential()
spambase_nn_3 = Sequential()
spambase_nn_4 = Sequential()

# Input Layers
spambase_nn_1.add(Input((57,)))
spambase_nn_2.add(Input((57,)))
spambase_nn_3.add(Input((57,)))
spambase_nn_4.add(Input((57,)))

# Hidden Layers 
spambase_nn_1.add(Dense(30,activation="sigmoid"))
spambase_nn_1.add(Dense(10,activation="sigmoid"))

spambase_nn_2.add(Dense(30,activation="relu"))
spambase_nn_2.add(Dense(10,activation="relu"))

spambase_nn_3.add(Dense(15,activation="sigmoid"))
spambase_nn_3.add(Dense(5,activation="sigmoid"))

spambase_nn_4.add(Dense(15,activation="relu"))
spambase_nn_4.add(Dense(5,activation="relu"))

# Output Layers
spambase_nn_1.add(Dense(2,activation="softmax"))
spambase_nn_2.add(Dense(2,activation="softmax"))
spambase_nn_3.add(Dense(2,activation="softmax"))
spambase_nn_4.add(Dense(2,activation="softmax"))

# Complie Models
spambase_nn_1.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
spambase_nn_2.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
spambase_nn_3.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
spambase_nn_4.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Fit Models
spambase_history_1 = spambase_nn_1.fit(spambase_x_train, spambase_y_train, epochs=200,validation_data=(spambase_x_val,spambase_y_val),verbose=1)
spambase_history_2 = spambase_nn_2.fit(spambase_x_train, spambase_y_train, epochs=200,validation_data=(spambase_x_val,spambase_y_val),verbose=1)
spambase_history_3 = spambase_nn_3.fit(spambase_x_train, spambase_y_train, epochs=200,validation_data=(spambase_x_val,spambase_y_val),verbose=1)
spambase_history_4 = spambase_nn_4.fit(spambase_x_train, spambase_y_train, epochs=200,validation_data=(spambase_x_val,spambase_y_val),verbose=1)

# Plot Graphs
plt.title("Model 1 - 30 Node Layer ; 10 Node Layer ; Sigmoid Activation ")
plt.plot(spambase_history_1.history["accuracy"], label="Training")
plt.plot(spambase_history_1.history["val_accuracy"], label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.legend()
plt.show()


plt.title("Model 2 - 30 Node Layer ; 10 Node Layer ; ReLu Activation ")
plt.plot(spambase_history_2.history["accuracy"], label="Training")
plt.plot(spambase_history_2.history["val_accuracy"], label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.legend()
plt.show()

plt.title("Model 3 - 15 Node Layer ; 5 Node Layer ; Sigmoid Activation ")
plt.plot(spambase_history_3.history["accuracy"], label="Training")
plt.plot(spambase_history_3.history["val_accuracy"], label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.legend()
plt.show()

plt.title("Model 4 - 15 Node Layer ; 5 Node Layer ; ReLu Activation ")
plt.plot(spambase_history_4.history["accuracy"], label="Training")
plt.plot(spambase_history_4.history["val_accuracy"], label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.legend()
plt.show()

# Print Table of Max Accuracy for each Model
Max_accuracy = [
    ("Model 1", max(spambase_history_1.history["val_accuracy"]), min(spambase_history_1.history["val_accuracy"])),
    ("Model 2", max(spambase_history_2.history["val_accuracy"]), min(spambase_history_2.history["val_accuracy"])),
    ("Model 3", max(spambase_history_3.history["val_accuracy"]), min(spambase_history_3.history["val_accuracy"])),
    ("Model 4", max(spambase_history_4.history["val_accuracy"]), min(spambase_history_4.history["val_accuracy"]))
]

# Display table
print(tabulate(Max_accuracy, headers=["Model", "Max Accuracy","Min Accuracy"]))

