#############################################################################################################
#   File Name: JQ-CompSci711-Assign2.py   
#   Course: CompSci 711 - Intro to Machine Learing
#   Author: John Quinn
#   Purpose: 
#       Complete assignment 2 requiremnts 
#               1. Compare rmse of  k-nearest neighbor regressor obtained using three different k values by plotting 
#                   their learning curves obtained using 10-fold cross-validation. In a table, show rmse for the last point of learning curves (i.e. with maximum training data).
#               2.Compare rmse of k-nearest neighbor regressor (with parameter k tuned with at least 5 values),
#                    decision tree regressor (with parameter min_sample_leaf tuned with at least 5 values) and linear regression,
#                    by plotting their learning curves obtained using 10-fold cross-validation. In a table, show rmse for the last 
#                    point of learning curves (i.e. with maximum training data).
#               3.Choose a dataset for regression task. You can choose it from OpenML (see the slides about Regression Datasets). The dataset must have:
#               4. At least one nominal feature
#               5.At least 1000 examples
#               6.Choose a dataset in which the number of examples are far more than the number of features.
#               7.Avoid dataset with missing feature values.
#               8.Handle nominal features as described in the A2.pptx slides.
#
#   Completed + Submitted : 3/7/2024
#############################################################################################################
##      Package Imports     ##
    #   General Packages  #
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time

    #   SkLearn Imports     
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.impute import SimpleImputer
from sklearn.model_selection import learning_curve
from sklearn.neighbors  import KNeighborsRegressor

#############################################################################################################
##     Start Program Timer       ##
Whole_Time_start = time.time()

#############################################################################################################
##   Loading Data Sets      ##
Moneyball_data = datasets.fetch_openml(data_id=41021) # https://www.openml.org/search?type=data&status=active&sort=runs&qualities.NumberOfInstances=between_1000_10000&qualities.NumberOfClasses=lte_1&id=41021
Moneyball_data_df = pd.DataFrame(Moneyball_data.data, columns=Moneyball_data.feature_names)
Moneyball_data_df['target'] = pd.Series(Moneyball_data.target)

#############################################################################################################
##   Preprocessig - Dropping Data   ##

Moneyball_data_df = Moneyball_data_df.drop(["OOBP", "OSLG"],axis=1) # Dropping these made Results WAYYYYYYY etter ; most of this was missing any way ; pitcher stats?

#############################################################################################################
##   Preprocessig - Missing Values Imputation  ##
#     # Used Mean Imputing to Fill values with most mean of non missing column 

imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value="10") # This is replacing playoff and season rank ; if unranked they did not do well enough to make it to the ranking so a value lower than both ranges used to denote worse performance 

imp_moneyball = imp.fit_transform(Moneyball_data_df)
imp_moneyball = pd.DataFrame(imp_moneyball, columns= imp.get_feature_names_out())

#############################################################################################################
##   Preprocessing - One Hot Encoding  ##

ct_Moneyball = ColumnTransformer([("encoder", OneHotEncoder(sparse_output=False), [0,1])], remainder="passthrough")
trans_moneyball = ct_Moneyball.fit_transform(imp_moneyball)
trans_moneyball = pd.DataFrame(trans_moneyball, columns= ct_Moneyball.get_feature_names_out())

#############################################################################################################
##   K Nearest Neighbor    ##

    # K = 3
knn_3 = KNeighborsRegressor(n_neighbors=3)

knn_3_scores_moneyball = cross_validate(knn_3, trans_moneyball, Moneyball_data.target, cv=10, scoring="neg_root_mean_squared_error")
knn_3_moneyball_training_examples, knn_3_moneyball_training_scores, knn_3_moneyball_test_scores = learning_curve(knn_3, trans_moneyball,  Moneyball_data.target, 
                                                                                                           train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0], cv=10, scoring="neg_root_mean_squared_error")

    # K = 7
knn_7 = KNeighborsRegressor(n_neighbors=7)

knn_7_scores_moneyball = cross_validate(knn_7, trans_moneyball, Moneyball_data.target, cv=10, scoring="neg_root_mean_squared_error")
knn_7_moneyball_training_examples, knn_7_moneyball_training_scores, knn_7_moneyball_test_scores = learning_curve(knn_7, trans_moneyball,  Moneyball_data.target, 
                                                                                                           train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0], cv=10, scoring="neg_root_mean_squared_error")

    # K = 11
knn_11 = KNeighborsRegressor(n_neighbors=11)

knn_11_scores_moneyball = cross_validate(knn_11, trans_moneyball, Moneyball_data.target, cv=10, scoring="neg_root_mean_squared_error")
knn_11_moneyball_training_examples, knn_11_moneyball_training_scores, knn_11_moneyball_test_scores = learning_curve(knn_11, trans_moneyball,  Moneyball_data.target, 
                                                                                                           train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0], cv=10, scoring="neg_root_mean_squared_error")


 #  Plot and Table Results and Display

plt.plot(knn_3_moneyball_training_examples,-knn_3_moneyball_test_scores.mean(axis=1),"bo-",label= 'K = 3')
plt.plot(knn_7_moneyball_training_examples,-knn_7_moneyball_test_scores.mean(axis=1),"go-", label= 'K = 7')
plt.plot(knn_11_moneyball_training_examples,-knn_11_moneyball_test_scores.mean(axis=1),"ro-", label= 'K = 11')
plt.legend()
plt.xlabel("Number of Trianing Examples")
plt.ylabel("RMSE")
plt.title("'Moneyball' K Nearest Neighbor with Different K Values")
plt.show()

RMSE_Values = {'K = 3': [-np.mean(knn_3_moneyball_test_scores[-1,:])],
                'K = 7': [-np.mean(knn_7_moneyball_test_scores[-1,:])],
                'K = 11': [-np.mean(knn_11_moneyball_test_scores[-1,:])]}
RMSE_Values_df = pd.DataFrame(RMSE_Values)
print("RMSE Values for Varied K Vales of K Nearest Neighbor on 'Moneyball' Data Set \n", RMSE_Values_df)

#############################################################################################################
#      Parameter Tuning and Comparison of Models      ##

    # K Nearest Neigbor Tuning
knn_Tuning = KNeighborsRegressor()

n_neigbors_search = {"n_neighbors": [i for i in range(1, 15,3)] } # Paramaters Tuning list  1 - 15 increments of 3

knn_Tuned = model_selection.GridSearchCV(knn_Tuning, n_neigbors_search, cv=10)
knn_Tuned_scores = cross_validate(knn_Tuned, trans_moneyball, Moneyball_data.target, cv=10, scoring="neg_root_mean_squared_error")
knn_Tuned_training_examples, knn_Tuned_training_scores, knn_Tuned_test_scores = learning_curve(knn_Tuned, trans_moneyball,  Moneyball_data.target, 
                                                                                                           train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0], cv=10, scoring="neg_root_mean_squared_error")


    # Decison Tree Tuning
dct = DecisionTreeRegressor()

min_samples_leaf_Search = {"min_samples_leaf": [i for i in range(1, 15,3)] } # Paramaters Tuning list  1 - 15 increments of 3

Dec_tree_Tuned = model_selection.GridSearchCV(dct, min_samples_leaf_Search, cv=10)
Dec_tree_Tuned_scores = cross_validate(Dec_tree_Tuned, trans_moneyball, Moneyball_data.target,cv=10)
Dec_tree_Tuned_training_examples, Dec_tree_Tuned_training_scores, Dec_tree_Tuned_test_scores = learning_curve(Dec_tree_Tuned, trans_moneyball,  Moneyball_data.target, 
                                                                                                           train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0], cv=10, scoring="neg_root_mean_squared_error")

    # Linear Regression Tuning 
lr = LinearRegression()

lr_scores = cross_validate(lr, trans_moneyball, Moneyball_data.target, cv=10, scoring="neg_root_mean_squared_error")
lr_training_examples, lr_training_scores, lr_test_scores = learning_curve(lr, trans_moneyball,  Moneyball_data.target, 
                                                                                                           train_sizes=[.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1.0], cv=10, scoring="neg_root_mean_squared_error")

plt.plot(lr_training_examples, -lr_test_scores.mean(axis=1), label= "test")
plt.plot(lr_training_examples, -lr_training_scores.mean(axis=1),  label= "training")
plt.legend()
plt.xlabel("Number of Trianing Examples")
plt.ylabel("RMSE")
plt.title("Learning Curve for Linear Regression - to see odd very low RMSE")
plt.show()

    #  Plot and Table Results and Display
plt.plot(knn_Tuned_training_examples,-knn_Tuned_test_scores.mean(axis=1),"go-", label= 'K Nearest Neighbor')
plt.plot(Dec_tree_Tuned_training_examples,-Dec_tree_Tuned_test_scores.mean(axis=1),"bo-",label= 'Decision Tree')
plt.plot(lr_training_examples,-lr_test_scores.mean(axis=1),"ro-", label= 'Linear Regression')
plt.legend()
plt.xlabel("Number of Trianing Examples")
plt.ylabel("RMSE")
plt.title("Comparison of Differnet ML Models Training on 'Moneyball' Dataset")
plt.show()

RMSE_Values_Models = {'K Nearest Neigbor': [-np.mean(knn_Tuned_test_scores[-1,:])],
                        'Decision Tree': [-np.mean(Dec_tree_Tuned_test_scores[-1,:])],
                        'Linear Regression': [-np.mean(lr_test_scores[-1,:])]}
RMSE_Values_Models_df = pd.DataFrame(RMSE_Values_Models)
print("RMSE Values for K Nearest Neighbor, Decision Tree, and Linear Regression for the 'Moneyball' Data Set \n", RMSE_Values_Models_df)

# #############################################################################################################
##     End Timer       ##
Whole_Time_end = time.time()
print("Whole Program Time (Seconds): ", round(Whole_Time_end-Whole_Time_start,3))