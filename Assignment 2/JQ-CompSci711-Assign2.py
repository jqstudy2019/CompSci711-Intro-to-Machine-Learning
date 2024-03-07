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
# 
#############################################################################################################
##      Package Imports     ##
    #   General Packages  #
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time

# from ucimlrepo import fetch_ucirepo 

# from scipy.io.arff import loadarff

    #   SkLearn Imports     
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
# from sklearn import tree
# from sklearn import metrics
# from sklearn.metrics import roc_curve
# from sklearn.metrics import roc_auc_score
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

# seoul_bike_sharing_demand = pd.read_csv('SeoulBikeData.csv', encoding=None) # Downloaded as csv from: https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand
# seoul_bike_sharing_demand_array = np.array(seoul_bike_sharing_demand)

#############################################################################################################
##   Preprocessig - Dropping Data   ##

Moneyball_data_df = Moneyball_data_df.drop(["OOBP", "OSLG"],axis=1) # Dropping these made Results WAYYYYYYY etter ; most of this was missing any way ; pitcher stats?
# print(Moneyball_data_df)

#############################################################################################################
##   Preprocessig - Missing Values Imputation  ##
#     # Used Mean Imputing to Fill values with most mean of non missing column 

imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value="10") # This is replacing playoff and season rank ; if unranked they did not do well enough to make it to the ranking so a value lower than both ranges used to denote worse performance 

imp_moneyball = imp.fit_transform(Moneyball_data_df)
imp_moneyball = pd.DataFrame(imp_moneyball, columns= imp.get_feature_names_out())
print(imp_moneyball)

#############################################################################################################
##   Preprocessing - One Hot Encoding  ##

ct_Moneyball = ColumnTransformer([("encoder", OneHotEncoder(sparse_output=False), [0,1])], remainder="passthrough")
trans_moneyball = ct_Moneyball.fit_transform(imp_moneyball)
print(trans_moneyball)

#############################################################################################################
##   K Nearest Neighbor    ##

knn_3 = KNeighborsRegressor(n_neighbors=3)
knn_7 = KNeighborsRegressor(n_neighbors=7)
knn_11 = KNeighborsRegressor(n_neighbors=11)

knn_3_scores_moneyball = cross_validate(knn_3, trans_moneyball, Moneyball_data.target, cv=10, scoring="neg_root_mean_squared_error")
knn_3_moneyball_rmse = 0-knn_3_scores_moneyball['test_score']
print('Final RMSE K=3: ',knn_3_moneyball_rmse[len(knn_3_moneyball_rmse)-1])
knn_3_moneyball_training_examples, knn_3_moneyball_training_scores, knn_3_moneyball_test_scores = learning_curve(knn_3, trans_moneyball,  Moneyball_data.target, 
                                                                                                           train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0], cv=10, scoring="neg_root_mean_squared_error")
knn_3_moneyball_test_scores_rmse = 0 - knn_3_moneyball_test_scores


knn_7_scores_moneyball = cross_validate(knn_7, trans_moneyball, Moneyball_data.target, cv=10, scoring="neg_root_mean_squared_error")
knn_7_moneyball_rmse = 0-knn_7_scores_moneyball['test_score']
print('Final RMSE K=7: ',knn_7_moneyball_rmse[len(knn_7_moneyball_rmse)-1])
knn_7_moneyball_training_examples, knn_7_moneyball_training_scores, knn_7_moneyball_test_scores = learning_curve(knn_7, trans_moneyball,  Moneyball_data.target, 
                                                                                                           train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0], cv=10, scoring="neg_root_mean_squared_error")
knn_7_moneyball_test_scores_rmse = 0 - knn_7_moneyball_test_scores


knn_11_scores_moneyball = cross_validate(knn_11, trans_moneyball, Moneyball_data.target, cv=10, scoring="neg_root_mean_squared_error")
knn_11_moneyball_rmse = 0-knn_11_scores_moneyball['test_score']
print('Final RMSE K=11: ',knn_11_moneyball_rmse[len(knn_11_moneyball_rmse)-1])
knn_11_moneyball_training_examples, knn_11_moneyball_training_scores, knn_11_moneyball_test_scores = learning_curve(knn_11, trans_moneyball,  Moneyball_data.target, 
                                                                                                           train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0], cv=10, scoring="neg_root_mean_squared_error")
knn_11_moneyball_test_scores_rmse = 0 - knn_11_moneyball_test_scores


plt.plot(knn_3_moneyball_training_examples,-knn_3_moneyball_test_scores.mean(axis=1),"bo-",label= 'K = 3')
plt.plot(knn_7_moneyball_training_examples,-knn_7_moneyball_test_scores.mean(axis=1),"go-", label= 'K = 7')
plt.plot(knn_11_moneyball_training_examples,-knn_11_moneyball_test_scores.mean(axis=1),"ro-", label= 'K = 11')
plt.legend()
plt.show()

RMSE_Values = {'K = 3': [knn_3_moneyball_rmse[len(knn_3_moneyball_rmse)-1]],
                'K = 7': [knn_7_moneyball_rmse[len(knn_7_moneyball_rmse)-1]],
                'K = 11': [knn_11_moneyball_rmse[len(knn_11_moneyball_rmse)-1]]}
RMSE_Values_df = pd.DataFrame(RMSE_Values)
print("RMSE Values for K Nearest Neighbor of Moneyball Data Set \n", RMSE_Values_df)


# #############################################################################################################
# ##   Linear Regression     ##

# lr = LinearRegression()

# lrr_scores_moneyball = cross_validate(lr, imp_moneyball, Moneyball_data.target, cv=10, scoring="neg_root_mean_squared_error")
# # lrr_moneyball_rmse = 0-lrr_scores_moneyball['test_score']
# # print(lrr_moneyball_rmse.mean())

# training_examples, training_scores, test_scores = learning_curve(lr, imp_moneyball,  Moneyball_data.target, train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0], cv=10, scoring="neg_root_mean_squared_error")
# training_scores_rmse = 0 - training_scores
# # plt.plot(training_scores_rmse[:,0],training_scores_rmse[:,1])
# # plt.show()

# lrr_seoul_bike = cross_validate(lr, seoul_bike_trans_data, seoul_bike_sharing_demand.loc[:,'Rented Bike Count'], cv=10, scoring="neg_root_mean_squared_error")
# # print(lrr_seoul_bike)

# #############################################################################################################
# ##   K Nearest Neighbor    ##

# knn_3 = KNeighborsRegressor(n_neighbors=3)
# knn_7 = KNeighborsRegressor(n_neighbors=3)
# knn_11 = KNeighborsRegressor(n_neighbors=11)

# knn_scores_moneyball = cross_validate(knn, imp_moneyball, Moneyball_data.target, cv=10, scoring="neg_root_mean_squared_error")
# print(knn_scores_moneyball)
# knn_moneyball_rmse = 0-knn_scores_moneyball['test_score']
# print(knn_moneyball_rmse[len(knn_moneyball_rmse)-1])

# knn_moneyball_training_examples, knn_moneyball_training_scores, knn_moneyball_test_scores = learning_curve(lr, imp_moneyball,  Moneyball_data.target, train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0], cv=10, scoring="neg_root_mean_squared_error")
# knn_moneyball_training_scores_rmse = 0 - knn_moneyball_training_scores
# knn_moneyball_test_scores_rmse = 0 - knn_moneyball_test_scores
# plt.plot(knn_moneyball_training_examples,knn_moneyball_test_scores_rmse)
# plt.show()

# #############################################################################################################
# ##   Decison Tree   ##

# dct = DecisionTreeClassifier()

# dct_scores_moneyball = cross_validate(dct, imp_moneyball, Moneyball_data.target, cv=10, scoring="neg_root_mean_squared_error")
# print(dct_scores_moneyball)
# # knn_moneyball_rmse = 0-knn_scores_moneyball['test_score']
# # print(knn_moneyball_rmse.mean())

# dct_moneyball_training_examples, dct_moneyball_training_scores, dct_moneyball_test_scores = learning_curve(dct, imp_moneyball,  Moneyball_data.target, train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0], cv=10, scoring="neg_root_mean_squared_error")
# dct_moneyball_training_scores_rmse = 0 - dct_moneyball_training_scores
# plt.plot(dct_moneyball_training_examples,dct_moneyball_test_scores)
# plt.show()

############################################################################################
# ##   Cross Validation Prediction ; ROC Plotting ; AUC Calc   ##
# Cross_Valid_start = time.time()

# # Create a figure with 2x2 subplots
# fig, axs = plt.subplots(2, 2, figsize=(10, 8))

#     # OVA Prostate Entropy 
# OVA_Prostate_entropy_y_scores = model_selection.cross_val_predict(OVA_Prostate_entropy,OVA_Prostate.data,OVA_Prostate.target,method='predict_proba',cv=10)
# OVA_Prostate_entropy_fpr,OVA_Prostate_entropy_tpr,OVA_Prostate_entropy_th = roc_curve(OVA_Prostate.target,OVA_Prostate_entropy_y_scores[:,1],pos_label="Prostate")
# axs[0,0].set_xlabel("1 - Specificity")
# axs[0,0].set_ylabel("Sensitivity")
# axs[0,0].set_xlim(0,1)
# axs[0,0].set_ylim(0,1)
# axs[0,0].plot(OVA_Prostate_entropy_fpr,OVA_Prostate_entropy_tpr)
# axs[0,0].set_title("OVA Prostate Entropy Trained ROC")
# OVA_Prostate_entropy_AUC = roc_auc_score(OVA_Prostate.target, OVA_Prostate_entropy_y_scores[:,1])

#     # Steel Plate Falult Entropy
# Steel_Plate_Fault_entropy_y_scores = model_selection.cross_val_predict(Steel_plate_falult_entropy,Steel_plate_falult.data,Steel_plate_falult.target,method='predict_proba',cv=10)
# Steel_Plate_Fault_entropy_fpr,Steel_Plate_Fault_entropy_tpr,Steel_Plate_Fault_entropy_th = roc_curve(Steel_plate_falult.target,Steel_Plate_Fault_entropy_y_scores[:,1],pos_label="2")
# axs[0,1].set_xlabel("1 - Specificity")
# axs[0,1].set_ylabel("Sensitivity")
# axs[0,1].set_xlim(0,1)
# axs[0,1].set_ylim(0,1)
# axs[0,1].plot(Steel_Plate_Fault_entropy_fpr,Steel_Plate_Fault_entropy_tpr)
# axs[0,1].set_title("Steel Plate Fault Entropy Trained ROC")
# Steel_Plate_Fault_entropy_AUC = roc_auc_score(Steel_plate_falult.target, Steel_Plate_Fault_entropy_y_scores[:,1])

#     # OVA Prostate Gini Index
# OVA_Prostate_Gini_y_scores = model_selection.cross_val_predict(OVA_Prostate_Gini,OVA_Prostate.data,OVA_Prostate.target,method='predict_proba',cv=10)
# OVA_Prostate_Gini_fpr,OVA_Prostate_Gini_tpr,OVA_Prostate_th = roc_curve(OVA_Prostate.target,OVA_Prostate_Gini_y_scores[:,1],pos_label="Prostate")
# axs[1,0].set_xlabel("1 - Specificity")
# axs[1,0].set_ylabel("Sensitivity")
# axs[1,0].set_xlim(0,1)
# axs[1,0].set_ylim(0,1)
# axs[1,0].plot(OVA_Prostate_Gini_fpr,OVA_Prostate_Gini_tpr)
# axs[1,0].set_title("OVA Prostate Gini Index Trained ROC")
# OVA_Prostate_Gini_AUC = roc_auc_score(OVA_Prostate.target, OVA_Prostate_Gini_y_scores[:,1])

#     # Steel Plate Falult Gini Index
# Steel_Plate_Fault_Gini_y_scores = model_selection.cross_val_predict(Steel_plate_falult_Gini,Steel_plate_falult.data,Steel_plate_falult.target,method='predict_proba',cv=10)
# Steel_plate_falult_Gini_fpr,Steel_plate_falult_Gini_tpr,Steel_plate_falult_Gini_th = roc_curve(Steel_plate_falult.target,Steel_Plate_Fault_Gini_y_scores[:,1],pos_label="2")
# axs[1,1].set_xlabel("1 - Specificity")
# axs[1,1].set_ylabel("Sensitivity")
# axs[1,1].set_xlim(0,1)
# axs[1,1].set_ylim(0,1)
# axs[1,1].plot(Steel_plate_falult_Gini_fpr,Steel_plate_falult_Gini_tpr)
# axs[1,1].set_title("Steel Plate Fault Gini Index Trained ROC")
# Steel_Plate_Fault_Gini_AUC = roc_auc_score(Steel_plate_falult.target, Steel_Plate_Fault_Gini_y_scores[:,1])
   
#     # Display Tables
# plt.tight_layout()
# plt.show()

# AUC_Values = {'OVA Prostate Entropy AUC': [OVA_Prostate_entropy_AUC],
#                 'Steel Fault Entropy AUC': [Steel_Plate_Fault_entropy_AUC],
#                 'OVA Prostate Gini AUC': [OVA_Prostate_Gini_AUC],
#                 'Steel Fault Gini AUC': [Steel_Plate_Fault_Gini_AUC]}
# AUC_Values_df = pandas.DataFrame(AUC_Values)
# print(AUC_Values_df)

# Cross_Valid__end = time.time()
# print("Cross Validation Time (Seconds): ", round(Cross_Valid__end- Cross_Valid_start,3))

# #############################################################################################################
# #      Parameter Tuning      ##
# Param_Tuning_start = time.time()

# min_samples_leaf_Search = {"min_samples_leaf": [i for i in range(1, 6)] } # Paramaters Tuning list (1- 5)

#     # OVA Prostate Entropy Tuning
# OVA_Prostate_entropy_Tuned = model_selection.GridSearchCV(OVA_Prostate_entropy, min_samples_leaf_Search, scoring="roc_auc", cv=5)
# OVA_Prostate_entropy_Tuned_y_scores = model_selection.cross_val_predict(OVA_Prostate_entropy_Tuned, OVA_Prostate.data, OVA_Prostate.target, method="predict_proba",cv=10)
# OVA_Prostate_entropy_Tuned.fit(OVA_Prostate.data,OVA_Prostate.target)


# #############################################################################################################
##     End Timer       ##
Whole_Time_end = time.time()
print("Whole Program Time (Seconds): ", round(Whole_Time_end-Whole_Time_start,3))