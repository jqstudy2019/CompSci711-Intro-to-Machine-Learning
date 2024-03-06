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

from ucimlrepo import fetch_ucirepo 

from scipy.io.arff import loadarff

    #   SkLearn Imports     
from sklearn import datasets
# from sklearn.tree import DecisionTreeClassifier
# from sklearn import tree
# from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.impute import SimpleImputer

#############################################################################################################
##     Start Program Timer       ##
Whole_Time_start = time.time()

#############################################################################################################
##   Loading Data Sets      ##
loading_Moneyball_start = time.time()
Moneyball_data = datasets.fetch_openml(data_id=41021) # https://www.openml.org/search?type=data&sort=runs&status=active&qualities.NumberOfInstances=between_1000_10000&qualities.NumberOfClasses=lte_1&id=688
loading_Moneyball_end = time.time()
print("Moneyball Dataset Loading Time (Seconds): ", round(loading_Moneyball_end-loading_Moneyball_start,3))
Moneyball_data.data.info()
print(Moneyball_data)

# UCI Dataset
# fetch dataset 
seoul_bike_sharing_demand = pd.read_csv('SeoulBikeData.csv', encoding=None)
seoul_bike_sharing_demand_array = np.array(seoul_bike_sharing_demand)
print(seoul_bike_sharing_demand)


#############################################################################################################
##   One Hot Encoding  ##

ct_seoul_bike = ColumnTransformer([("encoder", OneHotEncoder(sparse_output=False), [0,11,12,13])], remainder="passthrough")
seoul_bike_trans_data = ct_seoul_bike.fit_transform(seoul_bike_sharing_demand_array)
seoul_bike_trans_data = pd.DataFrame(seoul_bike_trans_data, columns = ct_seoul_bike.get_feature_names_out())
print(seoul_bike_trans_data)

ct_Moneyball = ColumnTransformer([("encoder", OneHotEncoder(sparse_output=False), [0,1,8,9,10,11])], remainder="passthrough")
trans_moneyball = ct_Moneyball.fit_transform(Moneyball_data.data)
moneyball_new_data = pd.DataFrame(trans_moneyball, columns = ct_Moneyball.get_feature_names_out(), index = Moneyball_data.data.index)


#############################################################################################################
##   Missing Values Imputation  ##
#     # Used Mean Imputing to Fill values with most mean of non missing column 

imp = SimpleImputer(missing_values=np.nan, strategy='mean')

imp_moneyball = imp.fit_transform(moneyball_new_data)
imp_moneyball = pd.DataFrame(imp_moneyball, columns= imp.get_feature_names_out())
print(imp_moneyball)

#############################################################################################################
##   Linear Regression     ##

lr = LinearRegression()

lrr_moneyball = cross_validate(lr, imp_moneyball, Moneyball_data.target, cv=10, scoring="neg_root_mean_squared_error")
print(lrr_moneyball)

lrr_seoul_bike = cross_validate(lr, seoul_bike_trans_data, seoul_bike_sharing_demand.loc[:,'Rented Bike Count'], cv=10, scoring="neg_root_mean_squared_error")
print(lrr_seoul_bike)



# lrr_pbceq_y_scores = model_selection.cross_val_predict(lrr_moneyball,Moneyball_data.data,Moneyball_data.target,method='predict_proba',cv=10)
# Steel_Plate_Fault_entropy_fpr,Steel_Plate_Fault_entropy_tpr,Steel_Plate_Fault_entropy_th = roc_curve(Steel_plate_falult.target,Steel_Plate_Fault_entropy_y_scores[:,1],pos_label="2")
# OVA_Prostate_entropy_AUC = roc_auc_score(OVA_Prostate.target, OVA_Prostate_entropy_y_scores[:,1])

# #############################################################################################################
# ##   Create Decision Trees     ##
# Fitting_start = time.time()
#     #   Entropy Training and Prediction #
#         #   OVA Prostate Entropy 
# OVA_Prostate_entropy = DecisionTreeClassifier(criterion='entropy')
# OVA_Prostate_entropy.fit(OVA_Prostate.data,OVA_Prostate.target)

#         # Stee Plate Falult Entropy
# Steel_plate_falult_entropy = DecisionTreeClassifier(criterion='entropy')
# Steel_plate_falult_entropy.fit(Steel_plate_falult.data,Steel_plate_falult.target)

#     #   Gini Index Training and Prediction #
#         #   OVA Prostate Gini Index
# OVA_Prostate_Gini = DecisionTreeClassifier(criterion='gini')
# OVA_Prostate_Gini.fit(OVA_Prostate.data,OVA_Prostate.target)

#         # Stee Plate Falult Gini Index
# Steel_plate_falult_Gini = DecisionTreeClassifier(criterion='gini')
# Steel_plate_falult_Gini.fit(Steel_plate_falult.data,Steel_plate_falult.target)

# Fitting_end = time.time()
# print("Fitting Time (Seconds): ", round(Fitting_end-Fitting_start,3))
# #############################################################################################################
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

#     # Steel Fault Entropy Tuning
# Steel_plate_falult_entropy_Tuned = model_selection.GridSearchCV(Steel_plate_falult_entropy, min_samples_leaf_Search, scoring="roc_auc", cv=5)
# Steel_plate_falult_entropy_Tuned_y_scores = model_selection.cross_val_predict(Steel_plate_falult_entropy_Tuned, Steel_plate_falult.data, Steel_plate_falult.target, method="predict_proba",cv=10)
# Steel_plate_falult_entropy_Tuned.fit(Steel_plate_falult.data, Steel_plate_falult.target)

#     # OVA Prostate Gini Tuning
# OVA_Prostate_gini_Tuned = model_selection.GridSearchCV(OVA_Prostate_Gini, min_samples_leaf_Search, scoring="roc_auc", cv=5)
# OVA_Prostate_gini_Tuned_y_scores = model_selection.cross_val_predict(OVA_Prostate_gini_Tuned, OVA_Prostate.data, OVA_Prostate.target, method="predict_proba",cv=10)
# OVA_Prostate_gini_Tuned.fit(OVA_Prostate.data,OVA_Prostate.target)

#     # Steel Fault Gini Tuning
# Steel_plate_falult_gini_Tuned = model_selection.GridSearchCV(Steel_plate_falult_Gini, min_samples_leaf_Search, scoring="roc_auc", cv=5)
# Steel_plate_falult_gini_Tuned_y_scores = model_selection.cross_val_predict(Steel_plate_falult_gini_Tuned, Steel_plate_falult.data, Steel_plate_falult.target, method="predict_proba",cv=10)
# Steel_plate_falult_gini_Tuned.fit(Steel_plate_falult.data, Steel_plate_falult.target)

# Best_Params = {'OVA Prostate Entropy Best Params': [OVA_Prostate_entropy_Tuned.best_params_],
#         'Steel Fault Entropy Best Params': [Steel_plate_falult_entropy_Tuned.best_params_],
#         'OVA Prostate Gini Best Params': [OVA_Prostate_gini_Tuned.best_params_],
#         'Steel Fault Gini Best Params': [Steel_plate_falult_gini_Tuned.best_params_]}
# Best_Params_df = pandas.DataFrame(Best_Params)
# print(Best_Params_df)

# Param_Tuning_end = time.time()
# print("Parameter Tuning Time (Seconds): ", round(Param_Tuning_end-Param_Tuning_start,3))

# #############################################################################################################
##     End Timer       ##
Whole_Time_end = time.time()
print("Whole Program Time (Seconds): ", round(Whole_Time_end-Whole_Time_start,3))