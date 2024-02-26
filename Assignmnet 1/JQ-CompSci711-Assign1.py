#############################################################################################################
#   File Name: JQ-CompSci711-Assign1.py   
#   Course: CompSci 711 - Intro to Machine Learing
#   Author: John Quinn
#   Assignment Tasks:
#       Write a program to do the entire task (i.e. do not do it from the Python prompt as in the slides).
#           Separately on each dataset, 
#               compare decision trees created using 
#                   entropy
#                   gini index criteria 
#               Compare by: 
#                   plotting ROC curves    
#                   computing the AUC values   
#               Generate results using 10-fold cross-validation
#               Do parameter search on min_samples_leaf parameter using GridsearchCV  (at least 5 different values). 
#
#       Also Write report - See Canvas for details 
#
#
#
#       Not needed but included timers in the program as some steps take a long time to run and I wanted to make sure I had an idea of how long they would run/ show they are running
#
#       On my laptop this took ~2000 Seconds/  to run 
#############################################################################################################
##      Package Imports     ##

    #   General Packages  #
import pandas
import numpy
from matplotlib import pyplot as plt
import time

    #   SkLearn Imports     #
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import model_selection

#############################################################################################################
##     Start Program Timer       ##
Whole_Time_start = time.time()

#############################################################################################################
##   Loading Data Sets      ##
loading_OVA_start = time.time()
OVA_Prostate = datasets.fetch_openml(data_id=1146) # https://www.openml.org/search?type=data&sort=qualities.NumberOfNumericFeatures&status=any&qualities.NumberOfClasses=%3D_2&qualities.NumberOfFeatures=gte_0&qualities.NumberOfInstances=between_1000_10000&id=1146
loading_OVA_end = time.time()
print("OVA_Prostate Dataset Loading Time (Seconds): ", round(loading_OVA_end-loading_OVA_start,3))

loading_Steel_start = time.time()
Steel_plate_falult = datasets.fetch_openml(data_id=1504) # https://www.openml.org/search?type=data&sort=runs&status=any&qualities.NumberOfClasses=%3D_2&qualities.NumberOfFeatures=gte_0&qualities.NumberOfInstances=between_1000_10000&id=1504
loading_Steel_end = time.time()
print("Steel Plate Fault Dataset Loading Time (Seconds): ", round(loading_Steel_end-loading_Steel_start,3))

#############################################################################################################
##   Create Decision Trees     ##
Fitting_start = time.time()
    #   Entropy Training and Prediction #
        #   OVA Prostate Entropy 
OVA_Prostate_entropy = DecisionTreeClassifier(criterion='entropy')
OVA_Prostate_entropy.fit(OVA_Prostate.data,OVA_Prostate.target)

        # Stee Plate Falult Entropy
Steel_plate_falult_entropy = DecisionTreeClassifier(criterion='entropy')
Steel_plate_falult_entropy.fit(Steel_plate_falult.data,Steel_plate_falult.target)

    #   Gini Index Training and Prediction #
        #   OVA Prostate Gini Index
OVA_Prostate_Gini = DecisionTreeClassifier(criterion='gini')
OVA_Prostate_Gini.fit(OVA_Prostate.data,OVA_Prostate.target)

        # Stee Plate Falult Gini Index
Steel_plate_falult_Gini = DecisionTreeClassifier(criterion='gini')
Steel_plate_falult_Gini.fit(Steel_plate_falult.data,Steel_plate_falult.target)

Fitting_end = time.time()
print("Fitting Time (Seconds): ", round(Fitting_end-Fitting_start,3))
#############################################################################################################
##   Cross Validation Prediction ; ROC Plotting ; AUC Calc   ##
Cross_Valid_start = time.time()

# Create a figure with 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # OVA Prostate Entropy 
OVA_Prostate_entropy_y_scores = model_selection.cross_val_predict(OVA_Prostate_entropy,OVA_Prostate.data,OVA_Prostate.target,method='predict_proba',cv=10)
OVA_Prostate_entropy_fpr,OVA_Prostate_entropy_tpr,OVA_Prostate_entropy_th = roc_curve(OVA_Prostate.target,OVA_Prostate_entropy_y_scores[:,1],pos_label="Prostate")
axs[0,0].set_xlabel("1 - Specificity")
axs[0,0].set_ylabel("Sensitivity")
axs[0,0].set_xlim(0,1)
axs[0,0].set_ylim(0,1)
axs[0,0].plot(OVA_Prostate_entropy_fpr,OVA_Prostate_entropy_tpr)
axs[0,0].set_title("OVA Prostate Entropy Trained ROC")
OVA_Prostate_entropy_AUC = roc_auc_score(OVA_Prostate.target, OVA_Prostate_entropy_y_scores[:,1])

    # Steel Plate Falult Entropy
Steel_Plate_Fault_entropy_y_scores = model_selection.cross_val_predict(Steel_plate_falult_entropy,Steel_plate_falult.data,Steel_plate_falult.target,method='predict_proba',cv=10)
Steel_Plate_Fault_entropy_fpr,Steel_Plate_Fault_entropy_tpr,Steel_Plate_Fault_entropy_th = roc_curve(Steel_plate_falult.target,Steel_Plate_Fault_entropy_y_scores[:,1],pos_label="2")
axs[0,1].set_xlabel("1 - Specificity")
axs[0,1].set_ylabel("Sensitivity")
axs[0,1].set_xlim(0,1)
axs[0,1].set_ylim(0,1)
axs[0,1].plot(Steel_Plate_Fault_entropy_fpr,Steel_Plate_Fault_entropy_tpr)
axs[0,1].set_title("Steel Plate Fault Entropy Trained ROC")
Steel_Plate_Fault_entropy_AUC = roc_auc_score(Steel_plate_falult.target, Steel_Plate_Fault_entropy_y_scores[:,1])

    # OVA Prostate Gini Index
OVA_Prostate_Gini_y_scores = model_selection.cross_val_predict(OVA_Prostate_Gini,OVA_Prostate.data,OVA_Prostate.target,method='predict_proba',cv=10)
OVA_Prostate_Gini_fpr,OVA_Prostate_Gini_tpr,OVA_Prostate_th = roc_curve(OVA_Prostate.target,OVA_Prostate_Gini_y_scores[:,1],pos_label="Prostate")
axs[1,0].set_xlabel("1 - Specificity")
axs[1,0].set_ylabel("Sensitivity")
axs[1,0].set_xlim(0,1)
axs[1,0].set_ylim(0,1)
axs[1,0].plot(OVA_Prostate_Gini_fpr,OVA_Prostate_Gini_tpr)
axs[1,0].set_title("OVA Prostate Gini Index Trained ROC")
OVA_Prostate_Gini_AUC = roc_auc_score(OVA_Prostate.target, OVA_Prostate_Gini_y_scores[:,1])

    # Steel Plate Falult Gini Index
Steel_Plate_Fault_Gini_y_scores = model_selection.cross_val_predict(Steel_plate_falult_Gini,Steel_plate_falult.data,Steel_plate_falult.target,method='predict_proba',cv=10)
Steel_plate_falult_Gini_fpr,Steel_plate_falult_Gini_tpr,Steel_plate_falult_Gini_th = roc_curve(Steel_plate_falult.target,Steel_Plate_Fault_Gini_y_scores[:,1],pos_label="2")
axs[1,1].set_xlabel("1 - Specificity")
axs[1,1].set_ylabel("Sensitivity")
axs[1,1].set_xlim(0,1)
axs[1,1].set_ylim(0,1)
axs[1,1].plot(Steel_plate_falult_Gini_fpr,Steel_plate_falult_Gini_tpr)
axs[1,1].set_title("Steel Plate Fault Gini Index Trained ROC")
Steel_Plate_Fault_Gini_AUC = roc_auc_score(Steel_plate_falult.target, Steel_Plate_Fault_Gini_y_scores[:,1])
   
    # Display Tables
plt.tight_layout()
plt.show()

AUC_Values = {'OVA Prostate Entropy AUC': [OVA_Prostate_entropy_AUC],
                'Steel Fault Entropy AUC': [Steel_Plate_Fault_entropy_AUC],
                'OVA Prostate Gini AUC': [OVA_Prostate_Gini_AUC],
                'Steel Fault Gini AUC': [Steel_Plate_Fault_Gini_AUC]}
AUC_Values_df = pandas.DataFrame(AUC_Values)
print(AUC_Values_df)

Cross_Valid__end = time.time()
print("Cross Validation Time (Seconds): ", round(Cross_Valid__end- Cross_Valid_start,3))

#############################################################################################################
#      Parameter Tuning      ##
Param_Tuning_start = time.time()

min_samples_leaf_Search = {"min_samples_leaf": [i for i in range(1, 6)] } # Paramaters Tuning list (1- 5)

    # OVA Prostate Entropy Tuning
OVA_Prostate_entropy_Tuned = model_selection.GridSearchCV(OVA_Prostate_entropy, min_samples_leaf_Search, scoring="roc_auc", cv=5)
OVA_Prostate_entropy_Tuned_y_scores = model_selection.cross_val_predict(OVA_Prostate_entropy_Tuned, OVA_Prostate.data, OVA_Prostate.target, method="predict_proba",cv=10)
OVA_Prostate_entropy_Tuned.fit(OVA_Prostate.data,OVA_Prostate.target)

    # Steel Fault Entropy Tuning
Steel_plate_falult_entropy_Tuned = model_selection.GridSearchCV(Steel_plate_falult_entropy, min_samples_leaf_Search, scoring="roc_auc", cv=5)
Steel_plate_falult_entropy_Tuned_y_scores = model_selection.cross_val_predict(Steel_plate_falult_entropy_Tuned, Steel_plate_falult.data, Steel_plate_falult.target, method="predict_proba",cv=10)
Steel_plate_falult_entropy_Tuned.fit(Steel_plate_falult.data, Steel_plate_falult.target)

    # OVA Prostate Gini Tuning
OVA_Prostate_gini_Tuned = model_selection.GridSearchCV(OVA_Prostate_Gini, min_samples_leaf_Search, scoring="roc_auc", cv=5)
OVA_Prostate_gini_Tuned_y_scores = model_selection.cross_val_predict(OVA_Prostate_gini_Tuned, OVA_Prostate.data, OVA_Prostate.target, method="predict_proba",cv=10)
OVA_Prostate_gini_Tuned.fit(OVA_Prostate.data,OVA_Prostate.target)

    # Steel Fault Gini Tuning
Steel_plate_falult_gini_Tuned = model_selection.GridSearchCV(Steel_plate_falult_Gini, min_samples_leaf_Search, scoring="roc_auc", cv=5)
Steel_plate_falult_gini_Tuned_y_scores = model_selection.cross_val_predict(Steel_plate_falult_gini_Tuned, Steel_plate_falult.data, Steel_plate_falult.target, method="predict_proba",cv=10)
Steel_plate_falult_gini_Tuned.fit(Steel_plate_falult.data, Steel_plate_falult.target)

Best_Params = {'OVA Prostate Entropy Best Params': [OVA_Prostate_entropy_Tuned.best_params_],
        'Steel Fault Entropy Best Params': [Steel_plate_falult_entropy_Tuned.best_params_],
        'OVA Prostate Gini Best Params': [OVA_Prostate_gini_Tuned.best_params_],
        'Steel Fault Gini Best Params': [Steel_plate_falult_gini_Tuned.best_params_]}
Best_Params_df = pandas.DataFrame(Best_Params)
print(Best_Params_df)

Param_Tuning_end = time.time()
print("Parameter Tuning Time (Seconds): ", round(Param_Tuning_end-Param_Tuning_start,3))

#############################################################################################################
##     End Timer       ##
Whole_Time_end = time.time()
print("Whole Program Time (Seconds): ", round(Whole_Time_end-Whole_Time_start,3))