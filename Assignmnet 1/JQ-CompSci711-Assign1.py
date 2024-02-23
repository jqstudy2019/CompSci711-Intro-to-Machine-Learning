#############################################################################################################
#   File Name: JQ-CompSci711-Assign1.py   
#   Course: CompSci 711 - Intro to Machine Learing
#   Author: John Quinn
#   Assignment Task:
#       Separately on each dataset, 
#            compare decision trees created using 
#                entropy
#                gini index criteria 
#            Compare by: 
#                plotting ROC curves    
#                computing the AUC values   
#           Generate results using 10-fold cross-validation
#            Do parameter search on min_samples_leaf parameter using GridsearchCV  (at least 5 different values). 
#
#       Write a program to do the entire task (i.e. do not do it from the Python prompt as in the slides).
#
#############################################################################################################
##     Start Program Timer       ##
Whole_Time_start = time.time()

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

#############################################################################################################
##   Cross Validation Prediction ; ROC Plotting ; AUC Calc   ##

    # OVA Prostate Entropy 
OVA_Prostate_entropy_y_scores = model_selection.cross_val_predict(OVA_Prostate_entropy,OVA_Prostate.data,OVA_Prostate.target,method='predict_proba',cv=10)
OVA_Prostate_entropy_fpr,OVA_Prostate_entropy_tpr,OVA_Prostate_entropy_th = roc_curve(OVA_Prostate.target,OVA_Prostate_entropy_y_scores[:,1],pos_label="Prostate")
plt.xlabel("1 - Specificity")
plt.ylabel("Sensitivity")
plt.xlim(0,1)
plt.ylim(0,1)
plt.plot(OVA_Prostate_entropy_fpr,OVA_Prostate_entropy_tpr,label="Decision Tree")
plt.legend()
plt.show()
OVA_Prostate_entropy_AUC = roc_auc_score(OVA_Prostate.target, OVA_Prostate_entropy_y_scores[:,1])
print("OVA Prostate Entropy AUC= ",OVA_Prostate_entropy_AUC)

    # Steel Plate Falult Entropy
Steel_Plate_Fault_entropy_y_scores = model_selection.cross_val_predict(Steel_plate_falult_entropy,Steel_plate_falult.data,Steel_plate_falult.target,method='predict_proba',cv=10)
Steel_Plate_Fault_entropy_fpr,Steel_Plate_Fault_entropy_tpr,Steel_Plate_Fault_entropy_th = roc_curve(Steel_plate_falult.target,Steel_Plate_Fault_entropy_y_scores[:,1],pos_label="2")
plt.xlabel("1 - Specificity")
plt.ylabel("Sensitivity")
plt.xlim(0,1)
plt.ylim(0,1)
plt.plot(Steel_Plate_Fault_entropy_fpr,Steel_Plate_Fault_entropy_tpr,label="Decision Tree")
plt.legend()
plt.show()
Steel_Plate_Fault_entropy_AUC = roc_auc_score(Steel_plate_falult.target, Steel_Plate_Fault_entropy_y_scores[:,1])
print("Steel Fault Entropy AUC= ",Steel_Plate_Fault_entropy_AUC)

    # OVA Prostate Gini Index
OVA_Prostate_Gini_y_scores = model_selection.cross_val_predict(OVA_Prostate_Gini,OVA_Prostate.data,OVA_Prostate.target,method='predict_proba',cv=10)
OVA_Prostate_Gini_fpr,OVA_Prostate_Gini_tpr,OVA_Prostate_th = roc_curve(OVA_Prostate.target,OVA_Prostate_Gini_y_scores[:,1],pos_label="Prostate")
plt.xlabel("1 - Specificity")
plt.ylabel("Sensitivity")
plt.xlim(0,1)
plt.ylim(0,1)
plt.plot(OVA_Prostate_Gini_fpr,OVA_Prostate_Gini_tpr,label="Decision Tree")
plt.legend()
plt.show()
OVA_Prostate_Gini_AUC = roc_auc_score(OVA_Prostate.target, OVA_Prostate_Gini_y_scores[:,1])
print("OVA Prostate Gini Index AUC= ",OVA_Prostate_Gini_AUC)

    # Steel Plate Falult Gini Index
Steel_Plate_Fault_Gini_y_scores = model_selection.cross_val_predict(Steel_plate_falult_Gini,Steel_plate_falult.data,Steel_plate_falult.target,method='predict_proba',cv=10)
Steel_plate_falult_Gini_fpr,Steel_plate_falult_Gini_tpr,Steel_plate_falult_Gini_th = roc_curve(Steel_plate_falult.target,Steel_Plate_Fault_Gini_y_scores[:,1],pos_label="2")
plt.xlabel("1 - Specificity")
plt.ylabel("Sensitivity")
plt.xlim(0,1)
plt.ylim(0,1)
plt.plot(Steel_plate_falult_Gini_fpr,Steel_plate_falult_Gini_tpr,label="Decision Tree")
plt.legend()
plt.show()
Steel_Plate_Fault_Gini_AUC = roc_auc_score(Steel_plate_falult.target, Steel_Plate_Fault_Gini_y_scores[:,1])
print("Steel Fault Gini Index AUC= ",Steel_Plate_Fault_Gini_AUC)
#############################################################################################################
#      Parameter Tuning Testing     ##
min_samples_leaf_Search = {"min_samples_leaf": [i for i in range(1, 6)] }

loading_dtc_start = time.time()
OVA_Prostate_entropy = DecisionTreeClassifier(criterion='entropy')
loading_dtc_end = time.time()
print("DTC Time (Seconds): ", round(loading_dtc_end-loading_dtc_start,3))

loading_GridCV_start = time.time()
OVA_Prostate_entropy_Tuned = model_selection.GridSearchCV(OVA_Prostate_entropy, min_samples_leaf_Search, scoring="roc_auc", cv=5)
loading_GridCV_end = time.time()
print("GridCV Time (Seconds): ", round(loading_GridCV_end-loading_GridCV_start,3))

loading_CrossVal_start = time.time()
OVA_Prostate_entropy_Tuned_y_scores = model_selection.cross_val_predict(OVA_Prostate_entropy_Tuned, OVA_Prostate.data, OVA_Prostate.target, method="predict_proba",cv=10)
loading_CrossVal_end = time.time()
print("CrossVal Time (Seconds): ", round(loading_CrossVal_end-loading_CrossVal_start,3))

loading_Fit_start = time.time()
OVA_Prostate_entropy_Tuned.fit(OVA_Prostate.data,OVA_Prostate.target)
loading_Fit_end = time.time()
print("Fit Time (Seconds): ", round(loading_Fit_end-loading_Fit_start,3))

print("OVA Prostate Best Params:", OVA_Prostate_entropy_Tuned.best_params_)

#############################################################################################################
##     End Timer       ##
Whole_Time_end = time.time()
print("Whole Program Time (Seconds): ", round(Whole_Time_end-Whole_Time_start,3))

