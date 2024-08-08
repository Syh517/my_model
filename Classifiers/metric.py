import os
import numpy as np
from sklearn import metrics
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

def metric(y_true,y_pred):
    # TP,TN,FP,FN,precision,recall,F1
    TP=0
    TN=0
    FP=0
    FN=0

    for i in range(len(y_true)):
        if y_true[i] ==0 and y_pred[i] ==0:
            TP+=1
        elif y_true[i] ==1 and y_pred[i] ==1:
            TN+=1
        elif y_true[i] ==1 and y_pred[i] ==0:
            FP+=1
        else: # y_true[i] ==0 and y_pred[i] ==1
            FN+=1

    if TP+FP==0:
        Precision='NULL'
    else:
        Precision=TP/(TP+FP)

    if TP +FN ==0:
        Recall='NULL'
    else:
        Recall=TP/(TP+FN)

    if Precision=='NULL' or Recall=='NULL' or Precision+Recall==0:
        F1='NULL'
    else:
        F1=2*Precision*Recall/(Precision+Recall)

    # Precision = TP / (TP + FP + 0.00001)
    # Recall = TP / (TP + FN + 0.00001)
    # F1 = 2 * Precision * Recall / (Precision + Recall + 0.00001)
    Accuracy=(TP+TN)/(TP+TN+FP+FN+ 0.00001)
    AUC=metrics.roc_auc_score(y_true, y_pred)


    print(TP,TN,FP,FN)
    t= {}
    t['Precision']=Precision
    t['Recall']=Recall
    t['F1']=F1
    t['Accuracy'] =Accuracy
    t['AUC']=AUC

    return t
