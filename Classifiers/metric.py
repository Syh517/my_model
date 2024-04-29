import os
import numpy as np

def metrics(y_true,y_pred):
    # TP,TN,FP,FN,precision,recall,F1
    TP=0
    TN=0
    FP=0
    FN=0
    for i in range(len(y_true)):
        for j in range(len(y_pred)):
            if y_true[i] ==0 and y_pred[j] ==0:
                TP+=1
            elif y_true[i] ==1 and y_pred[j] ==1:
                TN+=1
            elif y_true[i] ==1 and y_pred[j] ==0:
                FP+=1
            else:
                FN+=1

    print(TP,TN,FP,FN)
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

    Accuracy=(TP+TN)/(TP+TN+FP+FN)

    t= {}
    t['Precision']=Precision
    t['Recall']=Recall
    t['F1']=F1
    t['Accuracy'] =Accuracy

    return t