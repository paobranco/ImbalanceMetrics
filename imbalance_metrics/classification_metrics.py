from sklearn.metrics import  precision_recall_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics._ranking import _binary_clf_curve
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import copy
import math
import warnings


def get_minority(y):
    y = pd.Series(y)
    value_counts = y.value_counts()
    if len(value_counts) == 0:
        raise Exception("Dataset can not be empty") 
    else:
        minor = value_counts.idxmin()
        #print("Using minority class "+str(minor)+" as positive class")
        return minor

def get_pr(y_true, y_probabilities,pos_label= None):
    precision, recall, _ = precision_recall_curve(y_true, y_probabilities,pos_label=pos_label)
    return precision, recall

def calculate_classification_phi(y_true,phi_option = 1, return_phi_per_class = False):
    """
    Calculates the Phi relvance value for each class of 'y'.

    Parameters
    ----------
    y_true : array-like
        Input data for which phi value needs to be calculated.

    phi_option : int, default = 1
        If 1, Classes weighted by inverse frequency 
        If 2, Same as Option 1 but divided by class frequency 

    return_phi_per_class : bool, default = False
        Whether to return the Phi relevance value of each class or each sample
    Returns
    -------
    y_phi : array-like
        If return_phi_per_class = False, Phi values for each element of 'y_true'.
        If return_phi_per_class = True, Phi values for each class of 'y_true'.
    """
    sum_inverse = y_true.value_counts().apply(lambda x: 1 / x).sum()
    phi = y_true.value_counts().apply(lambda x: (1 / x) / sum_inverse)
    if return_phi_per_class:
        return phi
    else:
        y_phi = y_true.map(phi)
        if phi_option == 1:
            return y_phi
        elif phi_option == 2: 
            y_count = y_true.map(y_true.value_counts())
            new_y_phi= y_phi/y_count
            return new_y_phi
        else: 
            raise Exception()

def gmean_score(y_true, y_pred, weighted = True):
    """
    Calculates geometric mean score.

    Parameters
    ----------
    y_true : array-like
        True target values.

    y_pred : array-like
        Predicted target values.

    weighted : bool, default = True
        Whether to use Phi relevance value for calculation. 

    Returns
    -------
    float
        The geometric mean score.

    """
    classes = np.unique(y_true)

    matrix = confusion_matrix(y_true, y_pred)

    recalls = []
    for i in range(len(classes)):
        TP = matrix[i, i]
        FN = np.sum(matrix[i, :]) - TP
        recall = TP / (TP + FN)
        if weighted:
            phi = calculate_classification_phi(y_true, return_phi_per_class = True)
            recall = recall * phi[classes[i]]
        recalls.append(recall)

    recalls_product = 1
    for r in recalls:
        recalls_product *= r

    gmean = math.pow(recalls_product, 1 / len(classes))
    return gmean

def pr_davis(y_true, y_probabilities,return_pr=False, pos_label= None):
    """
    Calculates Precision-Recall AUC using Davis method.
    
    Parameters
    ----------
    y_true : array-like
        True target values.

    y_probabilities : array-like
        Predicted target values.

    return_pr : bool, default = False
        If True, return precision and recall values, and AUC score.
    
    pos_label : int or str, default = None
        The label of the positive class. When pos_label = None, minority value is selected.

        
    Returns
    -------
    float or tuple
        If return_pr=False, returns the precision-recall AUC score.
        If return_pr=True, returns the precision, recall and precision-recall AUC score.

    """
    if pos_label == None:
        pos_label = get_minority(y_true)

    labels = np.unique(y_true)
    try:
        index  = np.where(labels == pos_label)[0][0]
    except IndexError:
        warnings.warn("Positive label not found. Using minority class as positive")
        pos_label = get_minority(y_true)
        index  = np.where(labels == pos_label)[0][0]

    try:
        fps, tps, _ = _binary_clf_curve(y_true, y_probabilities[:,index],pos_label=pos_label,sample_weight=None)
    except IndexError:
        fps, tps, _ = _binary_clf_curve(y_true, y_probabilities,pos_label=pos_label,sample_weight=None)
    


    #Interpolate new TPs and FPs when diff between successive TP is >1
    for i in range(len(tps)-1):
        if (tps[i+1] - tps[i]) >= 2:
            local_skew = (fps[i+1]-fps[i])/(tps[i+1]-tps[i])
        
        for x in range(1,int(tps[i+1] - tps[i])):
            new_fp = fps[i]+(local_skew*x)
            tps = np.insert(tps, i+x, tps[i]+x)
            fps = np.insert(fps, i+x, new_fp)


    precision_davis = tps / (tps + fps)
    precision_davis[np.isnan(precision_davis)] = 0
    recall_davis = tps / tps[-1]
        
    # Stop when full recall is attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)

    precision_davis = np.r_[precision_davis[sl], 1]
    recall_davis = np.r_[recall_davis[sl], 0]
    pr_auc_davis = auc(recall_davis, precision_davis)

    if return_pr:
        return precision_davis, recall_davis, pr_auc_davis
    
    else:
        return pr_auc_davis


def pr_manning(y_true, y_probabilities,return_pr=False, pos_label= None):

    """
    Calculates Precision-Recall AUC using Manning method.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_probabilities : array-like
        Predicted target values.
    return_pr : bool, optional (default=False)
        Whether to return precision, recall, and AUC values.
    pos_label : int or str, default = None
        The label of the positive class. When pos_label=None, minority value is selected.


    Returns
    -------
    float or tuple of arrays and float
        If return_pr=False, returns the precision-recall AUC score.
        If return_pr=True, returns the precision, recall and precision-recall AUC score.
    """

    if pos_label == None:
        pos_label = get_minority(y_true)

    labels = np.unique(y_true)
    try:
        index  = np.where(labels == pos_label)[0][0]
    except IndexError:
        warnings.warn("Positive label not found. Using minority class as positive")
        pos_label = get_minority(y_true)
        index  = np.where(labels == pos_label)[0][0]

    try:
        precision, recall = get_pr(y_true, y_probabilities[:,index],pos_label=pos_label)
    except IndexError:
        precision, recall = get_pr(y_true, y_probabilities,pos_label=pos_label)

    precision_manning = copy.deepcopy(precision)
    recall_manning = copy.deepcopy(recall)
    prInv = np.fliplr([precision_manning])[0]
    recInv = np.fliplr([recall_manning])[0]
    j = recall_manning.shape[0]-2

    while j>=0:
        if prInv[j+1]>prInv[j]:
            prInv[j]=prInv[j+1]
        j=j-1

    decreasing_max_precision = np.maximum.accumulate(prInv[::-1])[::-1]
    pr_auc_manning = auc(recInv, decreasing_max_precision)

    if return_pr:
        return decreasing_max_precision, recInv, pr_auc_manning
    
    else:
        return pr_auc_manning


def cross_validate_auc(clf, X, y, scoring, cv, pr = False, pos_label= None):
    """
    Calculates Cross-validated Area Under the Curve (AUC) score.

    Parameters
    ----------
    clf : classifier object
        Classifier object used for cross-validation.
    y_true : array-like
        True target values.
    y_probabilities : array-like
        Predicted target values.
    pr : bool, optional (default=False)
        Whether to use precision and recall while calculating AUC values.
    pos_label : int or str, default = None
        The label of the positive class. When pos_label=None, minority value is selected.

    Returns
    -------
    float 
        Cross-validated Area Under the Curve (AUC) score.
    """
    

    cv = StratifiedKFold(n_splits=cv)
    y_probabilities = []
    y_true = []

    for train,test in cv.split(X, y) :
        clf.fit(X.iloc[train], y.iloc[train])
        
        #Predictions
        pred_proba = clf.predict_proba(X.iloc[test])
        
        y_true.append(y.iloc[test]) 
        y_probabilities.append(pred_proba)

    y_true = np.concatenate(y_true)
    y_probabilities = np.concatenate(y_probabilities)

    if pos_label == None:
        pos_label = get_minority(y_true)
    
    labels = np.unique(y_true)
    try:
        index  = np.where(labels == pos_label)[0][0]
    except IndexError:
        warnings.warn("Positive label not found. Using minority class as positive")
        pos_label = get_minority(y_true)
        index  = np.where(labels == pos_label)[0][0]
        
    if pr:
        precision, recall = get_pr(y_true, y_probabilities[:,index],pos_label=pos_label)  
        return scoring(recall,precision)
    else:
        return scoring(y_true,y_probabilities,pos_label=pos_label) 
