import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score



# ---------------------------------------------- #
# 各项指标检测                                    
# ---------------------------------------------- #
def evaluation(y_target, y_predict):
    _, ind = y_predict.topk(1, 1, True, True)
    y_target = y_target.cpu().detach().numpy()
    ind = ind.cpu().detach().numpy()
    accuracy=classification_report(y_target, ind,output_dict=True,zero_division=0.0)['accuracy']
    s=classification_report(y_target, ind,output_dict=True,zero_division=0.0)['weighted avg']
    precision=s['precision']
    recall=s['recall']
    f1_score=s['f1-score']
    return accuracy,precision,recall,f1_score #, kapp

# ---------------------------------------------- #
# 相似度指标检测                                   
# ---------------------------------------------- #
def evaluation_split(y_target, y_predict):
    y_target = y_target.cpu().detach().numpy()
    y_predict = y_predict.cpu().detach().numpy()
    accuracy=classification_report(y_target, y_predict,output_dict=True,zero_division=0.0)['accuracy']
    s=classification_report(y_target, y_predict,output_dict=True,zero_division=0.0)['weighted avg']
    precision=s['precision']
    recall=s['recall']
    f1_score=s['f1-score']
    return accuracy,precision,recall,f1_score #, kapp



# ---------------------------------------------- #
# 准确度计算                                          
# ---------------------------------------------- #
def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)