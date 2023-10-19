#!/usr/bin/env python
# coding:utf-8

import json
import os

import numpy as np
from scipy import interp
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             classification_report, confusion_matrix, f1_score,
                             hamming_loss, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve, zero_one_loss)
from sklearn.preprocessing import label_binarize

import helper.logger as logger
from helper.tree import lca_height


def evaluate(config, metrics, epoch_predict_probs, epoch_predict_labels, epoch_target_labels, n_classes):
    """
    :param epoch_predict_probs: for binary and multiclass classification, List[List[Float]], softmax-predicted probability list
                                for multilabel classification, List[List[Float], ...], sigmoid-activated probability list
    :param epoch_predict_labels: for binary and multiclass classification, List[int], softmax-predicted label list
                                 for multilabel classification, List[List[int], ...], with threshold>0.0 or 0.5
    :param epoch_target_labels: for binary and multiclass classification, List[int], ground truth
                                for multilabel classification, List[List[int], ...]
    :param n_classes: int, number of classes
    :return 
    """
    assert len(epoch_predict_probs) == len(epoch_predict_labels) == len(epoch_target_labels), \
        'mismatch between prediction and ground truth labels for evaluation'
    
    if config.model.type in ["PreAttnMMs_MTL_IMP2", "PreAttnMMs_GAT_IMP8_GC", "PreAttnMMs_GAT_IMP8_GC_WeightedLoss", "PreAttnMMs_MTL_IMP3"]:
        # transform into np.array([])
        epoch_predict_labels = np.asarray(epoch_predict_labels)
        epoch_target_labels = np.asarray(epoch_target_labels)
        sample_size = epoch_predict_labels.shape[1]

        label_pred = np.zeros((sample_size, 11))
        label_target = np.zeros((sample_size, 11))

        # transform label_target and label_pred into binarized label, i.e. [[1,0,1,0,0,0,0,0,0,0,0],...]
        for i in range(sample_size):
            for index, j in enumerate(epoch_target_labels[:, i]):
                if index == 0:
                    if j == 0:
                        label_target[i, 0] = 1
                        label_target[i, 1] = 0
                    elif j == 1:
                        label_target[i, 0] = 0
                        label_target[i, 1] = 1
                elif index == 1:
                    if j == 0:
                        label_target[i, 2] = 1
                        label_target[i, 3] = 0
                        label_target[i, 4] = 0
                    elif j == 1:
                        label_target[i, 2] = 0
                        label_target[i, 3] = 1
                        label_target[i, 4] = 0
                    elif j == 2:
                        label_target[i, 2] = 0
                        label_target[i, 3] = 0
                        label_target[i, 4] = 1
                    elif j == 3:
                        label_target[i, 2] = 0
                        label_target[i, 3] = 0
                        label_target[i, 4] = 0
                elif index == 2:
                    if j == 0:
                        label_target[i, 5] = 1
                        label_target[i, 6] = 0
                    elif j == 1:
                        label_target[i, 5] = 0
                        label_target[i, 6] = 1
                    elif j == 2:
                        label_target[i, 5] = 0
                        label_target[i, 6] = 0
                elif index == 3:
                    if j == 0:
                        label_target[i, 7] = 1
                        label_target[i, 8] = 0
                    elif j == 1:
                        label_target[i, 7] = 0
                        label_target[i, 8] = 1
                    elif j == 2:
                        label_target[i, 7] = 0
                        label_target[i, 8] = 0
                elif index == 4:
                    if j == 0:
                        label_target[i, 9] = 1
                        label_target[i, 10] = 0
                    elif j == 1:
                        label_target[i, 9] = 0
                        label_target[i, 10] = 1
                    elif j == 2:
                        label_target[i, 9] = 0
                        label_target[i, 10] = 0

            for index, j in enumerate(epoch_predict_labels[:, i]):
                if index == 0:
                    if j == 0:
                        label_pred[i, 0] = 1
                        label_pred[i, 1] = 0
                    elif j == 1:
                        label_pred[i, 0] = 0
                        label_pred[i, 1] = 1
                elif index == 1:
                    if j == 0:
                        label_pred[i, 2] = 1
                        label_pred[i, 3] = 0
                        label_pred[i, 4] = 0
                    elif j == 1:
                        label_pred[i, 2] = 0
                        label_pred[i, 3] = 1
                        label_pred[i, 4] = 0
                    elif j == 2:
                        label_pred[i, 2] = 0
                        label_pred[i, 3] = 0
                        label_pred[i, 4] = 1
                    elif j == 3:
                        label_pred[i, 2] = 0
                        label_pred[i, 3] = 0
                        label_pred[i, 4] = 0
                elif index == 2:
                    if j == 0:
                        label_pred[i, 5] = 1
                        label_pred[i, 6] = 0
                    elif j == 1:
                        label_pred[i, 5] = 0
                        label_pred[i, 6] = 1
                    elif j == 2:
                        label_pred[i, 5] = 0
                        label_pred[i, 6] = 0
                elif index == 3:
                    if j == 0:
                        label_pred[i, 7] = 1
                        label_pred[i, 8] = 0
                    elif j == 1:
                        label_pred[i, 7] = 0
                        label_pred[i, 8] = 1
                    elif j == 2:
                        label_pred[i, 7] = 0
                        label_pred[i, 8] = 0
                elif index == 4:
                    if j == 0:
                        label_pred[i, 9] = 1
                        label_pred[i, 10] = 0
                    elif j == 1:
                        label_pred[i, 9] = 0
                        label_pred[i, 10] = 1
                    elif j == 2:
                        label_pred[i, 9] = 0
                        label_pred[i, 10] = 0

        # calculate metrics
        # Exact Match Ratio
        metrics["exact_match_ratio"] = accuracy_score(label_target, label_pred)

        # 0-1 Loss
        metrics["01loss"] = zero_one_loss(label_target, label_pred)

        # Hamming Loss
        metrics["hamming_loss"] = hamming_loss(label_target, label_pred)

        # precision
        metrics["precision"] = precision_score(label_target, label_pred, average=None)
        metrics["macro-precision"] = precision_score(label_target, label_pred, average='macro')
        metrics["micro-precision"] = precision_score(label_target, label_pred, average='micro')

        # recall
        metrics["recall"] = recall_score(label_target, label_pred, average=None)
        metrics["macro-recall"] = recall_score(label_target, label_pred, average='macro')
        metrics["micro-recall"] = recall_score(label_target, label_pred, average='micro')

        # f1
        metrics["f1"] = f1_score(label_target, label_pred, average=None)
        metrics["macro-f1"] = f1_score(label_target, label_pred, average='macro')
        metrics["micro-f1"] = f1_score(label_target, label_pred, average='micro')

    elif config.model.type == "PreAttnMMs_MTL_LCL":
        # transform into np.array([])
        epoch_predict_labels = np.asarray(epoch_predict_labels)
        epoch_target_labels = np.asarray(epoch_target_labels)
        sample_size = epoch_predict_labels.shape[1]

        label_pred = np.zeros((sample_size, 11))
        label_target = np.zeros((sample_size, 11))

        for i in range(sample_size):
            for index, j in enumerate(epoch_target_labels[:, i]):
                if index == 0:
                    if j == 0:
                        label_target[i, 0] = 1
                        label_target[i, 1] = 0
                    elif j == 1:
                        label_target[i, 0] = 0
                        label_target[i, 1] = 1
                elif index == 1:
                    if j == 0:
                        label_target[i, 2] = 1
                        label_target[i, 3] = 0
                        label_target[i, 4] = 0
                        label_target[i, 5] = 0
                        label_target[i, 6] = 0
                    elif j == 1:
                        label_target[i, 2] = 0
                        label_target[i, 3] = 1
                        label_target[i, 4] = 0
                        label_target[i, 5] = 0
                        label_target[i, 6] = 0
                    elif j == 2:
                        label_target[i, 2] = 0
                        label_target[i, 3] = 0
                        label_target[i, 4] = 1
                        label_target[i, 5] = 0
                        label_target[i, 6] = 0
                    elif j == 3:
                        label_target[i, 2] = 0
                        label_target[i, 3] = 0
                        label_target[i, 4] = 0
                        label_target[i, 5] = 1
                        label_target[i, 6] = 0
                    elif j == 4:
                        label_target[i, 2] = 0
                        label_target[i, 3] = 0
                        label_target[i, 4] = 0
                        label_target[i, 5] = 0
                        label_target[i, 6] = 1
                elif index == 2:
                    if j == 0:
                        label_target[i, 7] = 0
                        label_target[i, 8] = 0
                        label_target[i, 9] = 0
                        label_target[i, 10] = 0
                    elif j == 1:
                        label_target[i, 7] = 0
                        label_target[i, 8] = 0
                        label_target[i, 9] = 0
                        label_target[i, 10] = 0
                    elif j == 2:
                        label_target[i, 7] = 0
                        label_target[i, 8] = 0
                        label_target[i, 9] = 0
                        label_target[i, 10] = 0
                    elif j == 3:
                        label_target[i, 7] = 1
                        label_target[i, 8] = 0
                        label_target[i, 9] = 0
                        label_target[i, 10] = 0
                    elif j == 4:
                        label_target[i, 7] = 0
                        label_target[i, 8] = 1
                        label_target[i, 9] = 0
                        label_target[i, 10] = 0
                    elif j == 5:
                        label_target[i, 7] = 0
                        label_target[i, 8] = 0
                        label_target[i, 9] = 1
                        label_target[i, 10] = 0
                    elif j == 6:
                        label_target[i, 7] = 0
                        label_target[i, 8] = 0
                        label_target[i, 9] = 0
                        label_target[i, 10] = 1

            for index, j in enumerate(epoch_predict_labels[:, i]):
                if index == 0:
                    if j == 0:
                        label_pred[i, 0] = 1
                        label_pred[i, 1] = 0
                    elif j == 1:
                        label_pred[i, 0] = 0
                        label_pred[i, 1] = 1
                elif index == 1:
                    if j == 0:
                        label_pred[i, 2] = 1
                        label_pred[i, 3] = 0
                        label_pred[i, 4] = 0
                        label_pred[i, 5] = 0
                        label_pred[i, 6] = 0
                    elif j == 1:
                        label_pred[i, 2] = 0
                        label_pred[i, 3] = 1
                        label_pred[i, 4] = 0
                        label_pred[i, 5] = 0
                        label_pred[i, 6] = 0
                    elif j == 2:
                        label_pred[i, 2] = 0
                        label_pred[i, 3] = 0
                        label_pred[i, 4] = 1
                        label_pred[i, 5] = 0
                        label_pred[i, 6] = 0
                    elif j == 3:
                        label_pred[i, 2] = 0
                        label_pred[i, 3] = 0
                        label_pred[i, 4] = 0
                        label_pred[i, 5] = 1
                        label_pred[i, 6] = 0
                    elif j == 4:
                        label_pred[i, 2] = 0
                        label_pred[i, 3] = 0
                        label_pred[i, 4] = 0
                        label_pred[i, 5] = 0
                        label_pred[i, 6] = 1
                elif index == 2:
                    if j == 0:
                        label_pred[i, 7] = 0
                        label_pred[i, 8] = 0
                        label_pred[i, 9] = 0
                        label_pred[i, 10] = 0
                    elif j == 1:
                        label_pred[i, 7] = 0
                        label_pred[i, 8] = 0
                        label_pred[i, 9] = 0
                        label_pred[i, 10] = 0
                    elif j == 2:
                        label_pred[i, 7] = 0
                        label_pred[i, 8] = 0
                        label_pred[i, 9] = 0
                        label_pred[i, 10] = 0
                    elif j == 3:
                        label_pred[i, 7] = 1
                        label_pred[i, 8] = 0
                        label_pred[i, 9] = 0
                        label_pred[i, 10] = 0
                    elif j == 4:
                        label_pred[i, 7] = 0
                        label_pred[i, 8] = 1
                        label_pred[i, 9] = 0
                        label_pred[i, 10] = 0
                    elif j == 5:
                        label_pred[i, 7] = 0
                        label_pred[i, 8] = 0
                        label_pred[i, 9] = 1
                        label_pred[i, 10] = 0
                    elif j == 6:
                        label_pred[i, 7] = 0
                        label_pred[i, 8] = 0
                        label_pred[i, 9] = 0
                        label_pred[i, 10] = 1
    
        # calculate metrics
        # Exact Match Ratio
        metrics["exact_match_ratio"] = accuracy_score(label_target, label_pred)

        # 0-1 Loss
        metrics["01loss"] = zero_one_loss(label_target, label_pred)

        # Hamming Loss
        metrics["hamming_loss"] = hamming_loss(label_target, label_pred)

        # precision
        metrics["precision"] = precision_score(label_target, label_pred, average=None)
        metrics["macro-precision"] = precision_score(label_target, label_pred, average='macro')
        metrics["micro-precision"] = precision_score(label_target, label_pred, average='micro')

        # recall
        metrics["recall"] = recall_score(label_target, label_pred, average=None)
        metrics["macro-recall"] = recall_score(label_target, label_pred, average='macro')
        metrics["micro-recall"] = recall_score(label_target, label_pred, average='micro')

        # f1
        metrics["f1"] = f1_score(label_target, label_pred, average=None)
        metrics["macro-f1"] = f1_score(label_target, label_pred, average='macro')
        metrics["micro-f1"] = f1_score(label_target, label_pred, average='micro')
    
    elif  config.model.type == "PreAttnMMs_FCLN":
        label_target = np.zeros((len(epoch_target_labels), 11))
        label_pred = np.zeros((len(epoch_predict_labels), 11))
        
        # reconstruct the target labels into binary form
        for i in range(label_target.shape[0]):
            if epoch_target_labels[i] == 0:
                label_target[i, 0] = 1
                label_target[i, 2] = 1
            elif epoch_target_labels[i] == 1:
                label_target[i, 0] = 1
                label_target[i, 3] = 1
            elif epoch_target_labels[i] == 2:
                label_target[i, 0] = 1
                label_target[i, 4] = 1
            elif epoch_target_labels[i] == 3:
                label_target[i, 1] = 1
                label_target[i, 5] = 1
                label_target[i, 7] = 1
            elif epoch_target_labels[i] == 4:
                label_target[i, 1] = 1
                label_target[i, 5] = 1
                label_target[i, 8] = 1
            elif epoch_target_labels[i] == 5:
                label_target[i, 1] = 1
                label_target[i, 6] = 1
                label_target[i, 9] = 1
            elif epoch_target_labels[i] == 6:
                label_target[i, 1] = 1
                label_target[i, 6] = 1
                label_target[i, 10] = 1
            else:
                logger.error("There is an error in TEST phase for PreAttnMMs_FCLN!!")

        # reconstruct the predict labels into binary form
        for i in range(label_pred.shape[0]):
            if epoch_predict_labels[i] == 0:
                label_pred[i, 0] = 1
                label_pred[i, 2] = 1
            elif epoch_predict_labels[i] == 1:
                label_pred[i, 0] = 1
                label_pred[i, 3] = 1
            elif epoch_predict_labels[i] == 2:
                label_pred[i, 0] = 1
                label_pred[i, 4] = 1
            elif epoch_predict_labels[i] == 3:
                label_pred[i, 1] = 1
                label_pred[i, 5] = 1
                label_pred[i, 7] = 1
            elif epoch_predict_labels[i] == 4:
                label_pred[i, 1] = 1
                label_pred[i, 5] = 1
                label_pred[i, 8] = 1
            elif epoch_predict_labels[i] == 5:
                label_pred[i, 1] = 1
                label_pred[i, 6] = 1
                label_pred[i, 9] = 1
            elif epoch_predict_labels[i] == 6:
                label_pred[i, 1] = 1
                label_pred[i, 6] = 1
                label_pred[i, 10] = 1
            else:
                logger.error("There is an error in TEST phase for PreAttnMMs_FCLN!!")

        # Exact Match Ratio
        metrics["exact_match_ratio"] = accuracy_score(label_target, label_pred)

        # 0-1 Loss
        metrics["01loss"] = zero_one_loss(label_target, label_pred)

        # Hamming Loss
        metrics["hamming_loss"] = hamming_loss(label_target, label_pred)

        # precision
        metrics["precision"] = precision_score(label_target, label_pred, average=None)
        metrics["macro-precision"] = precision_score(label_target, label_pred, average='macro')
        metrics["micro-precision"] = precision_score(label_target, label_pred, average='micro')

        # recall
        metrics["recall"] = recall_score(label_target, label_pred, average=None)
        metrics["macro-recall"] = recall_score(label_target, label_pred, average='macro')
        metrics["micro-recall"] = recall_score(label_target, label_pred, average='micro')

        # f1
        metrics["f1"] = f1_score(label_target, label_pred, average=None)
        metrics["macro-f1"] = f1_score(label_target, label_pred, average='macro')
        metrics["micro-f1"] = f1_score(label_target, label_pred, average='micro')

    # evaluation for binary classification
    elif config.experiment.evaluation == "binary classification":
        # accuracy
        metrics["acc"] = accuracy_score(epoch_target_labels, epoch_predict_labels)
        
        # precision
        metrics["precision"] = precision_score(epoch_target_labels, epoch_predict_labels, average=None)
        metrics["macro-precision"] = precision_score(epoch_target_labels, epoch_predict_labels, average='macro')
        metrics["micro-precision"] = precision_score(epoch_target_labels, epoch_predict_labels, average='micro')

        # recall
        metrics["recall"] = recall_score(epoch_target_labels, epoch_predict_labels, average=None)
        metrics["macro-recall"] = recall_score(epoch_target_labels, epoch_predict_labels, average='macro')
        metrics["micro-recall"] = recall_score(epoch_target_labels, epoch_predict_labels, average='micro')

        # f1
        metrics["f1"] = f1_score(epoch_target_labels, epoch_predict_labels, average=None)
        metrics["macro-f1"] = f1_score(epoch_target_labels, epoch_predict_labels, average='macro')
        metrics["micro-f1"] = f1_score(epoch_target_labels, epoch_predict_labels, average='micro')

        # AUC
        metrics["auc"] = roc_auc_score(np.asarray(epoch_target_labels), np.asarray(epoch_predict_probs)[:,1])

        # AP
        metrics["ap"] = average_precision_score(np.asarray(epoch_target_labels), np.asarray(epoch_predict_probs)[:,1])

        # confusion matrix
        metrics["cm"] = confusion_matrix(np.asarray(epoch_target_labels), np.asarray(epoch_predict_labels))

        # classification_report
        target_names = ['class {}'.format(i) for i in range(n_classes)]
        metrics["cr"] = classification_report(np.asarray(epoch_target_labels), np.asarray(epoch_predict_labels), target_names=target_names, output_dict=True)

    elif config.experiment.evaluation == "multiclass classification":
        # tansform target labels into onehot form
        epoch_target_labels_onehot = label_binarize(epoch_target_labels, classes=[i for i in range(n_classes)])
        
        # accuracy
        metrics["acc"] = accuracy_score(epoch_target_labels, epoch_predict_labels)

        # precision
        metrics["precision"] = precision_score(epoch_target_labels, epoch_predict_labels, average=None)
        metrics["macro-precision"] = precision_score(epoch_target_labels, epoch_predict_labels, average='macro')
        metrics["micro-precision"] = precision_score(epoch_target_labels, epoch_predict_labels, average='micro')

        # recall
        metrics["recall"] = recall_score(epoch_target_labels, epoch_predict_labels, average=None)
        metrics["macro-recall"] = recall_score(epoch_target_labels, epoch_predict_labels, average='macro')
        metrics["micro-recall"] = recall_score(epoch_target_labels, epoch_predict_labels, average='micro')

        # f1
        metrics["f1"] = f1_score(epoch_target_labels, epoch_predict_labels, average=None)
        metrics["macro-f1"] = f1_score(epoch_target_labels, epoch_predict_labels, average='macro')
        metrics["micro-f1"] = f1_score(epoch_target_labels, epoch_predict_labels, average='micro')

        # AUC
        metrics["macro-auc"] = roc_auc_score(np.asarray(epoch_target_labels), np.asarray(epoch_predict_probs), average='macro', multi_class='ovr')

        # AP
        metrics["ap"] = average_precision_score(epoch_target_labels_onehot, np.asarray(epoch_predict_probs), average=None)
        metrics["macro-ap"] = average_precision_score(epoch_target_labels_onehot, np.asarray(epoch_predict_probs), average='macro')
        metrics["micro-ap"] = average_precision_score(epoch_target_labels_onehot, np.asarray(epoch_predict_probs), average='micro')

        # confusion matrix
        metrics["cm"] = confusion_matrix(np.asarray(epoch_target_labels), np.asarray(epoch_predict_labels))

        # classification_report
        target_names = ['class {}'.format(i) for i in range(n_classes)]
        metrics["cr"] = classification_report(np.asarray(epoch_target_labels), np.asarray(epoch_predict_labels), target_names=target_names, output_dict=True)

    elif config.experiment.evaluation == "multilabel classification":
        # Exact Match Ratio
        metrics["exact_match_ratio"] = accuracy_score(epoch_target_labels, epoch_predict_labels)

        # 0-1 Loss
        metrics["01loss"] = zero_one_loss(epoch_target_labels, epoch_predict_labels)

        # Hamming Loss
        metrics["hamming_loss"] = hamming_loss(epoch_target_labels, epoch_predict_labels)

        # precision
        metrics["precision"] = precision_score(epoch_target_labels, epoch_predict_labels, average=None)
        metrics["macro-precision"] = precision_score(epoch_target_labels, epoch_predict_labels, average='macro')
        metrics["micro-precision"] = precision_score(epoch_target_labels, epoch_predict_labels, average='micro')

        # recall
        metrics["recall"] = recall_score(epoch_target_labels, epoch_predict_labels, average=None)
        metrics["macro-recall"] = recall_score(epoch_target_labels, epoch_predict_labels, average='macro')
        metrics["micro-recall"] = recall_score(epoch_target_labels, epoch_predict_labels, average='micro')

        # f1
        metrics["f1"] = f1_score(epoch_target_labels, epoch_predict_labels, average=None)
        metrics["macro-f1"] = f1_score(epoch_target_labels, epoch_predict_labels, average='macro')
        metrics["micro-f1"] = f1_score(epoch_target_labels, epoch_predict_labels, average='micro')

        # AUC
        metrics["macro-auc"] = roc_auc_score(epoch_target_labels, epoch_predict_probs, average='macro', multi_class='ovr')

        # AP
        metrics["ap"] = average_precision_score(epoch_target_labels, epoch_predict_probs, average=None)
        metrics["macro-ap"] = average_precision_score(epoch_target_labels, epoch_predict_probs, average='macro')
        metrics["micro-ap"] = average_precision_score(epoch_target_labels, epoch_predict_probs, average='micro')

    return metrics

def evaluate4test(config, target_labels_array, predcit_labels_array):
    """
    Function: evaluation metric for TEST, there is some subtle difference with upper `def evaluate()`
    Params: target_labels_array, np.array([[1,0,1,0,0,0,...],[],[],...]), shape=(sample size, 11)
            predcit_labels_array, np.array([[1,0,1,0,0,0,...],[],[],...]), shape=(sample size, 11)
    Return: metrics, Dict{}
    """
    metrics = {}

    # index of sample with 4 3-th level true label
    index_layer3 = [index for index, value in enumerate(list(map(lambda x: (x==1).any(), target_labels_array[:,7:]))) if value == True]
    
    # TopK Exact Match Ratio
    metrics["exact_match_ratio_top1"] = accuracy_score(target_labels_array[:, :2], predcit_labels_array[:, :2])
    metrics["exact_match_ratio_top2"] = accuracy_score(target_labels_array[:, :7], predcit_labels_array[:, :7])
    metrics["exact_match_ratio_overall"] = accuracy_score(target_labels_array, predcit_labels_array)
    
    # Level-wise Exact Match Ratio
    metrics["exact_match_ratio_layer1"] = accuracy_score(target_labels_array[:, :2], predcit_labels_array[:, :2])
    metrics["exact_match_ratio_layer2"] = accuracy_score(target_labels_array[:, 2:7], predcit_labels_array[:, 2:7])
    metrics["exact_match_ratio_layer3"] = accuracy_score(target_labels_array[:, 7:], predcit_labels_array[:, 7:])
    metrics["exact_match_ratio_layer3.1"] = accuracy_score(target_labels_array[index_layer3, 7:], predcit_labels_array[index_layer3, 7:])
    metrics["exact_match_ratio_layer_avg"] = (metrics["exact_match_ratio_layer1"] + metrics["exact_match_ratio_layer2"] + metrics["exact_match_ratio_layer3"]) / 3
    metrics["exact_match_ratio_layer_avg1"] = (metrics["exact_match_ratio_layer1"] + metrics["exact_match_ratio_layer2"] + metrics["exact_match_ratio_layer3.1"]) / 3
    
    # Accuracy for each class
    metrics["accuracy"] = np.zeros((11))
    metrics["accuracy"][0] = accuracy_score(target_labels_array[:, 0], predcit_labels_array[:, 0])
    metrics["accuracy"][1] = accuracy_score(target_labels_array[:, 1], predcit_labels_array[:, 1])
    metrics["accuracy"][2] = accuracy_score(target_labels_array[:, 2], predcit_labels_array[:, 2])
    metrics["accuracy"][3] = accuracy_score(target_labels_array[:, 3], predcit_labels_array[:, 3])
    metrics["accuracy"][4] = accuracy_score(target_labels_array[:, 4], predcit_labels_array[:, 4])
    metrics["accuracy"][5] = accuracy_score(target_labels_array[:, 5], predcit_labels_array[:, 5])
    metrics["accuracy"][6] = accuracy_score(target_labels_array[:, 6], predcit_labels_array[:, 6])
    metrics["accuracy"][7] = accuracy_score(target_labels_array[:, 7], predcit_labels_array[:, 7])
    metrics["accuracy"][8] = accuracy_score(target_labels_array[:, 8], predcit_labels_array[:, 8])
    metrics["accuracy"][9] = accuracy_score(target_labels_array[:, 9], predcit_labels_array[:, 9])
    metrics["accuracy"][10] = accuracy_score(target_labels_array[:, 10], predcit_labels_array[:, 10])

    # 0-1 Loss (i.e. 1- Exact Match Ratio)
    metrics["01loss_top1"] = zero_one_loss(target_labels_array[:, :2], predcit_labels_array[:, :2])
    metrics["01loss_top2"] = zero_one_loss(target_labels_array[:, :7], predcit_labels_array[:, :7])
    metrics["01loss_overall"] = zero_one_loss(target_labels_array, predcit_labels_array)
    metrics["01loss_layer1"] = zero_one_loss(target_labels_array[:, :2], predcit_labels_array[:, :2])
    metrics["01loss_layer2"] = zero_one_loss(target_labels_array[:, 2:7], predcit_labels_array[:, 2:7])
    metrics["01loss_layer3"] = zero_one_loss(target_labels_array[:, 7:], predcit_labels_array[:, 7:])
    metrics["01loss_layer3.1"] = zero_one_loss(target_labels_array[index_layer3, 7:], predcit_labels_array[index_layer3, 7:])
    metrics["01loss_layer_avg"] = (metrics["01loss_layer1"] + metrics["01loss_layer2"] + metrics["01loss_layer3"]) / 3
    metrics["01loss_layer_avg1"] = (metrics["01loss_layer1"] + metrics["01loss_layer2"] + metrics["01loss_layer3.1"]) / 3
    
    # Hamming Loss
    metrics["hamming_loss"] = hamming_loss(target_labels_array, predcit_labels_array)

    # Hamming score
    metrics["hamming_score"] = hamming_score(target_labels_array, predcit_labels_array)

    # precision
    metrics["precision"] = precision_score(target_labels_array, predcit_labels_array, average=None)
    metrics["macro-precision"] = precision_score(target_labels_array, predcit_labels_array, average='macro')
    metrics["micro-precision"] = precision_score(target_labels_array, predcit_labels_array, average='micro')
    # layer 1
    metrics["precision_layer1"] = precision_score(target_labels_array[:, :2], predcit_labels_array[:, :2], average=None)
    metrics["macro-precision_layer1"] = precision_score(target_labels_array[:, :2], predcit_labels_array[:, :2], average='macro')
    metrics["micro-precision_layer1"] = precision_score(target_labels_array[:, :2], predcit_labels_array[:, :2], average='micro')
    # layer 2
    metrics["precision_layer2"] = precision_score(target_labels_array[:, 2:7], predcit_labels_array[:, 2:7], average=None)
    metrics["macro-precision_layer2"] = precision_score(target_labels_array[:, 2:7], predcit_labels_array[:, 2:7], average='macro')
    metrics["micro-precision_layer2"] = precision_score(target_labels_array[:, 2:7], predcit_labels_array[:, 2:7], average='micro')
    # layer 3
    metrics["precision_layer3"] = precision_score(target_labels_array[:, 7:], predcit_labels_array[:, 7:], average=None)
    metrics["macro-precision_layer3"] = precision_score(target_labels_array[:, 7:], predcit_labels_array[:, 7:], average='macro')
    metrics["micro-precision_layer3"] = precision_score(target_labels_array[:, 7:], predcit_labels_array[:, 7:], average='micro')
    # layer 3.1
    metrics["precision_layer3.1"] = precision_score(target_labels_array[index_layer3, 7:], predcit_labels_array[index_layer3, 7:], average=None)
    metrics["macro-precision_layer3.1"] = precision_score(target_labels_array[index_layer3, 7:], predcit_labels_array[index_layer3, 7:], average='macro')
    metrics["micro-precision_layer3.1"] = precision_score(target_labels_array[index_layer3, 7:], predcit_labels_array[index_layer3, 7:], average='micro')
    # avg
    metrics["macro-precision_avg"] = (metrics["macro-precision_layer1"] + metrics["macro-precision_layer2"] + metrics["macro-precision_layer3"]) / 3
    metrics["macro-precision_avg1"] = (metrics["macro-precision_layer1"] + metrics["macro-precision_layer2"] + metrics["macro-precision_layer3.1"]) / 3
    metrics["micro-precision_avg"] = (metrics["micro-precision_layer1"] + metrics["micro-precision_layer2"] + metrics["micro-precision_layer3"]) / 3
    metrics["micro-precision_avg1"] = (metrics["micro-precision_layer1"] + metrics["micro-precision_layer2"] + metrics["micro-precision_layer3.1"]) / 3
    
    # recall
    metrics["recall"] = recall_score(target_labels_array, predcit_labels_array, average=None)
    metrics["macro-recall"] = recall_score(target_labels_array, predcit_labels_array, average='macro')
    metrics["micro-recall"] = recall_score(target_labels_array, predcit_labels_array, average='micro')
    # layer 1
    metrics["recall_layer1"] = recall_score(target_labels_array[:, :2], predcit_labels_array[:, :2], average=None)
    metrics["macro-recall_layer1"] = recall_score(target_labels_array[:, :2], predcit_labels_array[:, :2], average='macro')
    metrics["micro-recall_layer1"] = recall_score(target_labels_array[:, :2], predcit_labels_array[:, :2], average='micro')
    # layer 2
    metrics["recall_layer2"] = recall_score(target_labels_array[:, 2:7], predcit_labels_array[:, 2:7], average=None)
    metrics["macro-recall_layer2"] = recall_score(target_labels_array[:, 2:7], predcit_labels_array[:, 2:7], average='macro')
    metrics["micro-recall_layer2"] = recall_score(target_labels_array[:, 2:7], predcit_labels_array[:, 2:7], average='micro')
    # layer 3
    metrics["recall_layer3"] = recall_score(target_labels_array[:, 7:], predcit_labels_array[:, 7:], average=None)
    metrics["macro-recall_layer3"] = recall_score(target_labels_array[:, 7:], predcit_labels_array[:, 7:], average='macro')
    metrics["micro-recall_layer3"] = recall_score(target_labels_array[:, 7:], predcit_labels_array[:, 7:], average='micro')
    # layer 3.1
    metrics["recall_layer3.1"] = recall_score(target_labels_array[index_layer3, 7:], predcit_labels_array[index_layer3, 7:], average=None)
    metrics["macro-recall_layer3.1"] = recall_score(target_labels_array[index_layer3, 7:], predcit_labels_array[index_layer3, 7:], average='macro')
    metrics["micro-recall_layer3.1"] = recall_score(target_labels_array[index_layer3, 7:], predcit_labels_array[index_layer3, 7:], average='micro')
    # avg
    metrics["macro-recall_avg"] = (metrics["macro-recall_layer1"] + metrics["macro-recall_layer2"] + metrics["macro-recall_layer3"]) / 3
    metrics["macro-recall_avg1"] = (metrics["macro-recall_layer1"] + metrics["macro-recall_layer2"] + metrics["macro-recall_layer3.1"]) / 3
    metrics["micro-recall_avg"] = (metrics["micro-recall_layer1"] + metrics["micro-recall_layer2"] + metrics["micro-recall_layer3"]) / 3
    metrics["micro-recall_avg1"] = (metrics["micro-recall_layer1"] + metrics["micro-recall_layer2"] + metrics["micro-recall_layer3.1"]) / 3

    # f1
    metrics["f1"] = f1_score(target_labels_array, predcit_labels_array, average=None)
    metrics["macro-f1"] = f1_score(target_labels_array, predcit_labels_array, average='macro')
    metrics["micro-f1"] = f1_score(target_labels_array, predcit_labels_array, average='micro')
    # layer 1
    metrics["f1_layer1"] = f1_score(target_labels_array[:, :2], predcit_labels_array[:, :2], average=None)
    metrics["macro-f1_layer1"] = f1_score(target_labels_array[:, :2], predcit_labels_array[:, :2], average='macro')
    metrics["micro-f1_layer1"] = f1_score(target_labels_array[:, :2], predcit_labels_array[:, :2], average='micro')
    # layer 2
    metrics["f1_layer2"] = f1_score(target_labels_array[:, 2:7], predcit_labels_array[:, 2:7], average=None)
    metrics["macro-f1_layer2"] = f1_score(target_labels_array[:, 2:7], predcit_labels_array[:, 2:7], average='macro')
    metrics["micro-f1_layer2"] = f1_score(target_labels_array[:, 2:7], predcit_labels_array[:, 2:7], average='micro')
    # layer 3
    metrics["f1_layer3"] = f1_score(target_labels_array[:, 7:], predcit_labels_array[:, 7:], average=None)
    metrics["macro-f1_layer3"] = f1_score(target_labels_array[:, 7:], predcit_labels_array[:, 7:], average='macro')
    metrics["micro-f1_layer3"] = f1_score(target_labels_array[:, 7:], predcit_labels_array[:, 7:], average='micro')
    # layer 3.1
    metrics["f1_layer3.1"] = f1_score(target_labels_array[index_layer3, 7:], predcit_labels_array[index_layer3, 7:], average=None)
    metrics["macro-f1_layer3.1"] = f1_score(target_labels_array[index_layer3, 7:], predcit_labels_array[index_layer3, 7:], average='macro')
    metrics["micro-f1_layer3.1"] = f1_score(target_labels_array[index_layer3, 7:], predcit_labels_array[index_layer3, 7:], average='micro')
    # avg
    metrics["macro-f1_avg"] = (metrics["macro-f1_layer1"] + metrics["macro-f1_layer2"] + metrics["macro-f1_layer3"]) / 3
    metrics["macro-f1_avg1"] = (metrics["macro-f1_layer1"] + metrics["macro-f1_layer2"] + metrics["macro-f1_layer3.1"]) / 3
    metrics["micro-f1_avg"] = (metrics["micro-f1_layer1"] + metrics["micro-f1_layer2"] + metrics["micro-f1_layer3"]) / 3
    metrics["micro-f1_avg1"] = (metrics["micro-f1_layer1"] + metrics["micro-f1_layer2"] + metrics["micro-f1_layer3.1"]) / 3

    # hierarchy dependency sensitivity
    metrics["hd_sensitivity_with_partial"], metrics["hd_sensitivity_wo_partial"] = hierarchy_dependency_sensitivity(config, predcit_labels_array)

    # error inspection
    metrics["error"] = error_inspection(config, target_labels_array, predcit_labels_array)
    metrics["LCA"] = lca_height_related(config, target_labels_array, predcit_labels_array)

    # detailed error inspection terms
    metrics["error_details"] = ["{0}({1:.4f}%)".format(metrics["error"]["all_correct_sample"]["number"], 100 * metrics["error"]["all_correct_sample"]["ratio"]),
                                "{0}({1:.4f}%)".format(metrics["error"]["notall_correct_sample"]["number"], 100 * metrics["error"]["notall_correct_sample"]["ratio"]),
                                "{0}({1:.4f}%)".format(metrics["error"]["notall_correct_sample"]["partial_predict_sample"]["number"], 100 * metrics["error"]["notall_correct_sample"]["partial_predict_sample"]["ratio"]),
                                "{0}({1:.4f}%)".format(metrics["error"]["notall_correct_sample"]["partial_predict_sample"]["partial_correct_sample"]["number"], 100 * metrics["error"]["notall_correct_sample"]["partial_predict_sample"]["partial_correct_sample"]["ratio"]),
                                "{0}({1:.4f}%)".format(metrics["error"]["notall_correct_sample"]["partial_predict_sample"]["partial_notcorrect_sample"]["number"], 100 * metrics["error"]["notall_correct_sample"]["partial_predict_sample"]["partial_notcorrect_sample"]["ratio"]),
                                "{0}({1:.4f}%)".format(metrics["error"]["notall_correct_sample"]["complete_predict_sample"]["number"], 100 * metrics["error"]["notall_correct_sample"]["complete_predict_sample"]["ratio"]),
                                "{0}({1:.4f}%)".format(metrics["error"]["notall_correct_sample"]["complete_predict_sample"]["first_level_wrong_sample"]["number"], 100 * metrics["error"]["notall_correct_sample"]["complete_predict_sample"]["first_level_wrong_sample"]["ratio"]),
                                "{0}({1:.4f}%)".format(metrics["error"]["notall_correct_sample"]["complete_predict_sample"]["first_level_right_sample"]["number"], 100 * metrics["error"]["notall_correct_sample"]["complete_predict_sample"]["first_level_right_sample"]["ratio"]),
                                "{0}({1:.4f}%)".format(metrics["error"]["notall_correct_sample"]["complete_predict_sample"]["first_level_right_sample"]["wrong_under_infection_sample"]["number"], 100 * metrics["error"]["notall_correct_sample"]["complete_predict_sample"]["first_level_right_sample"]["wrong_under_infection_sample"]["ratio"]),
                                "{0}({1:.4f}%)".format(metrics["error"]["notall_correct_sample"]["complete_predict_sample"]["first_level_right_sample"]["wrong_under_noninfection_sample"]["number"], 100 * metrics["error"]["notall_correct_sample"]["complete_predict_sample"]["first_level_right_sample"]["wrong_under_noninfection_sample"]["ratio"]),
                                "{0}({1:.4f}%)".format(metrics["error"]["notall_correct_sample"]["complete_predict_sample"]["first_level_right_sample"]["wrong_under_noninfection_sample"]["wrong_at_NIID_and_Neo_sample"]["number"], 100 * metrics["error"]["notall_correct_sample"]["complete_predict_sample"]["first_level_right_sample"]["wrong_under_noninfection_sample"]["wrong_at_NIID_and_Neo_sample"]["ratio"]),
                                "{0}({1:.4f}%)".format(metrics["error"]["notall_correct_sample"]["complete_predict_sample"]["first_level_right_sample"]["wrong_under_noninfection_sample"]["wrong_under_NIID_and_Neo_sample"]["number"], 100 * metrics["error"]["notall_correct_sample"]["complete_predict_sample"]["first_level_right_sample"]["wrong_under_noninfection_sample"]["wrong_under_NIID_and_Neo_sample"]["ratio"]),
                                "{0}({1:.4f}%)".format(metrics["error"]["notall_correct_sample"]["complete_predict_sample"]["first_level_right_sample"]["wrong_under_noninfection_sample"]["wrong_under_NIID_and_Neo_sample"]["wrong_under_NIID"]["number"], 100 * metrics["error"]["notall_correct_sample"]["complete_predict_sample"]["first_level_right_sample"]["wrong_under_noninfection_sample"]["wrong_under_NIID_and_Neo_sample"]["wrong_under_NIID"]["ratio"]),
                                "{0}({1:.4f}%)".format(metrics["error"]["notall_correct_sample"]["complete_predict_sample"]["first_level_right_sample"]["wrong_under_noninfection_sample"]["wrong_under_NIID_and_Neo_sample"]["wrong_under_neo"]["number"], 100 * metrics["error"]["notall_correct_sample"]["complete_predict_sample"]["first_level_right_sample"]["wrong_under_noninfection_sample"]["wrong_under_NIID_and_Neo_sample"]["wrong_under_neo"]["ratio"])
                                ]

    # detailed LCA related information
    metrics["lca_details"] = ["{0}".format(metrics["LCA"]["test_sample_size"]["number"]),
                              "{0}({1:.4f}%)".format(metrics["LCA"]["exact_match_sample"]["number"], 100 * metrics["LCA"]["exact_match_sample"]["ratio"]),
                              "{0}({1:.4f}%)".format(metrics["LCA"]["not_exact_match_sample"]["number"], 100 * metrics["LCA"]["not_exact_match_sample"]["ratio"]),
                              "{0}({1:.4f}%)".format(metrics["LCA"]["not_exact_match_sample"]["partial_path_sample"]["number"], 100 * metrics["LCA"]["not_exact_match_sample"]["partial_path_sample"]["ratio"]),
                              "{0}({1:.4f}%)".format(metrics["LCA"]["not_exact_match_sample"]["partial_path_sample"]["partial_correct_sample"]["number"], 100 * metrics["LCA"]["not_exact_match_sample"]["partial_path_sample"]["partial_correct_sample"]["ratio"]),
                              "{0}({1:.4f}%)".format(metrics["LCA"]["not_exact_match_sample"]["complete_path_sample"]["number"], 100 * metrics["LCA"]["not_exact_match_sample"]["complete_path_sample"]["ratio"]),
                              "{0}({1:.4f}%)".format(metrics["LCA"]["not_exact_match_sample"]["complete_path_sample"]["notcorrect_predict_path_sample"]["number"], 100 * metrics["LCA"]["not_exact_match_sample"]["complete_path_sample"]["notcorrect_predict_path_sample"]["ratio"]),
                              "{0}({1:.4f}%)".format(metrics["LCA"]["not_exact_match_sample"]["complete_path_sample"]["correct_predict_path_sample"]["number"], 100 * metrics["LCA"]["not_exact_match_sample"]["complete_path_sample"]["correct_predict_path_sample"]["ratio"]),
                              "{0}".format(metrics["LCA"]["not_exact_match_sample"]["complete_path_sample"]["correct_predict_path_sample"]["lca_height"]),
                              "{0}({1:.4f}%)".format(metrics["LCA"]["not_exact_match_sample"]["correct_path_sample"]["number"], 100 * metrics["LCA"]["not_exact_match_sample"]["correct_path_sample"]["ratio"]),
                              "{0}({1:.4f}%)".format(metrics["LCA"]["not_exact_match_sample"]["notcorrect_path_sample"]["number"], 100 * metrics["LCA"]["not_exact_match_sample"]["notcorrect_path_sample"]["ratio"]),
                              "{0}".format(metrics["LCA"]["not_exact_match_sample"]["correct_path_sample"]["lca_heights_all_mean"]),
                              ]
    
    return metrics

def roc(n_classes, target_labels, predict_probs):
    """
    """
    if n_classes == 2:
        fpr, tpr, thresholds = roc_curve(target_labels, predict_probs[:, 1])
        roc_auc = auc(fpr, tpr)

        roc_value = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}
    else:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        target_labels_onehot = label_binarize(target_labels, classes=[i for i in range(n_classes)])
        # calculate auc of each class
        for m in range(n_classes):
            fpr[m], tpr[m], _ = roc_curve(target_labels_onehot[:, m], predict_probs[:, m])
            roc_auc[m] = auc(fpr[m], tpr[m])
        # Compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr[n] for n in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for k in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[k], tpr[k])
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # calculate micro-
        fpr["micro"], tpr["micro"], _ = roc_curve(target_labels_onehot.ravel(), predict_probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # save the data
        roc_value = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}

    return roc_value

def prc(n_classes, target_labels, predict_probs, predict_labels):
    """
    """
    # Compute PRC curve and area the curve
    if n_classes == 2:
        precision, recall, _ = precision_recall_curve(target_labels, predict_probs[:, 1])
        f1, auprc = f1_score(target_labels, predict_labels), auc(recall, precision)
        
        # save the data
        prc_value = {'precision': precision, 'recall': recall, 'auprc': auprc}

    else:
        precision = dict()
        recall = dict()
        average_precision = dict()
        target_labels_onehot = label_binarize(target_labels, classes=[i for i in range(n_classes)])
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(target_labels_onehot[:, i], predict_probs[:, i])
            average_precision[i] = average_precision_score(target_labels_onehot[:, i], predict_probs[:, i])
        
        precision["micro"], recall["micro"], _ = precision_recall_curve(target_labels_onehot.ravel(), predict_probs.ravel())
        average_precision["micro"] = average_precision_score(target_labels_onehot, predict_probs, average="micro")

        # save the data
        prc_value = {'precision': precision, 'recall': recall, 'auprc': average_precision}
    
    return prc_value

def hamming_score(y_true, y_pred):
    """
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    """
    count = 0
    for i in range(y_true.shape[0]):
        p = sum(np.logical_and(y_true[i], y_pred[i]))
        q = sum(np.logical_or(y_true[i], y_pred[i]))
        count += p / q
    return count / y_true.shape[0]

def hierarchy_dependency_sensitivity(config, pred_labels):
    """
    Function: evaluate the sensitivity to the hierarchical discipline structure of the model.
    Params: conifg
            pred_labels, np.array([[1, 0, 1, 0, 0, 0,...], [], ...])
    """
    path = os.path.join(config.vocab.dir, "hierarchy_dependency.json")
    with open(path, 'r') as fin:
        config = json.load(fin)
    
    sensitivity_with_partial = list(map(lambda x: np.all(config['dependency_path_with_partial'] == x, axis=1).any(), pred_labels)).count(True) / pred_labels.shape[0]
    sensitivity_wo_partial = list(map(lambda x: np.all(config['dependency_path_wo_partial'] == x, axis=1).any(), pred_labels)).count(True) / pred_labels.shape[0]

    return sensitivity_with_partial, sensitivity_wo_partial

def error_inspection(config, label_target, label_pred):
    """
    Function: inspect the detailed error information
    Params: label_target, np.array([[1,0,1,0,0,0,...],[],[],...]), shape=(sample size, 11)
            label_pred, np.array([[1,0,1,0,0,0,...],[],[],...]), shape=(sample size, 11)
    """
    path = os.path.join(config.vocab.dir, "hierarchy_dependency.json")
    with open(path, 'r') as fin:
        config = json.load(fin)
    partial_path = np.asarray(config["partial_path"])

    # 1. 获取完全预测正确和非完全预测正确的样本index
    all_correct_idx = []
    notall_correct_idx = []
    for index, value in enumerate(label_target):
        if (label_pred[index,:] == label_target[index,:]).all():
            all_correct_idx.append(index)
        else:
            notall_correct_idx.append(index)

    # 完全预测正确的样本占比
    all_correct_ratio = len(all_correct_idx) / label_target.shape[0]
    # 非完全预测正确的样本占比
    notall_correct_ratio = len(notall_correct_idx) / label_target.shape[0]

    # 2. 在非完全预测正确的样本中获取只预测了部分标签(stop early)的样本index（此处的部分预测，也只是从根节点出发的4种部分预测正确，即R->0, R->1, R->1->5, R->1->6）
    partial_path_idx = []
    complete_path_idx = []
    for index, value in enumerate(label_pred[notall_correct_idx]):
        if np.array(list((map(lambda x: np.all(value == x, axis=0), partial_path)))).any():
            partial_path_idx.append(notall_correct_idx[index])
        else:
            complete_path_idx.append(notall_correct_idx[index])

    # 计算只预测部分标签(i.e. R->0, R->1, R->1->5, R->1->6)在所有非完全正确预测样本中的比例
    partial_path_ratio = len(partial_path_idx) / label_target.shape[0]
    complete_path_ratio = len(complete_path_idx) / label_target.shape[0]

    # 3. 计算在上述不完整标签预测的样本中，预测的部分标签是正确的占比
    partial_correct_idx = []
    partial_notcorrect_idx = []
    if len(partial_path_idx) != 0:
        for index, value in enumerate(label_pred[partial_path_idx]):
            last_index = np.argwhere(value == 1)[-1][0]
            if (value[:last_index+1] == label_target[partial_path_idx][index][:last_index+1]).all():
                partial_correct_idx.append(partial_path_idx[index])
            else:
                partial_notcorrect_idx.append(partial_path_idx[index])

        # 计算在只被预测了部分标签的样本中，标签预测结果是正确的比例
        partial_correct_ratio = len(partial_correct_idx) / label_target.shape[0]
        partial_notcorrect_ratio = len(partial_notcorrect_idx) / label_target.shape[0]
    else:
        partial_correct_ratio = 0
        partial_notcorrect_ratio = 0

    # 对于既不是完全预测正确，也不是非完整预测标签的其他错误预测(other wrong)类别的样本情况进行分析
    # 首先是计算第一层分类(infection & non-infection)即分类错误的部分
    first_level_wrong_idx = []
    first_level_right_idx = []
    for index, value in enumerate(label_pred[complete_path_idx]):
        if (value[:2] == label_target[complete_path_idx][index][:2]).all():
            first_level_right_idx.append(complete_path_idx[index])
        else:
            first_level_wrong_idx.append(complete_path_idx[index])

    # 计算在other wrong类别中，第一层的预测即错误的样本量与占比
    first_level_wrong_ratio = len(first_level_wrong_idx) / label_target.shape[0]
    first_level_right_ratio = len(first_level_right_idx) / label_target.shape[0]

    # 对于第一层预测正确的部分，继续分析其下一层的类别预测错误情况，主要目的是看尽管不是完全预测正确，但是部分预测错误的标签是否落在与真实标签同一父类标签下
    """
    第一部分：是infection预测正确，但是细分感染预测错误
    """
    other_wrong_under_infection_idx = []
    if len(first_level_right_idx) != 0:
        for index, value in enumerate(label_pred[first_level_right_idx]):
            if value[0] == 1 and not (value[2:] == label_target[first_level_right_idx][index][2:]).all():
                other_wrong_under_infection_idx.append(first_level_right_idx[index])
            else:
                continue
        other_wrong_under_infection_ratio = len(other_wrong_under_infection_idx) / label_target.shape[0]
    else:
        other_wrong_under_infection_ratio = 0

    """
    第二部分：是non-infection预测正确，但是后续存在分类错误
    """
    other_wrong_under_noninfection_idx = []
    if len(first_level_right_idx) != 0:
        for index, value in enumerate(label_pred[first_level_right_idx]):
            if value[1] == 1 and (value[2:5]==0).all() and not (value[5:] == label_target[first_level_right_idx][index][5:]).all():
                other_wrong_under_noninfection_idx.append(first_level_right_idx[index])
            else:
                continue
        other_wrong_under_noninfection_ratio = len(other_wrong_under_noninfection_idx) / label_target.shape[0]
    else:
        other_wrong_under_noninfection_ratio = 0
        
    """
    第三部分：是non-infection预测正确，且NIID和Neo类别预测正确，但是最后一层预测错误
    """
    correct_under_noninfection_idx = []
    not_correct_under_noninfection_idx = []
    if len(other_wrong_under_noninfection_idx) != 0:
        for index, value in enumerate(label_pred[other_wrong_under_noninfection_idx]):
            if (value[5:7] == label_target[other_wrong_under_noninfection_idx][index][5:7]).all():
                correct_under_noninfection_idx.append(other_wrong_under_noninfection_idx[index])
            else:
                not_correct_under_noninfection_idx.append(other_wrong_under_noninfection_idx[index])
        correct_under_noninfection_ratio = len(correct_under_noninfection_idx) / label_target.shape[0]
        not_correct_under_noninfection_ratio = len(not_correct_under_noninfection_idx) / label_target.shape[0]
    else:
        correct_under_noninfection_ratio = 0
        not_correct_under_noninfection_ratio = 0
        
    """
    第四部分：是non-infection预测正确，正确预测为NIID类别，但是更细分类类别预测错误
    """
    other_wrong_under_niid_idx = []
    other_wrong_under_neo_idx = []
    if len(correct_under_noninfection_idx) != 0:
        for index, value in enumerate(label_pred[correct_under_noninfection_idx]):
            if value[5] == 1 and not (value[7:] == label_target[correct_under_noninfection_idx][index][7:]).all():
                other_wrong_under_niid_idx.append(correct_under_noninfection_idx[index])
            else:
                other_wrong_under_neo_idx.append(correct_under_noninfection_idx[index])
        
        other_wrong_under_niid_ratio = len(other_wrong_under_niid_idx) / label_target.shape[0]
        other_wrong_under_neo_ratio = len(other_wrong_under_neo_idx) / label_target.shape[0]
    else:
        other_wrong_under_niid_ratio = 0
        other_wrong_under_neo_ratio = 0
    

    error = {"all_correct_sample": {"number": len(all_correct_idx),
                                    "ratio": all_correct_ratio,
                                    "index": all_correct_idx},
             "notall_correct_sample": {"number": len(notall_correct_idx),
                                       "ratio": notall_correct_ratio,
                                       "index": notall_correct_idx,
                                       "partial_predict_sample": {"number": len(partial_path_idx),
                                                               "ratio": partial_path_ratio,
                                                               "index": partial_path_idx,
                                                               "partial_correct_sample": {"number": len(partial_correct_idx),
                                                                                          "ratio": partial_correct_ratio,
                                                                                          "index": partial_correct_idx},
                                                                "partial_notcorrect_sample": {"number": len(partial_notcorrect_idx),
                                                                                              "ratio": partial_notcorrect_ratio,
                                                                                              "index": partial_notcorrect_idx}},
                                       "complete_predict_sample": {"number": len(complete_path_idx),
                                                                   "ratio": complete_path_ratio,
                                                                   "index": complete_path_idx,
                                                                   "first_level_wrong_sample": {"number": len(first_level_wrong_idx),
                                                                                             "ratio": first_level_wrong_ratio,
                                                                                             "index": first_level_wrong_idx},
                                                                   "first_level_right_sample": {"number": len(first_level_right_idx),
                                                                                                "ratio": first_level_right_ratio,
                                                                                                "index": first_level_right_idx,
                                                                                                "wrong_under_infection_sample": {"number": len(other_wrong_under_infection_idx),
                                                                                                                                 "ratio": other_wrong_under_infection_ratio,
                                                                                                                                 "index": other_wrong_under_infection_idx},
                                                                                                "wrong_under_noninfection_sample": {"number": len(other_wrong_under_noninfection_idx),
                                                                                                                                    "ratio": other_wrong_under_noninfection_ratio,
                                                                                                                                    "index": other_wrong_under_noninfection_idx,
                                                                                                                                    "wrong_under_NIID_and_Neo_sample": {"number": len(correct_under_noninfection_idx),
                                                                                                                                                                        "ratio": correct_under_noninfection_ratio,
                                                                                                                                                                        "index": correct_under_noninfection_idx,
                                                                                                                                                                        "wrong_under_NIID": {'number': len(other_wrong_under_niid_idx),
                                                                                                                                                                                             'ratio': other_wrong_under_niid_ratio,
                                                                                                                                                                                             'index': other_wrong_under_niid_idx},
                                                                                                                                                                        "wrong_under_neo": {'number': len(other_wrong_under_neo_idx),
                                                                                                                                                                                            'ratio': other_wrong_under_neo_ratio,
                                                                                                                                                                                            'index': other_wrong_under_neo_idx}},
                                                                                                                                    "wrong_at_NIID_and_Neo_sample": {"number": len(not_correct_under_noninfection_idx),
                                                                                                                                                                     "ratio": not_correct_under_noninfection_ratio,
                                                                                                                                                                     "index": not_correct_under_noninfection_idx}}}}}}
    
    logger.info("本实验中，完全匹配样本数为：{0}，占比：{1} \n \
        非完全匹配样本数为：{2}， 占比：{3} \n \
            不完整路径预测的样本数为：{4}， 占比：{5} \n \
                不完整路径预测中预测正确的样本数为：{6}， 占比：{7} \n \
                不完整路径预测中预测错误的样本数为：{8}，占比：{9} \n \
            完整路径预测的样本数为：{10}，占比：{11} \n \
                第一层预测错误的样本数为：{12}，占比：{13} \n \
                第一层预测正确的样本数为：{14}，占比：{15} \n \
                    第二层感染细分类预测错误的样本数为：{16}，占比：{17} \n \
                    第二层非感染细分类预测错误的样本数为：{18}，占比：{19} \n \
                        NIID和Neo层即分类预测错误的样本数为：{20}，占比：{21} \n \
                        NIID和Neo层预测正确，但第三层预测错误的样本数为：{22}，占比：{23} \n \
                            正确预测为NIID，但是更细分类错误的样本数为：{24}，占比：{25} \n \
                            正确预测为NEO，但是更细分类错误的样本数为：{26}，占比：{27}".format(len(all_correct_idx), all_correct_ratio, \
                                                          len(notall_correct_idx), notall_correct_ratio, \
                                                          len(partial_path_idx), partial_path_ratio, \
                                                          len(partial_correct_idx), partial_correct_ratio, \
                                                          len(partial_notcorrect_idx), partial_notcorrect_ratio, \
                                                          len(complete_path_idx), complete_path_ratio, \
                                                          len(first_level_wrong_idx), first_level_wrong_ratio, \
                                                          len(first_level_right_idx), first_level_right_ratio, \
                                                          len(other_wrong_under_infection_idx), other_wrong_under_infection_ratio, \
                                                          len(other_wrong_under_noninfection_idx), other_wrong_under_noninfection_ratio, \
                                                          len(not_correct_under_noninfection_idx), not_correct_under_noninfection_ratio, \
                                                          len(correct_under_noninfection_idx), correct_under_noninfection_ratio, \
                                                          len(other_wrong_under_niid_idx), other_wrong_under_niid_ratio, \
                                                          len(other_wrong_under_neo_idx), other_wrong_under_neo_ratio))
    
    return error

def lca_height_related(config, label_target, label_pred):
    """
    Function: get the LCA height related information to characterize that the model are proned to predict the incorrect
              sample into label under same parent label with true label of the sample.
    Params: label_target, np.array([[1,0,1,0,0,0,...],[],[],...]), shape=(sample size, 11)
            label_pred, np.array([[1,0,1,0,0,0,...],[],[],...]), shape=(sample size, 11)
    """
    path = os.path.join(config.vocab.dir, "hierarchy_dependency.json")
    with open(path, 'r') as fin:
        config = json.load(fin)
    partial_path = np.asarray(config["partial_path"])
    complete_path_wo_partial = np.asarray(config["dependency_path_wo_partial"])
    complete_path_wt_partial = np.asarray(config["dependency_path_with_partial"])
    node_mapping = config["node_mapping"]

    # 1. 获取完全预测正确和非完全预测正确的样本index
    all_correct_idx = []
    notall_correct_idx = []
    for index, value in enumerate(label_target):
        if (label_pred[index,:] == label_target[index,:]).all():
            all_correct_idx.append(index)
        else:
            notall_correct_idx.append(index)

    # 完全预测正确的样本占比
    all_correct_ratio = len(all_correct_idx) / label_target.shape[0]
    # 非完全预测正确的样本占比
    notall_correct_ratio = len(notall_correct_idx) / label_target.shape[0]

    # 2. 在非完全预测正确的样本中获取只预测了部分标签(stop early)的样本index（此处的部分预测，也只是从根节点出发的4种部分预测正确，即R->0, R->1, R->1->5, R->1->6）
    partial_path_idx = []
    complete_path_idx = []
    for index, value in enumerate(label_pred[notall_correct_idx]):
        if np.array(list((map(lambda x: np.all(value == x, axis=0), partial_path)))).any():
            partial_path_idx.append(notall_correct_idx[index])
        else:
            complete_path_idx.append(notall_correct_idx[index])

    # 计算只预测部分标签(i.e. R->0, R->1, R->1->5, R->1->6)在所有预测样本中的比例
    partial_path_ratio = len(partial_path_idx) / label_target.shape[0]
    complete_path_ratio = len(complete_path_idx) / label_target.shape[0]

    # 3. 计算在上述不完整标签预测的样本中，预测的部分标签是正确的占比
    partial_correct_idx = []
    partial_notcorrect_idx = []
    if len(partial_path_idx) != 0:
        for index, value in enumerate(label_pred[partial_path_idx]):
            last_index = np.argwhere(value == 1)[-1][0]
            if (value[:last_index+1] == label_target[partial_path_idx][index][:last_index+1]).all():
                partial_correct_idx.append(partial_path_idx[index])
            else:
                partial_notcorrect_idx.append(partial_path_idx[index])

        # 计算在只被预测了部分标签的样本中，标签预测结果是正确的比例
        partial_correct_ratio = len(partial_correct_idx) / label_target.shape[0]
        partial_notcorrect_ratio = len(partial_notcorrect_idx) / label_target.shape[0]
    else:
        partial_correct_ratio = 0
        partial_notcorrect_ratio = 0

    # 4. 对于不完全预测正确的样本中，其预测路径又不是部分路径的样本，求其预测路径不符合Label hierarchy规则的样本
    notcorrect_predict_path_idx = []
    correct_predict_path_idx = []
    for index, value in enumerate(label_pred[complete_path_idx]):
        if np.array(list((map(lambda x: np.all(value == x, axis=0), complete_path_wo_partial)))).any():
            correct_predict_path_idx.append(complete_path_idx[index])
        else:
            notcorrect_predict_path_idx.append(complete_path_idx[index])

    correct_predict_path_ratio = len(correct_predict_path_idx) / label_target.shape[0]
    notcorrect_predict_path_ratio = len(notcorrect_predict_path_idx) / label_target.shape[0]
    
    # 对于预测路径完整，且其预测路径符合label hierarchy的约束样本，求其LCA height
    lca_heights = []
    for trg, pred in zip(label_target[correct_predict_path_idx], label_pred[correct_predict_path_idx]):
        leaf_node_trg = node_mapping[str(complete_path_wt_partial.tolist().index(trg.tolist()))]
        leaf_node_pred = node_mapping[str(complete_path_wt_partial.tolist().index(pred.tolist()))]
        lca_heights.append(lca_height(leaf_node_trg, leaf_node_pred))

    lca_height_mean = np.asarray(lca_heights).mean()
    
    # 5. 对于不是完全匹配的样本，但是预测的标签部分或全部符合label hierarchy的样本，计算占比及其LCA height
    correct_path_idx = []
    notcorrect_path_idx = []
    for index, value in enumerate(label_pred[notall_correct_idx]):
        if np.array(list((map(lambda x: np.all(value == x, axis=0), complete_path_wt_partial)))).any():
            correct_path_idx.append(notall_correct_idx[index])
        else:
            notcorrect_path_idx.append(notall_correct_idx[index])
            
    # 计算非完全匹配样本，预测标签部分或全部遵守label Hierarchy约束的占比，以及完全不遵守label hierarchy约束的占比
    correct_path_ratio = len(correct_path_idx) / label_target.shape[0]
    notcorrect_path_ratio = len(notcorrect_path_idx) / label_target.shape[0]
    
    # 6. 对于不是完全匹配，但是预测标签部分或全部遵守label hierarchy约束的样本，计算其LCA height
    lca_heights_all = []
    for trg, pred in zip(label_target[correct_path_idx], label_pred[correct_path_idx]):
        leaf_node_trg = node_mapping[str(complete_path_wt_partial.tolist().index(trg.tolist()))]
        leaf_node_pred = node_mapping[str(complete_path_wt_partial.tolist().index(pred.tolist()))]
        lca_heights_all.append(lca_height(leaf_node_trg, leaf_node_pred))

    lca_heights_all_mean = np.asarray(lca_heights_all).mean()
    
    # organize into a dict
    lca = {"test_sample_size": {"number": label_target.shape[0],
                                "label_target": label_target,
                                "label_pred": label_pred},
           "exact_match_sample": {"number": len(all_correct_idx),
                                  "ratio": all_correct_ratio,
                                  "index": all_correct_idx},
           "not_exact_match_sample": {"number": len(notall_correct_idx),
                                      "ratio": notall_correct_ratio,
                                      "index": notall_correct_idx,
                                      "partial_path_sample": {"number": len(partial_path_idx),
                                                              "ratio": partial_path_ratio,
                                                              "index": partial_path_idx,
                                                              "partial_correct_sample": {"number": len(partial_correct_idx),
                                                                                         "ratio": partial_correct_ratio,
                                                                                         "index": partial_correct_idx}},
                                      "complete_path_sample": {"number": len(complete_path_idx),
                                                              "ratio": complete_path_ratio,
                                                              "index": complete_path_idx,
                                                              "notcorrect_predict_path_sample": {"number": len(notcorrect_predict_path_idx),
                                                                                                 "ratio": notcorrect_predict_path_ratio,
                                                                                                 "index": notcorrect_predict_path_idx},
                                                              "correct_predict_path_sample": {"number": len(correct_predict_path_idx),
                                                                                              "ratio": correct_predict_path_ratio,
                                                                                              "index": correct_predict_path_idx,
                                                                                              "lca_height": lca_height_mean,
                                                                                              "lca_heights": lca_heights}},
                                      "correct_path_sample": {"number": len(correct_path_idx),
                                                              "ratio": correct_path_ratio,
                                                              "index": correct_path_idx,
                                                              "lca_heights_all": lca_heights_all,
                                                              "lca_heights_all_mean": lca_heights_all_mean},
                                      "notcorrect_path_sample": {"number": len(notcorrect_path_idx),
                                                                "ratio": notcorrect_path_ratio,
                                                                "index": notcorrect_path_idx}}}
    
    logger.info("本实验中, 测试样本数量为: {0}, \n \
        完全匹配样本数为：{1}, 占比: {2} \n \
        非完全匹配样本数为: {3}, 占比: {4} \n \
            不完整路径预测的样本数为: {5}, 占全部测试样本比例为: {6} \n \
                不完整路径预测中预测正确的样本数为: {7}, 占全部测试样本比例为: {8} \n \
            完整路径预测的样本数为: {9}, 占全部测试样本比例为: {10} \n \
                违反类别层次关系约束的样本数为: {11}, 占全部测试样本比例为: {12} \n \
                不违反类别层次关系约束的样本数为: {13}, 占全部测试样本比例为: {14} \n \
                    其中预测标签与真实标签之间的LCA高度的均值为: {15} \n \
            预测路径部分或全部遵守层次关系约束的样本数为：{16}, 占全部测试样本比例为：{17} \n \
                其中预测标签与真实标签之间的LCA高度的均值为: {18} \n \
            预测路径完全不遵守层次关系约束的样本数为：{19}, 占全部测试样本比例为：{20}".format(label_target.shape[0], \
                                                          len(all_correct_idx), all_correct_ratio, \
                                                          len(notall_correct_idx), notall_correct_ratio, \
                                                          len(partial_path_idx), partial_path_ratio, \
                                                          len(partial_correct_idx), partial_correct_ratio, \
                                                          len(complete_path_idx), complete_path_ratio, \
                                                          len(notcorrect_predict_path_idx), notcorrect_predict_path_ratio, \
                                                          len(correct_predict_path_idx), correct_predict_path_ratio, \
                                                          lca_height_mean, len(correct_path_idx), correct_path_ratio, \
                                                          lca_heights_all_mean, len(notcorrect_path_idx), notcorrect_path_ratio))
    return lca
