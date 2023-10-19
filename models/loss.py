#!/usr/bin/env python
# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
        
class MultitaskLoss(nn.Module):
    def __init__(self, nclasses):
        """
        Function: define the loss func for multi-task learning
        """
        super(MultitaskLoss, self).__init__()
        self.nclasses = nclasses
        self.total_task = len(nclasses)

    def forward(self, pred_probs, target_label):
        """
        params: pred_probs: List, List[tensor[], tensor[], ...]
        params: target_label
        """
        loss = 0
        for i in range(self.total_task):
            loss += nn.NLLLoss()(F.log_softmax(pred_probs[i], dim=1), target_label[i])

        return loss

class MultitaskWeightedLoss(nn.Module):
    def __init__(self, nclasses):
        """
        Function: define the loss func for multi-task learning
        """
        super(MultitaskWeightedLoss, self).__init__()
        self.nclasses = nclasses
        self.total_task = len(nclasses)
        self.log_vars = nn.Parameter(torch.zeros((self.total_task)))

    def forward(self, pred_probs, target_label):
        """
        Reference: https://github.com/thiagodma/Pytorch_exs/blob/master/MultiTaskLearning/multitask_age_gender_ethnicity_resnet34.ipynb
                   Kendall, A., Y. Gal, and R. Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. in Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
        params: pred_probs: List, List[tensor[], tensor[], ...]
        params: target_label
        """
        loss0 = nn.NLLLoss()(F.log_softmax(pred_probs[0], dim=1), target_label[0])
        loss1 = nn.NLLLoss()(F.log_softmax(pred_probs[1], dim=1), target_label[1])
        loss2 = nn.NLLLoss()(F.log_softmax(pred_probs[2], dim=1), target_label[2])
        loss3 = nn.NLLLoss()(F.log_softmax(pred_probs[3], dim=1), target_label[3])
        loss4 = nn.NLLLoss()(F.log_softmax(pred_probs[4], dim=1), target_label[4])

        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0*loss0 + self.log_vars[0]

        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1*loss1 + self.log_vars[1]

        precision2 = torch.exp(-self.log_vars[2])
        loss2 = precision2*loss2 + self.log_vars[2]

        precision3 = torch.exp(-self.log_vars[3])
        loss3 = precision3*loss3 + self.log_vars[3]

        precision4 = torch.exp(-self.log_vars[4])
        loss4 = precision4*loss4 + self.log_vars[4]

        return loss0 + loss1 + loss2 + loss3 + loss4
