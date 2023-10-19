#!/usr/bin/env python
# coding:utf-8

import numpy as np
import torch

import helper.logger as logger


class EarlyStopping():
    """Early stops the training if auc doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time auc improved.
                            Default: 7
            verbose (bool): If True, prints a message for each auc improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'  
        Reference: https://github.com/Bjarten/early-stopping-pytorch       
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_metric_max = 0.0
        self.delta = delta
        self.path = path
    def __call__(self, val_metric, model, optimizer, best_epoch, best_performance, val_metric_name='auc'):

        score = -val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model, optimizer, best_epoch, best_performance, val_metric_name)
        # if set val_loss as the monitored target, can change > to <
        elif score >= self.best_score + self.delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model, optimizer, best_epoch, best_performance, val_metric_name)
            self.counter = 0

    def save_checkpoint(self, val_metric, model, optimizer, best_epoch, best_performance, val_metric_name):
        '''Saves model when validation auc increase.'''
        if self.verbose:
            logger.info(f'Validation {str.upper(val_metric_name)} increased ({self.val_metric_max:.6f} --> {val_metric:.6f}).  Saving model ...')
        torch.save({
                'epoch': best_epoch,
                'best_performance': best_performance,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}, self.path)
        self.val_metric_max = val_metric