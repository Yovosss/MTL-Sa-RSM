#!/usr/bin/env python
# coding: utf-8
import json
import os

import numpy as np
from torch.utils.data.dataset import Dataset

import helper.logger as logger


class HcDataset(Dataset):
    def __init__(self, config):
        """
        Dataset for FUO hierarchical classification based on torch.utils.data.dataset.Dataset
        :param config: Object
        :param data: 
        :param label:
        :param indices:
        """
        self.config = config
        self.kfold = config.data.kfold
        self.max_timestamp = config.data.max_timestamp
        self.norm_type = config.data.norm_type

    def __len__(self):
        """
        Get the number of samples
        :return: self.
        """
        return self.sample_size
    
    def __getitem__(self, index):
        """
        sample from the overall dataset
        :param index: int, should be smaller than len(dataset)
        :return: sample -> Dict{'':, '':, '':, '':} 
        """
        if index >= self.__len__():
            raise IndexError

        raw_sample = [self.data[s][index] for s in ['X_t', 'T_t', 'X_t_mask', 'deltaT_t', 'X_val' , 'X_cat', 'y_classes_unique']]
        
        return self._preprocess_sample(raw_sample)

    def _preprocess_sample(self, raw_sample):
        """"
        normalize the sample data
        :param: raw_sample -> List[array(), array(), array(), array(), List[List[]]]
        :return: Dict{'X_t': np.array([]),
                      'X': np.array([]),
                      'X_t_mask': np.array([]),
                      'deltaT_t': np.array([]),
                      'X_t_filledLOCF': np.array([]),
                      'y_classes': List[List[int]],
                      'empirical_mean': np.array([])}
        """
        sample = {}

        # normalize the X_t
        if self.norm_type == "Normalization":
            sample['X_t'] = self._rescale_norm(raw_sample[0], self.data['X_t_max'], self.data['X_t_min'])
        elif self.norm_type == "Standardization":
            sample['X_t'] = self._rescale_stdize(raw_sample[0], self.data['X_t_mean'], self.data['X_t_std'])
        sample['X_t'] = np.nan_to_num(sample['X_t'])
        
        # fill the nan value in X_val and normalize X_val
        raw_sample[4] = self._fillnan(raw_sample[4], self.data['X_val_mean'])
        if self.norm_type == "Normalization":
            raw_sample[4] = self._rescale_norm(raw_sample[4], self.data['X_val_max'], self.data['X_val_min'])
        elif self.norm_type == "Standardization":
            raw_sample[4] = self._rescale_stdize(raw_sample[4], self.data['X_val_mean'], self.data['X_val_std'])
        # concatenate the static variables
        sample['X'] = np.concatenate((raw_sample[4], raw_sample[5]))

        # forward fill nan value in np.array
        sample['X_t_filledLOCF'] = self._locf_numpy(sample['X_t'], raw_sample[0])

        sample['empirical_mean'] = self.empirical_mean

        sample['X_t_mask'] = raw_sample[2]
        sample['deltaT_t'] = raw_sample[3] / 86400 # 24*60*60
        sample['y_classes'] = raw_sample[-1]

        return sample
    
    def f_empirical_mean(self):
        if self.norm_type == "Normalization":
            X_rescaled = self._rescale_norm(self.data['X_t'], self.data['X_t_max'], self.data['X_t_min'])
        elif self.norm_type == "Standardization":
            X_rescaled = self._rescale_stdize(self.data['X_t'], self.data['X_t_mean'], self.data['X_t_std'])
        X_rescaled = X_rescaled.reshape(-1, self.data['X_t'].shape[-1])
        empirical_mean = np.nanmean(X_rescaled, axis=0)

        return empirical_mean

    def _rescale_norm(self, x, max, min):
        """
        normalize the non-time-series data and time-series data
        :param x: A np.array witn shape (t_i, d)
               mean: A np.array with shape (d,)
               std: A np.array with shape (d,)
        :return A np.array with same shape as x with rescaled values
        """
        if x.ndim == 1:
            return (x - min) / (max - min)
        elif x.ndim == 2:
            return (x - min[np.newaxis, :]) / (max[np.newaxis, :] - min[np.newaxis, :])
        elif x.ndim == 3:
            return np.asarray([(xx - min[np.newaxis, :]) / (max[np.newaxis, :]-min[np.newaxis, :]) for xx in x])

    def _rescale_stdize(self, x, mean, std):
        """
        standardize the non-time-series data and time-series data
        :param x: A np.array witn shape (t_i, d)
               mean: A np.array with shape (d,)
               std: A np.array with shape (d,)
        :return A np.array with same shape as x with rescaled values
        """
        if x.ndim == 1:
            return (x - mean) / std
        elif x.ndim == 2:
            return (x - mean[np.newaxis, :]) / std[np.newaxis, :]
        elif x.ndim == 3:
            return np.asarray([(xx - mean[np.newaxis, :]) / std[np.newaxis, :] for xx in x])

    def _fillnan(self, x, mean):
        """
        fill the nan value in non-time-series data
        :param x: A np.array of static variables with shape (d,)
               mean: A np.array of mean value of each variable with shape (d,)
        :return A np.array without nan value
        """
        x[np.isnan(x)] = mean[np.isnan(x)]

        return x  

    def _locf_numpy(self, X, X_nan):
        """Numpy implementation of LOCF.

        Parameters
        ----------
        X : np.ndarray,
            Time series containing missing values (NaN) to be imputed.

        Returns
        -------
        X_imputed : array,
            Imputed time series.

        Notes
        -----
        This implementation gets inspired by the question on StackOverflow:
        https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
        """
        trans_X = X.transpose((1, 0))
        trans_X_nan = X_nan.transpose((1, 0))
        mask = np.isnan(trans_X_nan)
        n_features, n_steps  = mask.shape
        idx = np.where(~mask, np.arange(n_steps), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)

        X_imputed = trans_X[np.arange(n_features)[:, None], idx]
        X_imputed = X_imputed.transpose((1, 0))

        # If there are values still missing,
        # they are missing at the beginning of the time-series sequence.
        # Impute them with self.nan
        if np.isnan(X_imputed).any():
            X_imputed = np.nan_to_num(X_imputed, nan=0)

        return X_imputed

class FlatDataset(HcDataset):
    def __init__(self, config, data, label, indices, stage="TRAIN"):
        """
        Dataset for Flat FUO classification based on torch.utils.data.dataset.Dataset
        :param config: Object
        :param data: 
        :param label:
        :param indices:
        """
        super().__init__(config)
        # indice the data
        logger.info('Loading {}-fold data of {} Dataset for Flat classification...'.format(self.kfold, stage))
        self.indice = indices['folds_idx_with_txy'] \
                                ['parent-node-0'] \
                                    [self.kfold,:] \
                                        [{'TRAIN': 0, 'VALIDATION': 1, 'TEST': 2}[stage]]
        
        self.data = {
            'X_t': data['X_t'][self.indice], 
            'T_t': data['T_t_rel'][self.indice],
            'X_t_mask': data['X_t_mask'][self.indice],
            'deltaT_t': data['deltaT_t'][self.indice],
            'X_val': data['static_data_val'][self.indice],
            'X_cat': data['static_data_cat_onehot'][self.indice],
            'y_classes_unique': np.array(label['y_classes_unique'], dtype=object)[self.indice].tolist(),
            'X_val_mean': indices['folds_stats']['parent-node-0'][self.kfold][0,0],
            'X_val_std': indices['folds_stats']['parent-node-0'][self.kfold][0,1],
            'X_val_max': indices['folds_stats']['parent-node-0'][self.kfold][0,2],
            'X_val_min': indices['folds_stats']['parent-node-0'][self.kfold][0,3],
            'X_t_mean': indices['folds_stats']['parent-node-0'][self.kfold][1,0],
            'X_t_std': indices['folds_stats']['parent-node-0'][self.kfold][1,1],
            'X_t_max': indices['folds_stats']['parent-node-0'][self.kfold][1,2],
            'X_t_min': indices['folds_stats']['parent-node-0'][self.kfold][1,3]
        }

        assert len(self.data['X_t']) == \
            len(self.data['T_t']) == \
                len(self.data['X_t_mask']) == \
                    len(self.data['deltaT_t']) == \
                        len(self.data['X_val']) == \
                            len(self.data['X_cat']) == \
                                len(self.data['y_classes_unique']), \
                                    "The first dimensions of X_t, T_t, X_t_mask, deltaT_t, X_val, X_cat and y_classes_unique are not aligned!!"

        self.sample_size = self.indice.shape[0]
        self.empirical_mean = self.f_empirical_mean()

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        return super().__getitem__(index)

class LocalHcDataset(HcDataset):
    def __init__(self, config, data, label, indices, stage="TRAIN"):
        """
        Dataset for Local FUO hierarchical classification based on torch.utils.data.dataset.Dataset
        :param config: Object
        :param data: 
        :param label:
        :param indices:
        """
        super().__init__(config)
        self.local_task = self.config.experiment.local_task

        # indice the data
        logger.info('Loading {}-fold data of {} Dataset for parent-node-{} classification...'.format(self.kfold, stage, self.local_task))
        self.indice = indices['folds_idx_with_txy'] \
                                ['parent-node-{}'.format(self.local_task)] \
                                    [self.kfold,:] \
                                        [{'TRAIN': 0, 'VALIDATION': 1, 'TEST': 2}[stage]]
        
        self.data = {
            'X_t': data['X_t'][self.indice], 
            'T_t': data['T_t_rel'][self.indice],
            'X_t_mask': data['X_t_mask'][self.indice],
            'deltaT_t': data['deltaT_t'][self.indice],
            'X_val': data['static_data_val'][self.indice],
            'X_cat': data['static_data_cat_onehot'][self.indice],
            'y_classes_unique': np.array(label['y_classes_unique'], dtype=object)[self.indice].tolist(),
            'X_val_mean': indices['folds_stats']['parent-node-{}'.format(self.local_task)][self.kfold][0,0],
            'X_val_std': indices['folds_stats']['parent-node-{}'.format(self.local_task)][self.kfold][0,1],
            'X_val_max': indices['folds_stats']['parent-node-{}'.format(self.local_task)][self.kfold][0,2],
            'X_val_min': indices['folds_stats']['parent-node-{}'.format(self.local_task)][self.kfold][0,3],
            'X_t_mean': indices['folds_stats']['parent-node-{}'.format(self.local_task)][self.kfold][1,0],
            'X_t_std': indices['folds_stats']['parent-node-{}'.format(self.local_task)][self.kfold][1,1],
            'X_t_max': indices['folds_stats']['parent-node-{}'.format(self.local_task)][self.kfold][1,2],
            'X_t_min': indices['folds_stats']['parent-node-{}'.format(self.local_task)][self.kfold][1,3]
        }
        assert len(self.data['X_t']) == \
            len(self.data['T_t']) == \
                len(self.data['X_t_mask']) == \
                    len(self.data['deltaT_t']) == \
                        len(self.data['X_val']) == \
                            len(self.data['X_cat']) == \
                                len(self.data['y_classes_unique']), \
                                    "The first dimensions of X_t, T_t, X_t_mask, deltaT_t, X_val, X_cat and y_classes_unique are not aligned!!"
        
        self.sample_size = self.indice.shape[0]
        self.empirical_mean = self.f_empirical_mean()

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        return super().__getitem__(index)

class GlobalHcDataset(HcDataset):
    def __init__(self, config, data, label, indices, stage="TRAIN"):
        """
        Dataset for Global FUO hierarchical classification based on torch.utils.data.dataset.Dataset
        :param config: Object
        :param data: 
        :param label:
        :param indices:
        """
        super().__init__(config)

        # indice the data
        logger.info('Loading {}-fold data of {} Dataset for global classification...'.format(self.kfold, stage))
        self.indice = indices['folds_idx_with_txy'] \
                                ['parent-node-0'] \
                                    [self.kfold,:] \
                                        [{'TRAIN': 0, 'VALIDATION': 1, 'TEST': 2}[stage]]

        self.data = {
            'X_t': data['X_t'][self.indice], 
            'T_t': data['T_t_rel'][self.indice],
            'X_t_mask': data['X_t_mask'][self.indice],
            'deltaT_t': data['deltaT_t'][self.indice],
            'X_val': data['static_data_val'][self.indice],
            'X_cat': data['static_data_cat_onehot'][self.indice],
            'y_classes_unique': np.array(label['y_classes_unique'], dtype=object)[self.indice].tolist(),
            'X_val_mean': indices['folds_stats']['parent-node-0'][self.kfold][0,0],
            'X_val_std': indices['folds_stats']['parent-node-0'][self.kfold][0,1],
            'X_val_max': indices['folds_stats']['parent-node-0'][self.kfold][0,2],
            'X_val_min': indices['folds_stats']['parent-node-0'][self.kfold][0,3],
            'X_t_mean': indices['folds_stats']['parent-node-0'][self.kfold][1,0],
            'X_t_std': indices['folds_stats']['parent-node-0'][self.kfold][1,1],
            'X_t_max': indices['folds_stats']['parent-node-0'][self.kfold][1,2],
            'X_t_min': indices['folds_stats']['parent-node-0'][self.kfold][1,3],
            'unique_label_number': label['unique_label_number']
        }

        assert len(self.data['X_t']) == \
            len(self.data['T_t']) == \
                len(self.data['X_t_mask']) == \
                    len(self.data['deltaT_t']) == \
                        len(self.data['X_val']) == \
                            len(self.data['X_cat']) == \
                                len(self.data['y_classes_unique']), \
                                    "The first dimensions of X_t, T_t, X_t_mask, deltaT_t, X_val, X_cat and y_classes_unique are not aligned!!"

        self.sample_size = self.indice.shape[0]
        self.empirical_mean = self.f_empirical_mean()
    
    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        return super().__getitem__(index)

    def _preprocess_sample(self, raw_sample):
        
        sample = super()._preprocess_sample(raw_sample)
        sample['label_node_inputs'] = [i for i in range(self.data['unique_label_number'])]

        return sample