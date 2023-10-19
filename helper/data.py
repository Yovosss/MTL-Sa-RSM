#!/usr/bin/env python
# coding:utf-8

import logging
import os
import pickle as pkl
import random
import re
from collections import Counter

import numpy as np
import pandas
import torch
import torch.utils.data as data
from sklearn.model_selection import StratifiedKFold

import helper.logger as logger

BASE_LOC = str(os.path.dirname(os.path.realpath(__file__)).split('helper')[0])

class DataPreprocess():

    def __init__(self, config):
        """
        Generate research-ready data for different experiments
        param: config object
        """
        self.seed = config.seed
        self.exp_type = config.experiment.type
        self.data_type = config.data.data_type
        self.foldk = config.experiment.k_folds

        self.data_dir = os.path.join(BASE_LOC, config.data.data_dir)
        self.data_loc = os.path.join(self.data_dir, config.data.data_loc)
        self.max_timestamp = config.data.max_timestamp
        self.split_ratio = config.experiment.dataset_split_ratio
        self.save_path_base = os.path.join(self.data_dir, config.data.data_saved)
        self.save_loc = os.path.join(self.save_path_base, 
                                    'data({}hrs)(ratio={}).pkl'.format(
                                    config.data.max_timestamp,
                                    self.split_ratio))
        if not os.path.exists(self.save_path_base):
            os.makedirs(self.save_path_base)

    def load(self):
        """
        Load the original data as input of Dataset class
        """

        if not os.path.exists(self.save_loc):
            logger.info("Preprocessing...")
            processed_dict = self.preprocess()
        
        else:
            logger.info("Loading previously preprocessed data...")
            with open(self.save_loc, 'rb') as f:
                processed_dict = pkl.load(f, encoding='bytes', errors='ignore')

        data = processed_dict['data']
        label = processed_dict['label']
        indices = processed_dict['indices']
        
        return data, label, indices

    def preprocess(self):
        """
        preprocess original data in an uniform format for different experiment setting.
        func1: split data into train, validation and test dataset
        func2: do normalization based on training dataset
        """
        data = {}
        label = {}
        indices = {}

        logger.info("Reading {}hrs original data!".format(self.max_timestamp))

        # load data
        ori_data = np.load(self.data_loc, allow_pickle=True)

        self._data = {}
        for attr in ['X_t','X_t_mask','T_t','T_t_rel','deltaT_t','static_data_val','static_data_cat','static_data_cat_onehot','label_data']:
            self._data[attr] = ori_data[attr]

        # process label data
        y_classes, label2id, y_classes_unique, unique_label_number, taxonomy = self._label_preprocess()

        # generate K-fold idx
        folds_idx = self._gen_folds_ids(y_classes_unique)

        # regenerate the K-fold idx in terms of label taxonomy
        folds_idx_with_txy = self._gen_folds_ids_refined(folds_idx, taxonomy, y_classes_unique)
        
        # check the shape of folds_idx_with_txy
        for key, value in folds_idx_with_txy.items():
            logger.info(key)
            for idx in folds_idx_with_txy[key]:
                logger.info([id.shape for id in idx])

        # calculate the mean, std according the training dataset on each fold
        folds_stats = self._cal_stats_for_folds(folds_idx_with_txy)

        logger.info("Split done!")

        # calculate the max timesteps
        self.max_step = max(np.asarray([np.sum(tt - tt[0] <= self.max_timestamp*60*60) for tt in self._data['T_t_rel']]))

        # filter and padding X_t, X_t_mask, T_t, deltaT_t
        lens = self._filter(self._data['T_t_rel'], self.max_timestamp)
        X_t_padded = self._padding(self._data['X_t'], lens)
        X_t_mask_padded = self._padding(self._data['X_t_mask'], lens)
        T_t_rel_padded = self._padding(self._data['T_t_rel'], lens)
        deltaT_t_padded = self._padding(self._data['deltaT_t'], lens)

        # re-fill missing value with np.nan
        X_t_padded_nan = masked_fill(X_t_padded, 1 - X_t_mask_padded, np.nan)

        logger.info("Preprocessing done!")

        # save the data, label and indices into dictionary
        data['X_t'] = X_t_padded_nan
        data['X_t_mask'] = X_t_mask_padded
        data['T_t'] = self._data['T_t']
        data['T_t_rel'] = T_t_rel_padded
        data['deltaT_t'] = deltaT_t_padded
        data['static_data_val'] = self._data['static_data_val']
        data['static_data_cat'] = self._data['static_data_cat']
        data['static_data_cat_onehot'] = self._data['static_data_cat_onehot']

        data['X_t_features'] = X_t_padded_nan.shape[-1]
        data['X_t_steps'] = self.max_step
        data['X_features'] = self._data['static_data_val'].shape[-1] + self._data['static_data_cat_onehot'].shape[-1]

        label['label_data'] = self._data['label_data']
        label['y_classes'] = y_classes
        label['label2id'] = label2id
        label['y_classes_unique'] = y_classes_unique
        label['unique_label_number'] = unique_label_number
        label['taxonomy'] = taxonomy

        indices['folds_idx'] = folds_idx
        indices['folds_idx_with_txy'] = folds_idx_with_txy
        indices['folds_stats'] = folds_stats

        output = {
            'data': data,
            'label': label,
            'indices': indices
        }

        pkl.dump(output, open(self.save_loc, 'wb'))
        logger.info("Saved in {}".format(self.save_loc))

        return output

    def _label_preprocess(self):
        """
        process label data into taxonomy
        return: y_classes -> [[0,0], [0,1], [1,3,1],...], col1: 0-1, col2: 0-4, col3: 0-3
                label2id -> {'l0_0': 1,'l0_1': 2,'l1_0': 3,'l1_1': 4,...}
                unique_labels -> [[0,1,3], [0,1,4], [0,2,6,9],...]
                unique_label_number -> 12
                taxonomy -> {0:{1,2}, 1:{3,4,5}, 2:{6,7},...}
        """
        y_classes = []
        assert self._data['label_data'].shape[0] == self._data['X_t'].shape[0]

        for lb in range(self._data['label_data'].shape[0]):
            # bacterial infection
            if self._data['label_data'][lb][0] == '1-1':
                y_classes.append([0, 0]) 
            # viral infection
            elif self._data['label_data'][lb][0] == '1-2':
                y_classes.append([0, 1]) 
            # fungal infection    
            elif self._data['label_data'][lb][0] == '1-3':
                y_classes.append([0, 2])
            # NIID disease 
            elif self._data['label_data'][lb][0] == '2-1-1':
                y_classes.append([1, 3, 0])
            # neoplastic disease 
            elif self._data['label_data'][lb][0] == '2-1-2':
                y_classes.append([1, 3, 1])
            # HM disease     
            elif self._data['label_data'][lb][0] == '2-2-1':
                y_classes.append([1, 4, 2])
            # SM disease     
            elif self._data['label_data'][lb][0] == '2-2-2':
                y_classes.append([1, 4, 3]) 
            else:
                logger.fatal("There are wrong mapped labels!!!")
        
        assert len(y_classes) == self._data['X_t'].shape[0]

        # in previous step, the labels were taken as unique id per level, here we resign all-level unique id to labels.
        label2id = {}
        taxonomy = {}
        logger.info("Building taxonomy...")
        max_levels = max([len(label) for label in y_classes])
        logger.info("Max number of levels : {}".format(max_levels))

        ct = 1
        for i in range(max_levels):
            labels_in_level = set([label[i] for label in y_classes if len(label) > i])
            for lb in labels_in_level:
                label2id['l{}_{}'.format(i,lb)] = ct
                ct += 1

        y_classes_unique = []
        for labels in y_classes:
            row_labels = [0] # start with the root label
            for i, label in enumerate(labels):
                row_labels.append(label2id['l{}_{}'.format(i,label)])
            y_classes_unique.append(row_labels)
        unique_label_number = ct

        # build the taxonomy
        taxonomy[0] = set()
        for labels in y_classes:
            new_dec_labels = [label2id['l{}_{}'.format(level,label)] for level,label in enumerate(labels)]
            parent = 0
            for dec_label in new_dec_labels:
                if parent not in taxonomy:
                    taxonomy[parent] = set()
                taxonomy[parent].add(dec_label)
                parent = dec_label
        logger.info("Label preprocessing done!")

        return y_classes, label2id, y_classes_unique, unique_label_number, taxonomy

    def _gen_folds_ids(self, y):
        """
        generate stratified K fold index list
        param: y -> List[List] label data
        return: folds_idx -> array(array(array(), array(), array()), ...), shape=(k, 3)
        """
        tmpy = np.array([xx[-1] for xx in y])
        try:
            folds_idx = self._make_split(tmpy)

        except:
            logger.error("When split the data, there is an error raised!")
            folds_idx = None

        return folds_idx

    def _gen_folds_ids_refined(self, folds_idx, taxonomy, y_classes_unique):
        """
        Regenarate K-fold index based on label taxonomy
        """
        y_classes_dict = {}
        for key, value in taxonomy.items():
            y_classes_dict[key] = []
            for index, label in enumerate(y_classes_unique):
                # if bool(set(value).intersection(set(label))):
                if not set(value).isdisjoint(label):
                # if any(i in value for i in label):
                    y_classes_dict[key].append(index)

        folds_idx_with_txy = {}
        for parnd, sampleid in y_classes_dict.items():
            folds_idx_with_txy['parent-node-%d'%parnd] = np.empty([self.foldk, 3], dtype=object)
            for k in range(self.foldk):
                folds_idx_with_txy['parent-node-%d'%parnd][k][0] = np.array([idx for idx in folds_idx[k][0] if idx in sampleid])
                folds_idx_with_txy['parent-node-%d'%parnd][k][1] = np.array([idx for idx in folds_idx[k][1] if idx in sampleid])
                folds_idx_with_txy['parent-node-%d'%parnd][k][2] = np.array([idx for idx in folds_idx[k][2] if idx in sampleid])

        return folds_idx_with_txy

    def _make_split(self, y):
        """
        split into K fold, each fold contains index of train, validation and test data
        param: y -> leaf-node labels of all samples
        return: idx_list -> 
        """
        assert self.foldk > 2

        idx_trva_list = []
        idx_te_list = []
        skf = StratifiedKFold(n_splits=self.foldk, random_state=self.seed, shuffle=True)
        for trn_index, te_index in skf.split(self._data['static_data_val'], y):
            idx_trva_list.append(trn_index)
            idx_te_list.append(te_index)

        idx_list = np.empty([self.foldk, 3], dtype=object)
        for i in range(self.foldk):
            idx_list[i][0] = np.setdiff1d(idx_trva_list[i], idx_te_list[(i + 1) % self.foldk], True)
            idx_list[i][1] = idx_te_list[(i + 1) % self.foldk]
            idx_list[i][2] = idx_te_list[i]

        return idx_list

    def _cal_stats_for_folds(self, folds_idx):
        """
        Calculate mean, std of training data based on each fold index
        param: folds_idx -> indices of train, val and test data in K folds
        return: stats_dict -> {'parent-node-0':[array([[array([mean of nts]), array([std of nts])],
                                                       [array([mean of ts]), array([std of ts])]]), 
                                                array([]),...], 
                               'parent-node-1':[]}
        """
        stats_dict = {}
        for parnd, index in folds_idx.items():
            stats_dict[parnd] = []
            for k in range(self.foldk):
                trn_idx = index[k, 0]
                stats_nts, stats_ts = self._cal_stats_of_trn(trn_idx)
                stats_nts_list = [stats_nts[:, 0], stats_nts[:, 1], stats_nts[:, 2], stats_nts[:, 3]]
                stats_ts_list = [stats_ts[:, 0], stats_ts[:, 1], stats_ts[:, 2], stats_ts[:, 3]]

                stats_dict[parnd].append(np.array([stats_nts_list, stats_ts_list]))

        return stats_dict

    def _cal_stats_of_trn(self, trn_idx):
        """
        Calculate the mean, std of training data
        param: trn_idx -> the training data index in k-th fold
        """
        # index data
        tsdata_trn = np.concatenate(self._data["X_t"][trn_idx])
        tsdata_trn_mask = np.concatenate(self._data["X_t_mask"][trn_idx])
        ntsdata_trn_val = self._data["static_data_val"][trn_idx]

        # cal the mean and std of tsdata under each fold
        tsdata_dim = tsdata_trn.shape[1]
        tsdata_stats = np.empty((tsdata_dim, 4)) * np.nan
        for d in range(tsdata_dim):
            d_mask = tsdata_trn_mask[:, d].flatten()
            d_tsdata = tsdata_trn[:, d].flatten()[np.where(d_mask == 1)]
            d_mean = np.nanmean(d_tsdata)
            d_std = np.nanstd(d_tsdata)
            d_max = np.nanmax(d_tsdata)
            d_min = np.nanmin(d_tsdata)
            tsdata_stats[d, :] = np.array([d_mean, d_std, d_max, d_min])
        
        # cal the mean and std of ntsdata under each fold
        ntsdata_dim = ntsdata_trn_val.shape[1]
        ntsdata_stats = np.empty((ntsdata_dim, 4)) * np.nan
        for d in range(ntsdata_dim):
            d_ntsdata = ntsdata_trn_val[:, d].flatten()
            d_mean = np.nanmean(d_ntsdata)
            d_std = np.nanstd(d_ntsdata)
            d_max = np.nanmax(d_ntsdata)
            d_min = np.nanmin(d_ntsdata)
            ntsdata_stats[d, :] = np.array([d_mean, d_std, d_max, d_min])

        return ntsdata_stats, tsdata_stats

    def _filter(self, ts, max_timestamp=None):
        """
        :param ts: An np.array of n np.array with shape (t_i,).
               max_timestamp: an Integer > 0 or None, here is 48, 72, 96 or 120.
        
        :returns A np.array of n Integers, Tts i-th element (x_i) indicates that 
                 we will take the first x_i numbers from i-th data sample.
        """
        if max_timestamp is None:
            ret = np.asarray([len(tt) for tt in ts])
        else:
            ret = np.asarray([np.sum(tt - tt[0] <= max_timestamp*60*60) for tt in ts])
        return ret

    def _padding(self, x, lens):
        """
        :params x: A np.array of n np.array with shape (t_i, d).
                lens: A np.array of n Integers > 0.
        :returns 
                A np.array with shape (n, t, d), where t = lens
        """
        n = len(x)
        t = self.max_step
        d = 1 if x[0].ndim == 1 else x[0].shape[1]
        ret = np.zeros([n, t, d], dtype=float)
        if x[0].ndim == 1:
            for i, xx in enumerate(x):
                ret[i, :lens[i]] = xx[:lens[i], np.newaxis]
        else:
            for i, xx in enumerate(x):
                ret[i, :lens[i]] = xx[:lens[i]]

        return ret   

def masked_fill(X, mask, val):
    """ Like torch.Tensor.masked_fill(), fill elements in given `X` with `val` where `mask` is True.

    Parameters
    ----------
    X : array-like,
        The data vector.

    mask : array-like,
        The boolean mask.

    val : float
        The value to fill in with.

    Returns
    -------
    array,
        mask
    """
    assert X.shape == mask.shape, 'Shapes of X and mask must match, ' \
                                f'but X.shape={X.shape}, mask.shape={mask.shape}'
    assert type(X) == type(mask), 'Data types of X and mask must match, ' \
                                f'but got {type(X)} and {type(mask)}'

    if isinstance(X, list):
        X = np.asarray(X)
        mask = np.asarray(mask)

    if isinstance(X, np.ndarray):
        mask = mask.astype(bool)
        X[mask] = val
    elif isinstance(X, torch.Tensor):
        mask = mask.type(torch.bool)
        X[mask] = val
    else:
        raise TypeError('X must be type of list/numpy.ndarray/torch.Tensor, '
                        f'but got {type(X)}')

    return X






        

