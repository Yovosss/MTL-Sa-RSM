#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch

import helper.logger as logger


class FlatCollator(object):
    def __init__(self, config, label):
        """
        Collator object for the collator_fn in data_modules.data_loader
        :param config: Object
        """
        super(FlatCollator, self).__init__()
        self.config = config
        self.device = config.train.device_setting.device

        # define class dict and n_classes for flat leaf-node classification
        if config.model.type == "PreAttnMMs_FCLN":
            self.classes = list(set([i[-1] for i in label['y_classes_unique']]))
            self.n_classes = len(self.classes)
            self.class_dict = {j:i for i, j in enumerate(self.classes)}

        # define n_classes for flat all-node classification    
        elif config.model.type == "PreAttnMMs_FCAN":
            self.n_classes = label['unique_label_number'] - 1

    def _leaf_node_label(self, batch_label):
        """
        tranform y_classes_unique to leaf-node label
        :params: batch_label, List[List[], List[], ...]-->[[0,1,3], [0,1,5], [0,2,6,8], ...]
        :Return: leaf_node_labels, List[...]-->List[3, 4, 5, 3, 4, 8, 10, ...]
        """
        leaf_node_labels = []
        for label in batch_label:
            leaf_node_labels.append(self.class_dict[label[-1]])

        return leaf_node_labels
    
    def _all_node_label_wo_root(self, batch_label):
        """
        tranform y_classes_unique to all-node label without ROOT node
        :params: batch_label, List[List[], List[], ...]-->[[0,1,3], [0,1,5], [0,2,6,8], ...]
        :Return: all_node_labels without root node, List[List[], List[], ...]-->[[1,0,1,0,0,0,0,0,0,0,0], [1,0,0,0,1,0,0,0,0,0,0],...]
        """
        all_node_labels = np.zeros((len(batch_label), self.n_classes))
        for i, label in enumerate(batch_label):
            for j in label[1:]:
                all_node_labels[i][j-1] = 1

        return all_node_labels

    def __call__(self, batch):
        """
        transform data into batch form for training
        :param batch: [Dict{'X_t': np.array([]),  
                            'X_t_mask': np.array([]), 
                            'deltaT_t': np.array([]), 
                            'X': np.array([]), 
                            'X_t_filledLOCF': np.array([]), 
                            'y_classes': List[List[int]], 
                            'empirical_mean: np.array([])}, ...]
        :return: batch -> Dict{'X_t': torch.FloatTensor,
                               'X_t_mask': torch.FloatTensor,
                               'deltaT_t': torch.FloatTensor,
                               'X': torch.FloatTensor,
                               'X_t_filledLOCF': torch.FloatTensor,
                               'y_classes': List[...],
                               'empirical_mean': torch.FloatTensor}
        """
        batch_X_t = []
        batch_X_t_mask = []
        batch_deltaT_t = []
        batch_X_t_filledLOCF = []
        batch_empirical_mean = []
        batch_X = []
        batch_label = []
        for sample in batch:
            batch_X_t.append(sample['X_t'])
            batch_X_t_mask.append(sample['X_t_mask'])
            batch_deltaT_t.append(sample['deltaT_t'])
            batch_X_t_filledLOCF.append(sample['X_t_filledLOCF'])
            batch_empirical_mean.append(sample['empirical_mean'])
            batch_X.append(sample['X'])
            batch_label.append(sample['y_classes'])

        # extract the leaf-node labels
        if self.config.model.type == "PreAttnMMs_FCLN":
            batch_label_new = self._leaf_node_label(batch_label)
        
        # extract the all-node labels
        elif self.config.model.type == "PreAttnMMs_FCAN":
            batch_label_new = self._all_node_label_wo_root(batch_label)

        return {
            'X': torch.tensor(np.array(batch_X)).to(torch.float32),
            'X_t': torch.tensor(np.array(batch_X_t)).to(torch.float32),
            'X_t_mask': torch.tensor(np.array(batch_X_t_mask)).to(torch.float32),
            'deltaT_t': torch.tensor(np.array(batch_deltaT_t)).to(torch.float32),
            'X_t_filledLOCF': torch.tensor(np.array(batch_X_t_filledLOCF)).to(torch.float32),
            'empirical_mean': torch.tensor(np.array(batch_empirical_mean)).to(torch.float32),
            'label': torch.tensor(batch_label_new).to(torch.long)
        }

class LocalCollator(object):
    def __init__(self, config, label):
        """
        Collator object for the collator_fn in data_modules.data_loader
        :param config: Object
        """
        super(LocalCollator, self).__init__()
        self.config = config
        self.device = config.train.device_setting.device
        self.taxonomy = label['taxonomy']

    def _localize_label(self, batch_labels):
        """
        :param batch_labels: label idx of one batch, List[List[int]], e.g. [[0, 1, 3], [0, 2, 6, 9],...]
        :return batch_local_label: np.array([int]), e.g. np.array([0, 1, ...])
        """
        label_dict = {}
        for idx, value in enumerate(self.taxonomy[self.config.experiment.local_task]):
            label_dict[value] = idx
        
        local_labels = []
        for label in batch_labels:
            for label_idx in label:
                if label_idx in label_dict:
                    local_labels.append(label_dict[label_idx])
                    break

        assert len(local_labels) == len(batch_labels), "The labels are missed during localization, please recheck!"

        return local_labels

    def __call__(self, batch):
        """
        transform data into batch form for training
        :param batch: [Dict{'X_t': np.array([]),  
                            'X_t_mask': np.array([]), 
                            'deltaT_t': np.array([]), 
                            'X': np.array([]), 
                            'X_t_filledLOCF': np.array([]), 
                            'y_classes': List[List[int]], 
                            'empirical_mean: np.array([])}, ...]
        :return: batch -> Dict{'X_t': torch.FloatTensor,
                               'X_t_mask': torch.FloatTensor,
                               'deltaT_t': torch.FloatTensor,
                               'X': torch.FloatTensor,
                               'X_t_filledLOCF': torch.FloatTensor,
                               'y_classes': List[List[int]],
                               'empirical_mean': torch.FloatTensor}
        """
        batch_X_t = []
        batch_X_t_mask = []
        batch_deltaT_t = []
        batch_X_t_filledLOCF = []
        batch_empirical_mean = []
        batch_X = []
        batch_label = []
        for sample in batch:
            batch_X_t.append(sample['X_t'])
            batch_X_t_mask.append(sample['X_t_mask'])
            batch_deltaT_t.append(sample['deltaT_t'])
            batch_X_t_filledLOCF.append(sample['X_t_filledLOCF'])
            batch_empirical_mean.append(sample['empirical_mean'])
            batch_X.append(sample['X'])
            batch_label.append(sample['y_classes'])

        batch_local_label = self._localize_label(batch_label)

        return {
            'X': torch.tensor(np.array(batch_X)).to(torch.float32),
            'X_t': torch.tensor(np.array(batch_X_t)).to(torch.float32),
            'X_t_mask': torch.tensor(np.array(batch_X_t_mask)).to(torch.float32),
            'deltaT_t': torch.tensor(np.array(batch_deltaT_t)).to(torch.float32),
            'X_t_filledLOCF': torch.tensor(np.array(batch_X_t_filledLOCF)).to(torch.float32),
            'empirical_mean': torch.tensor(np.array(batch_empirical_mean)).to(torch.float32),
            'label': torch.tensor(batch_local_label).to(torch.long)
        }

class GlobalCollator(object):
    def __init__(self, config, label):
        """
        Collator object for the collator_fn in data_modules.data_loader
        :param config: Object
        """
        super(GlobalCollator, self).__init__()
        self.config = config
        self.device = config.train.device_setting.device
        if self.config.model.type in ["PreAttnMMs_GCN_MAP_V1"]:
            self.n_classes = label['unique_label_number']
        elif self.config.model.type == "PreAttnMMs_HMCN":
            self.n_classes = label['unique_label_number'] - 1
        elif self.config.model.type in ["PreAttnMMs_MTL_IMP2", "PreAttnMMs_MTL_LCL", "PreAttnMMs_GAT_IMP8_GC", "PreAttnMMs_GAT_IMP8_GC_WeightedLoss"]:
            self.leaf_classes = list(set([i[-1] for i in label['y_classes_unique']]))

    def _globalize_label(self, batch_label):
        """
        Function: generate all classes label matrix for global classification with ROOT node
        Params-->batch_label: List[List[],...]
        Return: global_label: np.array([int,...], [int,...],...)--> np.array([0,1,1,0,0,1,0,0,0,0,0,0],[],...)
        """
        global_label = np.zeros((len(batch_label), self.n_classes))
        for i, label in enumerate(batch_label):
            for j in label:
                global_label[i][j] = 1

        return global_label

    def _globalize_label_wo_root(self, batch_label):
        """
        Function: generate all classes label matrix for global classification without ROOT node
        Params-->batch_label: List[List[],...]
        Return: global_label: np.array([int,...], [int,...],...)--> np.array([0,1,1,0,0,1,0,0,0,0,0],[],...)
        """
        global_label = np.zeros((len(batch_label), self.n_classes))
        for i, label in enumerate(batch_label):
            for j in label[1:]:
                global_label[i][j-1] = 1

        return global_label

    def _seperate_label_for_mtl(self, batch_label):
        """
        Function: seperate y_classes_unique to multi-task form
        Params: batch_label: List[List[],...]
        Return: multi-task label
        """
        mtl_label = [[],
                     [],
                     [],
                     [],
                     []]
        for i, label in enumerate(batch_label):
            if label[-1] == 3:
                mtl_label[0].append(0)
                mtl_label[1].append(0)
                mtl_label[2].append(2)
                mtl_label[3].append(2)
                mtl_label[4].append(2)
            elif label[-1] == 4:
                mtl_label[0].append(0)
                mtl_label[1].append(1)
                mtl_label[2].append(2)
                mtl_label[3].append(2)
                mtl_label[4].append(2)
            elif label[-1] == 5:
                mtl_label[0].append(0)
                mtl_label[1].append(2)
                mtl_label[2].append(2)
                mtl_label[3].append(2)
                mtl_label[4].append(2)
            elif label[-1] == 8:
                mtl_label[0].append(1)
                mtl_label[1].append(3)
                mtl_label[2].append(0)
                mtl_label[3].append(0)
                mtl_label[4].append(2)
            elif label[-1] == 9:
                mtl_label[0].append(1)
                mtl_label[1].append(3)
                mtl_label[2].append(0)
                mtl_label[3].append(1)
                mtl_label[4].append(2)
            elif label[-1] == 10:
                mtl_label[0].append(1)
                mtl_label[1].append(3)
                mtl_label[2].append(1)
                mtl_label[3].append(2)
                mtl_label[4].append(0)
            elif label[-1] == 11:
                mtl_label[0].append(1)
                mtl_label[1].append(3)
                mtl_label[2].append(1)
                mtl_label[3].append(2)
                mtl_label[4].append(1)

        return mtl_label                

    def _seperate_label_for_mtl_clc(self, batch_label):
        """
        Function: seperate y_classes_unique to multi-task LCL form
        Params: batch_label: List[List[],...]
        Return: multi-task LCL label
        """
        # three sub-list represent the three layer of FUO hierarchy
        mtl_label = [[],
                     [],
                     []]
        for i, label in enumerate(batch_label):
            if label[-1] == 3:
                mtl_label[0].append(0)
                mtl_label[1].append(0)
                mtl_label[2].append(0)
            elif label[-1] == 4:
                mtl_label[0].append(0)
                mtl_label[1].append(1)
                mtl_label[2].append(1)
            elif label[-1] == 5:
                mtl_label[0].append(0)
                mtl_label[1].append(2)
                mtl_label[2].append(2)
            elif label[-1] == 8:
                mtl_label[0].append(1)
                mtl_label[1].append(3)
                mtl_label[2].append(3)
            elif label[-1] == 9:
                mtl_label[0].append(1)
                mtl_label[1].append(3)
                mtl_label[2].append(4)
            elif label[-1] == 10:
                mtl_label[0].append(1)
                mtl_label[1].append(4)
                mtl_label[2].append(5)
            elif label[-1] == 11:
                mtl_label[0].append(1)
                mtl_label[1].append(4)
                mtl_label[2].append(6)

        return mtl_label  

    def __call__(self, batch):
        """
        transform data into batch form for training
        :param batch: [Dict{'X_t': np.array([]),  
                            'X_t_mask': np.array([]), 
                            'deltaT_t': np.array([]), 
                            'X': np.array([]), 
                            'X_t_filledLOCF': np.array([]), 
                            'y_classes': List[List[int]], 
                            'empirical_mean: np.array([])}, ...]
        :return: batch -> Dict{'X_t': torch.FloatTensor,
                               'X_t_mask': torch.FloatTensor,
                               'deltaT_t': torch.FloatTensor,
                               'X': torch.FloatTensor,
                               'X_t_filledLOCF': torch.FloatTensor,
                               'y_classes': List[List[int]],
                               'empirical_mean': torch.FloatTensor}
        """
        batch_X_t = []
        batch_X_t_mask = []
        batch_deltaT_t = []
        batch_X_t_filledLOCF = []
        batch_empirical_mean = []
        batch_X = []
        batch_label = []
        batch_node_inputs = []
        for sample in batch:
            batch_X_t.append(sample['X_t'])
            batch_X_t_mask.append(sample['X_t_mask'])
            batch_deltaT_t.append(sample['deltaT_t'])
            batch_X_t_filledLOCF.append(sample['X_t_filledLOCF'])
            batch_empirical_mean.append(sample['empirical_mean'])
            batch_X.append(sample['X'])
            batch_label.append(sample['y_classes'])
            batch_node_inputs.append(sample['label_node_inputs'])
        
        if self.config.model.type in ["PreAttnMMs_GCN_MAP_V1"]: 
            batch_label_new = self._globalize_label(batch_label)

        elif self.config.model.type == "PreAttnMMs_HMCN":
            batch_label_new = self._globalize_label_wo_root(batch_label)
        
        elif self.config.model.type in ["PreAttnMMs_MTL_IMP2", "PreAttnMMs_GAT_IMP8_GC", "PreAttnMMs_GAT_IMP8_GC_WeightedLoss"]:
            batch_label_new = self._seperate_label_for_mtl(batch_label)

        elif self.config.model.type == "PreAttnMMs_MTL_LCL":
            batch_label_new = self._seperate_label_for_mtl_clc(batch_label)

        return {
            'X': torch.tensor(np.array(batch_X)).to(torch.float32),
            'X_t': torch.tensor(np.array(batch_X_t)).to(torch.float32),
            'X_t_mask': torch.tensor(np.array(batch_X_t_mask)).to(torch.float32),
            'deltaT_t': torch.tensor(np.array(batch_deltaT_t)).to(torch.float32),
            'X_t_filledLOCF': torch.tensor(np.array(batch_X_t_filledLOCF)).to(torch.float32),
            'empirical_mean': torch.tensor(np.array(batch_empirical_mean)).to(torch.float32),
            'node_input': torch.tensor(np.array(batch_node_inputs)).to(torch.long),
            'label': torch.tensor(batch_label_new).to(torch.long)
        }