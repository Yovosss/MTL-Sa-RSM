#!/usr/bin/env python
# coding:utf-8

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(self, config, in_features):
        """
        :params config: Dict
        :params input_dim: the dimension of static inputs
        """
        super(MLP, self).__init__()
        self.config = config
        self.in_features = in_features

        layers = []
        layers.append(nn.Linear(in_features, config.model.static_encoder.layer1.dimension))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(config.model.static_encoder.layer1.dropout))

        if config.model.static_encoder.num_layer == 2:
            layers.append(nn.Linear(config.model.static_encoder.layer1.dimension, config.model.static_encoder.layer2.dimension))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.model.static_encoder.layer2.dropout))
        
        if config.model.static_encoder.num_layer == 3:
            layers.append(nn.Linear(config.model.static_encoder.layer1.dimension, config.model.static_encoder.layer2.dimension))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.model.static_encoder.layer2.dropout))

            layers.append(nn.Linear(config.model.static_encoder.layer2.dimension, config.model.static_encoder.layer3.dimension))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.model.static_encoder.layer3.dropout))

        self.mlp = nn.Sequential(*layers)
        if config.model.static_encoder.num_layer == 1:
            self.out_features = config.model.static_encoder.layer1.dimension
        elif config.model.static_encoder.num_layer == 2:
            self.out_features = config.model.static_encoder.layer2.dimension
        elif config.model.static_encoder.num_layer == 3:
            self.out_features = config.model.static_encoder.layer3.dimension
        
    def forward(self, inputs):

        x = self.mlp(inputs)

        return x

class MLPHp(nn.Module):
    def __init__(self, trial, config, in_features):
        """
        :params config: Dict
        :params input_dim: the dimension of static inputs
        """
        super(MLPHp, self).__init__()
        self.config = config
        self.in_features = in_features

        # We optimize the number of layers, hidden units and dropout ratio in each layer.
        n_layers = trial.suggest_int("mlp_nlayers", 1, 2)
        layers = []

        for i in range(n_layers):
            out_features = trial.suggest_int("mlp_units_l{}".format(i), 128, 512, step=16)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            p = trial.suggest_float("mlp_dropout_l{}".format(i), 0.0, 0.5, step=0.05)
            layers.append(nn.Dropout(p))

            in_features = out_features
        
        self.out_features = out_features
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, inputs):

        x = self.mlp(inputs)

        return x

class MLP_IMP5(nn.Module):
    def __init__(self, config, in_features, dim_0, dropout_0, dim_1, dropout_1):
        """
        :params config: Dict
        :params input_dim: the dimension of static inputs
        """
        super(MLP_IMP5, self).__init__()
        self.config = config
        self.in_features = in_features

        layers = []
        layers.append(nn.Linear(in_features, dim_0))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout_0))

        layers.append(nn.Linear(dim_0, dim_1))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout_1))

        self.mlp = nn.Sequential(*layers)
        
    def forward(self, inputs):

        x = self.mlp(inputs)

        return x