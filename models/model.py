#!/usr/bin/env python
# coding:utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import PreSpatialAttn
from models.gat import (GATlayer_IMP8, GATlayer_IMP8_Hp)
from models.gcn import GraphConvolution
from models.grud_layer import GRUD_cell, GRUD_cells
from models.mlp import MLP, MLP_IMP5, MLPHp

class PreAttnMMs(nn.Module):
    def __init__(self, config, n_steps, n_t_features, n_features, n_classes):
        super(PreAttnMMs, self).__init__()
        self.config = config
        self.device = config.train.device_setting.device

        self.n_steps = n_steps
        self.n_t_features = n_t_features
        self.n_features = n_features
        self.n_classes = n_classes

        self.rnn_hidden_size_1 = config.model.temporal_encoder.layer1.rnn_hidden_size
        self.rnn_hidden_size_2 = config.model.temporal_encoder.layer2.rnn_hidden_size

        self.static_encoder = MLP(config, n_features)
        self.pre_attention = PreSpatialAttn(config, n_t_features)
        self.grud = GRUD_cells(config=config, 
                              input_dim=n_t_features, 
                              hidden_dim=self.rnn_hidden_size_1,
                              batch_size=config.train.batch_size,
                              dropout=config.model.temporal_encoder.layer1.dropout,
                              dropout_type=config.model.temporal_encoder.layer1.dropout_type)

        if config.model.temporal_encoder.num_layer > 1:
            self.gru = torch.nn.GRU(input_size = self.rnn_hidden_size_1, 
                                    hidden_size = self.rnn_hidden_size_2,
                                    batch_first =True,
                                    dropout=config.model.temporal_encoder.layer2.dropout)

            self.classifier = nn.Linear(self.rnn_hidden_size_2 + self.static_encoder.out_features, n_classes)
        else:
            self.classifier = nn.Linear(self.rnn_hidden_size_1 + self.static_encoder.out_features, n_classes)

    def forward(self, inputs):
        """
        :param: inputs: Dict, including X_t, 
                                        X_t_mask, 
                                        deltaT_t, 
                                        empirical_mean, 
                                        X_t_filledLOCF
        """
        x = inputs['X']
        values = inputs["X_t"]
        masks = inputs["X_t_mask"]
        deltas = inputs["deltaT_t"]
        empirical_mean = inputs["empirical_mean"]
        X_filledLOCF = inputs["X_t_filledLOCF"]

        # encode the static features
        static_output = self.static_encoder(x)

        # encode the temporal features
        attention_outputs = self.pre_attention(values)
        hidden_states, hidden_state = self.grud(attention_outputs, masks, deltas, empirical_mean, X_filledLOCF)

        if self.config.model.temporal_encoder.num_layer > 1:
            output, hidden_state = self.gru(hidden_states)
        
        # concatenate
        embedding = torch.cat([static_output, hidden_state.squeeze(0)], dim=1)
        logits = self.classifier(embedding)

        return logits

class PreAttnMMsHp(nn.Module):
    def __init__(self, trial, config, n_steps, n_t_features, n_features, n_classes):
        super(PreAttnMMsHp, self).__init__()
        self.config = config
        self.device = config.train.device_setting.device

        self.n_steps = n_steps
        self.n_t_features = n_t_features
        self.n_features = n_features
        self.n_classes = n_classes

        self.rnn_hidden_size = trial.suggest_int("rnn_hidden_size", 64, 256, step=16)
        dropout_type = trial.suggest_categorical("dropout_type", ["None", "Gal", "Moon", "mloss"])
        if dropout_type == "None":
            dropout = 0.0
        elif dropout_type == "Gal":
            dropout = trial.suggest_float("dropout_gal", 0.0, 0.5, step=0.05)
        elif dropout_type == "Moon":
            dropout = trial.suggest_float("dropout_Moon", 0.0, 0.5, step=0.05)
        elif dropout_type == "mloss":
            dropout = trial.suggest_float("dropout_mloss", 0.0, 0.5, step=0.05)

        self.static_encoder = MLPHp(trial, config, n_features)
        self.pre_attention = PreSpatialAttn(config, n_t_features)
        self.grud = GRUD_cells(config=config, 
                              input_dim=n_t_features, 
                              hidden_dim=self.rnn_hidden_size,
                              batch_size=config.train.batch_size,
                              dropout=dropout,
                              dropout_type=dropout_type)

        self.gru_layer = trial.suggest_categorical("gru_layer", [True, False])
        if self.gru_layer == True:
            hidden_size = trial.suggest_int("rnn_hidden_size_1", 64, 256, step=16)
            dropout_1 = trial.suggest_float("dropout_1", 0.0, 0.5, step=0.05)
            self.gru = torch.nn.GRU(input_size = self.rnn_hidden_size, 
                                    hidden_size = hidden_size,
                                    batch_first =True,
                                    dropout=dropout_1)

            self.classifier = nn.Linear(hidden_size + self.static_encoder.out_features, n_classes)
        else:
            self.classifier = nn.Linear(self.rnn_hidden_size + self.static_encoder.out_features, n_classes)

    def forward(self, inputs):
        """
        :param: inputs: Dict, including X_t, 
                                        X_t_mask, 
                                        deltaT_t, 
                                        empirical_mean, 
                                        X_t_filledLOCF
        """
        x = inputs['X']
        values = inputs["X_t"]
        masks = inputs["X_t_mask"]
        deltas = inputs["deltaT_t"]
        empirical_mean = inputs["empirical_mean"]
        X_filledLOCF = inputs["X_t_filledLOCF"]

        # encode the static features
        static_output = self.static_encoder(x)

        # encode the temporal features
        attention_outputs = self.pre_attention(values)
        hidden_states, hidden_state = self.grud(attention_outputs, masks, deltas, empirical_mean, X_filledLOCF)

        if self.gru_layer == True:
            output, hidden_state = self.gru(hidden_states)
        
        # concatenate
        embedding = torch.cat([static_output, hidden_state.squeeze(0)], dim=1)
        logits = self.classifier(embedding)

        return logits

class PreAttnMMs_FCLN(nn.Module):
    def __init__(self, config, n_steps, n_t_features, n_features, n_classes):
        """
        Flat classifier for all-node classification
        """
        super(PreAttnMMs_FCLN, self).__init__()
        self.config = config
        self.device = config.train.device_setting.device

        self.n_steps = n_steps
        self.n_t_features = n_t_features
        self.n_features = n_features
        self.n_classes = n_classes

        self.rnn_hidden_size_1 = config.model.temporal_encoder.layer1.rnn_hidden_size
        self.rnn_hidden_size_2 = config.model.temporal_encoder.layer2.rnn_hidden_size

        self.static_encoder = MLP(config, n_features)
        self.pre_attention = PreSpatialAttn(config, n_t_features)
        self.grud = GRUD_cells(config=config, 
                              input_dim=n_t_features, 
                              hidden_dim=self.rnn_hidden_size_1,
                              batch_size=config.train.batch_size,
                              dropout=config.model.temporal_encoder.layer1.dropout,
                              dropout_type=config.model.temporal_encoder.layer1.dropout_type)

        if config.model.temporal_encoder.num_layer > 1:
            self.gru = torch.nn.GRU(input_size = self.rnn_hidden_size_1, 
                                    hidden_size = self.rnn_hidden_size_2,
                                    batch_first =True,
                                    dropout=config.model.temporal_encoder.layer2.dropout)
            self.output_dim = self.rnn_hidden_size_2 + self.static_encoder.out_features
        else:
            self.output_dim = self.rnn_hidden_size_1 + self.static_encoder.out_features

        if config.model.fully_connected_layer.num_layer == 1:
            self.fc = nn.Linear(self.output_dim, config.model.fully_connected_layer.dimension)
            self.dropout_fc = nn.Dropout(config.model.fully_connected_layer.dropout)
            self.classifier = nn.Linear(config.model.fully_connected_layer.dimension, self.n_classes)
        else:
            self.classifier = nn.Linear(self.output_dim, self.n_classes)
                
    def forward(self, inputs):
        """
        :param: inputs: Dict, including X_t, 
                                        X_t_mask, 
                                        deltaT_t, 
                                        empirical_mean, 
                                        X_t_filledLOCF
        """
        x = inputs['X']
        values = inputs["X_t"]
        masks = inputs["X_t_mask"]
        deltas = inputs["deltaT_t"]
        empirical_mean = inputs["empirical_mean"]
        X_filledLOCF = inputs["X_t_filledLOCF"]

        # encode the static features
        static_output = self.static_encoder(x)

        # encode the temporal features
        attention_outputs = self.pre_attention(values)
        hidden_states, hidden_state = self.grud(attention_outputs, masks, deltas, empirical_mean, X_filledLOCF)

        if self.config.model.temporal_encoder.num_layer > 1:
            output, hidden_state = self.gru(hidden_states)
        
        # concatenate
        embedding = torch.cat([static_output, hidden_state.squeeze(0)], dim=1)

        # FC layer
        if self.config.model.fully_connected_layer.num_layer == 1:
            x = F.relu(self.fc(embedding))
            x = self.dropout_fc(x)
            logits = self.classifier(x)

        else:
            logits = self.classifier(embedding)

        return logits

class PreAttnMMs_FCLN_Hp(nn.Module):
    def __init__(self, trial, config, n_steps, n_t_features, n_features, n_classes):
        super(PreAttnMMs_FCLN_Hp, self).__init__()
        self.config = config
        self.device = config.train.device_setting.device

        self.n_steps = n_steps
        self.n_t_features = n_t_features
        self.n_features = n_features
        self.n_classes = n_classes

        rnn_hidden_size = trial.suggest_int("rnn_hidden_size", 64, 256, step=16)
        dropout_type = trial.suggest_categorical("grud_dropout_type", ["None", "Gal", "Moon", "mloss"])
        if dropout_type == "None":
            dropout = 0.0
        elif dropout_type == "Gal":
            dropout = trial.suggest_float("dropout_gal", 0.0, 0.5, step=0.05)
        elif dropout_type == "Moon":
            dropout = trial.suggest_float("dropout_Moon", 0.0, 0.5, step=0.05)
        elif dropout_type == "mloss":
            dropout = trial.suggest_float("dropout_mloss", 0.0, 0.5, step=0.05)

        self.static_encoder = MLPHp(trial, config, n_features)
        self.pre_attention = PreSpatialAttn(config, n_t_features)
        self.grud = GRUD_cells(config=config, 
                              input_dim=n_t_features, 
                              hidden_dim=rnn_hidden_size,
                              batch_size=config.train.batch_size,
                              dropout=dropout,
                              dropout_type=dropout_type)
        self.gru_layer = trial.suggest_categorical("gru_layer", [True, False])
        if self.gru_layer == True:
            rnn_hidden_size_1 = trial.suggest_int("rnn_hidden_size_1", 64, 256, step=16)
            dropout_gru = trial.suggest_float("dropout_gru", 0.0, 0.5, step=0.05)
            self.gru = torch.nn.GRU(input_size = rnn_hidden_size, 
                                    hidden_size = rnn_hidden_size_1,
                                    batch_first =True,
                                    dropout=dropout_gru)
            self.output_dim = rnn_hidden_size_1 + self.static_encoder.out_features
        else:
            self.output_dim = rnn_hidden_size + self.static_encoder.out_features

        self.fc_layer = trial.suggest_categorical("fc_layer", [True, False])
        if self.fc_layer == True:
            fc_dimension = trial.suggest_int("fc_dimension", 64, 512, step=16)
            fc_dropout = trial.suggest_float("fc_dropout", 0.0, 0.5, step=0.05)
            self.fc = nn.Linear(self.output_dim, fc_dimension)
            self.dropout_fc = nn.Dropout(fc_dropout)
            self.classifier = nn.Linear(fc_dimension, self.n_classes)
        else:
            self.classifier = nn.Linear(self.output_dim, self.n_classes)

    def forward(self, inputs):
        """
        :param: inputs: Dict, including X_t, 
                                        X_t_mask, 
                                        deltaT_t, 
                                        empirical_mean, 
                                        X_t_filledLOCF
        """
        x = inputs['X']
        values = inputs["X_t"]
        masks = inputs["X_t_mask"]
        deltas = inputs["deltaT_t"]
        empirical_mean = inputs["empirical_mean"]
        X_filledLOCF = inputs["X_t_filledLOCF"]

        # encode the static features
        static_output = self.static_encoder(x)

        # encode the temporal features
        attention_outputs = self.pre_attention(values)
        hidden_states, hidden_state = self.grud(attention_outputs, masks, deltas, empirical_mean, X_filledLOCF)

        if self.gru_layer:
            output, hidden_state = self.gru(hidden_states)
        
        # concatenate
        embedding = torch.cat([static_output, hidden_state.squeeze(0)], dim=1)
        if self.fc_layer:
            x = F.relu(self.fc(embedding))
            x = self.dropout_fc(x)
            logits = self.classifier(x)
        else:
            logits = self.classifier(embedding)

        return logits

class PreAttnMMs_FCAN(nn.Module):
    def __init__(self, config, n_steps, n_t_features, n_features, n_classes):
        """
        Flat classifier for all-node classification
        """
        super(PreAttnMMs_FCAN, self).__init__()
        self.config = config
        self.device = config.train.device_setting.device

        self.n_steps = n_steps
        self.n_t_features = n_t_features
        self.n_features = n_features
        self.n_classes = n_classes

        self.rnn_hidden_size_1 = config.model.temporal_encoder.layer1.rnn_hidden_size
        self.rnn_hidden_size_2 = config.model.temporal_encoder.layer2.rnn_hidden_size

        self.static_encoder = MLP(config, n_features)
        self.pre_attention = PreSpatialAttn(config, n_t_features)
        self.grud = GRUD_cells(config=config, 
                              input_dim=n_t_features, 
                              hidden_dim=self.rnn_hidden_size_1,
                              batch_size=config.train.batch_size,
                              dropout=config.model.temporal_encoder.layer1.dropout,
                              dropout_type=config.model.temporal_encoder.layer1.dropout_type)

        if config.model.temporal_encoder.num_layer > 1:
            self.gru = torch.nn.GRU(input_size = self.rnn_hidden_size_1, 
                                    hidden_size = self.rnn_hidden_size_2,
                                    batch_first =True,
                                    dropout=config.model.temporal_encoder.layer2.dropout)
            if config.model.fully_connected_layer.num_layer == 1:
                self.fc = nn.Linear(self.rnn_hidden_size_2 + self.static_encoder.out_features, 
                                    config.model.fully_connected_layer.dimension)
                self.dropout_fc = nn.Dropout(config.model.fully_connected_layer.dropout)
                self.classifier = nn.Linear(config.model.fully_connected_layer.dimension, self.n_classes)
            else:
                self.classifier = nn.Linear(self.rnn_hidden_size_2 + self.static_encoder.out_features, 
                                            self.n_classes)
        else:
            if config.model.fully_connected_layer.num_layer == 1:
                self.fc = nn.Linear(self.rnn_hidden_size_1 + self.static_encoder.out_features, 
                                    config.model.fully_connected_layer.dimension)
                self.dropout_fc = nn.Dropout(config.model.fully_connected_layer.dropout)
                self.classifier = nn.Linear(config.model.fully_connected_layer.dimension, self.n_classes)
            else:
                self.classifier = nn.Linear(self.rnn_hidden_size_1 + self.static_encoder.out_features, self.n_classes)
    def forward(self, inputs):
        """
        :param: inputs: Dict, including X_t, 
                                        X_t_mask, 
                                        deltaT_t, 
                                        empirical_mean, 
                                        X_t_filledLOCF
        """
        x = inputs['X']
        values = inputs["X_t"]
        masks = inputs["X_t_mask"]
        deltas = inputs["deltaT_t"]
        empirical_mean = inputs["empirical_mean"]
        X_filledLOCF = inputs["X_t_filledLOCF"]

        # encode the static features
        static_output = self.static_encoder(x)

        # encode the temporal features
        attention_outputs = self.pre_attention(values)
        hidden_states, hidden_state = self.grud(attention_outputs, masks, deltas, empirical_mean, X_filledLOCF)

        if self.config.model.temporal_encoder.num_layer > 1:
            output, hidden_state = self.gru(hidden_states)
        
        # concatenate
        embedding = torch.cat([static_output, hidden_state.squeeze(0)], dim=1)

        # FC layer
        if self.config.model.fully_connected_layer.num_layer == 1:
            x = F.relu(self.fc(embedding))
            x = self.dropout_fc(x)
            logits = self.classifier(x)

        else:
            logits = self.classifier(embedding)

        return logits

class PreAttnMMs_FCAN_Hp(nn.Module):
    def __init__(self, trial, config, n_steps, n_t_features, n_features, n_classes):
        super(PreAttnMMs_FCAN_Hp, self).__init__()
        self.config = config
        self.device = config.train.device_setting.device

        self.n_steps = n_steps
        self.n_t_features = n_t_features
        self.n_features = n_features
        self.n_classes = n_classes

        rnn_hidden_size = trial.suggest_int("rnn_hidden_size", 64, 256, step=16)
        dropout_type = trial.suggest_categorical("grud_dropout_type", ["None", "Gal", "Moon", "mloss"])
        if dropout_type == "None":
            dropout = 0.0
        elif dropout_type == "Gal":
            dropout = trial.suggest_float("dropout_gal", 0.0, 0.5, step=0.05)
        elif dropout_type == "Moon":
            dropout = trial.suggest_float("dropout_Moon", 0.0, 0.5, step=0.05)
        elif dropout_type == "mloss":
            dropout = trial.suggest_float("dropout_mloss", 0.0, 0.5, step=0.05)
        
        self.static_encoder = MLPHp(trial, config, n_features)
        self.pre_attention = PreSpatialAttn(config, n_t_features)
        self.grud = GRUD_cells(config=config, 
                              input_dim=n_t_features, 
                              hidden_dim=rnn_hidden_size,
                              batch_size=config.train.batch_size,
                              dropout=dropout,
                              dropout_type=dropout_type)
        
        self.gru_layer = trial.suggest_categorical("gru_layer", [True, False])
        if self.gru_layer == True:
            rnn_hidden_size_1 = trial.suggest_int("rnn_hidden_size_1", 64, 256, step=16)
            dropout_gru = trial.suggest_float("dropout_gru", 0.0, 0.5, step=0.05)
            self.gru = torch.nn.GRU(input_size = rnn_hidden_size, 
                                    hidden_size = rnn_hidden_size_1,
                                    batch_first =True,
                                    dropout=dropout_gru)
            self.fc_layer = trial.suggest_categorical("fc_layer_0", [True, False])
            if self.fc_layer == True:
                fc_dimension = trial.suggest_int("fc_dimension_0", 64, 512, step=16)
                self.dropout_fc = nn.Dropout(trial.suggest_float("fc_dropout_0", 0.0, 0.5, step=0.05))
                self.fc = nn.Linear(rnn_hidden_size_1 + self.static_encoder.out_features, fc_dimension)
                self.classifier = nn.Linear(fc_dimension, self.n_classes)
            else:
                self.classifier = nn.Linear(rnn_hidden_size_1 + self.static_encoder.out_features, self.n_classes)
        else:
            self.fc_layer = trial.suggest_categorical("fc_layer_1", [True, False])
            if self.fc_layer == True:
                fc_dimension = trial.suggest_int("fc_dimension_1", 64, 512, step=16)
                self.dropout_fc = nn.Dropout(trial.suggest_float("fc_dropout_1", 0.0, 0.5, step=0.05))
                self.fc = nn.Linear(rnn_hidden_size + self.static_encoder.out_features, fc_dimension)
                self.classifier = nn.Linear(fc_dimension, self.n_classes)
            else:
                self.classifier = nn.Linear(rnn_hidden_size + self.static_encoder.out_features, self.n_classes)

    def forward(self, inputs):
        """
        :param: inputs: Dict, including X_t, 
                                        X_t_mask, 
                                        deltaT_t, 
                                        empirical_mean, 
                                        X_t_filledLOCF
        """
        x = inputs['X']
        values = inputs["X_t"]
        masks = inputs["X_t_mask"]
        deltas = inputs["deltaT_t"]
        empirical_mean = inputs["empirical_mean"]
        X_filledLOCF = inputs["X_t_filledLOCF"]

        # encode the static features
        static_output = self.static_encoder(x)

        # encode the temporal features
        attention_outputs = self.pre_attention(values)
        hidden_states, hidden_state = self.grud(attention_outputs, masks, deltas, empirical_mean, X_filledLOCF)

        if self.gru_layer:
            output, hidden_state = self.gru(hidden_states)
        
        # concatenate
        embedding = torch.cat([static_output, hidden_state.squeeze(0)], dim=1)
        if self.fc_layer:
            x = F.relu(self.fc(embedding))
            x = self.dropout_fc(x)
            logits = self.classifier(x)
        else:
            logits = self.classifier(embedding)

        return logits

class PreAttnMMs_HMCN(nn.Module):
    def __init__(self, config, n_steps, n_t_features, n_features, n_classes):
        super(PreAttnMMs_HMCN, self).__init__()
        self.config = config
        self.device = config.train.device_setting.device

        self.n_steps = n_steps
        self.n_t_features = n_t_features
        self.n_features = n_features
        self.local_classes = n_classes[0]
        self.global_classes = n_classes[1]

        # static encoder
        self.static_encoder = MLP(config, n_features)

        # temporal encoder
        self.rnn_hidden_size_1 = config.model.temporal_encoder.layer1.rnn_hidden_size
        self.rnn_hidden_size_2 = config.model.temporal_encoder.layer2.rnn_hidden_size
        self.pre_attention = PreSpatialAttn(config, n_t_features)
        self.grud = GRUD_cells(config=config, 
                              input_dim=n_t_features, 
                              hidden_dim=self.rnn_hidden_size_1,
                              batch_size=config.train.batch_size,
                              dropout=config.model.temporal_encoder.layer1.dropout,
                              dropout_type=config.model.temporal_encoder.layer1.dropout_type)

        if config.model.temporal_encoder.num_layer > 1:
            self.gru = torch.nn.GRU(input_size = self.rnn_hidden_size_1, 
                                    hidden_size = self.rnn_hidden_size_2,
                                    batch_first =True,
                                    dropout=config.model.temporal_encoder.layer2.dropout)

            self.output_dim = self.rnn_hidden_size_2 + self.static_encoder.out_features
        else:
            self.output_dim = self.rnn_hidden_size_1 + self.static_encoder.out_features

        # HMCN block
        self.beta = config.model.hmcn_block.beta
        self.layer_num = len(self.local_classes)
        neuron_each_layer = [config.model.hmcn_block.hidden_dim] * len(self.local_classes)
        neuron_each_local_l1 = [config.model.hmcn_block.hidden_dim] * len(self.local_classes)

        self.linear_layers = nn.ModuleList([])
        self.local_linear_l1 = nn.ModuleList([])
        self.local_linear_l2 = nn.ModuleList([])

        self.batchnorms = nn.ModuleList([])
        self.batchnorms_local_1 = nn.ModuleList([])
        for idx, neuron_number in enumerate(neuron_each_layer):
            if idx == 0:
                self.linear_layers.append(
                    nn.Linear(self.output_dim, neuron_number))
            else:
                self.linear_layers.append(
                    nn.Linear(neuron_each_layer[idx - 1] + self.output_dim, neuron_number))
            self.batchnorms.append(nn.BatchNorm1d(neuron_number))

        for idx, neuron_number in enumerate(neuron_each_local_l1):
            self.local_linear_l1.append(
                nn.Linear(neuron_each_layer[idx], neuron_each_local_l1[idx]))
            self.batchnorms_local_1.append(nn.BatchNorm1d(neuron_each_local_l1[idx]))

        for idx, neuron_numnber in enumerate(self.local_classes):
            self.local_linear_l2.append(
                nn.Linear(neuron_each_local_l1[idx], self.local_classes[idx]))

        self.final_linear_layer = nn.Linear(neuron_each_layer[-1] + self.output_dim, self.global_classes)

    def forward(self, inputs):
        """
        :param: inputs: Dict, including X_t, 
                                        X_t_mask, 
                                        deltaT_t, 
                                        empirical_mean, 
                                        X_t_filledLOCF
        """
        x = inputs['X']
        values = inputs["X_t"]
        masks = inputs["X_t_mask"]
        deltas = inputs["deltaT_t"]
        empirical_mean = inputs["empirical_mean"]
        X_filledLOCF = inputs["X_t_filledLOCF"]

        # encode the static features
        static_output = self.static_encoder(x)

        # encode the temporal features
        attention_outputs = self.pre_attention(values)
        hidden_states, hidden_state = self.grud(attention_outputs, masks, deltas, empirical_mean, X_filledLOCF)

        if self.config.model.temporal_encoder.num_layer > 1:
            output, hidden_state = self.gru(hidden_states)
        
        # concatenate
        embedding = torch.cat([static_output, hidden_state.squeeze(0)], dim=1) # torch.Size([64, 240])

        # hmcn block
        local_outputs = []
        output = embedding
        for layer_idx, layer in enumerate(self.linear_layers):
            if layer_idx == 0:
                output = layer(output)
                output = F.relu(output)
            else:
                output = layer(torch.cat([output, embedding], dim=1))
                output = F.relu(output)
            output = self.batchnorms[layer_idx](output)

            local_output = self.local_linear_l1[layer_idx](output) # torch.Size([64, 512])
            local_output = F.relu(local_output)
            local_output = self.batchnorms_local_1[layer_idx](output)
            
            local_output = self.local_linear_l2[layer_idx](local_output)
            local_outputs.append(local_output)

        global_outputs = torch.sigmoid(
            self.final_linear_layer(torch.cat([output, embedding], dim=1)))

        local_outputs = torch.sigmoid(torch.cat(local_outputs, dim=1))

        output = self.beta * global_outputs + (1 - self.beta) * local_outputs

        return output

class PreAttnMMs_HMCN_Hp(nn.Module):
    def __init__(self, trial, config, n_steps, n_t_features, n_features, n_classes):
        super(PreAttnMMs_HMCN_Hp, self).__init__()
        self.config = config
        self.device = config.train.device_setting.device

        self.n_steps = n_steps
        self.n_t_features = n_t_features
        self.n_features = n_features
        self.local_classes = n_classes[0]
        self.global_classes = n_classes[1]

        rnn_hidden_size = trial.suggest_int("rnn_hidden_size", 64, 256, step=16)
        dropout_type = trial.suggest_categorical("grud_dropout_type", ["None", "Gal", "Moon", "mloss"])
        if dropout_type == "None":
            dropout = 0.0
        elif dropout_type == "Gal":
            dropout = trial.suggest_float("dropout_gal", 0.0, 0.5, step=0.05)
        elif dropout_type == "Moon":
            dropout = trial.suggest_float("dropout_Moon", 0.0, 0.5, step=0.05)
        elif dropout_type == "mloss":
            dropout = trial.suggest_float("dropout_mloss", 0.0, 0.5, step=0.05)
        
        self.static_encoder = MLPHp(trial, config, n_features)
        self.pre_attention = PreSpatialAttn(config, n_t_features)
        self.grud = GRUD_cells(config=config, 
                              input_dim=n_t_features, 
                              hidden_dim=rnn_hidden_size,
                              batch_size=config.train.batch_size,
                              dropout=dropout,
                              dropout_type=dropout_type)

        self.gru_layer = trial.suggest_categorical("gru_layer", [True, False])
        if self.gru_layer == True:
            rnn_hidden_size_1 = trial.suggest_int("rnn_hidden_size_1", 64, 256, step=16)
            dropout_gru = trial.suggest_float("dropout_gru", 0.0, 0.5, step=0.05)
            self.gru = torch.nn.GRU(input_size = rnn_hidden_size, 
                                    hidden_size = rnn_hidden_size_1,
                                    batch_first =True,
                                    dropout=dropout_gru)
            self.output_dim = rnn_hidden_size_1 + self.static_encoder.out_features
        else:
            self.output_dim = rnn_hidden_size + self.static_encoder.out_features

        # HMCN block
        self.beta = trial.suggest_float("beta", 0.0, 1.0, step=0.05)
        self.hmcn_hidden_dim = trial.suggest_int("hmcn_hidden_dim", 256, 1024, step=32)
        self.layer_num = len(self.local_classes)
        neuron_each_layer = [self.hmcn_hidden_dim] * len(self.local_classes)
        neuron_each_local_l1 = [self.hmcn_hidden_dim] * len(self.local_classes)

        self.linear_layers = nn.ModuleList([])
        self.local_linear_l1 = nn.ModuleList([])
        self.local_linear_l2 = nn.ModuleList([])
        self.batchnorms = nn.ModuleList([])
        self.batchnorms_local_1 = nn.ModuleList([])
        for idx, neuron_number in enumerate(neuron_each_layer):
            if idx == 0:
                self.linear_layers.append(
                    nn.Linear(self.output_dim, neuron_number))
            else:
                self.linear_layers.append(
                    nn.Linear(neuron_each_layer[idx - 1] + self.output_dim, neuron_number))
            self.batchnorms.append(nn.BatchNorm1d(neuron_number))

        for idx, neuron_number in enumerate(neuron_each_local_l1):
            self.local_linear_l1.append(
                nn.Linear(neuron_each_layer[idx], neuron_each_local_l1[idx]))
            self.batchnorms_local_1.append(nn.BatchNorm1d(neuron_each_local_l1[idx]))

        for idx, neuron_numnber in enumerate(self.local_classes):
            self.local_linear_l2.append(
                nn.Linear(neuron_each_local_l1[idx], self.local_classes[idx]))

        self.final_linear_layer = nn.Linear(neuron_each_layer[-1] + self.output_dim, self.global_classes)
        
    def forward(self, inputs):
        """
        :param: inputs: Dict, including X_t, 
                                        X_t_mask, 
                                        deltaT_t, 
                                        empirical_mean, 
                                        X_t_filledLOCF
        """
        x = inputs['X']
        values = inputs["X_t"]
        masks = inputs["X_t_mask"]
        deltas = inputs["deltaT_t"]
        empirical_mean = inputs["empirical_mean"]
        X_filledLOCF = inputs["X_t_filledLOCF"]

        # encode the static features
        static_output = self.static_encoder(x)

        # encode the temporal features
        attention_outputs = self.pre_attention(values)
        hidden_states, hidden_state = self.grud(attention_outputs, masks, deltas, empirical_mean, X_filledLOCF)

        if self.gru_layer:
            output, hidden_state = self.gru(hidden_states)
        
        # concatenate
        embedding = torch.cat([static_output, hidden_state.squeeze(0)], dim=1)

        # hmcn block
        local_outputs = []
        output = embedding
        for layer_idx, layer in enumerate(self.linear_layers):
            if layer_idx == 0:
                output = layer(output)
                output = F.relu(output)
            else:
                output = layer(torch.cat([output, embedding], dim=1))
                output = F.relu(output)
            output = self.batchnorms[layer_idx](output)

            local_output = self.local_linear_l1[layer_idx](output) # torch.Size([64, 512])
            local_output = F.relu(local_output)
            local_output = self.batchnorms_local_1[layer_idx](output)
            
            local_output = self.local_linear_l2[layer_idx](local_output)
            local_outputs.append(local_output)

        global_outputs = torch.sigmoid(
            self.final_linear_layer(torch.cat([output, embedding], dim=1)))

        local_outputs = torch.sigmoid(torch.cat(local_outputs, dim=1))

        output = self.beta * global_outputs + (1 - self.beta) * local_outputs

        return output

class PreAttnMMs_MTL_IMP2(nn.Module):
    def __init__(self, config, n_steps, n_t_features, n_features, n_classes):
        super(PreAttnMMs_MTL_IMP2, self).__init__()
        self.config = config
        self.device = config.train.device_setting.device

        self.n_steps = n_steps
        self.n_t_features = n_t_features
        self.n_features = n_features
        self.n_classes = n_classes

        self.rnn_hidden_size_1 = config.model.temporal_encoder.layer1.rnn_hidden_size
        self.rnn_hidden_size_2 = config.model.temporal_encoder.layer2.rnn_hidden_size

        self.static_encoder = MLP(config, n_features)
        self.pre_attention = PreSpatialAttn(config, n_t_features)
        self.grud = GRUD_cells(config=config, 
                              input_dim=n_t_features, 
                              hidden_dim=self.rnn_hidden_size_1,
                              batch_size=config.train.batch_size,
                              dropout=config.model.temporal_encoder.layer1.dropout,
                              dropout_type=config.model.temporal_encoder.layer1.dropout_type)

        if config.model.temporal_encoder.num_layer > 1:
            self.gru = torch.nn.GRU(input_size = self.rnn_hidden_size_1, 
                                    hidden_size = self.rnn_hidden_size_2,
                                    batch_first =True,
                                    dropout=config.model.temporal_encoder.layer2.dropout)
            self.output_dim = self.rnn_hidden_size_2 + self.static_encoder.out_features
        else:
            self.output_dim = self.rnn_hidden_size_1 + self.static_encoder.out_features
            
        # head 0
        self.fc_0 = nn.Linear(self.output_dim, config.model.mtl_head_block.head0.hidden_dim)
        self.dropout_0 = nn.Dropout(config.model.mtl_head_block.head0.dropout)
        self.classifier_0 = nn.Linear(config.model.mtl_head_block.head0.hidden_dim, 
                                n_classes[0])
        # head 1
        self.fc_1 = nn.Linear(self.output_dim, config.model.mtl_head_block.head1.hidden_dim)
        self.dropout_1 = nn.Dropout(config.model.mtl_head_block.head1.dropout)
        self.classifier_1 = nn.Linear(config.model.mtl_head_block.head1.hidden_dim, 
                                n_classes[1])
        # head 2
        self.fc_2 = nn.Linear(self.output_dim, config.model.mtl_head_block.head2.hidden_dim)
        self.dropout_2 = nn.Dropout(config.model.mtl_head_block.head2.dropout)
        self.classifier_2 = nn.Linear(config.model.mtl_head_block.head2.hidden_dim, 
                                n_classes[2])
        # head 3
        self.fc_3 = nn.Linear(self.output_dim, config.model.mtl_head_block.head3.hidden_dim)
        self.dropout_3 = nn.Dropout(config.model.mtl_head_block.head3.dropout)
        self.classifier_3 = nn.Linear(config.model.mtl_head_block.head3.hidden_dim, 
                                n_classes[6])
        # head 4
        self.fc_4 = nn.Linear(self.output_dim, config.model.mtl_head_block.head4.hidden_dim)
        self.dropout_4 = nn.Dropout(config.model.mtl_head_block.head4.dropout)
        self.classifier_4 = nn.Linear(config.model.mtl_head_block.head4.hidden_dim, 
                                n_classes[7])
    
    def forward(self, inputs):
        """
        :param: inputs: Dict, including X_t, 
                                        X_t_mask, 
                                        deltaT_t, 
                                        empirical_mean, 
                                        X_t_filledLOCF
        """
        x = inputs['X']
        values = inputs["X_t"]
        masks = inputs["X_t_mask"]
        deltas = inputs["deltaT_t"]
        empirical_mean = inputs["empirical_mean"]
        X_filledLOCF = inputs["X_t_filledLOCF"]

        # encode the static features
        static_output = self.static_encoder(x)

        # encode the temporal features
        attention_outputs = self.pre_attention(values)
        hidden_states, hidden_state = self.grud(attention_outputs, masks, deltas, empirical_mean, X_filledLOCF)

        if self.config.model.temporal_encoder.num_layer > 1:
            output, hidden_state = self.gru(hidden_states)
        
        # concatenate
        embedding = torch.cat([static_output, hidden_state.squeeze(0)], dim=1)

        # multi-task block
        x0 = F.relu(self.fc_0(embedding))
        x0 = self.dropout_0(x0)
        head0 = self.classifier_0(x0)

        x1 = F.relu(self.fc_1(embedding))
        x1 = self.dropout_1(x1)
        head1 = self.classifier_1(x1)

        x2 = F.relu(self.fc_2(embedding))
        x2 = self.dropout_2(x2)
        head2 = self.classifier_2(x2)

        x3 = F.relu(self.fc_3(embedding))
        x3 = self.dropout_3(x3)
        head3 = self.classifier_3(x3)

        x4 = F.relu(self.fc_4(embedding))
        x4 = self.dropout_4(x4)
        head4 = self.classifier_4(x4)
        
        return [head0, head1, head2, head3, head4]

class PreAttnMMs_MTL_IMP2_Hp(nn.Module):
    def __init__(self, trial, config, n_steps, n_t_features, n_features, n_classes):
        super(PreAttnMMs_MTL_IMP2_Hp, self).__init__()
        self.config = config
        self.device = config.train.device_setting.device

        self.n_steps = n_steps
        self.n_t_features = n_t_features
        self.n_features = n_features
        self.n_classes = n_classes

        rnn_hidden_size = trial.suggest_int("rnn_hidden_size", 64, 256, step=16)
        dropout_type = trial.suggest_categorical("grud_dropout_type", ["None", "Gal", "Moon", "mloss"])
        if dropout_type == "None":
            dropout = 0.0
        elif dropout_type == "Gal":
            dropout = trial.suggest_float("dropout_gal", 0.0, 0.5, step=0.05)
        elif dropout_type == "Moon":
            dropout = trial.suggest_float("dropout_Moon", 0.0, 0.5, step=0.05)
        elif dropout_type == "mloss":
            dropout = trial.suggest_float("dropout_mloss", 0.0, 0.5, step=0.05)

        self.static_encoder = MLPHp(trial, config, n_features)
        self.pre_attention = PreSpatialAttn(config, n_t_features)
        self.grud = GRUD_cells(config=config, 
                              input_dim=n_t_features, 
                              hidden_dim=rnn_hidden_size,
                              batch_size=config.train.batch_size,
                              dropout=dropout,
                              dropout_type=dropout_type)
        self.gru_layer = trial.suggest_categorical("gru_layer", [True, False])
        if self.gru_layer == True:
            rnn_hidden_size_1 = trial.suggest_int("rnn_hidden_size_1", 64, 256, step=16)
            dropout_gru = trial.suggest_float("dropout_gru", 0.0, 0.5, step=0.05)
            self.gru = torch.nn.GRU(input_size = rnn_hidden_size, 
                                    hidden_size = rnn_hidden_size_1,
                                    batch_first =True,
                                    dropout=dropout_gru)
            self.output_dim = rnn_hidden_size_1 + self.static_encoder.out_features
        else:
            self.output_dim = rnn_hidden_size + self.static_encoder.out_features
            
        # head 0
        hidden_dim_head_0 = trial.suggest_int("hidden_dim_head_0", 64, 512, step=16)
        dropout_head_0 = trial.suggest_float("dropout_head_0", 0.0, 0.5, step=0.05)
        self.fc_0 = nn.Linear(self.output_dim, hidden_dim_head_0)
        self.dropout_0 = nn.Dropout(dropout_head_0)
        self.classifier_0 = nn.Linear(hidden_dim_head_0, n_classes[0])

        # head 1
        hidden_dim_head_1 = trial.suggest_int("hidden_dim_head_1", 64, 512, step=16)
        dropout_head_1 = trial.suggest_float("dropout_head_1", 0.0, 0.5, step=0.05)
        self.fc_1 = nn.Linear(self.output_dim, hidden_dim_head_1)
        self.dropout_1 = nn.Dropout(dropout_head_1)
        self.classifier_1 = nn.Linear(hidden_dim_head_1, n_classes[1])

        # head 2
        hidden_dim_head_2 = trial.suggest_int("hidden_dim_head_2", 64, 512, step=16)
        dropout_head_2 = trial.suggest_float("dropout_head_2", 0.0, 0.5, step=0.05)
        self.fc_2 = nn.Linear(self.output_dim, hidden_dim_head_2)
        self.dropout_2 = nn.Dropout(dropout_head_2)
        self.classifier_2 = nn.Linear(hidden_dim_head_2, n_classes[2])

        # head 3
        hidden_dim_head_3 = trial.suggest_int("hidden_dim_head_3", 64, 512, step=16)
        dropout_head_3 = trial.suggest_float("dropout_head_3", 0.0, 0.5, step=0.05)
        self.fc_3 = nn.Linear(self.output_dim, hidden_dim_head_3)
        self.dropout_3 = nn.Dropout(dropout_head_3)
        self.classifier_3 = nn.Linear(hidden_dim_head_3, n_classes[6])

        # head 4
        hidden_dim_head_4 = trial.suggest_int("hidden_dim_head_4", 64, 512, step=16)
        dropout_head_4 = trial.suggest_float("dropout_head_4", 0.0, 0.5, step=0.05)
        self.fc_4 = nn.Linear(self.output_dim, hidden_dim_head_4)
        self.dropout_4 = nn.Dropout(dropout_head_4)
        self.classifier_4 = nn.Linear(hidden_dim_head_4, n_classes[7])

    def forward(self, inputs):
        """
        :param: inputs: Dict, including X_t, 
                                        X_t_mask, 
                                        deltaT_t, 
                                        empirical_mean, 
                                        X_t_filledLOCF
        """
        x = inputs['X']
        values = inputs["X_t"]
        masks = inputs["X_t_mask"]
        deltas = inputs["deltaT_t"]
        empirical_mean = inputs["empirical_mean"]
        X_filledLOCF = inputs["X_t_filledLOCF"]

        # encode the static features
        static_output = self.static_encoder(x)

        # encode the temporal features
        attention_outputs = self.pre_attention(values)
        hidden_states, hidden_state = self.grud(attention_outputs, masks, deltas, empirical_mean, X_filledLOCF)

        if self.gru_layer:
            output, hidden_state = self.gru(hidden_states)
        
        # concatenate
        embedding = torch.cat([static_output, hidden_state.squeeze(0)], dim=1)
        # multi-task block
        x0 = F.relu(self.fc_0(embedding))
        x0 = self.dropout_0(x0)
        head0 = self.classifier_0(x0)

        x1 = F.relu(self.fc_1(embedding))
        x1 = self.dropout_1(x1)
        head1 = self.classifier_1(x1)

        x2 = F.relu(self.fc_2(embedding))
        x2 = self.dropout_2(x2)
        head2 = self.classifier_2(x2)

        x3 = F.relu(self.fc_3(embedding))
        x3 = self.dropout_3(x3)
        head3 = self.classifier_3(x3)

        x4 = F.relu(self.fc_4(embedding))
        x4 = self.dropout_4(x4)
        head4 = self.classifier_4(x4)

        return [head0, head1, head2, head3, head4]

class PreAttnMMs_MTL_LCL(nn.Module):
    def __init__(self, config, n_steps, n_t_features, n_features, n_classes):
        super(PreAttnMMs_MTL_LCL, self).__init__()
        self.config = config
        self.device = config.train.device_setting.device

        self.n_steps = n_steps
        self.n_t_features = n_t_features
        self.n_features = n_features
        self.n_classes = n_classes

        self.rnn_hidden_size_1 = config.model.temporal_encoder.layer1.rnn_hidden_size
        self.rnn_hidden_size_2 = config.model.temporal_encoder.layer2.rnn_hidden_size

        self.static_encoder = MLP(config, n_features)
        self.pre_attention = PreSpatialAttn(config, n_t_features)
        self.grud = GRUD_cells(config=config, 
                              input_dim=n_t_features, 
                              hidden_dim=self.rnn_hidden_size_1,
                              batch_size=config.train.batch_size,
                              dropout=config.model.temporal_encoder.layer1.dropout,
                              dropout_type=config.model.temporal_encoder.layer1.dropout_type)

        if config.model.temporal_encoder.num_layer > 1:
            self.gru = torch.nn.GRU(input_size = self.rnn_hidden_size_1, 
                                    hidden_size = self.rnn_hidden_size_2,
                                    batch_first =True,
                                    dropout=config.model.temporal_encoder.layer2.dropout)
            self.output_dim = self.rnn_hidden_size_2 + self.static_encoder.out_features
        else:
            self.output_dim = self.rnn_hidden_size_1 + self.static_encoder.out_features
        
        # head 0
        self.fc_0 = nn.Linear(self.output_dim, config.model.mtl_head_block.head0.hidden_dim)
        self.dropout_0 = nn.Dropout(config.model.mtl_head_block.head0.dropout)
        self.classifier_0 = nn.Linear(config.model.mtl_head_block.head0.hidden_dim, 
                                n_classes[0])
        # head 1
        self.fc_1 = nn.Linear(self.output_dim, config.model.mtl_head_block.head1.hidden_dim)
        self.dropout_1 = nn.Dropout(config.model.mtl_head_block.head1.dropout)
        self.classifier_1 = nn.Linear(config.model.mtl_head_block.head1.hidden_dim, 
                                n_classes[1])
        # head 2
        self.fc_2 = nn.Linear(self.output_dim, config.model.mtl_head_block.head2.hidden_dim)
        self.dropout_2 = nn.Dropout(config.model.mtl_head_block.head2.dropout)
        self.classifier_2 = nn.Linear(config.model.mtl_head_block.head2.hidden_dim, 
                                n_classes[2])
    
    def forward(self, inputs):
        """
        :param: inputs: Dict, including X_t, 
                                        X_t_mask, 
                                        deltaT_t, 
                                        empirical_mean, 
                                        X_t_filledLOCF
        """
        x = inputs['X']
        values = inputs["X_t"]
        masks = inputs["X_t_mask"]
        deltas = inputs["deltaT_t"]
        empirical_mean = inputs["empirical_mean"]
        X_filledLOCF = inputs["X_t_filledLOCF"]

        # encode the static features
        static_output = self.static_encoder(x)

        # encode the temporal features
        attention_outputs = self.pre_attention(values)
        hidden_states, hidden_state = self.grud(attention_outputs, masks, deltas, empirical_mean, X_filledLOCF)

        if self.config.model.temporal_encoder.num_layer > 1:
            output, hidden_state = self.gru(hidden_states)
        
        # concatenate
        embedding = torch.cat([static_output, hidden_state.squeeze(0)], dim=1)

        # multi-task block
        x0 = F.relu(self.fc_0(embedding))
        x0 = self.dropout_0(x0)
        head0 = self.classifier_0(x0)

        x1 = F.relu(self.fc_1(embedding))
        x1 = self.dropout_1(x1)
        head1 = self.classifier_1(x1)

        x2 = F.relu(self.fc_2(embedding))
        x2 = self.dropout_2(x2)
        head2 = self.classifier_2(x2)
        
        return [head0, head1, head2]

class PreAttnMMs_MTL_LCL_Hp(nn.Module):
    def __init__(self, trial, config, n_steps, n_t_features, n_features, n_classes):
        super(PreAttnMMs_MTL_LCL_Hp, self).__init__()
        self.config = config
        self.device = config.train.device_setting.device

        self.n_steps = n_steps
        self.n_t_features = n_t_features
        self.n_features = n_features
        self.n_classes = n_classes

        rnn_hidden_size = trial.suggest_int("rnn_hidden_size", 64, 256, step=16)
        dropout_type = trial.suggest_categorical("grud_dropout_type", ["None", "Gal", "Moon", "mloss"])
        if dropout_type == "None":
            dropout = 0.0
        elif dropout_type == "Gal":
            dropout = trial.suggest_float("dropout_gal", 0.0, 0.5, step=0.05)
        elif dropout_type == "Moon":
            dropout = trial.suggest_float("dropout_Moon", 0.0, 0.5, step=0.05)
        elif dropout_type == "mloss":
            dropout = trial.suggest_float("dropout_mloss", 0.0, 0.5, step=0.05)

        self.static_encoder = MLPHp(trial, config, n_features)
        self.pre_attention = PreSpatialAttn(config, n_t_features)
        self.grud = GRUD_cells(config=config, 
                              input_dim=n_t_features, 
                              hidden_dim=rnn_hidden_size,
                              batch_size=config.train.batch_size,
                              dropout=dropout,
                              dropout_type=dropout_type)
        self.gru_layer = trial.suggest_categorical("gru_layer", [True, False])
        if self.gru_layer == True:
            rnn_hidden_size_1 = trial.suggest_int("rnn_hidden_size_1", 64, 256, step=16)
            dropout_gru = trial.suggest_float("dropout_gru", 0.0, 0.5, step=0.05)
            self.gru = torch.nn.GRU(input_size = rnn_hidden_size, 
                                    hidden_size = rnn_hidden_size_1,
                                    batch_first =True,
                                    dropout=dropout_gru)
            self.output_dim = rnn_hidden_size_1 + self.static_encoder.out_features
        else:
            self.output_dim = rnn_hidden_size + self.static_encoder.out_features
            
        # head 0
        hidden_dim_head_0 = trial.suggest_int("hidden_dim_head_0", 64, 512, step=16)
        dropout_head_0 = trial.suggest_float("dropout_head_0", 0.0, 0.5, step=0.05)
        self.fc_0 = nn.Linear(self.output_dim, hidden_dim_head_0)
        self.dropout_0 = nn.Dropout(dropout_head_0)
        self.classifier_0 = nn.Linear(hidden_dim_head_0, n_classes[0])

        # head 1
        hidden_dim_head_1 = trial.suggest_int("hidden_dim_head_1", 64, 512, step=16)
        dropout_head_1 = trial.suggest_float("dropout_head_1", 0.0, 0.5, step=0.05)
        self.fc_1 = nn.Linear(self.output_dim, hidden_dim_head_1)
        self.dropout_1 = nn.Dropout(dropout_head_1)
        self.classifier_1 = nn.Linear(hidden_dim_head_1, n_classes[1])

        # head 2
        hidden_dim_head_2 = trial.suggest_int("hidden_dim_head_2", 64, 512, step=16)
        dropout_head_2 = trial.suggest_float("dropout_head_2", 0.0, 0.5, step=0.05)
        self.fc_2 = nn.Linear(self.output_dim, hidden_dim_head_2)
        self.dropout_2 = nn.Dropout(dropout_head_2)
        self.classifier_2 = nn.Linear(hidden_dim_head_2, n_classes[2])

    def forward(self, inputs):
        """
        :param: inputs: Dict, including X_t, 
                                        X_t_mask, 
                                        deltaT_t, 
                                        empirical_mean, 
                                        X_t_filledLOCF
        """
        x = inputs['X']
        values = inputs["X_t"]
        masks = inputs["X_t_mask"]
        deltas = inputs["deltaT_t"]
        empirical_mean = inputs["empirical_mean"]
        X_filledLOCF = inputs["X_t_filledLOCF"]

        # encode the static features
        static_output = self.static_encoder(x)

        # encode the temporal features
        attention_outputs = self.pre_attention(values)
        hidden_states, hidden_state = self.grud(attention_outputs, masks, deltas, empirical_mean, X_filledLOCF)

        if self.gru_layer:
            output, hidden_state = self.gru(hidden_states)
        
        # concatenate
        embedding = torch.cat([static_output, hidden_state.squeeze(0)], dim=1)
        # multi-task block
        x0 = F.relu(self.fc_0(embedding))
        x0 = self.dropout_0(x0)
        head0 = self.classifier_0(x0)

        x1 = F.relu(self.fc_1(embedding))
        x1 = self.dropout_1(x1)
        head1 = self.classifier_1(x1)

        x2 = F.relu(self.fc_2(embedding))
        x2 = self.dropout_2(x2)
        head2 = self.classifier_2(x2)

        return [head0, head1, head2]
    
class PreAttnMMs_GCN_MAP(nn.Module):
    def __init__(self, config, n_steps, n_t_features, n_features, n_classes, adj):
        super(PreAttnMMs_GCN_MAP, self).__init__()
        self.config = config
        self.device = config.train.device_setting.device

        self.n_steps = n_steps
        self.n_t_features = n_t_features
        self.n_features = n_features
        self.n_classes = n_classes
        self.adj = adj.to(self.device)

        # static encoder
        self.static_encoder = MLP(config, n_features)

        # temporal encoder
        self.rnn_hidden_size_1 = config.model.temporal_encoder.layer1.rnn_hidden_size
        self.rnn_hidden_size_2 = config.model.temporal_encoder.layer2.rnn_hidden_size
        self.pre_attention = PreSpatialAttn(config, n_t_features)
        self.grud = GRUD_cells(config=config, 
                              input_dim=n_t_features, 
                              hidden_dim=self.rnn_hidden_size_1,
                              batch_size=config.train.batch_size,
                              dropout=config.model.temporal_encoder.layer1.dropout,
                              dropout_type=config.model.temporal_encoder.layer1.dropout_type)

        if config.model.temporal_encoder.num_layer > 1:
            self.gru = torch.nn.GRU(input_size = self.rnn_hidden_size_1, 
                                    hidden_size = self.rnn_hidden_size_2,
                                    batch_first =True,
                                    dropout=config.model.temporal_encoder.layer2.dropout)

            self.output_dim = self.rnn_hidden_size_2 + self.static_encoder.out_features
        else:
            self.output_dim = self.rnn_hidden_size_1 + self.static_encoder.out_features

        # label encoder
        self.label_embedding = nn.Embedding(n_classes, config.model.label_encoder.embedding_dim)
        self.gc1 = GraphConvolution(config.model.label_encoder.embedding_dim, config.model.label_encoder.hidden_dim)
        self.gc2 = GraphConvolution(config.model.label_encoder.hidden_dim, self.output_dim)
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = config.model.label_encoder.dropout
    
    def forward(self, inputs):
        """
        :param: inputs: Dict, including X_t, 
                                        X_t_mask, 
                                        deltaT_t, 
                                        empirical_mean, 
                                        X_t_filledLOCF
        """
        x = inputs['X']
        values = inputs["X_t"]
        masks = inputs["X_t_mask"]
        deltas = inputs["deltaT_t"]
        empirical_mean = inputs["empirical_mean"]
        X_filledLOCF = inputs["X_t_filledLOCF"]
        node_inputs = inputs["node_input"]

        # encode the static features
        static_output = self.static_encoder(x)

        # encode the temporal features
        attention_outputs = self.pre_attention(values)
        hidden_states, hidden_state = self.grud(attention_outputs, masks, deltas, empirical_mean, X_filledLOCF)

        if self.config.model.temporal_encoder.num_layer > 1:
            output, hidden_state = self.gru(hidden_states)
        
        # concatenate
        feature = torch.cat([static_output, hidden_state.squeeze(0)], dim=1) # torch.Size([64, 240])

        # embedding label and model the graph
        node_input = node_inputs[0]
        embedding = self.label_embedding(node_input) # [12, 300]
        x = self.gc1(embedding, self.adj)
        x = self.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, self.adj)

        x_t = x.transpose(0, 1) # torch.Size([240, 12])

        logits = torch.matmul(feature, x_t)

        return (logits, x)

class PreAttnMMs_GCN_MAP_Hp(nn.Module):
    def __init__(self, trial, config, n_steps, n_t_features, n_features, n_classes, adj):
        super(PreAttnMMs_GCN_MAP_Hp, self).__init__()
        self.config = config
        self.device = config.train.device_setting.device

        self.n_steps = n_steps
        self.n_t_features = n_t_features
        self.n_features = n_features
        self.n_classes = n_classes
        self.adj = adj.to(self.device)

        # static encoder
        self.static_encoder = MLPHp(trial, config, n_features)

        # temporal encoder
        rnn_hidden_size = trial.suggest_int("rnn_hidden_size", 64, 256, step=16)
        dropout_type = trial.suggest_categorical("dropout_type", ["None", "Gal", "Moon", "mloss"])
        if dropout_type == "None":
            dropout = 0.0
        elif dropout_type == "Gal":
            dropout = trial.suggest_float("dropout_gal", 0.0, 0.5, step=0.05)
        elif dropout_type == "Moon":
            dropout = trial.suggest_float("dropout_Moon", 0.0, 0.5, step=0.05)
        elif dropout_type == "mloss":
            dropout = trial.suggest_float("dropout_mloss", 0.0, 0.5, step=0.05)

        self.pre_attention = PreSpatialAttn(config, n_t_features)
        self.grud = GRUD_cells(config=config, 
                              input_dim=n_t_features, 
                              hidden_dim=rnn_hidden_size,
                              batch_size=config.train.batch_size,
                              dropout=dropout,
                              dropout_type=dropout_type)

        self.gru_layer = trial.suggest_categorical("gru_layer", [True, False])
        if self.gru_layer == True:
            rnn_hidden_size_1 = trial.suggest_int("rnn_hidden_size_1", 64, 256, step=16)
            dropout_gru = trial.suggest_float("dropout_gru", 0.0, 0.5, step=0.05)

            self.gru = torch.nn.GRU(input_size = rnn_hidden_size, 
                                    hidden_size = rnn_hidden_size_1,
                                    batch_first =True,
                                    dropout=dropout_gru)

            self.output_dim = rnn_hidden_size_1 + self.static_encoder.out_features
        else:
            self.output_dim = rnn_hidden_size + self.static_encoder.out_features

        # label encoder
        self.graph_embedding_dim = trial.suggest_int("graph_embedding_dim", 32, 256, step=16)
        self.graph_hidden_dim = trial.suggest_int("graph_hidden_dim", 32, 256, step=16)
        self.graph_dropout = nn.Dropout(trial.suggest_float("dropout_graph", 0.0, 0.5, step=0.05))

        self.label_embedding = nn.Embedding(n_classes, self.graph_embedding_dim)
        self.gc1 = GraphConvolution(self.graph_embedding_dim, self.graph_hidden_dim)
        self.gc2 = GraphConvolution(self.graph_hidden_dim, self.output_dim)
        self.relu = nn.LeakyReLU(0.2)
    
    def forward(self, inputs):
        """
        :param: inputs: Dict, including X_t, 
                                        X_t_mask, 
                                        deltaT_t, 
                                        empirical_mean, 
                                        X_t_filledLOCF
        """
        x = inputs['X']
        values = inputs["X_t"]
        masks = inputs["X_t_mask"]
        deltas = inputs["deltaT_t"]
        empirical_mean = inputs["empirical_mean"]
        X_filledLOCF = inputs["X_t_filledLOCF"]
        node_inputs = inputs["node_input"]

        # encode the static features
        static_output = self.static_encoder(x)

        # encode the temporal features
        attention_outputs = self.pre_attention(values)
        hidden_states, hidden_state = self.grud(attention_outputs, masks, deltas, empirical_mean, X_filledLOCF)

        if self.gru_layer:
            output, hidden_state = self.gru(hidden_states)
        
        # concatenate
        feature = torch.cat([static_output, hidden_state.squeeze(0)], dim=1) # torch.Size([64, 240])

        # embedding label and model the graph
        node_input = node_inputs[0]
        embedding = self.label_embedding(node_input) # [12, 300]
        x = self.gc1(embedding, self.adj)
        x = self.relu(x)
        x = self.graph_dropout(x)
        x = self.gc2(x, self.adj)

        x_t = x.transpose(0, 1) # torch.Size([240, 12])

        logits = torch.matmul(feature, x_t)

        return (logits, x)

class PreAttnMMs_GAT_IMP8_GC(nn.Module):
    def __init__(self, config, n_steps, n_t_features, n_features, n_classes, adj):
        super(PreAttnMMs_GAT_IMP8_GC, self).__init__()
        self.config = config
        self.device = config.train.device_setting.device

        self.n_steps = n_steps
        self.n_t_features = n_t_features
        self.n_features = n_features
        self.n_classes = n_classes
        self.adj = torch.tensor(adj, dtype=torch.float).to(self.device)

        # static encoder
        self.static_encoder = MLP(config, n_features)

        # temporal encoder
        self.rnn_hidden_size_1 = config.model.temporal_encoder.layer1.rnn_hidden_size
        self.rnn_hidden_size_2 = config.model.temporal_encoder.layer2.rnn_hidden_size
        self.pre_attention = PreSpatialAttn(config, n_t_features)
        self.grud = GRUD_cells(config=config, 
                              input_dim=n_t_features, 
                              hidden_dim=self.rnn_hidden_size_1,
                              batch_size=config.train.batch_size,
                              dropout=config.model.temporal_encoder.layer1.dropout,
                              dropout_type=config.model.temporal_encoder.layer1.dropout_type)

        if config.model.temporal_encoder.num_layer > 1:
            self.gru = torch.nn.GRU(input_size = self.rnn_hidden_size_1, 
                                    hidden_size = self.rnn_hidden_size_2,
                                    batch_first =True,
                                    dropout=config.model.temporal_encoder.layer2.dropout)

            self.output_dim = self.rnn_hidden_size_2 + self.static_encoder.out_features
        else:
            self.output_dim = self.rnn_hidden_size_1 + self.static_encoder.out_features

        self.transformation_0 = nn.Linear(self.output_dim, self.output_dim)
        self.transformation_1 = nn.Linear(self.output_dim, self.output_dim)
        self.transformation_2 = nn.Linear(self.output_dim, self.output_dim)
        self.transformation_3 = nn.Linear(self.output_dim, self.output_dim)
        self.transformation_4 = nn.Linear(self.output_dim, self.output_dim)

        self.batchnorm_0 = nn.BatchNorm1d(self.output_dim)
        self.batchnorm_1 = nn.BatchNorm1d(self.output_dim)
        self.batchnorm_2 = nn.BatchNorm1d(self.output_dim)
        self.batchnorm_3 = nn.BatchNorm1d(self.output_dim)
        self.batchnorm_4 = nn.BatchNorm1d(self.output_dim)

        self.gatlayer = GATlayer_IMP8(config, self.output_dim)

        if config.model.gat_layer.is_concat:
            self.gat_out_features = config.model.gat_layer.num_out_features * config.model.gat_layer.num_of_heads
        else:
            self.gat_out_features = config.model.gat_layer.num_out_features

        # head 0
        self.classifier_0 = nn.Linear(self.gat_out_features + self.output_dim, n_classes[0])
        # head 1
        self.classifier_1 = nn.Linear(self.gat_out_features + self.output_dim, n_classes[1])
        # head 2
        self.classifier_2 = nn.Linear(self.gat_out_features + self.output_dim, n_classes[2])
        # head 3
        self.classifier_3 = nn.Linear(self.gat_out_features + self.output_dim, n_classes[6])
        # head 4
        self.classifier_4 = nn.Linear(self.gat_out_features + self.output_dim, n_classes[7])

    def forward(self, inputs):
        """
        :param: inputs: Dict, including X_t, 
                                        X_t_mask, 
                                        deltaT_t, 
                                        empirical_mean, 
                                        X_t_filledLOCF
        """
        x = inputs['X']
        values = inputs["X_t"]
        masks = inputs["X_t_mask"]
        deltas = inputs["deltaT_t"]
        empirical_mean = inputs["empirical_mean"]
        X_filledLOCF = inputs["X_t_filledLOCF"]

        # encode the static features
        static_output = self.static_encoder(x)

        # encode the temporal features
        attention_outputs = self.pre_attention(values)
        hidden_states, hidden_state = self.grud(attention_outputs, masks, deltas, empirical_mean, X_filledLOCF)

        if self.config.model.temporal_encoder.num_layer > 1:
            output, hidden_state = self.gru(hidden_states)
        
        # concatenate
        # (B, FIN)
        embedding = torch.cat([static_output, hidden_state.squeeze(0)], dim=1)

        # gat
        # (B, FIN)
        x_0 = self.batchnorm_0(F.relu(self.transformation_0(embedding)))
        x_1 = self.batchnorm_1(F.relu(self.transformation_1(embedding)))
        x_2 = self.batchnorm_2(F.relu(self.transformation_2(embedding)))
        x_3 = self.batchnorm_3(F.relu(self.transformation_3(embedding)))
        x_4 = self.batchnorm_4(F.relu(self.transformation_4(embedding)))
        in_node_features = torch.concat([x_0.unsqueeze(1), x_1.unsqueeze(1), x_2.unsqueeze(1), x_3.unsqueeze(1), x_4.unsqueeze(1)], dim=1)

        # (B, N, FIN)-->(B, N, NH * FOUT) OR (B, N, FOUT)
        gat_output, adj = self.gatlayer(in_node_features, self.adj)

        # concatenation
        output_0 = torch.concat([x_0.detach(), gat_output[:, 0, :]], dim=-1)
        output_1 = torch.concat([x_1.detach(), gat_output[:, 1, :]], dim=-1)
        output_2 = torch.concat([x_2.detach(), gat_output[:, 2, :]], dim=-1)
        output_3 = torch.concat([x_3.detach(), gat_output[:, 3, :]], dim=-1)
        output_4 = torch.concat([x_4.detach(), gat_output[:, 4, :]], dim=-1)

        head0 = self.classifier_0(output_0)
        head1 = self.classifier_1(output_1)
        head2 = self.classifier_2(output_2)
        head3 = self.classifier_3(output_3)
        head4 = self.classifier_4(output_4)

        return [head0, head1, head2, head3, head4]

class PreAttnMMs_GAT_IMP8_GC_Hp(nn.Module):
    def __init__(self, trial, config, n_steps, n_t_features, n_features, n_classes, adj):
        super(PreAttnMMs_GAT_IMP8_GC_Hp, self).__init__()
        self.config = config
        self.device = config.train.device_setting.device

        self.n_steps = n_steps
        self.n_t_features = n_t_features
        self.n_features = n_features
        self.n_classes = n_classes
        self.adj = torch.tensor(adj, dtype=torch.float).to(self.device)

        rnn_hidden_size = trial.suggest_int("rnn_hidden_size", 64, 256, step=16)
        dropout_type = trial.suggest_categorical("grud_dropout_type", ["None", "Gal", "Moon", "mloss"])
        if dropout_type == "None":
            dropout = 0.0
        elif dropout_type == "Gal":
            dropout = trial.suggest_float("dropout_gal", 0.0, 0.5, step=0.05)
        elif dropout_type == "Moon":
            dropout = trial.suggest_float("dropout_Moon", 0.0, 0.5, step=0.05)
        elif dropout_type == "mloss":
            dropout = trial.suggest_float("dropout_mloss", 0.0, 0.5, step=0.05)

        self.static_encoder = MLPHp(trial, config, n_features)
        self.pre_attention = PreSpatialAttn(config, n_t_features)
        self.grud = GRUD_cells(config=config, 
                              input_dim=n_t_features, 
                              hidden_dim=rnn_hidden_size,
                              batch_size=config.train.batch_size,
                              dropout=dropout,
                              dropout_type=dropout_type)
        self.gru_layer = trial.suggest_categorical("gru_layer", [True, False])
        if self.gru_layer == True:
            rnn_hidden_size_1 = trial.suggest_int("rnn_hidden_size_1", 64, 256, step=16)
            dropout_gru = trial.suggest_float("dropout_gru", 0.0, 0.5, step=0.05)
            self.gru = torch.nn.GRU(input_size = rnn_hidden_size, 
                                    hidden_size = rnn_hidden_size_1,
                                    batch_first =True,
                                    dropout=dropout_gru)

            self.output_dim = rnn_hidden_size_1 + self.static_encoder.out_features
        else:
            self.output_dim = rnn_hidden_size + self.static_encoder.out_features

        self.transformation_0 = nn.Linear(self.output_dim, self.output_dim)
        self.transformation_1 = nn.Linear(self.output_dim, self.output_dim)
        self.transformation_2 = nn.Linear(self.output_dim, self.output_dim)
        self.transformation_3 = nn.Linear(self.output_dim, self.output_dim)
        self.transformation_4 = nn.Linear(self.output_dim, self.output_dim)

        self.batchnorm_0 = nn.BatchNorm1d(self.output_dim)
        self.batchnorm_1 = nn.BatchNorm1d(self.output_dim)
        self.batchnorm_2 = nn.BatchNorm1d(self.output_dim)
        self.batchnorm_3 = nn.BatchNorm1d(self.output_dim)
        self.batchnorm_4 = nn.BatchNorm1d(self.output_dim)

        self.is_bias = trial.suggest_categorical("is_bias", [True, False])
        self.is_concat = trial.suggest_categorical("is_concat", [True, False])
        self.num_of_heads = trial.suggest_int("num_of_heads", 1, 8, step=1)
        self.num_out_features = trial.suggest_int("num_out_features", 32, 128, step=16)
        self.is_add_skip_connection = trial.suggest_categorical("is_add_skip_connection", [True, False])

        self.gatlayer = GATlayer_IMP8_Hp(trial, config, self.output_dim, self.num_out_features, self.num_of_heads, self.is_concat, self.is_bias, self.is_add_skip_connection)

        if self.is_concat:
            gat_out_features = self.num_out_features * self.num_of_heads + self.output_dim
        else:
            gat_out_features = self.num_out_features + self.output_dim
        
        # head 0
        self.classifier_0 = nn.Linear(gat_out_features, n_classes[0])
        # head 1
        self.classifier_1 = nn.Linear(gat_out_features, n_classes[1])
        # head 2
        self.classifier_2 = nn.Linear(gat_out_features, n_classes[2])
        # head 3
        self.classifier_3 = nn.Linear(gat_out_features, n_classes[6])
        # head 4
        self.classifier_4 = nn.Linear(gat_out_features, n_classes[7])

    def forward(self, inputs):
        """
        :param: inputs: Dict, including X_t, 
                                        X_t_mask, 
                                        deltaT_t, 
                                        empirical_mean, 
                                        X_t_filledLOCF
        """
        x = inputs['X']
        values = inputs["X_t"]
        masks = inputs["X_t_mask"]
        deltas = inputs["deltaT_t"]
        empirical_mean = inputs["empirical_mean"]
        X_filledLOCF = inputs["X_t_filledLOCF"]

        # encode the static features
        static_output = self.static_encoder(x)

        # encode the temporal features
        attention_outputs = self.pre_attention(values)
        hidden_states, hidden_state = self.grud(attention_outputs, masks, deltas, empirical_mean, X_filledLOCF)

        if self.gru_layer:
            output, hidden_state = self.gru(hidden_states)
        
        # concatenate
        # (B, FIN)
        embedding = torch.cat([static_output, hidden_state.squeeze(0)], dim=1)

        # gat
        x_0 = self.batchnorm_0(F.relu(self.transformation_0(embedding)))
        x_1 = self.batchnorm_1(F.relu(self.transformation_1(embedding)))
        x_2 = self.batchnorm_2(F.relu(self.transformation_2(embedding)))
        x_3 = self.batchnorm_3(F.relu(self.transformation_3(embedding)))
        x_4 = self.batchnorm_4(F.relu(self.transformation_4(embedding)))
        in_node_features = torch.concat([x_0.unsqueeze(1), x_1.unsqueeze(1), x_2.unsqueeze(1), x_3.unsqueeze(1), x_4.unsqueeze(1)], dim=1)

        # (B, N, NH * FOUT) OR (B, N, FOUT)
        gat_output, adj = self.gatlayer(in_node_features, self.adj)

        # concatenation
        output_0 = torch.concat([x_0.detach(), gat_output[:, 0, :]], dim=-1)
        output_1 = torch.concat([x_1.detach(), gat_output[:, 1, :]], dim=-1)
        output_2 = torch.concat([x_2.detach(), gat_output[:, 2, :]], dim=-1)
        output_3 = torch.concat([x_3.detach(), gat_output[:, 3, :]], dim=-1)
        output_4 = torch.concat([x_4.detach(), gat_output[:, 4, :]], dim=-1)

        head0 = self.classifier_0(output_0)
        head1 = self.classifier_1(output_1)
        head2 = self.classifier_2(output_2)
        head3 = self.classifier_3(output_3)
        head4 = self.classifier_4(output_4)

        return [head0, head1, head2, head3, head4]

