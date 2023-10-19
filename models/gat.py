#!/usr/bin/env python
# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

class GATlayer_IMP8(nn.Module):
    def __init__(self, config, num_in_features, activation=nn.ELU()):
        super(GATlayer_IMP8, self).__init__()

        self.config = config
        self.num_in_features = num_in_features
        self.num_out_features = config.model.gat_layer.num_out_features
        self.num_of_heads = config.model.gat_layer.num_of_heads
        self.is_concat = config.model.gat_layer.is_concat
        self.is_bias = config.model.gat_layer.is_bias
        self.in_dropout_prob = config.model.gat_layer.in_dropout
        self.out_dropout_prob = config.model.gat_layer.out_dropout
        self.is_gat_activation = config.model.gat_layer.is_gat_activation
        self.is_add_skip_connection = config.model.gat_layer.is_add_skip_connection
        
        self.linear_proj = nn.Linear(num_in_features, self.num_of_heads * self.num_out_features, bias=False)

        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, self.num_of_heads, self.num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, self.num_of_heads, self.num_out_features))

        if self.is_bias and self.is_concat:
            self.bias = nn.Parameter(torch.Tensor(self.num_of_heads * self.num_out_features))
        elif self.is_bias and not self.is_concat:
            self.bias = nn.Parameter(torch.Tensor(self.num_out_features))
        else:
            self.register_parameter('bias', None)
        
        if self.is_add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, self.num_of_heads * self.num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        self.leakyReLU = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)
        if self.is_gat_activation:
            self.activation = activation
        else:
            self.activation = None 
        self.in_dropout = nn.Dropout(p=self.in_dropout_prob)
        self.out_dropout = nn.Dropout(p=self.out_dropout_prob)

        self.log_attention_weights = config.model.gat_layer.log_attention_weights
        self.attention_weights = None

        self.init_params()

    def init_params(self):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow

        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.

        """
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        
        if self.log_attention_weights:  
            self.attention_weights = attention_coefficients

        num_of_nodes = in_nodes_features.shape[1]
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()

        if self.is_add_skip_connection:
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]: 

                out_nodes_features += in_nodes_features.unsqueeze(2)
            
            else:
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, num_of_nodes, self.num_of_heads, self.num_out_features)
        
        if self.is_concat:
            out_nodes_features = out_nodes_features.view(-1, num_of_nodes, self.num_of_heads * self.num_out_features)
        else:
            out_nodes_features = out_nodes_features.mean(dim=2)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)

    def forward(self, in_nodes_features, adj):
        
        num_of_nodes = in_nodes_features.shape[1]
        assert adj.shape == (num_of_nodes, num_of_nodes), \
            f'Expected connectivity matrix with shape=({num_of_nodes},{num_of_nodes}), got shape={adj.shape}.'

        in_nodes_features = self.in_dropout(in_nodes_features)

        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, num_of_nodes, self.num_of_heads, self.num_out_features)

        nodes_features_proj = self.out_dropout(nodes_features_proj)

        scores_source = torch.sum((nodes_features_proj * self.scoring_fn_source), dim=-1, keepdim=True)
        scores_target = torch.sum((nodes_features_proj * self.scoring_fn_target), dim=-1, keepdim=True)

        scores_source = scores_source.transpose(1, 2)
        scores_target = scores_target.permute(0, 2, 3, 1)

        all_scores = self.leakyReLU(scores_source + scores_target)

        all_attention_coefficients = self.softmax(all_scores + adj)

        out_nodes_features = torch.matmul(all_attention_coefficients, nodes_features_proj.transpose(1, 2))

        out_nodes_features = out_nodes_features.permute(0, 2, 1, 3)

        out_nodes_features = self.skip_concat_bias(all_attention_coefficients, in_nodes_features, out_nodes_features)

        return (out_nodes_features, adj)

class GATlayer_IMP8_Hp(nn.Module):
    def __init__(self, trial, config, num_in_features, num_out_features, num_of_heads, is_concat, is_bias, is_add_skip_connection, activation=nn.ELU()):
        super(GATlayer_IMP8_Hp, self).__init__()
        self.config = config
        self.num_in_features = num_in_features
        self.num_out_features = num_out_features
        self.num_of_heads = num_of_heads
        self.is_concat = is_concat
        self.is_bias = is_bias
        self.is_add_skip_connection = is_add_skip_connection

        in_dropout = trial.suggest_float("in_dropout", 0.0, 0.7, step=0.05)
        out_dropout = trial.suggest_float("out_dropout", 0.0, 0.7, step=0.05)
        self.is_gat_activation = trial.suggest_categorical("is_gat_activation", [True, False])

        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        if self.is_bias and self.is_concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif self.is_bias and not self.is_concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if self.is_add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        self.leakyReLU = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)
        if self.is_gat_activation:
            self.activation = activation
        else:
            self.activation = None
        self.in_dropout = nn.Dropout(p=in_dropout)
        self.out_dropout = nn.Dropout(p=out_dropout)

        self.log_attention_weights = config.model.gat_layer.log_attention_weights
        self.attention_weights = None

        self.init_params()

    def init_params(self):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow

        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.

        """
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        
        if self.log_attention_weights:  
            self.attention_weights = attention_coefficients
            
        num_of_nodes = in_nodes_features.shape[1]

        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()

        if self.is_add_skip_connection:
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]: 
                out_nodes_features += in_nodes_features.unsqueeze(2)
            
            else:
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, num_of_nodes, self.num_of_heads, self.num_out_features)

        if self.is_concat:
            out_nodes_features = out_nodes_features.view(-1, num_of_nodes, self.num_of_heads * self.num_out_features)
        else:
            out_nodes_features = out_nodes_features.mean(dim=2)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)

    def forward(self, in_nodes_features, adj):
        
        num_of_nodes = in_nodes_features.shape[1]
        assert adj.shape == (num_of_nodes, num_of_nodes), \
            f'Expected connectivity matrix with shape=({num_of_nodes},{num_of_nodes}), got shape={adj.shape}.'
        
        in_nodes_features = self.in_dropout(in_nodes_features)

        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, num_of_nodes, self.num_of_heads, self.num_out_features)

        nodes_features_proj = self.out_dropout(nodes_features_proj)

        scores_source = torch.sum((nodes_features_proj * self.scoring_fn_source), dim=-1, keepdim=True)
        scores_target = torch.sum((nodes_features_proj * self.scoring_fn_target), dim=-1, keepdim=True)

        scores_source = scores_source.transpose(1, 2)
        scores_target = scores_target.permute(0, 2, 3, 1)

        all_scores = self.leakyReLU(scores_source + scores_target)

        all_attention_coefficients = self.softmax(all_scores + adj)

        out_nodes_features = torch.matmul(all_attention_coefficients, nodes_features_proj.transpose(1, 2))

        out_nodes_features = out_nodes_features.permute(0, 2, 1, 3)

        out_nodes_features = self.skip_concat_bias(all_attention_coefficients, in_nodes_features, out_nodes_features)

        return (out_nodes_features, adj)