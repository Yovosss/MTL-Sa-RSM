#!/usr/bin/env python
# coding:utf-8
import numbers
import warnings
from typing import List, Optional, Tuple, Union, cast, overload

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter

from .temporal_decay import TemporalDecay


class GRUD_cell(nn.Module):
    def __init__(self, config, n_steps, n_features, rnn_hidden_size, n_classes):
        super(GRUD_cell, self).__init__()
        self.config = config
        self.device = config.train.device_setting.device

        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.n_classes = n_classes

        self.rnn_cell = nn.GRUCell(
            self.n_features * 2 + self.rnn_hidden_size, self.rnn_hidden_size
        )
        self.temp_decay_h = TemporalDecay(
            input_size=self.n_features, output_size=self.rnn_hidden_size, diag=False
        )
        self.temp_decay_x = TemporalDecay(
            input_size=self.n_features, output_size=self.n_features, diag=True
        )

    def forward(self, values, masks, deltas, empirical_mean, X_filledLOCF):
        """Forward processing of GRU-D.

        Parameters
        ----------
        values, masks, deltas, empirical_mean, X_filledLOCF

        Returns
        -------
        dict,
            A dictionary includes all results.
        """
        hidden_state = torch.zeros(
            (values.size()[0], self.rnn_hidden_size), device=self.device
        )

        hidden_states = torch.empty(values.size()[0], 
                                    values.size()[1], 
                                    self.rnn_hidden_size, 
                                    dtype=values.dtype, 
                                    device = self.config.train.device_setting.device)

        for t in range(self.n_steps):
            # for data, [batch, time, features]
            x = values[:, t, :]  # values
            m = masks[:, t, :]  # mask
            d = deltas[:, t, :]  # delta, time gap
            x_filledLOCF = X_filledLOCF[:, t, :]

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)
            hidden_state = hidden_state * gamma_h

            x_h = gamma_x * x_filledLOCF + (1 - gamma_x) * empirical_mean
            x_replaced = m * x + (1 - m) * x_h
            inputs = torch.cat([x_replaced, hidden_state, m], dim=1)
            hidden_state = self.rnn_cell(inputs, hidden_state)

            hidden_states[:, t, :] = hidden_state

        return hidden_states, hidden_state

class GRUD_cells(nn.Module):
    def __init__(self, config,
                       input_dim: int,
                       hidden_dim: int, 
                       batch_size: int,
                       dropout: float = 0.,
                       dropout_type: str = 'mloss',
                       mode: str = 'GRU', 
                       proj_size: int = 0,
                       num_layers: int = 1, 
                       batch_first: bool = True, 
                       bidirectional: bool = False):
        """
        This is the GRUD-D model i complemented based on the original paper.
        """
        super(GRUD_cells, self).__init__()
        self.config = config
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.dropout_type = dropout_type
        self.mode = mode
        self.num_layers = num_layers
        self.proj_size = proj_size
        self.batch_size = batch_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
                isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout, num_layers))

        self._all_weights = []
        self._flat_weights_names = []

        # decay rates gamma
        w_gamma_x = nn.Parameter(torch.Tensor(input_dim))
        w_gamma_h = nn.Parameter(torch.Tensor(input_dim, hidden_dim))

        # r
        w_r_x = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        w_r_h = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        w_r_m = nn.Parameter(torch.Tensor(input_dim, hidden_dim))

        # z
        w_z_x = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        w_z_h = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        w_z_m = nn.Parameter(torch.Tensor(input_dim, hidden_dim))

        # h_tilde
        w_h_x = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        w_h_h = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        w_h_m = nn.Parameter(torch.Tensor(input_dim, hidden_dim))

        # bias
        b_gamma_x = nn.Parameter(torch.Tensor(input_dim))
        b_gamma_h = nn.Parameter(torch.Tensor(hidden_dim))
        b_r = nn.Parameter(torch.Tensor(hidden_dim))
        b_z = nn.Parameter(torch.Tensor(hidden_dim))
        b_h = nn.Parameter(torch.Tensor(hidden_dim))

        layer_params = (w_gamma_x, w_gamma_h,\
                        w_r_x, w_r_h, w_r_m, \
                        w_z_x, w_z_h, w_z_m, \
                        w_h_x, w_h_h, w_h_m, \
                        b_gamma_x, b_gamma_h, b_r, b_z, b_h)
        param_names = ['weight_gamma_x', 'weight_gamma_h',  \
                       'weight_rx', 'weight_rh','weight_rm',\
                       'weight_zx', 'weight_zh','weight_zm',\
                       'weight_hx', 'weight_hh','weight_hm',\
                       'bias_gamma_x', 'bias_gamma_h', 'bias_r', 'bias_z', 'bias_h']

        for name, param in zip(param_names, layer_params):
            setattr(self, name, param)
        self._all_weights.append(param_names)
        self._flat_weights_names.extend(param_names)

        hidden_state = torch.zeros((batch_size, hidden_dim), requires_grad = False)
        self.register_buffer('hidden_state', hidden_state)

        self._flat_weights = [(lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn) for wn in self._flat_weights_names]

        # self.flatten_parameters()
        self.reset_parameters()

    # def __setattr__(self, attr, value):
    #     if hasattr(self, "_flat_weights_names") and attr in self._flat_weights_names:
    #         # keep self._flat_weights up to date if you do self.weight = ...
    #         idx = self._flat_weights_names.index(attr)
    #         self._flat_weights[idx] = value
    #     super(GRUD_cell, self).__setattr__(attr, value)

    def reset_parameters(self):
        """
        init the parameters defined by nn.Parameter()
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    # def flatten_parameters(self) -> None:
    #     """Resets parameter data pointer so that they can use faster code paths.

    #     Right now, this works only if the module is on the GPU and cuDNN is enabled.
    #     Otherwise, it's a no-op.
    #     """
    #     # Short-circuits if _flat_weights is only partially instantiated
    #     if len(self._flat_weights) != len(self._flat_weights_names):
    #         return

    #     for w in self._flat_weights:
    #         if not isinstance(w, Tensor):
    #             return
    #     # Short-circuits if any tensor in self._flat_weights is not acceptable to cuDNN
    #     # or the tensors in _flat_weights are of different dtypes

    #     first_fw = self._flat_weights[0]
    #     dtype = first_fw.dtype
    #     for fw in self._flat_weights:
    #         if (not isinstance(fw.data, Tensor) or not (fw.data.dtype == dtype) or
    #                 not fw.data.is_cuda or
    #                 not torch.backends.cudnn.is_acceptable(fw.data)):
    #             return

    #     # If any parameters alias, we fall back to the slower, copying code path. This is
    #     # a sufficient check, because overlapping parameter buffers that don't completely
    #     # alias would break the assumptions of the uniqueness check in
    #     # Module.named_parameters().
    #     unique_data_ptrs = set(p.data_ptr() for p in self._flat_weights)
    #     if len(unique_data_ptrs) != len(self._flat_weights):
    #         return

    #     with torch.cuda.device_of(first_fw):
    #         import torch.backends.cudnn.rnn as rnn

    #         # Note: no_grad() is necessary since _cudnn_rnn_flatten_weight is
    #         # an inplace operation on self._flat_weights
    #         with torch.no_grad():
    #             if torch._use_cudnn_rnn_flatten_weight():
    #                 num_weights = len(self._flat_weights)
    #                 torch._cudnn_rnn_flatten_weight(
    #                     self._flat_weights, num_weights,
    #                     self.input_dim, rnn.get_cudnn_mode(self.mode),
    #                     self.hidden_dim, self.proj_size, self.num_layers,
    #                     self.batch_first, bool(self.bidirectional))
    
    # def _apply(self, fn):
    #     ret = super(GRUD_cell, self)._apply(fn)

    #     # Resets _flat_weights
    #     # Note: be v. careful before removing this, as 3rd party device types
    #     # likely rely on this behavior to properly .to() modules like LSTM.
    #     self._flat_weights = [(lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn) for wn in self._flat_weights_names]
    #     # Flattens params (on CUDA)
    #     self.flatten_parameters()

    #     return ret    
    
    def check_forward_args(self, input, hidden, batch_sizes):
        """
        used to check the input, hidden state and batch_size
        """
        is_input_packed = batch_sizes is not None
        expected_input_dim = 2 if is_input_packed else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

        if is_input_packed:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)

        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)
        
        def check_hidden_size(hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
            if tuple(hx.size()) != expected_hidden_size:
                raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

        check_hidden_size(hidden, expected_hidden_size)
    
    def extra_repr(self):
        s = '{input_dim}, {hidden_dim}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(GRUD_cells, self).__setstate__(state)
        if 'all_weights' in state:
            self._all_weights = state['all_weights']
        if 'proj_size' not in state:
            self.proj_size = 0

        if isinstance(self._all_weights[0][0], str):
            return
        
        self._flat_weights_names = []
        self._all_weights = []
        weights = ['weight_gamma_x', 'weight_gamma_h',  \
                    'weight_rx', 'weight_rh','weight_rm',\
                    'weight_zx', 'weight_zh','weight_zm',\
                    'weight_hx', 'weight_hh','weight_hm',\
                    'bias_gamma_x', 'bias_gamma_h', 'bias_r', 'bias_z', 'bias_h']
        self._all_weights += [weights]
        self._flat_weights_names.extend(weights)
        self._flat_weights = [(lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn) for wn in self._flat_weights_names]

    def forward(self, values, masks, deltas, empirical_mean, X_filledLOCF):
        """
        """

        # init hidden state
        h = getattr(self, 'hidden_state')

        # decay rates gamma
        w_gamma_x = getattr(self, 'weight_gamma_x')
        w_gamma_h = getattr(self, 'weight_gamma_h')

        #z
        w_z_x = getattr(self, 'weight_zx')
        w_z_h = getattr(self, 'weight_zh')
        w_z_m = getattr(self, 'weight_zm')

        # r
        w_r_x = getattr(self, 'weight_rx')
        w_r_h = getattr(self, 'weight_rh')
        w_r_m = getattr(self, 'weight_rm')

        # h_tilde
        w_h_x = getattr(self, 'weight_hx')
        w_h_h = getattr(self, 'weight_hh')
        w_h_m = getattr(self, 'weight_hm')

        # bias
        b_gamma_x = getattr(self, 'bias_gamma_x')
        b_gamma_h = getattr(self, 'bias_gamma_h')
        b_z = getattr(self, 'bias_z')
        b_r = getattr(self, 'bias_r')
        b_h = getattr(self, 'bias_h')

        # self.check_forward_args(values, h, values.size()[0])

        hidden_tensor = torch.empty(values.size()[0], values.size()[1], self.hidden_dim, dtype=values.dtype, device = self.config.train.device_setting.device)

        for t in range(values.size()[1]):
            x = values[:, t, :]  # values
            m = masks[:, t, :]  # mask
            d = deltas[:, t, :]  # delta, time gap
            x_filledLOCF = X_filledLOCF[:, t, :]

            # another option
            gamma_x = torch.exp(-F.relu(w_gamma_x * d + b_gamma_x))
            gamma_h = torch.exp(-F.relu(d @ w_gamma_h + b_gamma_h))

            x = m * x + (1-m) * (gamma_x * x_filledLOCF + (1 - gamma_x) * empirical_mean)

            if self.dropout_type == 'Moon':
                '''
                RNNDROP: a novel dropout for rnn in asr(2015)
                '''
                h = gamma_h * h
                z = torch.sigmoid(x @ w_z_x + h @ w_z_h + m @ w_z_m + b_z)
                r = torch.sigmoid(x @ w_r_x + h @ w_r_h + m @ w_r_m + b_r)
                h_tilde = torch.tanh(x @ w_h_x + (r * h) @ w_h_h + m @ w_h_m + b_h)

                h = (1 - z) * h + z * h_tilde

                dropout = torch.nn.Dropout(p=self.dropout)
                h = dropout(h)
            
            elif self.dropout_type == 'Gal':
                '''
                A Theoretically grounded application of dropout in recurrent neural networks(2015)
                '''
                dropout = torch.nn.Dropout(p=self.dropout)
                h = dropout(h)

                h = gamma_h * h
                z = torch.sigmoid(x @ w_z_x + h @ w_z_h + m @ w_z_m + b_z)
                r = torch.sigmoid(x @ w_r_x + h @ w_r_h + m @ w_r_m + b_r)
                h_tilde = torch.tanh(x @ w_h_x + (r * h) @ w_h_h + m @ w_h_m + b_h)

                h = (1 - z) * h + z * h_tilde
            
            elif self.dropout_type == 'mloss':
                '''
                recurrent dropout without memory loss arXiv 1603.05118
                g = h_tilde, p = the probability to not drop a neuron
                '''
                h = gamma_h * h
                z = torch.sigmoid(x @ w_z_x + h @ w_z_h + m @ w_z_m + b_z)
                r = torch.sigmoid(x @ w_r_x + h @ w_r_h + m @ w_r_m + b_r)
                h_tilde = torch.tanh(x @ w_h_x + (r * h) @ w_h_h + m @ w_h_m + b_h)

                dropout = torch.nn.Dropout(p=self.dropout)
                h_tilde = dropout(h_tilde)

                h = (1 - z) * h + z * h_tilde

            else:
                h = gamma_h * h
                z = torch.sigmoid(x @ w_z_x + h @ w_z_h + m @ w_z_m + b_z)
                r = torch.sigmoid(x @ w_r_x + h @ w_r_h + m @ w_r_m + b_r)
                h_tilde = torch.tanh(x @ w_h_x + (r * h) @ w_h_h + m @ w_h_m + b_h)

                h = (1 - z) * h + z * h_tilde

            hidden_tensor[:, t, :] = h

        return hidden_tensor, h


