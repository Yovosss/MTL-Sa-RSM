#!/usr/bin/env python
# coding:utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreSpatialAttn(nn.Module):
    def __init__(self, config, hidden_size):
        super(PreSpatialAttn, self).__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.fc = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, inputs):
        """
        param: inputs: (batch_size, time_steps, features)
        """
        scores = self.fc(inputs)
        attention_weights = F.softmax(scores, dim=2)
        output_attention_mul = torch.mul(inputs, attention_weights)

        return output_attention_mul


