# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from typing import List

from parsytongue.model.utils import DeviceGetterMixin


class FeedforwardPredictor(nn.Module, DeviceGetterMixin):
    def __init__(self, input_dim: int, output_dim: int, out_key_prefix: str = None):
        super(FeedforwardPredictor, self).__init__()

        self._output = nn.Linear(input_dim, output_dim)
        self._out_key_prefix = out_key_prefix + '_' or ''

    def forward(self, inputs: torch.Tensor):
        logits = self._output(inputs)
        probs = torch.softmax(logits, dim=-1)

        predictions = torch.max(probs, dim=-1)

        return {
            self._out_key_prefix + 'predictions': predictions.indices.data.cpu().numpy(),
            self._out_key_prefix + 'probs': predictions.values.data.cpu().numpy(),
        }
