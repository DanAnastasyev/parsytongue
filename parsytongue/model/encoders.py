# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from typing import Dict

from parsytongue.model.lstm_wrapper import LstmWrapper
from parsytongue.model.utils import DeviceGetterMixin, BaseFactory


class EncoderFactory(BaseFactory):
    registry = {}


class Encoder(object):
    def get_output_size(self):
        raise NotImplementedError


@EncoderFactory.register('lstm')
class LstmEncoder(nn.Module, Encoder, DeviceGetterMixin):
    def __init__(self, input_dim: int, config: Dict[str, int]):
        super(LstmEncoder, self).__init__()

        self._input_dim = input_dim
        self._config = config

        self._encoder = LstmWrapper(
            input_size=input_dim,
            hidden_size=config['hidden_dim'],
            num_layers=config['num_layers'],
            bidirectional=True,
            batch_first=True
        )

    def get_output_size(self):
        return self._config['hidden_dim'] * 2

    def forward(self, inputs: torch.Tensor, mask: torch.LongTensor) -> torch.Tensor:
        assert len(inputs.shape) == 3, \
            'Expected 3D tensor (batch_size, sent_len, word_emb_size), found {}'.format(len(inputs.shape))

        batch_size, sent_len, word_emb_size = inputs.shape
        assert word_emb_size == self._input_dim

        outputs, _ = self._encoder(inputs, mask)
        assert outputs.shape == (batch_size, sent_len, self.get_output_size())

        return outputs


@EncoderFactory.register('pass_through')
class PassThroughEncoder(nn.Module, Encoder, DeviceGetterMixin):
    def __init__(self, input_dim: int, config: Dict[str, int]):
        super(PassThroughEncoder, self).__init__()

        self._input_dim = input_dim
        self._config = config

    def get_output_size(self):
        return self._config['output_dim']

    def forward(self, inputs: torch.Tensor, mask: torch.LongTensor) -> torch.Tensor:
        assert len(inputs.shape) == 3, \
            'Expected 3D tensor (batch_size, sent_len, word_emb_size), found {}'.format(len(inputs.shape))
        assert inputs.shape[-1] == self._input_dim

        return inputs