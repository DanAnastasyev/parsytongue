# -*- coding: utf-8 -*-

import attr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import logging

from typing import Dict, List

from parsytongue.model.lstm_wrapper import LstmWrapper
from parsytongue.model.perf_counter import timed
from parsytongue.model.highway import Highway

logger = logging.getLogger(__name__)


class DeviceGetterMixin(object):
    @property
    def param(self):
        return next(self.parameters())

    @property
    def device(self):
        return next(self.parameters()).device


@attr.s
class EmbedderConfig(object):
    name = attr.ib()
    params = attr.ib(default=None)


class Embedder(object):
    def get_output_size(self):
        raise NotImplementedError


class CharacterLevelEmbedder(nn.Module, Embedder, DeviceGetterMixin):
    NAME = 'chars'

    def __init__(self, config: Dict[str, int]):
        super(CharacterLevelEmbedder, self).__init__()

        self._config = config
        self._input_name = config['input_name']

        self._embedding = nn.Embedding(config['char_count'], config['char_embedding_dim'])
        self._encoder = LstmWrapper(
            input_size=config['char_embedding_dim'],
            hidden_size=config['lstm_dim'],
            num_layers=config['lstm_num_layers'],
            bidirectional=True,
            batch_first=True
        )

    def get_output_size(self):
        return self._config['lstm_dim'] * 2

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = inputs[self._input_name]

        assert len(inputs.shape) == 3, \
            'Expected 3D tensor (batch_size, sent_len, word_len), found {}'.format(inputs.shape)

        batch_size, sent_len, word_len = inputs.shape

        outputs = self._embedding(inputs)
        outputs = outputs.view(batch_size * sent_len, word_len, self._config['char_embedding_dim'])

        mask = (inputs != 0).view(batch_size * sent_len, word_len)
        _, (last_hidden_state, _) = self._encoder(outputs, mask)
        assert last_hidden_state.shape == (2, batch_size * sent_len, self._config['lstm_dim'])

        last_hidden_state = torch.cat([last_hidden_state[0], last_hidden_state[1]], -1)
        last_hidden_state = last_hidden_state.view(batch_size, sent_len, self.get_output_size())

        return last_hidden_state


class UncontextualizedELMoEmbedder(nn.Module, Embedder, DeviceGetterMixin):
    NAME = 'uncontextualized_elmo'

    def __init__(self, config: Dict[str, int]):
        super(UncontextualizedELMoEmbedder, self).__init__()

        self._config = config
        self._input_name = config['input_name']

        char_embedding_dim = config['char_embedding_dim']
        self._embedding = nn.Embedding(config['char_count'], char_embedding_dim)

        self._convolutions = []
        convs_output_dim = 0
        for i, (kernel_size, out_channels) in enumerate(config['filters']):
            conv = torch.nn.Conv1d(
                in_channels=char_embedding_dim, out_channels=out_channels,
                kernel_size=kernel_size, bias=True
            )
            self._convolutions.append(conv)
            self.add_module('char_conv_{}'.format(i), conv)

            convs_output_dim += out_channels

        self._activation = F.relu

        if config['use_projections']:
            projection_dim = config['projection_dim']
            self._highways = Highway(convs_output_dim, config['highway_count'], activation=F.relu)
            self._projection = torch.nn.Linear(convs_output_dim, projection_dim, bias=True)
            self._output_dim = projection_dim
        else:
            self._output_dim = convs_output_dim

    def get_output_size(self):
        return self._output_dim

    def forward(self, inputs: Dict[str, torch.Tensor]):
        inputs = inputs[self._input_name]

        assert len(inputs.shape) == 3, \
            'Expected 3D tensor (batch_size, sent_len, word_len), found {}'.format(inputs.shape)

        batch_size, sent_len, word_len = inputs.shape

        outputs = self._embedding(inputs)
        outputs = outputs.view(batch_size * sent_len, word_len, self._config['char_embedding_dim'])

        outputs = torch.transpose(outputs, 1, 2)
        conv_outputs = []
        for i in range(len(self._convolutions)):
            conv = getattr(self, 'char_conv_{}'.format(i))
            convolved = conv(outputs)
            convolved, _ = torch.max(convolved, dim=-1)
            convolved = self._activation(convolved)
            conv_outputs.append(convolved)

        token_embeddings = torch.cat(conv_outputs, dim=-1)

        if self._config['use_projections']:
            token_embeddings = self._highways(token_embeddings)
            token_embeddings = self._projection(token_embeddings)

        return token_embeddings.view(batch_size, sent_len, self._output_dim)


class PassThroughEmbedder(nn.Module, Embedder, DeviceGetterMixin):
    NAME = 'pass_through'

    def __init__(self, config: Dict[str, int]):
        super(PassThroughEmbedder, self).__init__()

        self._dim = config['dim']
        self._input_name = config['input_name']

    def get_output_size(self):
        return self._dim

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = inputs[self._input_name]

        assert len(inputs.shape) == 3, \
            'Expected 3D tensor (batch_size, sent_len, embed_dim), found {}'.format(inputs.shape)
        assert inputs.shape[-1] == self._dim

        return inputs


class EmbedderStack(nn.Module, Embedder, DeviceGetterMixin):
    _REGISTERED_EMBEDDERS = [CharacterLevelEmbedder, PassThroughEmbedder, UncontextualizedELMoEmbedder]

    def __init__(self, embedders) -> None:
        super(EmbedderStack, self).__init__()

        self._embedders = nn.ModuleList(embedders)

    def get_output_size(self):
        return sum(embedder.get_output_size() for embedder in self._embedders)

    def forward(self, inputs):
        return torch.cat([embedder(inputs) for embedder in self._embedders], -1)

    @classmethod
    def from_config(cls, configs: List[EmbedderConfig]):
        embedders = []
        for config in configs:
            for embedder_cls in cls._REGISTERED_EMBEDDERS:
                if config.name == embedder_cls.NAME:
                    embedders.append(embedder_cls(config.params))

        return cls(embedders)


class CachableUncontextualizedEmbedderStack(nn.Module, Embedder, DeviceGetterMixin):
    _REGISTERED_EMBEDDERS = [CharacterLevelEmbedder, PassThroughEmbedder, UncontextualizedELMoEmbedder]

    def __init__(self, vectorizer, embedders) -> None:
        super(CachableUncontextualizedEmbedderStack, self).__init__()

        self._vectorizer = vectorizer
        self._embedders = nn.ModuleList(embedders)
        self._embeddings_cache = {}

    def get_output_size(self):
        return sum(embedder.get_output_size() for embedder in self._embedders)

    def forward(self, inputs):
        sentence_length = len(inputs)

        token_embeddings = torch.zeros((1, sentence_length, self.get_output_size())).to(self.device)

        unknown_tokens, unknown_token_indices = [], []
        for token_index, token in enumerate(inputs):
            token_embedding = self._embeddings_cache.get(token)
            if token_embedding is not None:
                token_embeddings[0, token_index] = token_embedding.to(self.device)
            else:
                unknown_tokens.append(token)
                unknown_token_indices.append(token_index)

        if unknown_tokens:
            vectorized_unknown_tokens = self._vectorizer(unknown_tokens)
            vectorized_unknown_tokens = {
                key: val.unsqueeze(0).to(self.device)
                for key, val in vectorized_unknown_tokens.items()
            }

            unknown_token_embeddings = torch.cat(
                [embedder(vectorized_unknown_tokens) for embedder in self._embedders], dim=-1
            )
            token_embeddings[0, unknown_token_indices] = unknown_token_embeddings

            for unknown_token, unknown_token_embedding in zip(unknown_tokens, unknown_token_embeddings[0]):
                self._embeddings_cache[unknown_token] = unknown_token_embedding

        # vectorized_inputs = self._vectorizer(inputs)
        # vectorized_inputs = {key: val.unsqueeze(0) for key, val in vectorized_inputs.items()}
        # token_embeddings = torch.cat([embedder(vectorized_inputs) for embedder in self._embedders], dim=-1)

        return {
            'token_embeddings': token_embeddings,
            'mask': torch.ones((1, sentence_length)).to(self.device)
        }

    @classmethod
    def from_config(cls, vectorizers, configs: List[EmbedderConfig]):
        embedders = []
        for config in configs:
            for embedder_cls in cls._REGISTERED_EMBEDDERS:
                if config.name == embedder_cls.NAME:
                    embedders.append(embedder_cls(config.params))

        return cls(vectorizers, embedders)
