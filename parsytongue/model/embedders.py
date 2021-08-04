# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import logging

from typing import Dict, List

from parsytongue.model.lstm_wrapper import LstmWrapper
from parsytongue.model.utils import DeviceGetterMixin, BaseFactory
from parsytongue.model.highway import Highway

logger = logging.getLogger(__name__)


class EmbedderFactory(BaseFactory):
    registry = {}

    @classmethod
    def build(cls, name: str, *args, **kwargs):
        instance = super(EmbedderFactory, cls).build(name, *args, **kwargs)
        instance.eval()
        return instance


class Embedder(object):
    def get_output_size(self):
        raise NotImplementedError


@EmbedderFactory.register('chars')
class CharacterLevelEmbedder(nn.Module, Embedder, DeviceGetterMixin):
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


@EmbedderFactory.register('uncontextualized_elmo')
class UncontextualizedELMoEmbedder(nn.Module, Embedder, DeviceGetterMixin):
    def __init__(self, params: Dict[str, int]):
        super(UncontextualizedELMoEmbedder, self).__init__()

        self._config = params
        self._input_name = params['input_name']

        char_embedding_dim = params['char_embedding_dim']
        self._embedding = nn.Embedding(params['char_count'], char_embedding_dim)

        self._convolutions = []
        convs_output_dim = 0
        for i, (kernel_size, out_channels) in enumerate(params['filters']):
            conv = torch.nn.Conv1d(
                in_channels=char_embedding_dim, out_channels=out_channels,
                kernel_size=kernel_size, bias=True
            )
            self._convolutions.append(conv)
            self.add_module('char_conv_{}'.format(i), conv)

            convs_output_dim += out_channels

        if params['use_projections']:
            projection_dim = params['projection_dim']
            self._highways = Highway(convs_output_dim, params['highway_count'], activation=F.relu)
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
            convolved = F.relu(convolved)
            conv_outputs.append(convolved)

        token_embeddings = torch.cat(conv_outputs, dim=-1)

        if self._config['use_projections']:
            token_embeddings = self._highways(token_embeddings)
            token_embeddings = self._projection(token_embeddings)

        return token_embeddings.view(batch_size, sent_len, self._output_dim)


@EmbedderFactory.register('pass_through')
class PassThroughEmbedder(nn.Module, Embedder, DeviceGetterMixin):
    def __init__(self, params: Dict[str, int]):
        super(PassThroughEmbedder, self).__init__()

        self._dim = params['dim']
        self._input_name = params['input_name']

    def get_output_size(self):
        return self._dim

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = inputs[self._input_name]

        assert len(inputs.shape) == 3, \
            'Expected 3D tensor (batch_size, sent_len, embed_dim), found {}'.format(inputs.shape)
        assert inputs.shape[-1] == self._dim

        return inputs


@EmbedderFactory.register('bert')
class BertEmbedder(nn.Module, Embedder, DeviceGetterMixin):
    def __init__(self, params: Dict):
        from transformers import BertModel, BertConfig

        super(BertEmbedder, self).__init__()

        self._config = BertConfig(**params)
        self._bert = BertModel(self._config)
        self._input_name = 'bert'

    def get_output_size(self):
        return self._config.hidden_size

    def forward(self, inputs):
        from allennlp.nn.util import batched_span_select

        inputs = inputs[self._input_name]

        # TODO: Process long inputs
        embeddings, _ = self._bert(
            input_ids=inputs['token_ids'],
            attention_mask=inputs['mask'],
            token_type_ids=inputs['segment_ids'],
        )

        # span_embeddings: (batch_size, num_orig_tokens, max_span_length, embedding_size)
        # span_mask: (batch_size, num_orig_tokens, max_span_length)
        span_embeddings, span_mask = batched_span_select(embeddings.contiguous(), inputs['offsets'])
        span_mask = span_mask.unsqueeze(-1)
        span_embeddings *= span_mask  # zero out paddings

        span_embeddings_sum = span_embeddings.sum(2)
        span_embeddings_len = span_mask.sum(2)
        # Shape: (batch_size, num_orig_tokens, embedding_size)
        orig_embeddings = span_embeddings_sum / span_embeddings_len

        # All the places where the span length is zero, write in zeros.
        orig_embeddings = torch.where(
            (span_embeddings_len == 0).expand_as(orig_embeddings),
            torch.zeros_like(orig_embeddings),
            orig_embeddings
        )

        return orig_embeddings


@EmbedderFactory.register('stack')
class EmbedderStack(nn.Module, Embedder, DeviceGetterMixin):
    def __init__(self, embedders) -> None:
        super(EmbedderStack, self).__init__()

        self._embedders = nn.ModuleList([EmbedderFactory.build(**params) for params in embedders])

    def get_output_size(self):
        return sum(embedder.get_output_size() for embedder in self._embedders)

    def forward(self, inputs):
        token_embeddings = torch.cat([embedder(inputs) for embedder in self._embedders], -1)
        return token_embeddings


import os

def _fix_path(config, dir_path):
    if isinstance(config, list):
        for element in config:
            if isinstance(element, dict):
                _fix_path(element, dir_path)
        return

    for key, value in config.items():
        if key.endswith('_path') and isinstance(value, str):
            config[key] = os.path.join(dir_path, value)
        if isinstance(value, (dict, list)):
            _fix_path(value, dir_path)


def main():
    from parsytongue.parser.sentence import Sentence
    from parsytongue.model.vectorizers import VectorizerFactory

    dir_path = "models"

    json_config = {
        "vectorizers": {
            "name": "stack",
            "vectorizers": [
                {
                    "name": "elmo",
                    "vector_key": "elmo"
                },
                {
                    "name": "morph",
                    "vector_key": "morpho_embeddings",
                    "grammeme_to_index_path": "grammeme_to_index.json"
                },
                {
                    "name": "mask",
                    "vector_key": "mask"
                }
            ]
        },
        "embedders": {
            "name": "stack",
            "embedders": [
                {
                    "name": "uncontextualized_elmo",
                    "params": {
                        "input_name": "elmo",
                        "char_count": 262,
                        "char_embedding_dim": 16,
                        "filters": [
                            [
                                1,
                                32
                            ],
                            [
                                2,
                                32
                            ],
                            [
                                3,
                                64
                            ],
                            [
                                4,
                                128
                            ],
                            [
                                5,
                                256
                            ],
                            [
                                6,
                                512
                            ],
                            [
                                7,
                                1024
                            ]
                        ],
                        "use_projections": False,
                        "highway_count": 0,
                        "projection_dim": -1
                    }
                },
                {
                    "name": "pass_through",
                    "params": {
                        "input_name": "morpho_embeddings",
                        "dim": 63
                    }
                }
            ]
        }
    }

    _fix_path(json_config, dir_path)

    print(EmbedderFactory.registry)

    vectorizer = VectorizerFactory.build(**json_config['vectorizers'])
    embedder = EmbedderFactory.build(**json_config['embedders'])

    sentences = [Sentence("мама раму")]

    batch = vectorizer(sentences)
    embedder.eval()
    with torch.no_grad():
        print(embedder(batch).shape)


if __name__ == '__main__':
    main()
