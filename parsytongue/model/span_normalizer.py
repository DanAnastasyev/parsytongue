# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from typing import List, Dict

from parsytongue.model.configs import SpanNormalizerConfig
from parsytongue.model.encoders import LstmEncoder
from parsytongue.model.feedforward_predictor import FeedforwardPredictor
from parsytongue.model.utils import DeviceGetterMixin
from parsytongue.model.vectorizers import VectorizerFactory
from parsytongue.parser.sentence import Span


class SpanNormalizer(nn.Module, DeviceGetterMixin):
    def __init__(self, encoder_dim: int, config: SpanNormalizerConfig):
        super(SpanNormalizer, self).__init__()

        self._config = config

        self._vectorizer = VectorizerFactory.build(**config.vectorizer)

        input_dim = encoder_dim
        if config.use_possible_lemmas_feature:
            input_dim += config.normal_form_count

        # self._span_embedding = nn.Embedding(3, 16)

        self._encoder = LstmEncoder(input_dim=input_dim, config=config.encoder_params)
        self._output = FeedforwardPredictor(self._encoder.get_output_size(), config.normal_form_count, 'norms')

    def forward(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor,
        lemmas: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        assert self._config.use_possible_lemmas_feature == (lemmas is not None)

        # spans = torch.ones(embeddings.shape[:-1], dtype=torch.long) * 2
        # spans[:, 0] = 1
        # spans = self._span_embedding(spans)
        # embeddings = torch.cat([embeddings, spans], -1)

        if lemmas is not None:
            embeddings = torch.cat([embeddings, lemmas], -1)

        embeddings = self._encoder(embeddings, mask)
        return self._output(embeddings)

    def apply(self, spans: List[Span]):
        with torch.no_grad():
            vectorized_inputs = self._vectorizer(spans)
            return self(**vectorized_inputs)
