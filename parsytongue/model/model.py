# -*- coding: utf-8 -*-

import logging

import torch
import torch.nn as nn

from typing import Dict, List

from parsytongue.model.configs import ModelConfig
from parsytongue.model.embedders import EmbedderFactory
from parsytongue.model.encoders import EncoderFactory
from parsytongue.model.feedforward_predictor import FeedforwardPredictor
from parsytongue.model.perf_counter import timed
from parsytongue.model.span_normalizer import SpanNormalizer
from parsytongue.model.syntax_parser import SyntaxParser
from parsytongue.model.utils import DeviceGetterMixin
from parsytongue.model.vectorizers import VectorizerFactory
from parsytongue.parser.sentence import Sentence

logger = logging.getLogger(__name__)


class Model(nn.Module, DeviceGetterMixin):
    def __init__(self, config: ModelConfig) -> None:
        super(Model, self).__init__()

        self._vectorizer = VectorizerFactory.build(**config.vectorizer)
        self._embedder = EmbedderFactory.build(**config.embedder)
        self._encoder = EncoderFactory.build(input_dim=self._embedder.get_output_size(), **config.encoder)

        encoder_dim = self._encoder.get_output_size()

        self._grammar_value_output = FeedforwardPredictor(encoder_dim, config.decoder.grammar_value_count, 'gram_vals')
        self._lemma_output = FeedforwardPredictor(encoder_dim, config.decoder.lemma_count, 'lemmas')
        self._syntax_parser = SyntaxParser(encoder_dim, config.decoder.syntax_parser)
        self.span_normalizer = SpanNormalizer(encoder_dim, config.span_normalizer)

    def forward(
        self,
        vectorized_inputs: Dict[str, torch.Tensor],
        predict_morphology: bool,
        predict_lemmas: bool,
        predict_syntax: bool,
        return_embeddings: bool
    ) -> Dict[str, torch.Tensor]:
        mask = vectorized_inputs['mask']

        with timed('Embedder'):
            embedded_text_input = self._embedder(vectorized_inputs)

        with timed('Encoder'):
            encoded_text = self._encoder(embedded_text_input, mask)

        output_dict = {}
        if return_embeddings:
            output_dict['embeddings'] = encoded_text

        if predict_morphology:
            with timed('GramValOut'):
                output_dict.update(self._grammar_value_output(encoded_text))

        if predict_lemmas:
            with timed('LemmaOut'):
                output_dict.update(self._lemma_output(encoded_text))

        if predict_syntax:
            with timed('SyntaxOut'):
                output_dict.update(self._syntax_parser(encoded_text, mask))

        return output_dict

    def apply(
        self,
        sentences: List[Sentence],
        predict_morphology: bool = True,
        predict_lemmas: bool = True,
        predict_syntax: bool = True,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            with timed('Vectorizer'):
                vectorized_inputs = self._vectorizer(sentences)
            return self(vectorized_inputs, predict_morphology, predict_lemmas, predict_syntax, return_embeddings)
