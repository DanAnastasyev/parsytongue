# -*- coding: utf-8 -*-

import attr
import logging
import json
import torch
import numpy as np

from typing import Dict, List

from parsytongue.model.model import Model, ModelConfig, DecoderConfig, EmbedderConfig, EncoderConfig, VectorizerConfig
from parsytongue.model.perf_counter import timed

from parsytongue.parser.lemmatize_helper import LemmatizeHelper
from parsytongue.parser.vocabulary import Vocabulary

logger = logging.getLogger(__name__)


@attr.s
class VocabularyConfig(object):
    vocab_path = attr.ib()
    lemmatizer_path = attr.ib()


@attr.s
class ParserConfig(object):
    model = attr.ib(validator=attr.validators.instance_of(ModelConfig))
    vocab = attr.ib(validator=attr.validators.instance_of(VocabularyConfig))

    @classmethod
    def load(cls, path):
        with open(path) as f:
            config_json = json.load(f)

        model_config_json = config_json['model']

        vectorizer_configs = [
            VectorizerConfig(**vectorizer_config) for vectorizer_config in model_config_json['vectorizers']
        ]
        embedder_configs = [
            EmbedderConfig(**embedder_config) for embedder_config in model_config_json['embedders']
        ]
        encoder_config = EncoderConfig(**model_config_json['encoder'])
        decoder_config = DecoderConfig(**model_config_json['decoder'])

        model_config = ModelConfig(
            vectorizers=vectorizer_configs,
            embedders=embedder_configs,
            encoder=encoder_config,
            decoder=decoder_config,
            weights_path=model_config_json['weights_path']
        )

        vocab_config = VocabularyConfig(**config_json['vocab'])

        return cls(model=model_config, vocab=vocab_config)


_PARAMS_MAPPING = {
    'text_field_embedder.token_embedder_char_bilstm._embedding._module': '_embedder._embedders.0._embedding',
    'text_field_embedder.token_embedder_char_bilstm._encoder._module._module': '_embedder._embedders.0._encoder',
    'encoder._module': '_encoder._encoder',
    'child_arc_feedforward._linear_layers': '_child_arc_feedforward',
    'child_tag_feedforward._linear_layers': '_child_tag_feedforward',
    'head_arc_feedforward._linear_layers': '_head_arc_feedforward',
    'head_tag_feedforward._linear_layers': '_head_tag_feedforward',
    'tag_bilinear': '_tag_bilinear',
    'arc_attention': '_arc_attention',
    '_gram_val_output': '_grammar_value_output',
    'text_field_embedder.token_embedder_elmo._elmo._char_embedding_weights': '_embedder._embedders.0._embedding.weight',
    'text_field_embedder.token_embedder_elmo._elmo.char': '_embedder._embedders.0.char',
    'text_field_embedder.token_embedder_elmo._elmo._highways': '_embedder._embedders.0._highways',
    'text_field_embedder.token_embedder_elmo._elmo._projection': '_embedder._embedders.0._projection',
    'text_field_embedder.token_embedder_ru_bert._matched_embedder.transformer_model': '_embedder._embedders.0._bert',
}


def _load_weights(path):
    weights = torch.load(path, map_location=torch.device('cpu'))

    renamed_weights = {}
    for name, weight in weights.items():
        for rename_from, rename_to in _PARAMS_MAPPING.items():
            if name.startswith(rename_from):
                name = name.replace(rename_from, rename_to)
                break
        renamed_weights[name] = weight

    return renamed_weights


class Parser(object):
    def __init__(self, config: ParserConfig):
        self._vocab = Vocabulary.from_files(config.vocab.vocab_path)
        self._lemmatize_helper = LemmatizeHelper.load(config.vocab.lemmatizer_path)

        self._vectorizer, self._model = Model.from_config(self._vocab, config.model)
        logger.info('Loaded model:\n%s', self._model)

        self._model.load_state_dict(_load_weights(config.model.weights_path))
        self._model.eval()

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._model.to(device)

    def parse(
        self,
        sentence: List[str],
        predict_morphology: bool = True,
        predict_lemmas: bool = False,
        predict_syntax: bool = True,
    ):
        with timed('Run model'):
            with torch.no_grad():
                inputs = self._vectorizer(sentence)
                bert_inputs = inputs['bert']
                bert_inputs = {key: val.unsqueeze(0).to(self._model.device) for key, val in bert_inputs.items()}
                predictions = self._model(bert_inputs, predict_morphology, predict_lemmas, predict_syntax)

        with timed('Decode predictions'):
            return self._decode(sentence, predictions)

    def _decode(self, sentence: List[str], predictions: Dict[str, torch.Tensor]):
        outputs = {}
        if 'gram_vals' in predictions:
            outputs['grammar_values'] = [
                self._vocab.get_token_from_index(grammar_value_index, 'grammar_value_tags')
                for grammar_value_index in predictions['gram_vals'][0].cpu().numpy()
            ]

        if 'lemmas' in predictions:
            outputs['lemmas'] = [
                self._lemmatize_helper.lemmatize(token, lemmatize_rule_index)
                for token, lemmatize_rule_index in zip(sentence, predictions['lemmas'][0].cpu().numpy())
            ]

        if 'head_tags' in predictions:
            outputs['head_tags'] = [
                self._vocab.get_token_from_index(grammar_value_index, 'head_tags')
                for grammar_value_index in predictions['head_tags'][0, 1:].cpu().numpy()
            ]

            outputs['heads'] = predictions['heads'][0, 1:].cpu().numpy()

        return outputs
