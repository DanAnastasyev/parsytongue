# -*- coding: utf-8 -*-

import attr
import logging
import json
import torch
import numpy as np

from typing import Dict, List

from parsytongue.model.model import Model, ModelConfig, DecoderConfig, EmbedderConfig, EncoderConfig, VectorizerConfig
from parsytongue.model.perf_counter import timed, show_perf_results, clear_perf_counters

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


class UncontextualizedVectorizer(object):
    def __init__(self):
        from parsytongue.model.vectorizers import ELMoVectorizer, MorphoVectorizer

        self._elmo_vectorizer = ELMoVectorizer()
        self._elmo_embedder = torch.jit.load('small_embedder.pth')
        self._morpho_vectorizer = MorphoVectorizer()

        self._output_dim = 2048 + 63

        self._embeddings_cache = {}

    def __call__(self, tokens):
        token_embeddings = torch.zeros((1, len(tokens), self._output_dim))

        unknown_tokens, unknown_token_indices = [], []
        for token_index, token in enumerate(tokens):
            token_embedding = self._embeddings_cache.get(token)
            if token_embedding is not None:
                token_embeddings[0, token_index] = token_embedding
            else:
                unknown_tokens.append(token)
                unknown_token_indices.append(token_index)

        if unknown_tokens:
            elmo_input = self._elmo_vectorizer(unknown_tokens)
            elmo_input = elmo_input.unsqueeze(0)
            elmo_embeddings = self._elmo_embedder(elmo_input)

            morpho_embeddings = self._morpho_vectorizer(unknown_tokens).unsqueeze(0)

            unknown_token_embeddings = torch.cat((elmo_embeddings, morpho_embeddings), -1)

            token_embeddings[0, unknown_token_indices] = unknown_token_embeddings

            for unknown_token, unknown_token_embedding in zip(unknown_tokens, unknown_token_embeddings[0]):
                self._embeddings_cache[unknown_token] = unknown_token_embedding

        return token_embeddings


class Parser(object):
    def __init__(self, config: ParserConfig):
        self._vocab = Vocabulary.from_files(config.vocab.vocab_path)
        self._lemmatize_helper = LemmatizeHelper.load(config.vocab.lemmatizer_path)

        self._vectorizer = UncontextualizedVectorizer()
        self._model = torch.jit.load('small_model.pth')

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._model.to(device)

    def parse(
        self,
        sentence: List[str],
        predict_morphology: bool = True,
        predict_lemmas: bool = True,
        predict_syntax: bool = True,
    ):
        with timed('Full parse'):
            with torch.no_grad():
                embeddings = self._vectorizer(sentence)
                mask = torch.ones(embeddings.shape[:-1])
                gram_vals, lemmas, batch_energy = self._model(
                    embeddings, mask, predict_morphology, predict_lemmas, predict_syntax
                )

                predictions = {}

                if gram_vals is not None:
                    predictions['gram_vals'] = gram_vals[0].cpu().numpy()

                if lemmas is not None:
                    predictions['lemmas'] = lemmas[0].cpu().numpy()

                if batch_energy is not None:
                    heads, head_tags = self._run_mst_decoding(batch_energy)
                    predictions['heads'] = heads
                    predictions['head_tags'] = head_tags

            return self._decode(sentence, predictions)

    def _run_mst_decoding(self, batch_energy):
        scores, tag_ids = batch_energy[0].max(dim=0)

        from dependency_decoding import chu_liu_edmonds

        scores = scores.numpy().astype('float64')
        instance_heads, _ = chu_liu_edmonds(scores.T)

        # Find the labels which correspond to the edges in the max spanning tree.
        instance_head_tags = []
        for child, parent in enumerate(instance_heads):
            instance_head_tags.append(tag_ids[parent, child].item())
        # We don't care what the head or tag is for the root token, but by default it's
        # not necessarily the same in the batched vs unbatched case, which is annoying.
        # Here we'll just set them to zero.
        instance_heads[0] = 0
        instance_head_tags[0] = 0

        return instance_heads, instance_head_tags

    def _decode(self, sentence: List[str], predictions: Dict[str, torch.Tensor]):
        outputs = {}
        if 'gram_vals' in predictions:
            outputs['grammar_values'] = [
                self._vocab.get_token_from_index(grammar_value_index, 'grammar_value_tags')
                for grammar_value_index in predictions['gram_vals']
            ]

        if 'lemmas' in predictions:
            outputs['lemmas'] = [
                self._lemmatize_helper.lemmatize(token, lemmatize_rule_index)
                for token, lemmatize_rule_index in zip(sentence, predictions['lemmas'])
            ]

        if 'head_tags' in predictions:
            outputs['head_tags'] = [
                self._vocab.get_token_from_index(grammar_value_index, 'head_tags')
                for grammar_value_index in predictions['head_tags'][1:]
            ]

            outputs['heads'] = predictions['heads'][1:]

        return outputs

    def show_perf_results(self):
        show_perf_results()

    def clear_perf_counters(self):
        clear_perf_counters()
