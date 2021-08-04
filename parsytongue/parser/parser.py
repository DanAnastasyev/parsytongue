# -*- coding: utf-8 -*-

import logging
import pymorphy2
import torch

from typing import Dict, List

from parsytongue.model.model import Model, ModelConfig
from parsytongue.model.perf_counter import timed, show_perf_results, clear_perf_counters

from parsytongue.parser.lemmatize_helper import LemmatizeHelper
from parsytongue.parser.vocabulary import Vocabulary
from parsytongue.parser.markup import GrammarValue, SyntaxRelation
from parsytongue.parser.sentence import Sentence, Span

logger = logging.getLogger(__name__)


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


_PARAMS_MAPPING = {
    'encoder._module': '_encoder._encoder',
    '_gram_val_output': '_grammar_value_output._output',
    '_lemma_output': '_lemma_output._output',
    'text_field_embedder.token_embedder_elmo._elmo._char_embedding': '_vectorizer._embedder._embedders.0._embedding',
    'text_field_embedder.token_embedder_elmo._elmo': '_vectorizer._embedder._embedders.0',
    '_head_sentinel': '_syntax_parser._head_sentinel',
    'head_arc_feedforward._linear_layers.0': '_syntax_parser._head_arc_feedforward.0',
    'child_arc_feedforward._linear_layers.0': '_syntax_parser._child_arc_feedforward.0',
    'head_tag_feedforward._linear_layers.0': '_syntax_parser._head_tag_feedforward.0',
    'child_tag_feedforward._linear_layers.0': '_syntax_parser._child_tag_feedforward.0',
    'arc_attention': '_syntax_parser._arc_attention',
    'tag_bilinear': '_syntax_parser._tag_bilinear',
    '_span_label_embedding': 'span_normalizer._span_embedding',
    'span_encoder._module': 'span_normalizer._encoder._encoder',
    '_span_norm_output': 'span_normalizer._output._output',
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
    def __init__(self):
        dir_path = '../../GramEval2020/models/span_normalization/' \
                   'multitask_trainable_elmo_convs_small_encoder_decoder_frozen_3'

        self._vocab = Vocabulary.from_files('vocab')
        self._lemmatize_helper = LemmatizeHelper.load(os.path.join(dir_path, 'lemmatizer_info.json'))
        self._span_normalize_helper = LemmatizeHelper.load(os.path.join(dir_path, 'span_norms_lemmatizer_info.json'))
        self._morph = pymorphy2.MorphAnalyzer()

        with open('models/tiny_model.json') as f:
            import json
            json_config = json.load(f)

        _fix_path(json_config, dir_path='models')

        config = ModelConfig(**json_config['model'])
        self._model = Model(config)

        state_dict = _load_weights(os.path.join(dir_path, 'best.th'))
        self._model.load_state_dict(state_dict)

        self._grammar_values = []
        for grammar_value_index in range(self._vocab.get_vocab_size('grammar_value_tags')):
            grammar_value_string = self._vocab.get_token_from_index(grammar_value_index, 'grammar_value_tags')
            self._grammar_values.append(GrammarValue.from_string(grammar_value_string))

        self._head_tags = []
        for head_tag_index in range(self._vocab.get_vocab_size('head_tags')):
            head_tag = self._vocab.get_token_from_index(head_tag_index, 'head_tags')
            self._head_tags.append(SyntaxRelation.from_string(head_tag))

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._model.to(device)

    def to(self, device: torch.device):
        self._model.to(device)

    def parse(
        self,
        sentences: List[Sentence],
        predict_morphology: bool = True,
        predict_lemmas: bool = True,
        predict_syntax: bool = True,
        return_embeddings: bool = False,
    ):
        for sentence in sentences:
            for token in sentence.tokens:
                if not token._pymorphy_forms:
                    token._pymorphy_forms = self._morph.parse(token.text)

        predictions = self._model.apply(
            sentences=sentences,
            predict_morphology=predict_morphology,
            predict_lemmas=predict_lemmas,
            predict_syntax=predict_syntax,
            return_embeddings=return_embeddings,
        )

        for sent_index, sentence in enumerate(sentences):
            self._decode(sentence, predictions, sent_index)

    def _decode(self, sentence: Sentence, predictions: Dict[str, torch.Tensor], sent_index: int):
        if 'gram_vals_predictions' in predictions:
            for grammar_value_index, token in zip(predictions['gram_vals_predictions'][sent_index], sentence.tokens):
                token.grammar_value = self._grammar_values[grammar_value_index]

        if 'lemmas_predictions' in predictions:
            for lemmatize_rule_index, token in zip(predictions['lemmas_predictions'][sent_index], sentence.tokens):
                token.lemma = self._lemmatize_helper.lemmatize(token.text, lemmatize_rule_index)

        if 'head_tags' in predictions:
            heads, head_tags = predictions['heads'][sent_index], predictions['head_tags'][sent_index]
            for head_index, head_tag_index, token in zip(heads[1:], head_tags[1:], sentence.tokens):
                token.head_tag = self._head_tags[head_tag_index]
                token.head_index = head_index

        if 'embeddings' in predictions:
            for embedding, token in zip(predictions['embeddings'][sent_index], sentence.tokens):
                token._embedding = embedding

    def normalize_spans(self, spans: List[Span]):
        for span in spans:
            for token in span.tokens:
                assert token._embedding is not None

        predictions = self._model.span_normalizer.apply(spans)
        for span_index, span in enumerate(spans):
            normal_form = []
            for lemmatize_rule_index, token in zip(predictions['norms_predictions'][span_index], span.tokens):
                normal_form.append(self._span_normalize_helper.lemmatize(token.text, lemmatize_rule_index))
            # TODO: implement it smarter
            span.normal_form = ' '.join(normal_form)

    def show_perf_results(self):
        show_perf_results()

    def clear_perf_counters(self):
        clear_perf_counters()


def iterate_sentences(path):
    with open(path) as f:
        sentence = []
        for line in f:
            line = line.strip()
            if not line:
                if sentence:
                    yield sentence
                    sentence = []
                continue
            _, word, _, span_label, *_ = line.split('\t')
            sentence.append((word, span_label))
    if sentence:
        yield sentence


def main():
    parser = Parser()
    path = '/mnt/storage/dan-anastasev/Documents/junk/GramEval2020/data/span_normalization/data_open_test/valid.conllu'
    for sentence in iterate_sentences(path):
        spans_indices = []
        for index, (_, span_label) in enumerate(sentence):
            if span_label == 'B':
                spans_indices.append([])
                spans_indices[-1].append(index)
            if span_label == 'I':
                spans_indices[-1].append(index)
        sentence = Sentence([token for token, _ in sentence])
        parser.parse([sentence], predict_morphology=False, predict_lemmas=False,
                     predict_syntax=False, return_embeddings=True)

        spans = [Span(tokens=[sentence.tokens[index] for index in span_indices]) for span_indices in spans_indices]
        parser.normalize_spans(spans)
        for span in spans:
            print(' '.join(token.text for token in span.tokens), span.normal_form, sep='\t')


if __name__ == '__main__':
    main()
