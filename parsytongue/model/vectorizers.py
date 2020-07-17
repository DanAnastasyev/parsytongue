# -*- coding: utf-8 -*-

import attr
import logging
import numpy as np
import pymorphy2
import torch

from typing import Dict, List

logger = logging.getLogger(__name__)


_MAX_WORD_LEN = 50


@attr.s
class VectorizerConfig(object):
    name = attr.ib()
    vector_key = attr.ib()


class Vectorizer(object):
    def __call__(self, tokens: List[str]):
        raise NotImplementedError


class CharacterLevelVectorizer(Vectorizer):
    NAME = 'chars'

    def __init__(self, vocab, **kwargs):
        self._vocab = vocab

    def __call__(self, tokens):
        max_word_len = min(max(len(token) for token in tokens), _MAX_WORD_LEN)

        char_ids = np.zeros((len(tokens), max_word_len), dtype=np.int64)
        for token_index, token in enumerate(tokens):
            token = token[:max_word_len]

            char_ids[token_index, :len(token)] = [
                self._vocab.get_token_index(symbol, 'token_characters') for symbol in token
            ]

        return torch.from_numpy(char_ids)


class ELMoVectorizer(Vectorizer):
    NAME = 'elmo'

    def __init__(self, **kwargs):
        self._beginning_of_word_character = 258  # <begin word>
        self._end_of_word_character = 259  # <end word>
        self._padding_character = 260  # <padding>

    def __call__(self, tokens):
        max_word_len = _MAX_WORD_LEN

        char_ids = np.full((len(tokens), max_word_len), self._padding_character, dtype=np.int64)
        char_ids[:, 0] = self._beginning_of_word_character
        for token_index, token in enumerate(tokens):
            token = token.encode('utf-8', 'ignore')
            token = list(token[: max_word_len - 2])

            char_ids[token_index, 1: len(token) + 1] = token
            char_ids[token_index, len(token) + 1] = self._end_of_word_character

        # +1 one for masking
        char_ids = char_ids + 1

        return torch.from_numpy(char_ids)


class MorphoVectorizer(Vectorizer):
    NAME = 'morph'

    def __init__(self, **kwargs):
        self._morph = pymorphy2.MorphAnalyzer()
        self._grammeme_to_index = self._build_grammeme_to_index()
        self._morpho_vector_dim = max(self._grammeme_to_index.values()) + 1

    @property
    def morpho_vector_dim(self):
        return self._morpho_vector_dim

    def _build_grammeme_to_index(self):
        grammar_categories = [
            self._morph.TagClass.PARTS_OF_SPEECH,
            self._morph.TagClass.ANIMACY,
            self._morph.TagClass.ASPECTS,
            self._morph.TagClass.CASES,
            self._morph.TagClass.GENDERS,
            self._morph.TagClass.INVOLVEMENT,
            self._morph.TagClass.MOODS,
            self._morph.TagClass.NUMBERS,
            self._morph.TagClass.PERSONS,
            self._morph.TagClass.TENSES,
            self._morph.TagClass.TRANSITIVITY,
            self._morph.TagClass.VOICES
        ]

        grammeme_to_index = {}
        shift = 0
        for category in grammar_categories:
            # TODO: Save grammeme_to_index
            for grammeme_index, grammeme in enumerate(sorted(category)):
                grammeme_to_index[grammeme] = grammeme_index + shift
            shift += len(category) + 1  # +1 to address lack of the category in a parse

        return grammeme_to_index

    def vectorize_word(self, word):
        grammar_vector = np.zeros(self._morpho_vector_dim, dtype=np.float32)
        sum_parses_score = 0.
        for parse in self._morph.parse(word):
            sum_parses_score += parse.score
            for grammeme in parse.tag.grammemes:
                grammeme_index = self._grammeme_to_index.get(grammeme)
                if grammeme_index:
                    grammar_vector[grammeme_index] += parse.score

        if sum_parses_score != 0.:
            grammar_vector /= sum_parses_score

        assert np.all(grammar_vector < 1. + 1e-5) and np.all(grammar_vector > 0. - 1e-5)

        return grammar_vector

    def __call__(self, tokens):
        matrix = np.stack([self.vectorize_word(token) for token in tokens], axis=0)

        return torch.from_numpy(matrix)


class VectorizerStack(Vectorizer):
    _REGISTERED_VECTORIZERS = [CharacterLevelVectorizer, MorphoVectorizer, ELMoVectorizer]

    def __init__(self, vectorizers: Dict[str, Vectorizer]):
        self._vectorizers = vectorizers

    def __call__(self, tokens: List[str]):
        return {
            vectorizer_key: vectorizer(tokens)
            for vectorizer_key, vectorizer in self._vectorizers.items()
        }

    @classmethod
    def from_config(cls, vocab, configs):
        vectorizers = {}
        for config in configs:
            for vectorizer_cls in cls._REGISTERED_VECTORIZERS:
                if config.name == vectorizer_cls.NAME:
                    vectorizers[config.vector_key] = vectorizer_cls(vocab=vocab)

        assert len(vectorizers) == len(configs)

        return cls(vectorizers)
