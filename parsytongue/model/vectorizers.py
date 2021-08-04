# -*- coding: utf-8 -*-

import json
import logging
import numpy as np
import torch

from pymorphy2.units.by_analogy import KnownSuffixAnalyzer
from pymorphy2.units.unkn import UnknAnalyzer
from typing import Dict, List

from parsytongue.model.embedders import EmbedderFactory
from parsytongue.model.utils import BaseFactory, DeviceGetterMixin
from parsytongue.parser.sentence import Sentence, Token
from parsytongue.parser.lemmatize_helper import LemmatizeHelper

logger = logging.getLogger(__name__)


_MAX_WORD_LEN = 50


class VectorizerFactory(BaseFactory):
    registry = {}


class Vectorizer(object):
    def __init__(self):
        self._device = torch.device('cpu')

    def to(self, device):
        self._device = device

    def process_instance(self, tokens: List[Token]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def get_padding_id(self, output_key: str = None):
        return 0

    def __call__(self, sentences: List[Sentence]) -> Dict[str, torch.Tensor]:
        vectorized_batch = []
        for sentence in sentences:
            vectorized_batch.append(self.process_instance(sentence.tokens))

        if not vectorized_batch:
            return vectorized_batch

        result = {}
        for key in vectorized_batch[0]:
            padding_id = self.get_padding_id(key)
            result[key] = self._batchify(vectorized_batch, padding_id, lambda element: element[key])
        return result

    @staticmethod
    def _batchify(samples, padding_id, get_field):
        first_sample = get_field(samples[0])
        max_length = max(len(get_field(sample)) for sample in samples)

        tensor = torch.full(
            (len(samples), max_length) + first_sample.shape[1:],
            fill_value=padding_id,
            dtype=first_sample.dtype,
            device=first_sample.device
        )

        for sample_id, sample in enumerate(samples):
            data = get_field(sample)
            tensor[sample_id, :len(data)] = data

        return tensor


@VectorizerFactory.register('chars')
class CharacterLevelVectorizer(Vectorizer):
    def __init__(self, vocab, **kwargs):
        super(CharacterLevelVectorizer, self).__init__()

        self._vocab = vocab

    def process_instance(self, tokens: List[Token]) -> Dict[str, torch.Tensor]:
        max_word_len = min(max(len(token) for token in tokens), _MAX_WORD_LEN)

        char_ids = np.zeros((len(tokens), max_word_len), dtype=np.int64)
        for token_index, token in enumerate(tokens):
            token = token.text[:max_word_len]

            char_ids[token_index, :len(token)] = [
                self._vocab.get_token_index(symbol, 'token_characters') for symbol in token
            ]

        return {
            'chars': torch.from_numpy(char_ids).to(self._device)
        }


@VectorizerFactory.register('elmo')
class ELMoVectorizer(Vectorizer):
    def __init__(self, **kwargs):
        super(ELMoVectorizer, self).__init__()

        self._beginning_of_word_character = 258  # <begin word>
        self._end_of_word_character = 259  # <end word>
        self._padding_character = 260  # <padding>

    def process_instance(self, tokens: List[Token]):
        max_word_len = _MAX_WORD_LEN

        char_ids = np.full((len(tokens), max_word_len), self._padding_character, dtype=np.int64)
        char_ids[:, 0] = self._beginning_of_word_character
        for token_index, token in enumerate(tokens):
            token = token.text.encode('utf-8', 'ignore')
            token = list(token[: max_word_len - 2])

            char_ids[token_index, 1: len(token) + 1] = token
            char_ids[token_index, len(token) + 1] = self._end_of_word_character

        # +1 one for masking
        char_ids = char_ids + 1

        return {
            'elmo': torch.from_numpy(char_ids).to(self._device)
        }


@VectorizerFactory.register('bert')
class BertVectorizer(Vectorizer):
    def __init__(self, tokenizer_path, **kwargs):
        from transformers import BertTokenizer
        super(BertVectorizer, self).__init__()

        self._tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=False)

    def get_padding_id(self, output_key: str = None):
        if output_key == 'token_ids':
            return self._tokenizer.pad_token_id
        return 0

    def process_instance(self, tokens: List[Token]):
        offsets, token_ids = [], []
        token_ids.append(self._tokenizer.cls_token_id)
        for token in tokens:
            subtoken_ids = self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(token.text))
            offsets.append([len(token_ids), len(token_ids) + len(subtoken_ids) - 1])
            token_ids.extend(subtoken_ids)

        token_ids.append(self._tokenizer.sep_token_id)

        token_ids = torch.tensor(token_ids, dtype=torch.long, device=self._device)
        offsets = torch.tensor(offsets, dtype=torch.long, device=self._device)
        mask = torch.ones_like(token_ids)
        segment_ids = torch.zeros_like(token_ids)

        return {
            'token_ids': token_ids,
            'mask': mask,
            'segment_ids': segment_ids,
            'offsets': offsets
        }


@VectorizerFactory.register('morph')
class MorphoVectorizer(Vectorizer):
    def __init__(self, grammeme_to_index_path, **kwargs):
        super(MorphoVectorizer, self).__init__()

        with open(grammeme_to_index_path) as f:
            self._grammeme_to_index = json.load(f)
        self._morpho_vector_dim = max(self._grammeme_to_index.values()) + 1

    @property
    def morpho_vector_dim(self):
        return self._morpho_vector_dim

    def vectorize_word(self, word: Token):
        grammar_vector = np.zeros(self._morpho_vector_dim, dtype=np.float32)
        sum_parses_score = 0.
        for parse in word._pymorphy_forms:
            sum_parses_score += parse.score
            for grammeme in parse.tag.grammemes:
                grammeme_index = self._grammeme_to_index.get(grammeme)
                if grammeme_index:
                    grammar_vector[grammeme_index] += parse.score

        if sum_parses_score != 0.:
            grammar_vector /= sum_parses_score

        grammar_vector = np.clip(grammar_vector, a_min=0., a_max=1.)
        return grammar_vector

    def process_instance(self, tokens: List[Token]):
        matrix = np.stack([self.vectorize_word(token) for token in tokens], axis=0)

        return {
            'morpho_embeddings': torch.from_numpy(matrix).to(self._device)
        }


@VectorizerFactory.register('lemma')
class LemmaVectorizer(Vectorizer):
    def __init__(self, lemmatize_helper_path: str, output_key: str = 'lemmas'):
        super(LemmaVectorizer, self).__init__()

        self._lemmatize_helper = LemmatizeHelper.load(lemmatize_helper_path)
        self._output_key = output_key

    @staticmethod
    def _is_unknown(parse):
        return any(isinstance(unit[0], (UnknAnalyzer, KnownSuffixAnalyzer.FakeDictionary))
                for unit in parse.methods_stack)

    def vectorize_word(self, word):
        lemma_vector = np.zeros(len(self._lemmatize_helper), dtype=np.float32)
        for parse in word._pymorphy_forms:
            if self._is_unknown(parse):
                continue
            for form in parse.lexeme:
                lemma_id = self._lemmatize_helper.get_rule_index(word.text, form.word)
                lemma_vector[lemma_id] = 1.
        return lemma_vector

    def process_instance(self, tokens: List[Token]):
        matrix = np.stack([self.vectorize_word(token) for token in tokens], axis=0)

        return {
            self._output_key: torch.from_numpy(matrix).to(self._device)
        }


@VectorizerFactory.register('mask')
class MaskVectorizer(Vectorizer):
    def __init__(self, **kwargs):
        super(MaskVectorizer, self).__init__()

    def process_instance(self, tokens: List[Token]):
        return {
            'mask': torch.ones(len(tokens), dtype=torch.long, device=self._device)
        }


@VectorizerFactory.register('embeddings')
class EmbeddingsVectorizer(Vectorizer):
    def __init__(self, **kwargs):
        super(EmbeddingsVectorizer, self).__init__()

    def process_instance(self, tokens: List[Token]):
        return {
            'embeddings': torch.stack([token._embedding for token in tokens], dim=0).to(self._device)
        }


@VectorizerFactory.register('stack')
class VectorizerStack(Vectorizer):
    def __init__(self, vectorizers: List[Dict], **kwargs):
        super(VectorizerStack, self).__init__()

        self._vectorizers = []
        for params in vectorizers:
            self._vectorizers.append(VectorizerFactory.build(**params))

    def to(self, device):
        super(VectorizerStack, self).to(device)

        for vectorizer in self._vectorizers.values():
            vectorizer.to(device)

    def process_instance(self, tokens: List[Token]):
        result = {}
        for vectorizer in self._vectorizers:
            result.update(vectorizer.process_instance(tokens))
        return result


@VectorizerFactory.register('precomputed_embeddings')
class PrecomputedEmbeddingsVectorizer(torch.nn.Module, Vectorizer, DeviceGetterMixin):
    def __init__(self, vectorizers, embedders) -> None:
        super(PrecomputedEmbeddingsVectorizer, self).__init__()

        self._device = torch.device('cpu')
        self._vectorizer = VectorizerFactory.build(**vectorizers)
        self._embedder = EmbedderFactory.build(**embedders)
        self._output_dim = self._embedder.get_output_size()
        self._embeddings_cache = {}

    def to(self, device):
        self._device = device
        self._vectorizer.to(device)
        self._embedder.to(device)

    def __call__(self, sentences: List[Sentence]):
        max_length = max(len(sentence.tokens) for sentence in sentences)

        token_embeddings = torch.zeros((len(sentences), max_length, self._output_dim), device=self._device)
        mask = torch.zeros((len(sentences), max_length), device=self._device)

        unknown_tokens, unknown_token_rows, unknown_token_columns = [], [], []
        for sent_index, sentence in enumerate(sentences):
            mask[sent_index, :len(sentence.tokens)] = 1

            for token_index, token in enumerate(sentence.tokens):
                token_embedding = self._embeddings_cache.get(token.text)
                if token_embedding is not None:
                    token_embeddings[sent_index, token_index] = token_embedding.to(self._device)
                else:
                    unknown_tokens.append(token)
                    unknown_token_rows.append(sent_index)
                    unknown_token_columns.append(token_index)

        if unknown_tokens:
            vectorized_unknown_tokens = self._vectorizer.process_instance(unknown_tokens)
            vectorized_unknown_tokens = {
                key: value.unsqueeze(0) for key, value in vectorized_unknown_tokens.items()
            }

            unknown_token_embeddings = self._embedder(vectorized_unknown_tokens)
            token_embeddings[unknown_token_rows, unknown_token_columns] = unknown_token_embeddings

            for unknown_token, unknown_token_embedding in zip(unknown_tokens, unknown_token_embeddings[0]):
                self._embeddings_cache[unknown_token.text] = unknown_token_embedding

        return {
            'token_embeddings': token_embeddings,
            'mask': mask
        }


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
    dir_path = "models"

    json_config = {
        "name": "precomputed_embeddings",
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
                    "name": "bert",
                    "vector_key": "bert",
                    "tokenizer_path": "bert_tokenizer"
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

    print(VectorizerFactory.registry)

    vectorizer = VectorizerFactory.build(**json_config)

    sentences = [Sentence("мама мыла раму"), Sentence("красивая мама нежно гладила бельё")]

    print(vectorizer(sentences)['token_embeddings'].shape)


if __name__ == '__main__':
    main()
