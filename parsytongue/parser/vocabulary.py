# -*- coding: utf-8 -*-

import json
import logging
import os
import codecs

from typing import Dict, List

logger = logging.getLogger(__name__)


_DEFAULT_PADDING_TOKEN = "@@PADDING@@"
_DEFAULT_OOV_TOKEN = "@@UNKNOWN@@"


class SimpleVocabulary(object):
    def __init__(self, index_to_token: List[str]):
        self._index_to_token = index_to_token
        self._token_to_index = {token: index for index, token in enumerate(index_to_token)}

    def get_token_index(self, token):
        return self._token_to_index[token]

    def get_token_from_index(self, index):
        return self._index_to_token[index]

    def __len__(self):
        return len(self._token_to_index)

    @classmethod
    def from_file(cls, path: str, is_padded: bool):
        with codecs.open(path, encoding='utf-8') as f:
            index_to_token = [line.rstrip() for line in f]

        if is_padded:
            index_to_token = [_DEFAULT_PADDING_TOKEN] + index_to_token

        return cls(index_to_token)


class Vocabulary(object):
    def __init__(self, vocabs: Dict[str, SimpleVocabulary]):
        self._vocabs = vocabs

    def get_token_index(self, token: str, vocab: str) -> int:
        return self._vocabs[vocab].get_token_index(token)

    def get_token_from_index(self, index: int, vocab: str) -> str:
        return self._vocabs[vocab].get_token_from_index(index)

    def get_vocab_size(self, vocab: str) -> int:
        return len(self._vocabs[vocab])

    @classmethod
    def from_files(cls, directory: str):
        logger.info('Loading vocabulary from %s.', directory)

        with open(os.path.join(directory, 'config.json')) as f:
            config = json.load(f)

        vocabs = {}
        for vocab_config in config['vocabs']:
            vocabs[vocab_config['name']] = SimpleVocabulary.from_file(
                path=os.path.join(directory, vocab_config['path']),
                is_padded=vocab_config.get('is_padded', False)
            )

        return cls(vocabs)
