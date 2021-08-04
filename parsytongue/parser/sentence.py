# -*- coding: utf-8 -*-

import attr
import numpy as np

from attr.validators import instance_of, optional
from pymorphy2.analyzer import Parse as PymorphyParse
from razdel import sentenize, tokenize
from typing import List, Union, Tuple

from parsytongue.parser.markup import lazy_property, GrammarValue, SyntaxRelation


def _get_attrib(cls):
    return attr.ib(default=None, repr=lambda val: str(val), validator=optional(instance_of(cls)))


@attr.s(auto_attribs=True)
class Token(object):
    text: str
    start: int = None
    stop: int = None
    grammar_value: GrammarValue = _get_attrib(GrammarValue)
    lemma: str = None
    head_index: int = None
    head_tag: SyntaxRelation = _get_attrib(SyntaxRelation)
    _pymorphy_forms: List[PymorphyParse] = attr.ib(factory=list, repr=False)
    _embedding: np.ndarray = attr.ib(default=None, repr=False)

    @lazy_property
    def pymorphy_form(self) -> PymorphyParse:
        # TODO
        pass


@attr.s(auto_attribs=True)
class Span(object):
    tokens: List[Token] = attr.ib(factory=list)
    normal_form: str = None


class Sentence(object):
    def __init__(self, tokens: Union[str, List[str]], start: int = None, stop: int = None):
        assert isinstance(tokens, (str, list))

        shift = start or 0
        if isinstance(tokens, list):
            self.text = ' '.join(tokens)
            self.tokens = []
            token_start = 0
            for token in tokens:
                self.tokens.append(Token(token, token_start + shift, token_start + len(token) + shift))
                token_start += len(token) + 1
        else:
            self.text = tokens
            self.tokens = [
                Token(token.text, token.start + shift, token.stop + shift) for token in tokenize(self.text)
            ]

        self.start = shift
        self.stop = stop or shift + len(self.text)

    def __str__(self) -> str:
        return 'Sentence(text={}, tokens={}, start={}, stop={})'.format(self.text, self.tokens, self.start, self.stop)

    def __repr__(self) -> str:
        return str(self)

    def extract_span(self, positions: Union[Tuple[int, int], List[Tuple[int, int]]]) -> Span:
        if isinstance(positions, tuple):
            positions = [positions]

        positions = [
            (span_begin, span_end)
            for span_begin, span_end in positions
            if span_begin >= self.start and span_end <= self.stop
        ]

        tokens = []
        for token in self.tokens:
            for span_begin, span_end in positions:
                if token.start >= span_begin and token.stop <= span_end:
                    tokens.append(token)

        return Span(tokens)


class Doc(object):
    def __init__(self, tokens: Union[str, List[str]], start: int = None, stop: int = None):
        assert isinstance(tokens, (str, list))
        if isinstance(tokens, list):
            assert all(isinstance(element, list) for element in tokens)
            self.sentences = list(map(Sentence, tokens))
            self.text = ' '.join(sentence.text for sentence in self.sentences)
        else:
            self.text = tokens

            shift = start or 0
            self.sentences = [
                Sentence(text.text, text.start + shift, text.stop + shift) for text in sentenize(self.text)
            ]

        self.start = start
        self.stop = stop

    def __str__(self) -> str:
        return 'Doc(text={}, sentences={}, start={}, stop={})'.format(self.text, self.sentences, self.start, self.stop)

    def __repr__(self) -> str:
        return str(self)

    def extract_span(self, positions: Union[Tuple[int, int], List[Tuple[int, int]]]) -> Span:
        if isinstance(positions, tuple):
            positions = [positions]

        tokens = []
        for sentence in self.sentences:
            tokens.extend(sentence.extract_span(positions).tokens)

        return Span(tokens)
