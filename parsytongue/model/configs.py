# -*- coding: utf-8 -*-

import attr

from attr.validators import instance_of


class ConvertableFromJsonMixin(object):
    @classmethod
    def converter(cls, config):
        return cls(**config)


@attr.s
class SyntaxParserConfig(ConvertableFromJsonMixin):
    tag_representation_dim = attr.ib()
    arc_representation_dim = attr.ib()
    head_tag_count = attr.ib()


@attr.s
class DecoderConfig(ConvertableFromJsonMixin):
    lemma_count = attr.ib()
    grammar_value_count = attr.ib()
    syntax_parser = attr.ib(validator=instance_of(SyntaxParserConfig), converter=SyntaxParserConfig.converter)


@attr.s
class SpanNormalizerConfig(ConvertableFromJsonMixin):
    vectorizer = attr.ib()
    use_possible_lemmas_feature = attr.ib()
    normal_form_count = attr.ib()
    encoder_params = attr.ib()


@attr.s
class ModelConfig(object):
    vectorizer = attr.ib()
    embedder = attr.ib()
    encoder = attr.ib()
    decoder = attr.ib(validator=instance_of(DecoderConfig), converter=DecoderConfig.converter)
    span_normalizer = attr.ib(validator=instance_of(SpanNormalizerConfig), converter=SpanNormalizerConfig.converter)
