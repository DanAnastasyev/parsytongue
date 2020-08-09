# -*- coding: utf-8 -*-

import attr
import copy
import json
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Optional, Tuple, Any, List

from parsytongue.model.bilinear_matrix_attention import BilinearMatrixAttention
from parsytongue.model.chu_liu_edmonds import decode_mst
from parsytongue.model.embedder import Embedder, CachableUncontextualizedEmbedderStack, EmbedderStack, EmbedderConfig, DeviceGetterMixin
from parsytongue.model.encoder import Encoder, LstmEncoder, PassThroughEncoder, EncoderConfig
from parsytongue.model.perf_counter import timed
from parsytongue.model.vectorizers import VectorizerStack, VectorizerConfig

logger = logging.getLogger(__name__)


@attr.s
class DecoderConfig(object):
    tag_representation_dim = attr.ib()
    arc_representation_dim = attr.ib()
    lemma_count = attr.ib()
    grammar_value_count = attr.ib()
    head_tag_count = attr.ib()


@attr.s
class ModelConfig(object):
    vectorizers = attr.ib()
    embedders = attr.ib(validator=attr.validators.deep_iterable(
        member_validator=attr.validators.instance_of(EmbedderConfig),
        iterable_validator=attr.validators.instance_of(list)
    ))
    encoder = attr.ib(validator=attr.validators.instance_of(EncoderConfig))
    decoder = attr.ib(validator=attr.validators.instance_of(DecoderConfig))
    weights_path = attr.ib()


class Model(nn.Module, DeviceGetterMixin):
    def __init__(
        self,
        embedder: Embedder,
        encoder: Encoder,
        config: DecoderConfig
    ) -> None:
        super(Model, self).__init__()

        self._embedder = embedder
        self._encoder = encoder
        self._config = config

        encoder_dim = encoder.get_output_size()

        self._head_arc_feedforward = nn.Sequential(
            nn.Linear(encoder_dim, config.arc_representation_dim),
            nn.ELU()
        )
        self._child_arc_feedforward = nn.Sequential(
            nn.Linear(encoder_dim, config.arc_representation_dim),
            nn.ELU()
        )

        self._arc_attention = BilinearMatrixAttention(config.arc_representation_dim,
                                                      config.arc_representation_dim,
                                                      use_input_biases=True)

        self._head_tag_feedforward = nn.Sequential(
            nn.Linear(encoder_dim, config.tag_representation_dim),
            nn.ELU()
        )
        self._child_tag_feedforward = nn.Sequential(
            nn.Linear(encoder_dim, config.tag_representation_dim),
            nn.ELU()
        )

        self._tag_bilinear = nn.modules.Bilinear(config.tag_representation_dim,
                                                 config.tag_representation_dim,
                                                 config.head_tag_count)

        self._head_sentinel = nn.Parameter(torch.randn([1, 1, encoder.get_output_size()]))

        self._grammar_value_output = nn.Linear(encoder_dim, config.grammar_value_count)
        self._lemma_output = nn.Linear(encoder_dim, config.lemma_count)

    def forward(
        self,
        inputs: Dict[str, torch.LongTensor],
        predict_morphology: bool,
        predict_lemmas: bool,
        predict_syntax: bool,
    ) -> Dict[str, torch.Tensor]:
        with timed('Embedder'):
            inputs = self._embedder(inputs)

        embedded_text_input = inputs['token_embeddings']
        mask = inputs['mask']

        with timed('Encoder'):
            encoded_text = self._encoder(embedded_text_input, mask)

        output_dict = {'mask': mask}

        if predict_morphology:
            output_dict['gram_vals'] = self._predict_morphology(encoded_text)

        if predict_lemmas:
            output_dict['lemmas'] = self._predict_lemmas(encoded_text)

        if predict_syntax:
            predicted_heads, predicted_head_tags = self._predict_syntax(encoded_text, mask)
            output_dict['heads'] = predicted_heads
            output_dict['head_tags'] = predicted_head_tags

        return output_dict

    def _predict_morphology(self, encoded_text: torch.Tensor) -> torch.LongTensor:
        with timed('Grammar value prediction'):
            grammar_value_logits = self._grammar_value_output(encoded_text)
            predicted_grammar_values = grammar_value_logits.argmax(-1)

            return predicted_grammar_values

    def _predict_lemmas(self, encoded_text: torch.Tensor) -> torch.LongTensor:
        with timed('Lemma prediction'):
            lemma_logits = self._lemma_output(encoded_text)
            predicted_lemmas = lemma_logits.argmax(-1)

            return predicted_lemmas

    def _predict_syntax(self, encoded_text, mask):
        with timed('Syntax prediction'):
            batch_size, _, encoding_dim = encoded_text.shape

            head_sentinel = self._head_sentinel.expand(batch_size, 1, encoding_dim)
            # Concatenate the head sentinel onto the sentence representation.
            encoded_text = torch.cat([head_sentinel, encoded_text], 1)
            mask = torch.cat([mask.new_ones(batch_size, 1), mask], 1)
            mask = mask.float()

            # shape (batch_size, sequence_length, arc_representation_dim)
            head_arc_representation = self._head_arc_feedforward(encoded_text)
            child_arc_representation = self._child_arc_feedforward(encoded_text)

            # shape (batch_size, sequence_length, tag_representation_dim)
            head_tag_representation = self._head_tag_feedforward(encoded_text)
            child_tag_representation = self._child_tag_feedforward(encoded_text)

            # shape (batch_size, sequence_length, sequence_length)
            attended_arcs = self._arc_attention(head_arc_representation, child_arc_representation)

            minus_mask = (1 - mask) * -1e32
            attended_arcs = attended_arcs + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

            predicted_heads, predicted_head_tags = self._mst_decode(
                head_tag_representation, child_tag_representation, attended_arcs, mask
            )

            return predicted_heads, predicted_head_tags

    def _mst_decode(self,
                    head_tag_representation: torch.Tensor,
                    child_tag_representation: torch.Tensor,
                    attended_arcs: torch.Tensor,
                    mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions using the Edmonds' Algorithm
        for finding minimum spanning trees on directed graphs. Nodes in the
        graph are the words in the sentence, and between each pair of nodes,
        there is an edge in each direction, where the weight of the edge corresponds
        to the most likely dependency label probability for that arc. The MST is
        then generated from this directed graph.

        Parameters
        ----------
        head_tag_representation : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : ``torch.Tensor``, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.

        Returns
        -------
        heads : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            greedily decoded heads of each word.
        head_tags : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            dependency tags of the optimally decoded heads of each word.
        """
        batch_size, sequence_length, tag_representation_dim = head_tag_representation.size()

        lengths = mask.data.sum(dim=1).long().cpu().numpy()

        expanded_shape = [batch_size, sequence_length, sequence_length, tag_representation_dim]
        head_tag_representation = head_tag_representation.unsqueeze(2)
        head_tag_representation = head_tag_representation.expand(*expanded_shape).contiguous()
        child_tag_representation = child_tag_representation.unsqueeze(1)
        child_tag_representation = child_tag_representation.expand(*expanded_shape).contiguous()
        # Shape (batch_size, sequence_length, sequence_length, num_head_tags)
        pairwise_head_logits = self._tag_bilinear(head_tag_representation, child_tag_representation)

        # Note that this log_softmax is over the tag dimension, and we don't consider pairs
        # of tags which are invalid (e.g are a pair which includes a padded element) anyway below.
        # Shape (batch, num_labels,sequence_length, sequence_length)
        normalized_pairwise_head_logits = F.log_softmax(pairwise_head_logits, dim=3).permute(0, 3, 1, 2)

        # Shape (batch_size, sequence_length, sequence_length)
        normalized_arc_logits = F.log_softmax(attended_arcs, dim=2).transpose(1, 2)

        # Shape (batch_size, num_head_tags, sequence_length, sequence_length)
        # This energy tensor expresses the following relation:
        # energy[i,j] = "Score that i is the head of j". In this
        # case, we have heads pointing to their children.
        batch_energy = torch.exp(normalized_arc_logits.unsqueeze(1) + normalized_pairwise_head_logits)
        return self._run_mst_decoding(batch_energy, lengths)

    @staticmethod
    def _run_mst_decoding(batch_energy: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        heads = []
        head_tags = []
        for energy, length in zip(batch_energy.detach().cpu(), lengths):
            scores, tag_ids = energy.max(dim=0)
            # Although we need to include the root node so that the MST includes it,
            # we do not want any word to be the parent of the root node.
            # Here, we enforce this by setting the scores for all word -> ROOT edges
            # edges to be 0.
            scores[0, :] = 0
            # Decode the heads. Because we modify the scores to prevent
            # adding in word -> ROOT edges, we need to find the labels ourselves.
            instance_heads, _ = decode_mst(scores.numpy(), length, has_labels=False)

            # Find the labels which correspond to the edges in the max spanning tree.
            instance_head_tags = []
            for child, parent in enumerate(instance_heads):
                instance_head_tags.append(tag_ids[parent, child].item())
            # We don't care what the head or tag is for the root token, but by default it's
            # not necessarily the same in the batched vs unbatched case, which is annoying.
            # Here we'll just set them to zero.
            instance_heads[0] = 0
            instance_head_tags[0] = 0
            heads.append(instance_heads)
            head_tags.append(instance_head_tags)

        return torch.from_numpy(np.stack(heads)), torch.from_numpy(np.stack(head_tags))

    @classmethod
    def from_config(cls, vocab, config):
        vectorizer = VectorizerStack.from_config(vocab, config.vectorizers)
        embedder = EmbedderStack.from_config(config.embedders)
        encoder = PassThroughEncoder(embedder.get_output_size(), config.encoder.params)

        return vectorizer, cls(embedder, encoder, config.decoder)
