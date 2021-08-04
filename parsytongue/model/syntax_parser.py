# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from dependency_decoding import chu_liu_edmonds

from parsytongue.model.bilinear_matrix_attention import BilinearMatrixAttention
from parsytongue.model.configs import SyntaxParserConfig
from parsytongue.model.utils import DeviceGetterMixin


class SyntaxParser(nn.Module, DeviceGetterMixin):
    def __init__(self, encoder_dim: int, config: SyntaxParserConfig):
        super(SyntaxParser, self).__init__()

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

        self._head_sentinel = nn.Parameter(torch.randn([1, 1, encoder_dim]))

    def forward(self, encoded_text: torch.Tensor, mask: torch.Tensor):
        batch_size, _, encoding_dim = encoded_text.shape

        head_sentinel = self._head_sentinel.expand(batch_size, 1, encoding_dim)
        encoded_text = torch.cat([head_sentinel, encoded_text], 1)

        head_arc_representation = self._head_arc_feedforward(encoded_text)
        child_arc_representation = self._child_arc_feedforward(encoded_text)

        head_tag_representation = self._head_tag_feedforward(encoded_text)
        child_tag_representation = self._child_tag_feedforward(encoded_text)

        attended_arcs = self._arc_attention(head_arc_representation, child_arc_representation)

        mask = torch.cat([mask.new_ones(batch_size, 1), mask], 1)
        mask = mask.float()
        minus_mask = (1 - mask) * -1e32
        attended_arcs = attended_arcs + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        return self._mst_decode(
            head_tag_representation, child_tag_representation, attended_arcs, mask
        )

    def _mst_decode(self,
                    head_tag_representation: torch.Tensor,
                    child_tag_representation: torch.Tensor,
                    attended_arcs: torch.Tensor,
                    mask: torch.Tensor):
        batch_size, sequence_length, tag_representation_dim = head_tag_representation.size()

        lengths = mask.data.sum(dim=1).long().cpu().numpy()

        expanded_shape = [batch_size, sequence_length, sequence_length, tag_representation_dim]

        head_tag_representation = head_tag_representation.unsqueeze(2)
        head_tag_representation = head_tag_representation.expand(*expanded_shape).contiguous()

        child_tag_representation = child_tag_representation.unsqueeze(1)
        child_tag_representation = child_tag_representation.expand(*expanded_shape).contiguous()

        pairwise_head_logits = self._tag_bilinear(head_tag_representation, child_tag_representation)

        normalized_pairwise_head_logits = F.log_softmax(pairwise_head_logits, dim=3).permute(0, 3, 1, 2)

        normalized_arc_logits = F.log_softmax(attended_arcs, dim=2).transpose(1, 2)

        batch_energy = torch.exp(normalized_arc_logits.unsqueeze(1) + normalized_pairwise_head_logits)
        return self._run_mst_decoding(batch_energy, lengths)

    @staticmethod
    def _run_mst_decoding(batch_energy: torch.Tensor, lengths: torch.Tensor):
        batch_energy = batch_energy.detach().cpu()

        heads, head_tags, parse_scores = [], [], []
        for energy, length in zip(batch_energy, lengths):
            scores, tag_ids = energy.max(dim=0)

            scores = scores[:length, :length].numpy().astype('float64')
            instance_heads, score = chu_liu_edmonds(scores.T)

            instance_head_tags = []
            for child, parent in enumerate(instance_heads):
                instance_head_tags.append(tag_ids[parent, child].item())

            instance_heads[0] = 0
            instance_head_tags[0] = 0

            heads.append(instance_heads)
            head_tags.append(instance_head_tags)
            parse_scores.append(score)

        return {
            'heads': heads,
            'head_tags': head_tags,
            'parse_scores': parse_scores,
        }
