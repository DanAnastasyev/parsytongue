# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class LstmWrapper(nn.LSTM):
    def __init__(self, *args, **kwargs):
        kwargs['batch_first'] = True
        super(LstmWrapper, self).__init__(*args, **kwargs)

    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        assert len(inputs.shape) == 3
        assert len(mask.shape) == 2
        assert inputs.shape[:-1] == mask.shape

        _, seq_len, _ = inputs.shape

        sequence_lengths = mask.sum(-1)

        packed_inputs = nn.utils.rnn.pack_padded_sequence(
            inputs, sequence_lengths, batch_first=True, enforce_sorted=False
        )

        packed_outputs, (h_n, c_n) = super(LstmWrapper, self).forward(packed_inputs)

        output, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True, total_length=seq_len)

        return output, (h_n, c_n)
