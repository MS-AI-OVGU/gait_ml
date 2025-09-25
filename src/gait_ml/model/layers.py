import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np


class Encoder(nn.Module):
    """
    The Encoder processes the input sequence and produces a context vector
    (the final hidden state).
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        dropout_prob,
        bidirectional=False,
        type="gru",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GRU (Gated Recurrent Unit) is a type of RNN.
        # batch_first=True means input and output tensors are (batch, seq, feature)
        if type == "gru":
            self.rnn = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers,
                dropout=dropout_prob,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif type == "lstm":
            self.rnn = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers,
                dropout=dropout_prob,
                batch_first=True,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError("Unsupported RNN type. Use 'gru' or 'lstm'.")

    def forward(self, src):
        """
        Args:
            src (torch.Tensor): Input sequence tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            outputs (torch.Tensor): Output features from the last layer of the GRU for each time step.
                                    Shape: (batch_size, seq_len, hidden_dim).
            hidden (torch.Tensor): The final hidden state for each layer.
                                   Shape: (num_layers, batch_size, hidden_dim).
        """
        outputs, hidden = self.rnn(src)
        return outputs, hidden


class Seq2Seq(nn.Module):
    """
    The complete Sequence-to-Sequence model combining Encoder and Decoder.
    Uses teacher forcing during training.
    """

    def __init__(self, encoder, device, bidirectional=False):
        super().__init__()
        self.encoder = encoder
        self.device = device
        bidirectional_factor = 2 if bidirectional else 1
        self.fc_out = nn.Linear(encoder.hidden_dim * bidirectional_factor, 3)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """#TODO"""
        # batch_size = src.shape[0]
        # output_seq_len = trg.shape[1]
        # output_dim = trg.shape[2]

        # Encode the source sequence
        encoder_outputs, hidden = self.encoder(src)

        linear_out = self.fc_out(encoder_outputs)
        # out = torch.softmax(linear_out, dim=-1)
        return linear_out
