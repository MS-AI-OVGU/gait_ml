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


class CNNEncoder(nn.Module):
    """
    The CNNEncoder processes the input sequence using 1D convolutional layers
    and produces a feature map for sequence modeling.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, kernel_size, dropout_prob):
        """
        Args:
            input_dim (int): Number of input features per timestep.
            hidden_dim (int): Number of output channels for each convolutional layer.
            num_layers (int): Number of convolutional layers.
            kernel_size (int): Size of the convolutional kernel.
            dropout_prob (float): Dropout probability for regularization.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers = []
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,  # Keep the sequence length the same
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))

        self.cnn = nn.Sequential(*layers)

    def forward(self, src):
        """
        Args:
            src (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            outputs (torch.Tensor): Output features of shape (batch_size, seq_len, hidden_dim).
        """
        # Convert (batch_size, seq_len, input_dim) -> (batch_size, input_dim, seq_len)
        src = src.permute(0, 2, 1)
        outputs = self.cnn(src)
        # Convert back to (batch_size, seq_len, hidden_dim)
        outputs = outputs.permute(0, 2, 1)
        return outputs


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

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        """#TODO"""
        # batch_size = src.shape[0]
        # output_seq_len = trg.shape[1]
        # output_dim = trg.shape[2]

        # Encode the source sequence
        encoder_outputs, hidden = self.encoder(src)

        linear_out = self.fc_out(encoder_outputs)
        # out = torch.softmax(linear_out, dim=-1)
        return linear_out
