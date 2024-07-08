from datetime import datetime
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
import wandb
from torch.nn import Dropout, LayerNorm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.config import config


class DiffusionConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiffusionConv, self).__init__()
        self.k = config.model.k
        self.theta_forward = torch.nn.Parameter(torch.Tensor(self.k, in_channels, out_channels))
        self.theta_backward = torch.nn.Parameter(torch.Tensor(self.k, in_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.theta_forward)
        torch.nn.init.xavier_uniform_(self.theta_backward)

    def forward(self, x, edge_index, edge_weight):
        Tx_0 = x
        Tx_1 = self.propagate(edge_index, x=x, norm=edge_weight)
        out = torch.matmul(Tx_0.unsqueeze(1), self.theta_forward[0]) + torch.matmul(Tx_1.unsqueeze(1),
                                                                                    self.theta_backward[0])

        for k in range(1, self.k):
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=edge_weight)
            out += torch.matmul(Tx_2.unsqueeze(1), self.theta_forward[k]) + torch.matmul(Tx_1.unsqueeze(1),
                                                                                         self.theta_backward[k])
            Tx_1, Tx_2 = Tx_2, Tx_1

        return out.squeeze(1)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def propagate(self, edge_index, x, norm):
        row, col = edge_index
        deg = torch.zeros_like(x[:, 0]).scatter_add_(0, row, norm)
        deg_inv = deg.pow(-0.5)
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * norm * deg_inv[col]
        out = torch.zeros_like(x).scatter_add_(0, row, self.message(x[col], norm))
        return out


class DiffusionConvGRUCell(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(DiffusionConvGRUCell, self).__init__()
        self.hidden_channels = hidden_channels
        self.diffusion_conv = DiffusionConv(in_channels + hidden_channels, 2 * hidden_channels)

    def forward(self, x, edge_index, edge_weight, h):
        combined = torch.cat([x, h], dim=-1)
        conv_out = self.diffusion_conv(combined, edge_index, edge_weight)
        r, u = torch.split(conv_out, self.hidden_channels, dim=-1)
        r, u = torch.sigmoid(r), torch.sigmoid(u)

        conv_out = self.diffusion_conv(torch.cat([x, r * h], dim=-1), edge_index, edge_weight)
        c = torch.tanh(conv_out)
        h = u * h + (1 - u) * c

        return h


class DCRNNEncoder(L.LightningModule):
    def __init__(self, in_channels, hidden_channels):
        super(DCRNNEncoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.dcgru_cell = DiffusionConvGRUCell(in_channels, hidden_channels)

    def forward(self, x, edge_index, edge_weight):
        h = torch.zeros(x.size(0), self.hidden_channels, device=x.device)
        for t in range(x.size(1)):
            h = self.dcgru_cell(x[:, t], edge_index, edge_weight, h)
        return h


class DCRNNDecoder(L.LightningModule):
    def __init__(self, out_channels, hidden_channels):
        super(DCRNNDecoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.dcgru_cell = DiffusionConvGRUCell(out_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, out_channels)
        self.target_length = config.model.target_length

    def forward(self, h, edge_index, edge_weight):
        predictions = []
        input = torch.zeros(h.size(0), 1, self.hidden_channels, device=h.device)  # Start with zero input

        for t in range(self.target_length):
            h = self.dcgru_cell(input, edge_index, edge_weight, h)
            out = self.fc(h)
            predictions.append(out.unsqueeze(1))
            input = out.unsqueeze(1)  # Use the output as the next input

        return torch.cat(predictions, dim=1)


class DCRNN(L.LightningModule):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(DCRNN, self).__init__()
        self.encoder = DCRNNEncoder(in_channels, hidden_channels)
        self.decoder = DCRNNDecoder(out_channels, hidden_channels)

        self.norm = LayerNorm(hidden_channels)
        self.dropout = Dropout(p=config.model.dropout_rate)

        self.loss_func = getattr(F, config.model.loss_function)
        self.activation = getattr(F, config.model.activation)
        self.save_hyperparameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        h = self.encoder(x, edge_index, edge_weight)
        h = self.activation(h)
        h = self.norm(h)
        h = self.dropout(h)
        out = self.decoder(h, edge_index, edge_weight)
        return out