from time import perf_counter
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from lightning import LightningModule, seed_everything, Trainer
from torch.utils.data import DataLoader, Dataset


class TrafficDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        super().__init__()
        self.x = x  # Shape: (num_samples, sequence_length, num_nodes, num_features)
        self.y = y  # Shape: same as x

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = torch.tensor(self.x[idx], dtype=torch.float)  # Shape: (sequence_length, num_nodes, num_features)
        y = torch.tensor(self.y[idx], dtype=torch.float)  # Shape: same as x

        return {
            'x': x,
            'y': y,
        }


class GraphConvLayer(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, K, W):
        super(GraphConvLayer, self).__init__(aggr='add')
        self.K = K
        self.register_buffer('W', torch.tensor(W, dtype=torch.float32))
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        edge_weight = self.W[edge_index[0], edge_index[1]]
        out = x
        for _ in range(self.K):
            out = self.propagate(edge_index, x=out, edge_weight=edge_weight)
        return self.lin(out)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


class DCGRUCell(LightningModule):
    def __init__(self, num_nodes: int, input_dim: int, hidden_dim: int, K: int, W: np.ndarray):
        super(DCGRUCell, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.graph_conv = GraphConvLayer(input_dim + hidden_dim, hidden_dim, K, W)

        self.W_r = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.W_u = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.W_c = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

    def forward(self, X: torch.Tensor, H: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        X_H = torch.cat([X, H], dim=-1)
        X_H = self.graph_conv(X_H, edge_index)

        r = torch.sigmoid(X_H @ self.W_r)
        u = torch.sigmoid(X_H @ self.W_u)
        c = torch.tanh(X_H @ self.W_c)

        H = u * H + (1 - u) * c
        return H


class DCRNN(LightningModule):
    def __init__(self, num_nodes: int, num_features: int, input_dim: int, hidden_dim: int, output_dim: int, K: int,
                 W: np.ndarray, num_layers: int, edge_index: np.ndarray):
        super(DCRNN, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.edge_index = torch.tensor(edge_index, dtype=torch.long)

        self.encoder = nn.ModuleList([DCGRUCell(num_nodes=self.num_nodes,
                                                input_dim=self.input_dim if i == 0 else self.hidden_dim,
                                                hidden_dim=self.hidden_dim,
                                                K=K,
                                                W=W) for i in range(self.num_layers)])

        self.decoder = nn.ModuleList([DCGRUCell(num_nodes=self.num_nodes,
                                                input_dim=self.hidden_dim,
                                                hidden_dim=self.hidden_dim,
                                                K=K,
                                                W=W) for i in range(self.num_layers)])

        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        device = X.device
        edge_index = self.edge_index.to(device)
        batch_size, seq_len, num_nodes, _ = X.size()
        H_enc = [torch.zeros(batch_size, self.num_nodes, self.hidden_dim, device=device) for _ in
                 range(self.num_layers)]
        H_dec = [torch.zeros(batch_size, self.num_nodes, self.hidden_dim, device=device) for _ in
                 range(self.num_layers)]
        outputs = []

        for t in range(seq_len):
            x_t = X[:, t, :, :]
            for i, layer in enumerate(self.encoder):
                H_enc[i] = layer(x_t, H_enc[i], edge_index)
                x_t = H_enc[i]
                x_t = torch.relu(x_t)

        dec_input = H_enc[-1]
        for t in range(seq_len):
            for i, layer in enumerate(self.decoder):
                H_dec[i] = layer(dec_input, H_dec[i], edge_index)
                dec_input = H_dec[i]
                dec_input = torch.relu(dec_input)

            out = self.linear(dec_input)
            outputs.append(out.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        return outputs

    def training_step(self, batch, batch_idx):
        x = batch['x']
        y = batch['y']
        x, y = x.to(self.device), y.to(self.device)
        out = self(x)
        loss = nn.MSELoss()(out, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


seed_everything(42)

num_samples = 10
sequence_length = 12
num_nodes = 207
num_features = 4

x = np.random.rand(num_samples, sequence_length, num_nodes, num_features)
y = np.random.rand(num_samples, sequence_length, num_nodes, num_features)

K = 2  # Number of diffusion steps
W = np.random.rand(num_nodes, num_nodes)  # Weighted adjacency matrix
edge_index = np.array([[i, j] for i in range(num_nodes) for j in range(num_nodes) if W[i, j] > 0]).T
dcgru_hidden_dim = 32

dataset = TrafficDataset(x, y)
data_loader = DataLoader(dataset,
                         batch_size=4,  # Increased batch size
                         num_workers=8,
                         persistent_workers=True)

model = DCRNN(num_nodes=num_nodes,
              num_features=num_features,
              input_dim=num_features,
              hidden_dim=dcgru_hidden_dim,
              output_dim=num_features,
              K=K,
              W=W,
              num_layers=2,
              edge_index=edge_index)

trainer = Trainer(max_epochs=10)
start = perf_counter()
trainer.fit(model, data_loader)
print(f"Time: {perf_counter() - start:.6f} s")
