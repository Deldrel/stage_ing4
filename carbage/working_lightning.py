from typing import Dict
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
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


class DiffusionConvolution(nn.Module):
    def __init__(self, num_nodes: int, num_features: int, K: int, W: np.ndarray):
        super(DiffusionConvolution, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.K = K
        self.W = torch.tensor(W, dtype=torch.float32)

        ones = torch.ones(self.num_nodes)
        out_degree = self.W @ ones
        self.DO_inv = torch.linalg.inv(torch.diag(out_degree))

        in_degree = self.W.T @ ones
        self.DI_inv = torch.linalg.inv(torch.diag(in_degree))

        self.W_powers = [torch.matrix_power(self.DO_inv @ self.W, k) for k in range(self.K)]
        self.WT_powers = [torch.matrix_power(self.DI_inv @ self.W.T, k) for k in range(self.K)]

    def forward(self, X: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        device = X.device
        W_powers = [W_power.to(device) for W_power in self.W_powers]
        WT_powers = [WT_power.to(device) for WT_power in self.WT_powers]

        X_out = X.clone()

        for p in range(self.num_features):
            for k in range(self.K):
                a = theta[k, 0] * W_powers[k]
                b = theta[k, 1] * WT_powers[k]
                X_out[:, p] += (a + b) @ X[:, p]

        return X_out


class DCGRUCell(LightningModule):
    def __init__(self, num_nodes: int, num_features: int, input_dim: int, hidden_dim: int, K: int, W: np.ndarray):
        super(DCGRUCell, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.K = K
        self.W = W

        self.diffusion_conv = DiffusionConvolution(self.num_nodes, self.num_features, self.K, self.W)

        self.theta_r = nn.Parameter(torch.randn(self.K, 2))
        self.theta_u = nn.Parameter(torch.randn(self.K, 2))
        self.theta_C = nn.Parameter(torch.randn(self.K, 2))

        self.W_r = nn.Parameter(torch.randn(self.input_dim + self.hidden_dim, self.hidden_dim))
        self.W_u = nn.Parameter(torch.randn(self.input_dim + self.hidden_dim, self.hidden_dim))
        self.W_C = nn.Parameter(torch.randn(self.input_dim + self.hidden_dim, self.hidden_dim))

        self.b_r = nn.Parameter(torch.zeros(self.hidden_dim))
        self.b_u = nn.Parameter(torch.zeros(self.hidden_dim))
        self.b_C = nn.Parameter(torch.zeros(self.hidden_dim))

    def forward(self, X: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        X_H = torch.cat([X, H], dim=1)

        r = torch.sigmoid(self.diffusion_conv(X_H, self.theta_r) @ self.W_r + self.b_r)
        u = torch.sigmoid(self.diffusion_conv(X_H, self.theta_u) @ self.W_u + self.b_u)

        X_rH = torch.cat([X, r * H], dim=1)
        C = torch.tanh(self.diffusion_conv(X_rH, self.theta_C) @ self.W_C + self.b_C)

        H = u * H + (1 - u) * C

        return H


class DCRNN(LightningModule):
    def __init__(self, num_nodes: int, num_features: int, input_dim: int, hidden_dim: int, output_dim: int, K: int,
                 W: np.ndarray, num_layers: int):
        super(DCRNN, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.K = K
        self.W = W
        self.num_layers = num_layers

        self.encoder = nn.ModuleList([DCGRUCell(num_nodes=self.num_nodes,
                                                num_features=self.num_features,
                                                input_dim=self.input_dim if i == 0 else self.hidden_dim,
                                                hidden_dim=self.hidden_dim,
                                                K=self.K,
                                                W=self.W) for i in range(self.num_layers)])

        self.decoder = nn.ModuleList([DCGRUCell(num_nodes=self.num_nodes,
                                                num_features=self.num_features,
                                                input_dim=self.hidden_dim,
                                                hidden_dim=self.hidden_dim,
                                                K=self.K,
                                                W=self.W) for i in range(self.num_layers)])

        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        device = X.device
        H_enc = [torch.zeros(self.num_nodes, self.hidden_dim, device=device) for _ in range(self.num_layers)]
        H_dec = [torch.zeros(self.num_nodes, self.hidden_dim, device=device) for _ in range(self.num_layers)]
        outputs = []

        for t in range(X.shape[0]):
            x_t = X[t, :, :]
            for i, layer in enumerate(self.encoder):
                H_enc[i] = layer(x_t, H_enc[i])
                x_t = H_enc[i]
                x_t = torch.relu(x_t)

        dec_input = H_enc[-1]
        for t in range(X.shape[0]):
            for i, layer in enumerate(self.decoder):
                H_dec[i] = layer(dec_input, H_dec[i])
                dec_input = H_dec[i]
                dec_input = torch.relu(dec_input)

            out = self.linear(dec_input)
            outputs.append(out.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs

    def training_step(self, batch, batch_idx):
        x = batch['x'].squeeze(0)
        y = batch['y'].squeeze(0)
        x, y = x.to(self.device), y.to(self.device)  # Move inputs to the correct device
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
dcgru_hidden_dim = 32

dataset = TrafficDataset(x, y)
data_loader = DataLoader(dataset,
                         batch_size=1,
                         num_workers=8,
                         persistent_workers=True)

model = DCRNN(num_nodes=num_nodes,
              num_features=num_features,
              input_dim=num_features,
              hidden_dim=dcgru_hidden_dim,
              output_dim=num_features,
              K=K,
              W=W,
              num_layers=2)

trainer = Trainer(max_epochs=10)
start = perf_counter()
trainer.fit(model, data_loader)
print(f"Time: {perf_counter() - start:.6f} s")
