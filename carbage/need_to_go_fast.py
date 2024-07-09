from typing import Dict
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader as GeometricDataLoader
from lightning import LightningModule, seed_everything, Trainer
from torch.utils.data import Dataset


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


class DCGRUCell(LightningModule):
    def __init__(self, num_nodes: int, num_features: int, input_dim: int, hidden_dim: int):
        super(DCGRUCell, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv_z = GCNConv(input_dim + hidden_dim, hidden_dim)
        self.conv_r = GCNConv(input_dim + hidden_dim, hidden_dim)
        self.conv_h = GCNConv(input_dim + hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x, h], dim=-1)

        z = torch.sigmoid(self.conv_z(combined, edge_index))
        r = torch.sigmoid(self.conv_r(combined, edge_index))
        combined_reset = torch.cat([x, r * h], dim=-1)
        h_tilde = torch.tanh(self.conv_h(combined_reset, edge_index))

        h = (1 - z) * h + z * h_tilde
        return h


class DCRNN(LightningModule):
    def __init__(self, num_nodes: int, num_features: int, input_dim: int, hidden_dim: int, output_dim: int,
                 edge_index: torch.Tensor, num_layers: int):
        super(DCRNN, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.edge_index = edge_index

        self.encoder = nn.ModuleList([DCGRUCell(num_nodes=self.num_nodes,
                                                num_features=self.num_features,
                                                input_dim=self.input_dim if i == 0 else self.hidden_dim,
                                                hidden_dim=self.hidden_dim) for i in range(self.num_layers)])

        self.decoder = nn.ModuleList([DCGRUCell(num_nodes=self.num_nodes,
                                                num_features=self.num_features,
                                                input_dim=self.hidden_dim,
                                                hidden_dim=self.hidden_dim) for i in range(self.num_layers)])

        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        h_enc = [torch.zeros(x.size(1), self.hidden_dim, device=device) for _ in range(self.num_layers)]
        h_dec = [torch.zeros(x.size(1), self.hidden_dim, device=device) for _ in range(self.num_layers)]
        outputs = []

        for t in range(x.size(0)):
            x_t = x[t, :, :]
            for i, layer in enumerate(self.encoder):
                h_enc[i] = layer(x_t, self.edge_index, h_enc[i])
                x_t = h_enc[i]

        dec_input = h_enc[-1]
        for t in range(x.size(0)):
            for i, layer in enumerate(self.decoder):
                h_dec[i] = layer(dec_input, self.edge_index, h_dec[i])
                dec_input = h_dec[i]

            out = self.linear(dec_input)
            outputs.append(out.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs

    def training_step(self, batch, batch_idx):
        x = batch['x'].squeeze(0)
        y = batch['y'].squeeze(0)
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

# Generate random edge index for demonstration purposes
edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))

dataset = TrafficDataset(x, y)
data_loader = GeometricDataLoader(dataset, batch_size=1, num_workers=8, persistent_workers=True)

model = DCRNN(num_nodes=num_nodes,
              num_features=num_features,
              input_dim=num_features,
              hidden_dim=32,
              output_dim=num_features,
              edge_index=edge_index,
              num_layers=2)

trainer = Trainer(max_epochs=10)
start = perf_counter()
trainer.fit(model, data_loader)
print(f"Time: {perf_counter() - start:.6f} s")
