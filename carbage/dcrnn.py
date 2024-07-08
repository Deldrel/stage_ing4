import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Dropout, LayerNorm
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv import GCNConv


class TrafficDataset(Dataset):
    def __init__(self, x, y, adj_matrix):
        super().__init__()
        self.x = x  # Shape: (num_samples, sequence_length, num_nodes, num_features)
        self.y = y  # Shape: same as x
        self.adj_matrix = adj_matrix  # Shape: (num_nodes, num_nodes)

        self.num_samples = x.shape[0]
        self.num_nodes = adj_matrix.shape[0]

        self.edge_index = np.array(np.nonzero(adj_matrix))
        self.edge_index = np.stack(self.edge_index, axis=0)
        self.edge_index = torch.tensor(self.edge_index, dtype=torch.long)
        self.edge_attr = torch.tensor(adj_matrix[self.edge_index[0], self.edge_index[1]], dtype=torch.float)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.float)  # Shape: (sequence_length, num_nodes, num_features)
        y = torch.tensor(self.y[idx], dtype=torch.float)  # Shape: same as x

        return Data(x=x, edge_index=self.edge_index, edge_attr=self.edge_attr, y=y)


class DCRNN(L.LightningModule):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(DCRNN, self).__init__()
        self.loss_func = F.mse_loss
        self.activation = F.relu
        self.save_hyperparameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        return x

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        y = batch.y
        loss = self.loss_func(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# Example data
num_samples = 10
sequence_length = 12
num_nodes = 207
num_features = 4

x = np.random.rand(num_samples, sequence_length, num_nodes, num_features)
y = np.random.rand(num_samples, sequence_length, num_nodes, num_features)
adj_matrix = np.random.randint(0, 2, size=(num_nodes, num_nodes))

# Dataset and DataLoader
dataset = TrafficDataset(x, y, adj_matrix)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model initialization
model = DCRNN(in_channels=num_features,
              hidden_channels=64,
              out_channels=num_features)

# Training setup using PyTorch Lightning's Trainer
trainer = L.Trainer(max_epochs=1, logger=False, accelerator='cpu')
trainer.fit(model, dataloader)
