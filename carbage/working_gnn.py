import torch
import torch.nn.functional as F
from torch.nn import LayerNorm, Dropout
from torch_geometric.nn import GCNConv
import pytorch_lightning as pl
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import numpy as np


# Configuration object
class Config:
    class model:
        dropout_rate = 0.5
        loss_function = 'mse_loss'  # example loss function
        activation = 'relu'  # example activation function


config = Config()


# TrafficDataset definition
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
        x = torch.tensor(self.x[idx], dtype=torch.float)  # Shape: (12, 207, 4)
        y = torch.tensor(self.y[idx], dtype=torch.float)  # Shape: same as x

        x = x.view(-1, self.x.shape[-1])  # Shape: (12*207, 4)
        y = y.view(-1, self.y.shape[-1])  # Shape: same as x

        return Data(x=x, edge_index=self.edge_index, edge_attr=self.edge_attr, y=y)


# DCRNN model definition
class DCRNN(pl.LightningModule):
    def __init__(self, in_channels, hidden_channels, out_channels, sequence_length, config):
        super(DCRNN, self).__init__()
        self.sequence_length = sequence_length
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

        self.norm1 = LayerNorm(hidden_channels)
        self.norm2 = LayerNorm(out_channels)

        self.dropout = Dropout(p=config.model.dropout_rate)

        self.loss_func = getattr(F, config.model.loss_function)
        self.activation = getattr(F, config.model.activation)
        self.save_hyperparameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        # First GCN layer
        x = self.conv1(x, edge_index, edge_weight)
        x = self.activation(x)
        x = self.norm1(x)
        x = self.dropout(x)

        # Second GCN layer
        x = self.conv2(x, edge_index, edge_weight)
        x = self.norm2(x)

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
num_samples = 100
sequence_length = 12
num_nodes = 207
num_features = 4

x = np.random.rand(num_samples, sequence_length, num_nodes,
                   num_features)  # 100 samples, 12 time steps, 207 nodes, 4 features
y = np.random.rand(num_samples, sequence_length, num_nodes, num_features)
adj_matrix = np.random.randint(0, 2, size=(num_nodes, num_nodes))

# Dataset and DataLoader
dataset = TrafficDataset(x, y, adj_matrix)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model initialization
model = DCRNN(in_channels=num_features, hidden_channels=16, out_channels=num_features, sequence_length=sequence_length,
              config=config)

# Training setup using PyTorch Lightning's Trainer
trainer = pl.Trainer(max_epochs=1, logger=False)
trainer.fit(model, dataloader)
