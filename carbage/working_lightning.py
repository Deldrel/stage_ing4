import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, Dataset
from torch_geometric.nn import GCNConv


class MyDataset(Dataset):
    def __init__(self, x, y, adj_matrix):
        super().__init__()
        self.x = x
        self.y = y
        self.adj_matrix = adj_matrix
        self.num_samples = x.shape[0]
        self.num_nodes = adj_matrix.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.float)  # Shape: (12, 207, 4)
        y = torch.tensor(self.y[idx], dtype=torch.float)  # Shape: (12, 207, 4)

        # Flatten the node features to match the expected input for GCN
        x = x.view(-1, 4)  # Shape: (12*207, 4)
        y = y.view(-1, 4)  # Shape: (12*207, 4)

        # Ensure edge indices are within valid range
        edge_index = torch.tensor(np.nonzero(self.adj_matrix), dtype=torch.long)
        edge_attr = torch.tensor(self.adj_matrix[edge_index[0], edge_index[1]], dtype=torch.float)

        if edge_index.max().item() >= self.num_nodes:
            raise ValueError(f"Edge index {edge_index.max().item()} is out of bounds for num_nodes {self.num_nodes}")

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return data


class MyDataModule(L.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size=32):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class MyGNN(L.LightningModule):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MyGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = F.mse_loss(y_hat, batch.y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = F.mse_loss(y_hat, batch.y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = F.mse_loss(y_hat, batch.y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


# Load data
train_data = np.load('data/sequences/train.npz')
val_data = np.load('data/sequences/val.npz')
test_data = np.load('data/sequences/test.npz')
adj_matrix = np.load('data/crafted/adj_mx.npy')

# Extract matrices
train_x = train_data['x']
train_y = train_data['y']
val_x = val_data['x']
val_y = val_data['y']
test_x = test_data['x']
test_y = test_data['y']

print(f"Train data shape: {train_x.shape}, {train_y.shape}")
print(f"Val data shape: {val_x.shape}, {val_y.shape}")
print(f"Test data shape: {test_x.shape}, {test_y.shape}")
print(f"Adjacency matrix shape: {adj_matrix.shape}")

# Ensure adjacency matrix indices are within bounds
if adj_matrix.shape[0] != adj_matrix.shape[1]:
    raise ValueError("Adjacency matrix is not square")
if adj_matrix.shape[0] != train_x.shape[2]:
    raise ValueError("Adjacency matrix size does not match the number of nodes")

# Create dataset instances
train_dataset = MyDataset(train_x, train_y, adj_matrix)
val_dataset = MyDataset(val_x, val_y, adj_matrix)
test_dataset = MyDataset(test_x, test_y, adj_matrix)

# Instantiate the data module
data_module = MyDataModule(train_dataset, val_dataset, test_dataset)

# Instantiate the model
model = MyGNN(in_channels=4, hidden_channels=64, out_channels=4)

# Initialize the trainer
trainer = L.Trainer(max_epochs=1)

# Train the model
trainer.fit(model, data_module)
trainer.test(datamodule=data_module)
