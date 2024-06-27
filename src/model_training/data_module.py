from pathlib import Path

import numpy as np
import torch
from lightning import LightningDataModule
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from src.config import config


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


class DataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.adj_mx = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        pass

    def setup(self, stage: str) -> None:
        sequences_path = Path('data/sequences')
        self.adj_mx = np.load(Path('data/crafted/adj_mx.npy'))
        self.adj_mx += np.eye(self.adj_mx.shape[0])

        train_data = np.load(sequences_path / 'train.npz')
        x_train, y_train = train_data['x'], train_data['y']

        val_data = np.load(sequences_path / 'val.npz')
        x_val, y_val = val_data['x'], val_data['y']

        test_data = np.load(sequences_path / 'test.npz')
        x_test, y_test = test_data['x'], test_data['y']

        self.train_dataset = TrafficDataset(x_train, y_train, self.adj_mx)
        self.val_dataset = TrafficDataset(x_val, y_val, self.adj_mx)
        self.test_dataset = TrafficDataset(x_test, y_test, self.adj_mx)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=config.data_module.num_workers,
                          persistent_workers=config.data_module.persistent_workers,
                          batch_size=config.data_module.batch_size,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          num_workers=config.data_module.num_workers,
                          persistent_workers=config.data_module.persistent_workers,
                          batch_size=config.data_module.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=config.data_module.num_workers,
                          persistent_workers=config.data_module.persistent_workers,
                          batch_size=config.data_module.batch_size)
