from pathlib import Path

import numpy as np
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader


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
        x = torch.tensor(self.x[idx], dtype=torch.float)
        y = torch.tensor(self.y[idx], dtype=torch.float)
        x = x.view(12, self.num_nodes, -1)
        y = y.view(12, self.num_nodes, -1)

        edge_index = np.array(np.nonzero(self.adj_matrix))
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(self.adj_matrix[edge_index[0], edge_index[1]], dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return data


class DataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.adj_mx = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        sequences_path = Path('data/sequences')
        self.adj_mx = np.load(Path('data/crafted/adj_mx.npy'))
        self.adj_mx += np.eye(self.adj_mx.shape[0])

        train_data = np.load(sequences_path / 'train.npz')
        x_train, y_train = train_data['x'], train_data['y']

        val_data = np.load(sequences_path / 'val.npz')
        x_val, y_val = val_data['x'], val_data['y']

        test_data = np.load(sequences_path / 'test.npz')
        x_test, y_test = test_data['x'], test_data['y']

        self.train_dataset = MyDataset(x_train, y_train, self.adj_mx)
        self.val_dataset = MyDataset(x_val, y_val, self.adj_mx)
        self.test_dataset = MyDataset(x_test, y_test, self.adj_mx)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=64, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=64, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=64, shuffle=False, num_workers=4)


class DiffusionConvolution(nn.Module):
    def __init__(self, num_nodes, k, supports):
        super(DiffusionConvolution, self).__init__()
        self.num_nodes = num_nodes
        self.k = k
        self.supports = supports
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(self.num_nodes, self.num_nodes)) for _ in range(self.k)])
        self.bias = nn.Parameter(torch.FloatTensor(self.num_nodes))

    def forward(self, x):
        batch_size, num_nodes, num_features = x.size()
        out = torch.zeros(batch_size, num_nodes, num_features)

        for i in range(self.k):
            support = self.supports[i]
            weight = self.weights[i]
            x_support = torch.einsum('bnf,ff->bnf', (x, support))
            out += torch.einsum('bnf,ff->bnf', (x_support, weight))

        return out + self.bias


class DCRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_nodes, k, supports):
        super(DCRNNCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.dcgru = DiffusionConvolution(num_nodes, k, supports)

        self.conv_gate = nn.Conv2d(in_channels=input_dim + hidden_dim, out_channels=2 * hidden_dim, kernel_size=(1, 1))
        self.conv_cand = nn.Conv2d(in_channels=input_dim + hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1))

    def forward(self, x, hidden):
        x_hidden = torch.cat([x, hidden], dim=2)
        x_hidden = x_hidden.permute(0, 2, 1).unsqueeze(3)

        gates = self.conv_gate(x_hidden)
        update_gate, reset_gate = torch.split(gates, self.hidden_dim, dim=1)
        update_gate = torch.sigmoid(update_gate)
        reset_gate = torch.sigmoid(reset_gate)

        reset_hidden = hidden * reset_gate.squeeze(3).permute(0, 2, 1)
        candidate = torch.cat([x, reset_hidden], dim=2)
        candidate = candidate.permute(0, 2, 1).unsqueeze(3)

        candidate = torch.tanh(self.conv_cand(candidate)).squeeze(3).permute(0, 2, 1)
        hidden = update_gate * hidden + (1 - update_gate) * candidate
        return hidden


class DCRNN(LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, num_nodes, k, supports, num_layers):
        super(DCRNN, self).__init__()
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList(
            [DCRNNCell(input_dim if i == 0 else hidden_dim, hidden_dim, num_nodes, k, supports) for i in
             range(num_layers)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.loss_func = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, data):
        x = data.x
        batch_size, seq_len, num_nodes, num_features = x.size()
        hidden_states = [torch.zeros(batch_size, num_nodes, self.hparams.hidden_dim).to(x.device) for _ in
                         range(self.num_layers)]

        for t in range(seq_len):
            input_t = x[:, t, :, :]
            for i, cell in enumerate(self.dcrnn_cells):
                hidden_states[i] = cell(input_t, hidden_states[i])
                input_t = hidden_states[i]

        output = self.output_layer(hidden_states[-1])
        return output

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = self.loss_func(y_hat, batch.y)
        self.log('train_loss', loss, batch_size=batch.x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = self.loss_func(y_hat, batch.y)
        self.log('val_loss', loss, batch_size=batch.x.size(0))
        return loss

    def test_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = self.loss_func(y_hat, batch.y)
        self.log('test_loss', loss, batch_size=batch.x.size(0))
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}


# Test Script
if __name__ == '__main__':
    datamodule = DataModule()
    supports = [torch.FloatTensor(np.load(Path('data/crafted/adj_mx.npy')))]
    model = DCRNN(input_dim=4, hidden_dim=64, output_dim=4, num_nodes=207, k=3, supports=supports, num_layers=3)
    trainer = Trainer(max_epochs=50)
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)
