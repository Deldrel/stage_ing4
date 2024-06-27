from datetime import datetime
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn.conv import GCNConv

from src.config import config


class DCRNN(L.LightningModule):
    def __init__(self, in_channels, hidden_channels, out_channels, sequence_length, num_nodes, num_features):
        super(DCRNN, self).__init__()
        self.sequence_length = sequence_length
        self.hidden_channels = hidden_channels
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

        self.loss_func = F.mse_loss
        self.activation = F.relu
        self.save_hyperparameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        seq_len, num_nodes, in_channels = x.size()
        x = x.view(seq_len * num_nodes, in_channels)

        x = self.conv1(x, edge_index, edge_weight)
        x = self.activation(x)

        x = self.conv2(x, edge_index, edge_weight)

        x = x.view(seq_len, num_nodes, -1)
        return x

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = self.loss_func(y_hat, batch.y)
        self.log('train_loss', loss, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = self.loss_func(y_hat, batch.y)
        self.log('val_loss', loss, batch_size=batch.num_graphs)

    def test_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = self.loss_func(y_hat, batch.y)
        self.log('test_loss', loss, batch_size=batch.num_graphs)

    def on_train_epoch_start(self) -> None:
        if config.trainer.log:
            self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'])

    def on_train_end(self) -> None:
        path = Path(config.trainer.save_dir)
        path.mkdir(parents=True, exist_ok=True)
        model_name = f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
        torch.save(self.state_dict(), path / model_name)

        if config.trainer.log:
            wandb.save(str(path / model_name))

    def configure_optimizers(self):
        optimizer_class = getattr(torch.optim, config.model.optimizer)
        optimizer = optimizer_class(self.parameters(), lr=config.model.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode=config.reduce_lr_on_plateau.mode,
                                      factor=config.reduce_lr_on_plateau.factor,
                                      patience=config.reduce_lr_on_plateau.patience)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'monitor': config.reduce_lr_on_plateau.monitor,
                'strict': True,
            }
        }
