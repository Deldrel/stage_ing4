from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from lightning import LightningModule
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.conv import GCNConv

from src.config import config


class GNN(LightningModule):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, num_layers=3):
        super(GNN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, out_channels))

        self.loss_func = getattr(nn, config.model.loss_function)()
        self.save_hyperparameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index, edge_weight)
        return x.view(-1, self.hparams.out_channels)

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
        optimizer_class = getattr(optim, config.model.optimizer)
        optimizer = optimizer_class(self.parameters(),
                                    lr=config.model.learning_rate)
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
