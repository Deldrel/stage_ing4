from functools import lru_cache
from pathlib import Path

import torch
import wandb

from src.config import config
from src.helpers.decorators import timer
from .data_module import DataModule
from .gnn import GNN
from .trainer import get_trainer


class LightningManager:
    def __init__(self):
        self.data_module = None
        self.model = None
        self.trainer = None

    @lru_cache(maxsize=1)
    def setup(self) -> None:
        self.data_module = DataModule()
        self.model = GNN(in_channels=4, hidden_channels=64, out_channels=4, num_layers=4)
        self.search_checkpoint()
        self.trainer = get_trainer()

    def search_checkpoint(self) -> None:
        path = Path(config.trainer.save_dir)
        if not path.exists():
            return
        checkpoints = sorted(path.glob('*.pt'))
        if not checkpoints:
            return

        if input(f"Load {checkpoints[-1]}? [y/n]: ") == "y":
            try:
                checkpoint = torch.load(checkpoints[-1])
                self.model.load_state_dict(checkpoint)
            except Exception as e:
                print(f"Error loading checkpoint: {e}, keeping new model.")

    @timer
    def train_model(self) -> None:
        self.setup()

        if config.trainer.log:
            wandb.init(project=config.wandb.project,
                       dir=config.logdir,
                       config=config.dump())

        print(f"NOTE: you can interrupt the training whenever you want with a keyboard interrupt (CTRL+C)")
        self.trainer.fit(self.model, self.data_module)
        self.trainer.test(self.model, self.data_module)

        if config.trainer.log:
            wandb.finish()


@lru_cache(maxsize=1)
def get_lightning_manager() -> LightningManager:
    return LightningManager()


lightning_manager = get_lightning_manager()
