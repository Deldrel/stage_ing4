from functools import lru_cache

import wandb

from src.config import config
from src.helpers.decorators import timer
from .data_module import DataModule
from .dcrnn import DCRNN
from .trainer import get_trainer


class LightningManager:
    def __init__(self):
        self.data_module = None
        self.model = None
        self.trainer = None

    @lru_cache(maxsize=1)
    def setup(self) -> None:
        self.data_module = DataModule()
        self.model = DCRNN(in_channels=config.model.in_channels,
                           hidden_channels=config.model.hidden_channels,
                           out_channels=config.model.out_channels,
                           sequence_length=config.sequencer.sequence_length,
                           num_nodes=207,
                           num_features=4)
        self.trainer = get_trainer()

    @timer
    def train_model(self) -> None:
        self.setup()

        if config.trainer.log:
            wandb.init(project=config.wandb.project,
                       entity=config.wandb.entity,
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
