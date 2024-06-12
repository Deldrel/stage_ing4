from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.config import config


def get_trainer() -> Trainer:
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint.dirpath,
        filename=config.checkpoint.filename,
        monitor=config.checkpoint.monitor,
        verbose=config.checkpoint.verbose,
        save_last=config.checkpoint.save_last,
        save_top_k=config.checkpoint.save_top_k,
        mode=config.checkpoint.mode
    )

    early_stopping_callback = EarlyStopping(
        monitor=config.early_stopping.monitor,
        # min_delta=config.early_stopping.min_delta,
        patience=config.early_stopping.patience,
        verbose=config.early_stopping.verbose,
        mode=config.early_stopping.mode
    )

    if config.trainer.log:
        logger = WandbLogger(
            project=config.wandb.project,
            save_dir=config.logdir
        )
    else:
        logger = False

    return Trainer(
        max_epochs=config.trainer.max_epochs,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
        ],
        logger=logger
    )
