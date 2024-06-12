from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.config import config
from src.helpers.csv_file_manager import load_csv_files


class DataModule(LightningDataModule):
    def __init__(self) -> None:
        """
        This class is responsible for preparing the datasets and setting up the data loaders for the model.
        """
        super(DataModule, self).__init__()
        self.dataframes = load_csv_files(config.preprocessing.training_ready_in, verbose=True)
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.predict_set = None

    def prepare_data(self) -> None:
        """
        This function is responsible for preparing the datasets. You can download the data here as it is called once by the trainer
        """
        pass

    def setup(self, stage: str) -> None:
        """
        Split data here, the trainer calls this function automatically.
        """
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set,
                          batch_size=config.data_module.batch_size,
                          num_workers=config.data_module.num_workers,
                          persistent_workers=config.data_module.persistent_workers,
                          shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set,
                          batch_size=config.data_module.batch_size,
                          num_workers=config.data_module.num_workers,
                          persistent_workers=config.data_module.persistent_workers,
                          shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set,
                          batch_size=config.data_module.batch_size,
                          num_workers=config.data_module.num_workers,
                          persistent_workers=config.data_module.persistent_workers,
                          shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.predict_set,
                          batch_size=config.data_module.batch_size,
                          num_workers=config.data_module.num_workers,
                          persistent_workers=config.data_module.persistent_workers,
                          shuffle=False)
