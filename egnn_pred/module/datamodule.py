import pytorch_lightning as pl
import os, torch, numpy and
from pathlib import Path

class DataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_dataset_path,
            val_dataset_path,
            test_dataset_path,
            batch_size,
            num_workers,
            pin_memory=False,
    ):
        super(DataModule, self).__init__()
        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.test_dataset_path = test_dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.save_hyperparameters()

    def setup(self, stage=None):
        train_data = torch.load(self.train_dataset_path, weights_only=False)
        val_data = torch.load(self.val_dataset_path, weights_only=False)
        test_data = torch.load(self.test_dataset_path, weights_only=False)
        