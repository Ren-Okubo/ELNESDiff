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
            normalize=None,
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
        self.normalize = normalize
        self.save_hyperparameters()

    def setup(self, stage=None):
        train_data = torch.load(self.train_dataset_path, weights_only=False)
        val_data = torch.load(self.val_dataset_path, weights_only=False)
        test_data = torch.load(self.test_dataset_path, weights_only=False)

        self.train_dataset = train_data
        self.val_dataset = val_data
        self.test_dataset = test_data

    def collate_fn(self, batch):
        pos_tensor_list = []
        atom_type_tensor_list = []
        spectrum_tensor_list = []
        for data in batch:
            pos_tensor_list.append(data.pos)
            atom_type_tensor_list(data.x)
            spectrum_tensor = data.spectrum_raw[0]
            if self.normalize is None:
                normalized_spectrum = spectrum_tensor.repeat(data.x.shape[0],1)
            elif self.normaize == 'max':
                max_value = spectrum_tensor.max()
                normalized_spectrum = (spectrum_tensor / max_value).repeat(data.x.shape[0],1)
            elif self.normalize == 'sum':
                sum_value = spectrum_tensor.sum()
                normalized_spectrum = (spectrum_tensor / sum_value).repeat(data.x.shape[0],1)
            else:
                raise ValueError(f"Invalid normalization method: {self.normalize}")
            spectrum_tensor_list.append(normalized_spectrum)            

        return pos_atom_type_tensor, spectrum_tensor, edge_index
        