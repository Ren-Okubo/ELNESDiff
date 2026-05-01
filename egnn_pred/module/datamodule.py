import pytorch_lightning as pl
import os, torch, numpy as np
from pathlib import Path
from itertools import permutations

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
        num_atoms_list = []
        id_list = []
        for data in batch:
            pos_tensor_list.append(data.pos)
            atom_type_tensor_list.append(data.x)
            num_atoms_list.append(data.x.shape[0])
            id_list.append(data.id)
            spectrum_tensor = data.spectrum_raw[0]
            if self.normalize is None:
                normalized_spectrum = spectrum_tensor.repeat(data.x.shape[0],1)
            elif self.normalize == 'max':
                max_value = spectrum_tensor.max()
                normalized_spectrum = (spectrum_tensor / max_value).repeat(data.x.shape[0],1)
            elif self.normalize == 'sum':
                sum_value = spectrum_tensor.sum()
                normalized_spectrum = (spectrum_tensor / sum_value).repeat(data.x.shape[0],1)
            else:
                raise ValueError(f"Invalid normalization method: {self.normalize}")
            spectrum_tensor_list.append(normalized_spectrum)
        total_pos_tensor = torch.cat(pos_tensor_list, dim=0)
        total_atom_type_tensor = torch.cat(atom_type_tensor_list, dim=0)
        total_spectrum_tensor = torch.cat(spectrum_tensor_list, dim=0)
        assert total_pos_tensor.shape[0] == total_atom_type_tensor.shape[0] == total_spectrum_tensor.shape[0], "Batch size mismatch"
        edge_index_list = []
        start_idx = 0
        for num_atom in num_atoms_list:
            perm = list(permutations(range(start_idx, start_idx + num_atom), 2))
            edge_index_list.append(torch.tensor(perm, dtype=torch.long).t())
            start_idx += num_atom
        edge_index = torch.cat(edge_index_list, dim=1) # shape: [2, num_edges]
        return total_pos_tensor, total_atom_type_tensor, total_spectrum_tensor, edge_index, num_atoms_list, id_list

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )
        