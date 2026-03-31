import pytorch_lightning as pl
from pathlib import Path

class LigntningModule(pl.LightningModule):
    def __init__(self, model, optimizer, scheduler):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]
    
    def training_step(self, batch, batch_idx):
        target_coords = batch.coords
        target_one_hot = batch.one_hot
        batch_edge_index = batch.edge_index
        batch_spectrum = batch.spectrum
        model_output = self.model(batch_edge_index, h=batch_spectrum, x=None)
        pred_coords = model_output[:, :3]
        pred_one_hot = model_output[:, 3:]
        loss_coords = ((pred_coords - target_coords) ** 2).mean()
        loss_one_hot = ((pred_one_hot - target_one_hot) ** 2).mean()
        loss = loss_coords + loss_one_hot
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        target_coords = batch.coords
        target_one_hot = batch.one_hot
        batch_edge_index = batch.edge_index
        batch_spectrum = batch.spectrum
        model_output = self.model(batch_edge_index, h=batch_spectrum, x=None)
        pred_coords = model_output[:, :3]
        pred_one_hot = model_output[:, 3:]
        loss_coords = ((pred_coords - target_coords) ** 2).mean()
        loss_one_hot = ((pred_one_hot - target_one_hot) ** 2).mean()
        loss = loss_coords + loss_one_hot
        self.log("val_loss", loss)
        return loss