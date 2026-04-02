import pytorch_lightning as pl
from pathlib import Path

class LightningModule(pl.LightningModule):
    def __init__(self, model, optimizer, scheduler):
        super().__init__()
        self.model = model
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.optimizer = None
        self.scheduler = None
        self.save_hyperparameters()

    def configure_optimizers(self):
        self.optimizer = self.optimizer_factory(self.model.parameters())
        self.scheduler = self.scheduler_factory(self.optimizer)
        return [{"optimizer": self.optimizer, "lr_scheduler": self.scheduler, "monitor": "val_loss"}]
    
    def training_step(self, batch, batch_idx):
        target_pos, target_atom_type, batch_spectrum, batch_edge_index = batch
        model_output = self.model(batch_edge_index, h=batch_spectrum, x=None)
        pred_coords = model_output[:, :3]
        pred_one_hot = model_output[:, 3:]
        loss_coords = ((pred_coords - target_pos) ** 2).mean()
        loss_one_hot = ((pred_one_hot - target_atom_type) ** 2).mean()
        loss = loss_coords + loss_one_hot
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        target_pos, target_atom_type, batch_spectrum, batch_edge_index = batch
        model_output = self.model(batch_edge_index, h=batch_spectrum, x=None)
        pred_coords = model_output[:, :3]
        pred_one_hot = model_output[:, 3:]
        loss_coords = ((pred_coords - target_pos) ** 2).mean()
        loss_one_hot = ((pred_one_hot - target_atom_type) ** 2).mean()
        loss = loss_coords + loss_one_hot
        self.log("val_loss", loss)
        return loss