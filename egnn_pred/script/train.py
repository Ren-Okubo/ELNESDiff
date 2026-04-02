import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf
import fire

@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg: DictConfig):
    # Set random seed for reproducibility
    pl.seed_everything(cfg.seed)
    
    # Instantiate modules
    data_module = hydra.utils.instantiate(cfg.module.data_module)
    lightning_module = hydra.utils.instantiate(cfg.module.lightning_module)

    # Instantiate trainer
    trainer = hydra.utils.instantiate(cfg.stage.trainer)

    # Train the model
    trainer.fit(lightning_module, datamodule=data_module)

def _main():
    fire.Fire(main)

if __name__ == "__main__":
    _main()