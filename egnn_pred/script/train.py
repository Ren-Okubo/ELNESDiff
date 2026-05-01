import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

@hydra.main(version_base=None, config_path="../config", config_name="default")
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

if __name__ == "__main__":
    main()
