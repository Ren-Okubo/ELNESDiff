import sys
from pathlib import Path

import hydra
import torch
import wandb
from pymatgen.core import Molecule
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module.lightningmodule import LightningModule


@hydra.main(version_base=None, config_path="../config", config_name="predict")
def main(cfg: DictConfig):

    # instantiate wandb
    run = wandb.init(
        project=cfg.stage.wandb.project_name,
        name=cfg.stage.wandb.run_name,
        tags=cfg.stage.wandb.tags,
        notes=cfg.stage.wandb.notes,
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    # setting device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load training config
    trained_model_dir_path = Path(f"/jbod/rokubo/ELNESDiff/egnn_pred/outputs/{cfg.stage.trained_model}/train")
    trained_model_config_path = trained_model_dir_path / ".hydra" / "config.yaml"
    
    # load config of the trained model
    trained_model_config = OmegaConf.load(trained_model_config_path)
    OmegaConf.resolve(trained_model_config)

    # Set random seed for reproducibility
    pl.seed_everything(trained_model_config.seed)
    
    # Reinstantiate modules
    checkpoint_path = Path("/jbod/rokubo/ELNESDiff/egnn_pred/outputs/") / cfg.stage.trained_model / "train" / "checkpoints" / "best_model.ckpt"
    lightning_module = LightningModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location=device
    )

    data_module = hydra.utils.instantiate(trained_model_config.module.data_module)
    data_module.setup(stage="test")

    predict_dataloader = data_module.test_dataloader()
    lightning_module.eval()
    lightning_module.to(device)

    save_dir = Path(cfg.stage.result_save_path) / "predictions"
    save_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in predict_dataloader:
            batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]
            target_pos, target_atom_type, pred_pos, pred_atom_type, num_atoms_list, id_list = lightning_module.predict(batch, None)
            # Process predictions as needed
            target_species = torch.argmax(target_atom_type, dim=-1)
            target_species_list = ["O" if atom_type == 0 else "Si" for atom_type in target_species.cpu().numpy()]
            pred_species = torch.argmax(pred_atom_type, dim=-1)
            pred_species_list = ["O" if atom_type == 0 else "Si" for atom_type in pred_species.cpu().numpy()]
            # Save the molecules
            start_idx = 0
            for i, num_atoms in enumerate(num_atoms_list):
                material_id = id_list[i]
                target_coords = target_pos[start_idx:start_idx+num_atoms].cpu().numpy()
                target_species = target_species_list[start_idx:start_idx+num_atoms]
                pred_coords = pred_pos[start_idx:start_idx+num_atoms].cpu().numpy()
                pred_species = pred_species_list[start_idx:start_idx+num_atoms]
                start_idx += num_atoms
                target_molecule = Molecule(target_species, target_coords)
                pred_molecule = Molecule(pred_species, pred_coords)
                save_dir = Path(cfg.stage.result_save_path) / "predictions" / material_id
                save_dir.mkdir(parents=True, exist_ok=True)
                target_molecule.to(fmt="xyz", filename=save_dir / f"original.xyz")
                pred_molecule.to(fmt="xyz", filename=save_dir / f"generated.xyz")
    
    print(f"Predictions saved to {cfg.stage.result_save_path}/predictions")

if __name__ == "__main__":
    main()

