# ELNESDiff

ELNESDiff is a diffusion model for generating corresponding 3D local coordination environments conditioned on ELNES.

## Usage

### Training
```bash
python main_cuda.py \
  --project_name <project_name> \
  --run_name <run_name> \
  --dataset_path <dataset_path> \
  --scaling max
```

### Generate with pre-trained model
```bash
python generate_with_learned_model.py \
  --learned_model <project_name/run_id> \
  --project_name <project_name> \
  --run_name <run_name> \
  --scaling max
```
