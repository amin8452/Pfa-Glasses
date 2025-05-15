# Model Checkpoints

This directory will store model checkpoints during training.

Checkpoints are saved in the following format:
- `best_model.pth`: The model with the best validation performance
- `model_epoch_N.pth`: Model checkpoint at epoch N

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Scheduler state dict
- Training and validation losses
- Training arguments
