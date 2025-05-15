# Glasses 3D Reconstruction

This project implements a 3D reconstruction pipeline for glasses based on Tencent's Hunyuan3D-2 model. It provides tools for reconstructing 3D glasses models from 2D images, evaluating the reconstructions, and visualizing the results.

## Features

- **Custom Data Loading**: Specialized dataset class for glasses images
- **3D Metrics**: Implementation of Chamfer distance, Earth Mover's Distance (EMD), and 3D IoU
- **Training and Evaluation Scripts**: Scripts for training and evaluating the model
- **Visualization Tools**: Utilities for visualizing 3D meshes and point clouds
- **Pipeline Notebook**: Comprehensive notebook demonstrating the complete workflow

## Project Structure

```
Hunyuan3D-Pfa/
├── data/                  # Directory for the glasses dataset (5000 images)
├── checkpoints/           # Directory for saving model checkpoints
├── notebooks/             # Directory for Jupyter notebooks
│   └── glasses_reconstruction_pipeline.ipynb  # Complete workflow notebook
├── results/               # Directory for saving reconstruction results
├── src/                   # Source code
│   ├── data/              # Data loading and processing
│   │   ├── __init__.py
│   │   ├── dataset.py     # Custom dataset for glasses
│   │   └── transforms.py  # Data transformations
│   ├── models/            # Model implementations
│   │   ├── __init__.py
│   │   ├── hunyuan_adapter.py  # Adapter for Hunyuan3D-2 model
│   │   └── reconstruction.py   # 3D reconstruction model
│   ├── metrics/           # 3D metrics implementation
│   │   ├── __init__.py
│   │   ├── chamfer.py     # Chamfer distance implementation
│   │   ├── emd.py         # Earth Mover's Distance implementation
│   │   └── iou.py         # IoU implementation
│   │   └── train.py       # Training script
│   │   └── evaluate.py    # Evaluation script
│   └── utils/             # Utility functions
│       ├── __init__.py
│       ├── visualization.py  # Visualization utilities
│       └── io.py          # I/O utilities
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/amin8452/Pfa-Glasses.git
   cd Pfa-Glasses
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Install PyTorch3D following the [official instructions](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).

## Dataset Preparation

The project expects the dataset to be organized as follows:

```
data/
├── images/           # Directory containing glasses images
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── 3d_models/        # Directory containing 3D models (optional)
│   ├── image1.obj
│   ├── image2.obj
│   └── ...
├── train.txt         # List of training samples (optional)
├── val.txt           # List of validation samples (optional)
└── test.txt          # List of test samples (optional)
```

If the split files (`train.txt`, `val.txt`, `test.txt`) are not provided, the dataset will be automatically split into training (80%), validation (10%), and test (10%) sets.

## Usage

### Training

To train the model, run:

```bash
python src/train.py --data_dir data --checkpoint_dir checkpoints --batch_size 16 --num_epochs 100
```

Additional options:
- `--image_size`: Size to resize images to (default: 256)
- `--model_path`: Path to Hunyuan3D-2 model (default: "tencent/Hunyuan3D-2")
- `--lr`: Learning rate (default: 1e-4)
- `--device`: Device to use (default: "cuda")

### Evaluation

To evaluate the model, run:

```bash
python src/evaluate.py --data_dir data --checkpoint checkpoints/best_model.pth --results_dir results
```

Additional options:
- `--image_size`: Size to resize images to (default: 256)
- `--model_path`: Path to Hunyuan3D-2 model (default: "tencent/Hunyuan3D-2")
- `--device`: Device to use (default: "cuda")

### Pipeline Notebook

The `notebooks/glasses_reconstruction_pipeline.ipynb` notebook provides a comprehensive demonstration of the complete workflow, including:

1. Loading and preprocessing the dataset
2. Loading the model
3. Generating 3D reconstructions
4. Visualizing the reconstructions
5. Evaluating the reconstructions using various metrics
6. Reconstructing custom images

## 3D Metrics

The project implements three common metrics for evaluating 3D reconstructions:

1. **Chamfer Distance**: Measures the average distance between points in two point clouds.
2. **Earth Mover's Distance (EMD)**: Measures the minimum cost of transforming one point cloud into another.
3. **Intersection over Union (IoU)**: Measures the overlap between two 3D volumes.

## Model Architecture

The model architecture consists of two main components:

1. **HunyuanAdapter**: An adapter for the Hunyuan3D-2 model that provides an interface for generating 3D meshes from images.
2. **GlassesReconstruction**: A model that adapts the Hunyuan3D-2 model for glasses reconstruction by adding domain-specific layers and loss functions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Tencent Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2) for the base 3D reconstruction model
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d) for 3D utilities and differentiable rendering
- [Trimesh](https://github.com/mikedh/trimesh) for mesh processing utilities
