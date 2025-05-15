# Glasses Dataset

This directory should contain the glasses dataset with the following structure:

```
data/
├── images/           # Directory containing glasses images (5000 images)
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

## Dataset Preparation

1. Place your glasses images in the `images/` directory.
2. If you have corresponding 3D models, place them in the `3d_models/` directory with the same base name as the images.
3. Optionally, create split files (`train.txt`, `val.txt`, `test.txt`) listing the image filenames for each split.
