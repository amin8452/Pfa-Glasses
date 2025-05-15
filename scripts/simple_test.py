import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import trimesh
import random
from tqdm import tqdm

# Add the project root to the path
import sys
sys.path.append('.')

from src.data import GlassesDataset, get_transforms
from src.models import HunyuanAdapter, GlassesReconstruction

def test_model(args):
    """Test the model on a sample from the dataset."""
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data transforms
    transforms = get_transforms(image_size=args.image_size, split='test')
    
    # Create dataset
    test_dataset = GlassesDataset(
        data_dir=args.data_dir,
        split='test',
        transform=transforms['image'],
        target_transform=transforms['model'],
        image_size=args.image_size,
        max_samples=args.num_samples
    )
    
    # Create model
    base_model = HunyuanAdapter(model_path=args.model_path, device=device)
    model = GlassesReconstruction(
        base_model=base_model,
        num_classes=args.num_classes,
        feature_dim=args.feature_dim
    )
    model = model.to(device)
    
    # Load checkpoint if available
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("No checkpoint found, using untrained model")
    
    # Set model to evaluation mode
    model.eval()
    
    # Create results directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Select random samples
    if args.num_samples > 0:
        indices = random.sample(range(len(test_dataset)), min(args.num_samples, len(test_dataset)))
    else:
        indices = range(len(test_dataset))
    
    # Test the model on selected samples
    for i in tqdm(indices, desc="Testing model"):
        # Get sample
        sample = test_dataset[i]
        image = sample['image'].unsqueeze(0).to(device)  # Add batch dimension
        image_path = sample['image_path']
        gt_mesh_path = sample['model_path']
        
        # Generate mesh
        with torch.no_grad():
            output_path = os.path.join(args.output_dir, f"sample_{i}_pred.obj")
            mesh = model.generate_mesh(
                image=image,
                with_texture=True,
                output_path=output_path
            )
        
        print(f"Generated mesh for sample {i}, saved to {output_path}")
        
        # Load ground truth mesh if available
        if gt_mesh_path is not None and os.path.exists(gt_mesh_path):
            print(f"Ground truth mesh available at {gt_mesh_path}")
    
    print(f"\nResults saved to {args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Test the glasses reconstruction model")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data", help="Path to dataset directory")
    parser.add_argument("--image_size", type=int, default=256, help="Size to resize images to")
    parser.add_argument("--num_samples", type=int, default=2, help="Number of samples to test")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default="tencent/Hunyuan3D-2", help="Path to Hunyuan3D-2 model")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of glasses classes")
    parser.add_argument("--feature_dim", type=int, default=512, help="Dimension of feature vector")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to model checkpoint")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="results/test", help="Directory to save results")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    
    args = parser.parse_args()
    
    test_model(args)

if __name__ == "__main__":
    main()
