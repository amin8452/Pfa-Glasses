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
from src.metrics import chamfer_distance, earth_movers_distance, iou_3d
from src.utils import visualize_mesh, visualize_comparison

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
    metrics = {
        'chamfer_distance': [],
        'earth_movers_distance': [],
        'iou_3d': []
    }
    
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
        
        # Load ground truth mesh if available
        if gt_mesh_path is not None and os.path.exists(gt_mesh_path):
            gt_mesh = trimesh.load(gt_mesh_path)
            
            # Compute metrics
            cd = chamfer_distance(
                mesh.vertices,
                gt_mesh.vertices,
                bidirectional=True,
                reduction='mean'
            )
            metrics['chamfer_distance'].append(float(cd))
            
            emd = earth_movers_distance(
                mesh.vertices,
                gt_mesh.vertices,
                reduction='mean'
            )
            metrics['earth_movers_distance'].append(float(emd))
            
            iou = iou_3d(mesh, gt_mesh)
            metrics['iou_3d'].append(float(iou))
            
            # Visualize comparison
            comparison_path = os.path.join(args.output_dir, f"sample_{i}_comparison.png")
            visualize_comparison(
                meshes=[mesh, gt_mesh],
                titles=["Predicted", "Ground Truth"],
                save_path=comparison_path,
                show=False
            )
            
            # Print metrics for this sample
            print(f"Sample {i}:")
            print(f"  Chamfer Distance: {cd:.4f}")
            print(f"  Earth Mover's Distance: {emd:.4f}")
            print(f"  IoU: {iou:.4f}")
        else:
            # Just visualize the predicted mesh
            mesh_path = os.path.join(args.output_dir, f"sample_{i}_mesh.png")
            visualize_mesh(
                mesh=mesh,
                save_path=mesh_path,
                show=False
            )
    
    # Compute average metrics
    if metrics['chamfer_distance']:
        avg_cd = np.mean(metrics['chamfer_distance'])
        avg_emd = np.mean(metrics['earth_movers_distance'])
        avg_iou = np.mean(metrics['iou_3d'])
        
        print("\nAverage Metrics:")
        print(f"  Chamfer Distance: {avg_cd:.4f}")
        print(f"  Earth Mover's Distance: {avg_emd:.4f}")
        print(f"  IoU: {avg_iou:.4f}")
    
    print(f"\nResults saved to {args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Test the glasses reconstruction model")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data", help="Path to dataset directory")
    parser.add_argument("--image_size", type=int, default=256, help="Size to resize images to")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to test")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default="tencent/Hunyuan3D-2", help="Path to Hunyuan3D-2 model")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of glasses classes")
    parser.add_argument("--feature_dim", type=int, default=512, help="Dimension of feature vector")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to model checkpoint")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="results/test", help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    test_model(args)

if __name__ == "__main__":
    main()
