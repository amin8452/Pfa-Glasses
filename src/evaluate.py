import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import trimesh
import json
from pathlib import Path

from data import GlassesDataset, get_transforms
from models import HunyuanAdapter, GlassesReconstruction
from metrics import chamfer_distance, earth_movers_distance, iou_3d
from utils import visualize_comparison, save_mesh

def evaluate(args):
    """Evaluate the model."""
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
        max_samples=args.max_samples
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    base_model = HunyuanAdapter(model_path=args.model_path, device=device)
    model = GlassesReconstruction(
        base_model=base_model,
        num_classes=args.num_classes,
        feature_dim=args.feature_dim
    )
    model = model.to(device)
    
    # Load checkpoint
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Create subdirectories
    meshes_dir = results_dir / "meshes"
    visualizations_dir = results_dir / "visualizations"
    meshes_dir.mkdir(exist_ok=True)
    visualizations_dir.mkdir(exist_ok=True)
    
    # Evaluation
    model.eval()
    metrics = {
        'chamfer_distance': [],
        'earth_movers_distance': [],
        'iou_3d': []
    }
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Evaluation")):
            # Move batch to device
            images = batch['image'].to(device)
            image_paths = batch['image_path']
            
            # Generate meshes
            for j, image in enumerate(images):
                # Get image path
                image_path = image_paths[j]
                image_name = os.path.basename(image_path).split('.')[0]
                
                # Generate mesh
                mesh = model.generate_mesh(
                    image=image.unsqueeze(0),
                    with_texture=True,
                    output_path=str(meshes_dir / f"{image_name}.obj")
                )
                
                # Get ground truth mesh path
                gt_mesh_path = batch['model_path'][j]
                
                # Compute metrics if ground truth mesh exists
                if gt_mesh_path is not None and os.path.exists(gt_mesh_path):
                    # Load ground truth mesh
                    gt_mesh = trimesh.load(gt_mesh_path)
                    
                    # Compute Chamfer distance
                    cd = chamfer_distance(
                        mesh.vertices,
                        gt_mesh.vertices,
                        bidirectional=True,
                        reduction='mean'
                    )
                    metrics['chamfer_distance'].append(float(cd))
                    
                    # Compute Earth Mover's Distance
                    emd = earth_movers_distance(
                        mesh.vertices,
                        gt_mesh.vertices,
                        reduction='mean'
                    )
                    metrics['earth_movers_distance'].append(float(emd))
                    
                    # Compute IoU
                    iou = iou_3d(mesh, gt_mesh)
                    metrics['iou_3d'].append(float(iou))
                    
                    # Visualize comparison
                    visualize_comparison(
                        meshes=[mesh, gt_mesh],
                        titles=["Predicted", "Ground Truth"],
                        save_path=str(visualizations_dir / f"{image_name}_comparison.png"),
                        show=False
                    )
    
    # Compute average metrics
    avg_metrics = {
        'chamfer_distance': np.mean(metrics['chamfer_distance']) if metrics['chamfer_distance'] else float('nan'),
        'earth_movers_distance': np.mean(metrics['earth_movers_distance']) if metrics['earth_movers_distance'] else float('nan'),
        'iou_3d': np.mean(metrics['iou_3d']) if metrics['iou_3d'] else float('nan')
    }
    
    # Save metrics
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            'individual': metrics,
            'average': avg_metrics
        }, f, indent=4)
    
    # Print average metrics
    print("Average Metrics:")
    print(f"Chamfer Distance: {avg_metrics['chamfer_distance']:.4f}")
    print(f"Earth Mover's Distance: {avg_metrics['earth_movers_distance']:.4f}")
    print(f"IoU: {avg_metrics['iou_3d']:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a 3D glasses reconstruction model")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data", help="Path to dataset directory")
    parser.add_argument("--image_size", type=int, default=256, help="Size to resize images to")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to use")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default="tencent/Hunyuan3D-2", help="Path to Hunyuan3D-2 model")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of glasses classes")
    parser.add_argument("--feature_dim", type=int, default=512, help="Dimension of feature vector")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    
    # Evaluation arguments
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    
    # Results arguments
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save results")
    
    args = parser.parse_args()
    
    evaluate(args)

if __name__ == "__main__":
    main()
