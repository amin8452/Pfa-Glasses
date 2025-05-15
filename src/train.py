import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
from datetime import datetime

from data import GlassesDataset, get_transforms
from models import HunyuanAdapter, GlassesReconstruction
from metrics import chamfer_distance, earth_movers_distance, iou_3d

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train(args):
    """Train the model."""
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data transforms
    transforms = get_transforms(image_size=args.image_size, split='train')
    
    # Create datasets
    train_dataset = GlassesDataset(
        data_dir=args.data_dir,
        split='train',
        transform=transforms['image'],
        target_transform=transforms['model'],
        image_size=args.image_size,
        max_samples=args.max_samples
    )
    
    val_dataset = GlassesDataset(
        data_dir=args.data_dir,
        split='val',
        transform=get_transforms(image_size=args.image_size, split='val')['image'],
        target_transform=get_transforms(image_size=args.image_size, split='val')['model'],
        image_size=args.image_size,
        max_samples=args.max_samples
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
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
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(args.checkpoint_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc="Training"):
            # Move batch to device
            images = batch['image'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            targets = {
                'labels': torch.zeros(images.size(0), dtype=torch.long, device=device)  # Placeholder
            }
            loss_dict = model.compute_loss(outputs, targets)
            loss = loss_dict['total_loss']
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * images.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                images = batch['image'].to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Compute loss
                targets = {
                    'labels': torch.zeros(images.size(0), dtype=torch.long, device=device)  # Placeholder
                }
                loss_dict = model.compute_loss(outputs, targets)
                loss = loss_dict['total_loss']
                
                # Update statistics
                val_loss += loss.item() * images.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print statistics
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, f"best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': args
            }, checkpoint_path)
            print(f"Saved best model checkpoint to {checkpoint_path}")
        
        # Save regular checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': args
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

def main():
    parser = argparse.ArgumentParser(description="Train a 3D glasses reconstruction model")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data", help="Path to dataset directory")
    parser.add_argument("--image_size", type=int, default=256, help="Size to resize images to")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to use")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default="tencent/Hunyuan3D-2", help="Path to Hunyuan3D-2 model")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of glasses classes")
    parser.add_argument("--feature_dim", type=int, default=512, help="Dimension of feature vector")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Checkpoint arguments
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save_freq", type=int, default=10, help="Frequency of saving checkpoints")
    
    args = parser.parse_args()
    
    train(args)

if __name__ == "__main__":
    main()
