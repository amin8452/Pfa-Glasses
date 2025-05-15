import os
import argparse
import subprocess
import time

def main():
    parser = argparse.ArgumentParser(description="Create glasses dataset")
    parser.add_argument("--total", type=int, default=5000, help="Total number of images to create")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic images instead of downloading")
    args = parser.parse_args()
    
    # Step 1: Download or generate images
    print("Step 1: Creating image dataset...")
    cmd = ["python", "scripts/download_dataset.py", f"--total={args.total}"]
    if args.synthetic:
        cmd.append("--synthetic")
    
    subprocess.run(cmd, check=True)
    
    # Step 2: Generate 3D models
    print("\nStep 2: Generating 3D models...")
    subprocess.run(["python", "scripts/generate_3d_models.py", f"--num_models={args.total}"], check=True)
    
    # Step 3: Verify dataset
    print("\nStep 3: Verifying dataset...")
    
    # Count images
    image_count = len([f for f in os.listdir("data/images") if f.endswith(('.jpg', '.png', '.jpeg'))])
    print(f"Found {image_count} images in data/images/")
    
    # Count 3D models
    model_count = len([f for f in os.listdir("data/3d_models") if f.endswith('.obj')])
    print(f"Found {model_count} 3D models in data/3d_models/")
    
    # Check split files
    if os.path.exists("data/train.txt"):
        with open("data/train.txt", "r") as f:
            train_count = len(f.readlines())
        print(f"Train split: {train_count} images")
    
    if os.path.exists("data/val.txt"):
        with open("data/val.txt", "r") as f:
            val_count = len(f.readlines())
        print(f"Validation split: {val_count} images")
    
    if os.path.exists("data/test.txt"):
        with open("data/test.txt", "r") as f:
            test_count = len(f.readlines())
        print(f"Test split: {test_count} images")
    
    print("\nDataset creation complete!")
    print(f"Total images: {image_count}")
    print(f"Total 3D models: {model_count}")
    print("\nYou can now use this dataset for training and evaluation.")

if __name__ == "__main__":
    main()
