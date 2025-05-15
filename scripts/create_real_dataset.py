import os
import argparse
import subprocess
import time

def main():
    parser = argparse.ArgumentParser(description="Create real glasses dataset")
    parser.add_argument("--total", type=int, default=5000, help="Total number of images to create")
    parser.add_argument("--kaggle_username", type=str, help="Kaggle username")
    parser.add_argument("--kaggle_key", type=str, help="Kaggle API key")
    args = parser.parse_args()
    
    # Step 1: Set up Kaggle credentials if provided
    if args.kaggle_username and args.kaggle_key:
        print("Step 1: Setting up Kaggle credentials...")
        subprocess.run([
            "python", "scripts/setup_kaggle.py",
            f"--username={args.kaggle_username}",
            f"--key={args.kaggle_key}"
        ], check=True)
    else:
        print("Step 1: Skipping Kaggle setup (no credentials provided)")
    
    # Step 2: Download real dataset
    print("\nStep 2: Downloading real glasses dataset...")
    subprocess.run([
        "python", "scripts/download_real_dataset.py",
        f"--num_images={args.total}"
    ], check=True)
    
    # Step 3: Generate 3D models
    print("\nStep 3: Generating 3D models...")
    subprocess.run([
        "python", "scripts/generate_3d_models.py",
        f"--num_models={args.total}"
    ], check=True)
    
    # Step 4: Verify dataset
    print("\nStep 4: Verifying dataset...")
    
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
