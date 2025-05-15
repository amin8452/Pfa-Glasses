import os
import requests
import argparse
import zipfile
import io
import shutil
from tqdm import tqdm
import random

def download_file(url, destination):
    """
    Download a file from a URL to a destination.
    
    Args:
        url (str): URL to download from.
        destination (str): Path to save the file.
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {os.path.basename(destination)}") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def download_and_extract_kaggle_dataset(dataset_name, output_dir):
    """
    Download and extract a dataset from Kaggle.
    
    Args:
        dataset_name (str): Name of the Kaggle dataset.
        output_dir (str): Directory to extract the dataset to.
    """
    # Note: This requires Kaggle API credentials to be set up
    try:
        import kaggle
        print(f"Downloading Kaggle dataset: {dataset_name}")
        kaggle.api.dataset_download_files(dataset_name, path=output_dir, unzip=True)
        print(f"Dataset downloaded and extracted to {output_dir}")
        return True
    except Exception as e:
        print(f"Error downloading Kaggle dataset: {e}")
        print("Make sure you have set up Kaggle API credentials.")
        return False

def download_glasses_dataset(output_dir, num_images=5000):
    """
    Download a glasses dataset from a public source.
    
    Args:
        output_dir (str): Directory to save the dataset.
        num_images (int): Number of images to download.
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Try to download from Kaggle first
    kaggle_datasets = [
        "luxoptical/glasses-dataset",
        "anirudhsharma/eyeglasses-dataset",
        "tapakah68/glasses-and-headwear"
    ]
    
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    total_images = 0
    
    for dataset in kaggle_datasets:
        if total_images >= num_images:
            break
            
        if download_and_extract_kaggle_dataset(dataset, temp_dir):
            # Find all image files in the extracted dataset
            image_extensions = ['.jpg', '.jpeg', '.png']
            image_files = []
            
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_files.append(os.path.join(root, file))
            
            # Copy images to the output directory
            print(f"Found {len(image_files)} images in dataset {dataset}")
            
            # Shuffle the images to get a random subset
            random.shuffle(image_files)
            
            # Copy images
            remaining = num_images - total_images
            for i, image_file in enumerate(image_files[:remaining]):
                dest_file = os.path.join(images_dir, f"glasses_{total_images + i:05d}.jpg")
                shutil.copy(image_file, dest_file)
                
            total_images += min(len(image_files), remaining)
            print(f"Copied {min(len(image_files), remaining)} images. Total: {total_images}/{num_images}")
    
    # Clean up temporary directory
    shutil.rmtree(temp_dir)
    
    # If we still don't have enough images, try to download from other sources
    if total_images < num_images:
        print(f"Only found {total_images} images from Kaggle datasets.")
        print(f"Downloading {num_images - total_images} more images from other sources...")
        
        # URLs of public glasses datasets
        dataset_urls = [
            "https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset/archive/refs/heads/master.zip",
            "https://github.com/prajnasb/observations/archive/refs/heads/master.zip"
        ]
        
        for url in dataset_urls:
            if total_images >= num_images:
                break
                
            try:
                # Download dataset
                print(f"Downloading dataset from {url}")
                response = requests.get(url)
                
                # Extract dataset
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    # Create temporary directory
                    temp_dir = os.path.join(output_dir, "temp")
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    # Extract all files
                    z.extractall(temp_dir)
                    
                    # Find all image files in the extracted dataset
                    image_extensions = ['.jpg', '.jpeg', '.png']
                    image_files = []
                    
                    for root, _, files in os.walk(temp_dir):
                        for file in files:
                            if any(file.lower().endswith(ext) for ext in image_extensions):
                                image_files.append(os.path.join(root, file))
                    
                    # Copy images to the output directory
                    print(f"Found {len(image_files)} images in dataset {url}")
                    
                    # Shuffle the images to get a random subset
                    random.shuffle(image_files)
                    
                    # Copy images
                    remaining = num_images - total_images
                    for i, image_file in enumerate(image_files[:remaining]):
                        dest_file = os.path.join(images_dir, f"glasses_{total_images + i:05d}.jpg")
                        shutil.copy(image_file, dest_file)
                        
                    total_images += min(len(image_files), remaining)
                    print(f"Copied {min(len(image_files), remaining)} images. Total: {total_images}/{num_images}")
                    
                    # Clean up temporary directory
                    shutil.rmtree(temp_dir)
            
            except Exception as e:
                print(f"Error downloading dataset from {url}: {e}")
    
    # If we still don't have enough images, use the synthetic generator as a fallback
    if total_images < num_images:
        print(f"Only found {total_images} real images.")
        print(f"Generating {num_images - total_images} synthetic images as fallback...")
        
        # Import the synthetic image generator
        from download_dataset import generate_synthetic_images
        
        # Generate synthetic images
        generate_synthetic_images(num_images - total_images, total_images)
        
        total_images = num_images
    
    print(f"Dataset creation complete. Total images: {total_images}")
    
    # Create train, val, test splits
    create_splits(output_dir, total_images)

def create_splits(data_dir, total_images, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Create train, val, test splits.
    
    Args:
        data_dir (str): Directory containing the dataset.
        total_images (int): Total number of images.
        train_ratio (float): Ratio of images for training.
        val_ratio (float): Ratio of images for validation.
        test_ratio (float): Ratio of images for testing.
    """
    # Get all image files
    image_files = []
    images_dir = os.path.join(data_dir, "images")
    
    for file in os.listdir(images_dir):
        if file.endswith(('.jpg', '.jpeg', '.png')):
            image_files.append(file)
    
    # Shuffle the files
    random.shuffle(image_files)
    
    # Calculate split sizes
    train_size = int(total_images * train_ratio)
    val_size = int(total_images * val_ratio)
    test_size = total_images - train_size - val_size
    
    # Create splits
    train_files = image_files[:train_size]
    val_files = image_files[train_size:train_size + val_size]
    test_files = image_files[train_size + val_size:train_size + val_size + test_size]
    
    # Write splits to files
    with open(os.path.join(data_dir, "train.txt"), "w") as f:
        f.write("\n".join(train_files))
    
    with open(os.path.join(data_dir, "val.txt"), "w") as f:
        f.write("\n".join(val_files))
    
    with open(os.path.join(data_dir, "test.txt"), "w") as f:
        f.write("\n".join(test_files))
    
    print(f"Created splits: train ({len(train_files)}), val ({len(val_files)}), test ({len(test_files)})")

def main():
    parser = argparse.ArgumentParser(description="Download real glasses dataset")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save the dataset")
    parser.add_argument("--num_images", type=int, default=5000, help="Number of images to download")
    args = parser.parse_args()
    
    download_glasses_dataset(args.output_dir, args.num_images)

if __name__ == "__main__":
    main()
