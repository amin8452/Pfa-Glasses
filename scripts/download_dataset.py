import os
import requests
import time
import random
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from io import BytesIO
import numpy as np

# Create directories if they don't exist
def create_directories():
    os.makedirs("data/images", exist_ok=True)
    os.makedirs("data/3d_models", exist_ok=True)
    print("Created directories: data/images, data/3d_models")

# Function to download an image from a URL
def download_image(url, save_path):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img = img.convert('RGB')  # Convert to RGB format
            img.save(save_path)
            return True
        return False
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

# Function to download images from Unsplash API
def download_unsplash_images(query, count, start_idx=0):
    # Unsplash API credentials (you need to register for an API key)
    # This is a placeholder - you should replace with your own API key
    api_key = "YOUR_UNSPLASH_API_KEY"
    
    # API endpoint
    endpoint = f"https://api.unsplash.com/search/photos?query={query}&per_page=30&client_id={api_key}"
    
    # Download images
    successful_downloads = 0
    page = 1
    
    with tqdm(total=count, desc=f"Downloading {query} images") as pbar:
        while successful_downloads < count:
            try:
                # Get images from API
                response = requests.get(f"{endpoint}&page={page}")
                if response.status_code != 200:
                    print(f"Error: API returned status code {response.status_code}")
                    break
                
                data = response.json()
                if not data.get('results'):
                    print("No more results available")
                    break
                
                # Download each image
                for img_data in data['results']:
                    if successful_downloads >= count:
                        break
                    
                    img_url = img_data['urls']['regular']
                    img_id = img_data['id']
                    save_path = f"data/images/{query}_{start_idx + successful_downloads:05d}.jpg"
                    
                    if download_image(img_url, save_path):
                        successful_downloads += 1
                        pbar.update(1)
                
                page += 1
                time.sleep(1)  # Respect API rate limits
                
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(5)
    
    return successful_downloads

# Function to download images from Pexels API
def download_pexels_images(query, count, start_idx=0):
    # Pexels API credentials (you need to register for an API key)
    # This is a placeholder - you should replace with your own API key
    api_key = "YOUR_PEXELS_API_KEY"
    
    # API endpoint
    endpoint = f"https://api.pexels.com/v1/search?query={query}&per_page=80"
    headers = {"Authorization": api_key}
    
    # Download images
    successful_downloads = 0
    page = 1
    
    with tqdm(total=count, desc=f"Downloading {query} images") as pbar:
        while successful_downloads < count:
            try:
                # Get images from API
                response = requests.get(f"{endpoint}&page={page}", headers=headers)
                if response.status_code != 200:
                    print(f"Error: API returned status code {response.status_code}")
                    break
                
                data = response.json()
                if not data.get('photos'):
                    print("No more results available")
                    break
                
                # Download each image
                for img_data in data['photos']:
                    if successful_downloads >= count:
                        break
                    
                    img_url = img_data['src']['large']
                    save_path = f"data/images/{query}_{start_idx + successful_downloads:05d}.jpg"
                    
                    if download_image(img_url, save_path):
                        successful_downloads += 1
                        pbar.update(1)
                
                page += 1
                time.sleep(1)  # Respect API rate limits
                
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(5)
    
    return successful_downloads

# Function to generate synthetic images (as a fallback)
def generate_synthetic_images(count, start_idx=0):
    from PIL import Image, ImageDraw
    
    glasses_types = ["round", "square", "oval", "rectangular", "cat_eye", "aviator", "wayfarer", "rimless"]
    colors = ["black", "brown", "blue", "red", "gold", "silver", "green", "purple"]
    
    with tqdm(total=count, desc="Generating synthetic glasses images") as pbar:
        for i in range(count):
            # Create a blank image
            img = Image.new('RGB', (512, 512), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            
            # Choose random glasses type and color
            glasses_type = random.choice(glasses_types)
            color = random.choice(colors)
            
            # Draw a simple glasses shape
            if glasses_type in ["round", "oval"]:
                # Draw round/oval glasses
                left_eye_x, left_eye_y = 150, 256
                right_eye_x, right_eye_y = 362, 256
                eye_size = random.randint(60, 80)
                
                # Convert color name to RGB
                rgb_color = {
                    "black": (0, 0, 0),
                    "brown": (139, 69, 19),
                    "blue": (0, 0, 255),
                    "red": (255, 0, 0),
                    "gold": (255, 215, 0),
                    "silver": (192, 192, 192),
                    "green": (0, 128, 0),
                    "purple": (128, 0, 128)
                }.get(color, (0, 0, 0))
                
                # Draw left lens
                draw.ellipse(
                    [(left_eye_x - eye_size, left_eye_y - eye_size), 
                     (left_eye_x + eye_size, left_eye_y + eye_size)], 
                    outline=rgb_color, width=5
                )
                
                # Draw right lens
                draw.ellipse(
                    [(right_eye_x - eye_size, right_eye_y - eye_size), 
                     (right_eye_x + eye_size, right_eye_y + eye_size)], 
                    outline=rgb_color, width=5
                )
                
                # Draw bridge
                draw.line(
                    [(left_eye_x + eye_size, left_eye_y), 
                     (right_eye_x - eye_size, right_eye_y)], 
                    fill=rgb_color, width=5
                )
                
                # Draw temples (arms)
                temple_length = random.randint(100, 150)
                draw.line(
                    [(left_eye_x - eye_size, left_eye_y), 
                     (left_eye_x - eye_size - temple_length, left_eye_y + temple_length // 2)], 
                    fill=rgb_color, width=5
                )
                draw.line(
                    [(right_eye_x + eye_size, right_eye_y), 
                     (right_eye_x + eye_size + temple_length, right_eye_y + temple_length // 2)], 
                    fill=rgb_color, width=5
                )
            
            else:  # rectangular, square, etc.
                # Draw rectangular/square glasses
                left_eye_x, left_eye_y = 150, 256
                right_eye_x, right_eye_y = 362, 256
                eye_width = random.randint(70, 90)
                eye_height = random.randint(50, 70)
                
                # Convert color name to RGB
                rgb_color = {
                    "black": (0, 0, 0),
                    "brown": (139, 69, 19),
                    "blue": (0, 0, 255),
                    "red": (255, 0, 0),
                    "gold": (255, 215, 0),
                    "silver": (192, 192, 192),
                    "green": (0, 128, 0),
                    "purple": (128, 0, 128)
                }.get(color, (0, 0, 0))
                
                # Draw left lens
                draw.rectangle(
                    [(left_eye_x - eye_width, left_eye_y - eye_height), 
                     (left_eye_x + eye_width, left_eye_y + eye_height)], 
                    outline=rgb_color, width=5
                )
                
                # Draw right lens
                draw.rectangle(
                    [(right_eye_x - eye_width, right_eye_y - eye_height), 
                     (right_eye_x + eye_width, right_eye_y + eye_height)], 
                    outline=rgb_color, width=5
                )
                
                # Draw bridge
                draw.line(
                    [(left_eye_x + eye_width, left_eye_y), 
                     (right_eye_x - eye_width, right_eye_y)], 
                    fill=rgb_color, width=5
                )
                
                # Draw temples (arms)
                temple_length = random.randint(100, 150)
                draw.line(
                    [(left_eye_x - eye_width, left_eye_y), 
                     (left_eye_x - eye_width - temple_length, left_eye_y + temple_length // 2)], 
                    fill=rgb_color, width=5
                )
                draw.line(
                    [(right_eye_x + eye_width, right_eye_y), 
                     (right_eye_x + eye_width + temple_length, right_eye_y + temple_length // 2)], 
                    fill=rgb_color, width=5
                )
            
            # Save the image
            save_path = f"data/images/synthetic_glasses_{start_idx + i:05d}.jpg"
            img.save(save_path)
            pbar.update(1)
    
    return count

# Create train, val, test splits
def create_splits(total_images, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # Get all image files
    image_files = [f for f in os.listdir("data/images") if f.endswith(('.jpg', '.png', '.jpeg'))]
    
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
    with open("data/train.txt", "w") as f:
        f.write("\n".join(train_files))
    
    with open("data/val.txt", "w") as f:
        f.write("\n".join(val_files))
    
    with open("data/test.txt", "w") as f:
        f.write("\n".join(test_files))
    
    print(f"Created splits: train ({len(train_files)}), val ({len(val_files)}), test ({len(test_files)})")

def main():
    parser = argparse.ArgumentParser(description="Download glasses dataset")
    parser.add_argument("--total", type=int, default=5000, help="Total number of images to download")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic images instead of downloading")
    args = parser.parse_args()
    
    # Create directories
    create_directories()
    
    # Download or generate images
    if args.synthetic:
        # Generate synthetic images
        generate_synthetic_images(args.total)
    else:
        # Try to download from multiple sources
        queries = ["glasses", "eyeglasses", "sunglasses", "spectacles", "eyewear"]
        images_per_query = args.total // len(queries)
        
        total_downloaded = 0
        for i, query in enumerate(queries):
            count = images_per_query
            if i == len(queries) - 1:
                # Make sure we get exactly the total number of images
                count = args.total - total_downloaded
            
            # Try Unsplash first
            downloaded = download_unsplash_images(query, count, total_downloaded)
            total_downloaded += downloaded
            
            # If we didn't get enough, try Pexels
            if downloaded < count:
                remaining = count - downloaded
                downloaded = download_pexels_images(query, remaining, total_downloaded)
                total_downloaded += downloaded
        
        # If we still don't have enough images, generate synthetic ones
        if total_downloaded < args.total:
            remaining = args.total - total_downloaded
            print(f"Downloaded only {total_downloaded} images. Generating {remaining} synthetic images...")
            generate_synthetic_images(remaining, total_downloaded)
            total_downloaded += remaining
    
    # Create train, val, test splits
    create_splits(args.total)
    
    print(f"Dataset creation complete. Total images: {args.total}")

if __name__ == "__main__":
    main()
