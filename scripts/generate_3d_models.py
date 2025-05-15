import os
import numpy as np
import trimesh
import random
import argparse
from tqdm import tqdm
import glob

def create_simple_glasses_model(output_path, style="round"):
    """
    Create a simple 3D model of glasses.
    
    Args:
        output_path (str): Path to save the model.
        style (str): Style of glasses ('round', 'square', 'aviator', etc.).
    """
    # Parameters
    lens_radius = random.uniform(0.8, 1.2)
    lens_thickness = random.uniform(0.05, 0.15)
    bridge_width = random.uniform(0.3, 0.5)
    temple_length = random.uniform(3.0, 4.0)
    temple_thickness = random.uniform(0.05, 0.1)
    
    # Create meshes for different parts
    if style == "round":
        # Create round lenses
        left_lens = trimesh.creation.cylinder(
            radius=lens_radius,
            height=lens_thickness,
            sections=32
        )
        right_lens = left_lens.copy()
        
        # Position lenses
        left_lens.apply_translation([-lens_radius - bridge_width/2, 0, 0])
        right_lens.apply_translation([lens_radius + bridge_width/2, 0, 0])
        
    elif style == "square":
        # Create square lenses
        lens_width = lens_radius * 2
        lens_height = lens_radius * 1.8
        left_lens = trimesh.creation.box(
            extents=[lens_width, lens_thickness, lens_height]
        )
        right_lens = left_lens.copy()
        
        # Position lenses
        left_lens.apply_translation([-lens_width/2 - bridge_width/2, 0, 0])
        right_lens.apply_translation([lens_width/2 + bridge_width/2, 0, 0])
        
    elif style == "aviator":
        # Create aviator-style lenses (teardrop shape)
        # Approximate with a cylinder and transform it
        left_lens = trimesh.creation.cylinder(
            radius=lens_radius,
            height=lens_thickness,
            sections=32
        )
        right_lens = left_lens.copy()
        
        # Apply scaling to create teardrop shape
        scale_matrix = np.eye(4)
        scale_matrix[2, 2] = 1.3  # Stretch vertically
        left_lens.apply_transform(scale_matrix)
        right_lens.apply_transform(scale_matrix)
        
        # Position lenses
        left_lens.apply_translation([-lens_radius - bridge_width/2, 0, 0])
        right_lens.apply_translation([lens_radius + bridge_width/2, 0, 0])
    
    else:  # Default to round
        # Create round lenses
        left_lens = trimesh.creation.cylinder(
            radius=lens_radius,
            height=lens_thickness,
            sections=32
        )
        right_lens = left_lens.copy()
        
        # Position lenses
        left_lens.apply_translation([-lens_radius - bridge_width/2, 0, 0])
        right_lens.apply_translation([lens_radius + bridge_width/2, 0, 0])
    
    # Create bridge
    bridge_height = lens_thickness
    bridge = trimesh.creation.box(
        extents=[bridge_width, lens_thickness, bridge_height/2]
    )
    
    # Create temples (arms)
    left_temple = trimesh.creation.box(
        extents=[temple_length, temple_thickness, temple_thickness]
    )
    right_temple = left_temple.copy()
    
    # Position bridge and temples
    bridge.apply_translation([0, 0, 0])
    
    # Position temples at the edges of the lenses
    if style == "round":
        left_temple.apply_translation([-lens_radius*2 - bridge_width/2 - temple_length/2, 0, 0])
        right_temple.apply_translation([lens_radius*2 + bridge_width/2 + temple_length/2, 0, 0])
    elif style == "square":
        left_temple.apply_translation([-lens_width - bridge_width/2 - temple_length/2, 0, 0])
        right_temple.apply_translation([lens_width + bridge_width/2 + temple_length/2, 0, 0])
    elif style == "aviator":
        left_temple.apply_translation([-lens_radius*2 - bridge_width/2 - temple_length/2, 0, 0])
        right_temple.apply_translation([lens_radius*2 + bridge_width/2 + temple_length/2, 0, 0])
    
    # Combine all parts
    glasses = trimesh.util.concatenate([left_lens, right_lens, bridge, left_temple, right_temple])
    
    # Save the model
    glasses.export(output_path)
    
    return glasses

def generate_models_for_dataset(num_models=5000):
    """
    Generate 3D models for all images in the dataset.
    
    Args:
        num_models (int): Number of models to generate.
    """
    # Create output directory if it doesn't exist
    os.makedirs("data/3d_models", exist_ok=True)
    
    # Get list of image files
    image_files = glob.glob("data/images/*.jpg") + glob.glob("data/images/*.png")
    
    # Limit to the specified number
    image_files = image_files[:num_models]
    
    # Styles of glasses
    styles = ["round", "square", "aviator"]
    
    # Generate a model for each image
    for image_file in tqdm(image_files, desc="Generating 3D models"):
        # Get base filename without extension
        base_name = os.path.basename(image_file).split('.')[0]
        
        # Output path for the model
        output_path = f"data/3d_models/{base_name}.obj"
        
        # Skip if model already exists
        if os.path.exists(output_path):
            continue
        
        # Choose a random style
        style = random.choice(styles)
        
        # Generate the model
        create_simple_glasses_model(output_path, style=style)

def main():
    parser = argparse.ArgumentParser(description="Generate 3D models for glasses dataset")
    parser.add_argument("--num_models", type=int, default=5000, help="Number of models to generate")
    args = parser.parse_args()
    
    generate_models_for_dataset(args.num_models)
    
    print(f"Generated {args.num_models} 3D models for the dataset")

if __name__ == "__main__":
    main()
