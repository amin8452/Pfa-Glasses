import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
import torch
from typing import List, Tuple, Union, Optional
import os

def visualize_mesh(
    mesh: Union[trimesh.Trimesh, str],
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 10),
    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> None:
    """
    Visualize a 3D mesh.
    
    Args:
        mesh (trimesh.Trimesh or str): Mesh to visualize or path to mesh file.
        save_path (str, optional): Path to save the visualization.
        show (bool): Whether to display the visualization.
        figsize (Tuple[int, int]): Figure size.
        background_color (Tuple[float, float, float]): Background color.
    """
    # Load mesh if path is provided
    if isinstance(mesh, str):
        mesh = trimesh.load(mesh)
    
    # Create a scene with the mesh
    scene = trimesh.Scene(mesh)
    
    # Set background color
    scene.set_camera(background_color=background_color)
    
    # Render the scene
    png = scene.save_image(resolution=(figsize[0] * 100, figsize[1] * 100))
    
    # Convert to numpy array
    img = plt.imread(png)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.axis('off')
    
    # Save if path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()

def visualize_point_cloud(
    points: Union[np.ndarray, torch.Tensor, str],
    colors: Optional[Union[np.ndarray, torch.Tensor]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 10),
    point_size: float = 0.1,
    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> None:
    """
    Visualize a 3D point cloud.
    
    Args:
        points (np.ndarray or torch.Tensor or str): Points to visualize or path to point cloud file.
        colors (np.ndarray or torch.Tensor, optional): Colors for each point.
        save_path (str, optional): Path to save the visualization.
        show (bool): Whether to display the visualization.
        figsize (Tuple[int, int]): Figure size.
        point_size (float): Size of the points.
        background_color (Tuple[float, float, float]): Background color.
    """
    # Load point cloud if path is provided
    if isinstance(points, str):
        points = np.loadtxt(points)
    
    # Convert torch tensor to numpy array if necessary
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    
    if isinstance(colors, torch.Tensor):
        colors = colors.detach().cpu().numpy()
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Set background color
    ax.set_facecolor(background_color)
    
    # Plot points
    if colors is not None:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=point_size)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=point_size)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Save if path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()

def visualize_comparison(
    meshes: List[Union[trimesh.Trimesh, str]],
    titles: List[str],
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (15, 5),
    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> None:
    """
    Visualize a comparison of multiple 3D meshes.
    
    Args:
        meshes (List[trimesh.Trimesh or str]): Meshes to visualize or paths to mesh files.
        titles (List[str]): Titles for each mesh.
        save_path (str, optional): Path to save the visualization.
        show (bool): Whether to display the visualization.
        figsize (Tuple[int, int]): Figure size.
        background_color (Tuple[float, float, float]): Background color.
    """
    # Check inputs
    assert len(meshes) == len(titles), "Number of meshes and titles must match"
    
    # Create temporary directory for images
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    # Visualize each mesh and save to temporary file
    temp_paths = []
    for i, mesh in enumerate(meshes):
        temp_path = os.path.join(temp_dir, f"mesh_{i}.png")
        visualize_mesh(mesh, save_path=temp_path, show=False, background_color=background_color)
        temp_paths.append(temp_path)
    
    # Create figure for comparison
    fig, axes = plt.subplots(1, len(meshes), figsize=figsize)
    
    # Handle case with only one mesh
    if len(meshes) == 1:
        axes = [axes]
    
    # Plot each mesh
    for i, (temp_path, title) in enumerate(zip(temp_paths, titles)):
        img = plt.imread(temp_path)
        axes[i].imshow(img)
        axes[i].set_title(title)
        axes[i].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    # Clean up temporary files
    for temp_path in temp_paths:
        os.remove(temp_path)
    os.rmdir(temp_dir)
