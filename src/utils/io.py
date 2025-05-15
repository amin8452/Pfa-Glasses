import os
import numpy as np
import trimesh
import torch
from typing import Union, Optional, Tuple, List, Dict

def load_mesh(
    path: str,
    return_type: str = 'trimesh'
) -> Union[trimesh.Trimesh, Dict[str, torch.Tensor]]:
    """
    Load a 3D mesh from a file.
    
    Args:
        path (str): Path to the mesh file.
        return_type (str): Type of object to return ('trimesh' or 'tensors').
        
    Returns:
        trimesh.Trimesh or Dict[str, torch.Tensor]: Loaded mesh.
    """
    # Check if file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mesh file not found: {path}")
    
    # Load mesh using trimesh
    mesh = trimesh.load(path)
    
    # Return as trimesh object
    if return_type == 'trimesh':
        return mesh
    
    # Convert to tensors
    elif return_type == 'tensors':
        vertices = torch.from_numpy(mesh.vertices).float()
        faces = torch.from_numpy(mesh.faces).long()
        
        # Get texture coordinates if available
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv'):
            uv = torch.from_numpy(mesh.visual.uv).float()
        else:
            uv = None
        
        # Get texture image if available
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'):
            if hasattr(mesh.visual.material, 'image'):
                texture = torch.from_numpy(np.array(mesh.visual.material.image)).float() / 255.0
            else:
                texture = None
        else:
            texture = None
        
        return {
            'vertices': vertices,
            'faces': faces,
            'uv': uv,
            'texture': texture
        }
    
    else:
        raise ValueError(f"Invalid return_type: {return_type}. Must be 'trimesh' or 'tensors'.")

def save_mesh(
    mesh: Union[trimesh.Trimesh, Dict[str, torch.Tensor]],
    path: str,
    file_type: Optional[str] = None
) -> None:
    """
    Save a 3D mesh to a file.
    
    Args:
        mesh (trimesh.Trimesh or Dict[str, torch.Tensor]): Mesh to save.
        path (str): Path to save the mesh.
        file_type (str, optional): File type to save as. If None, inferred from path.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    # If mesh is a dictionary of tensors, convert to trimesh
    if isinstance(mesh, dict):
        vertices = mesh['vertices'].detach().cpu().numpy()
        faces = mesh['faces'].detach().cpu().numpy()
        
        # Create trimesh object
        trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Add texture if available
        if 'uv' in mesh and mesh['uv'] is not None and 'texture' in mesh and mesh['texture'] is not None:
            uv = mesh['uv'].detach().cpu().numpy()
            texture = (mesh['texture'].detach().cpu().numpy() * 255).astype(np.uint8)
            
            # Create material
            material = trimesh.visual.material.SimpleMaterial(image=texture)
            
            # Create texture visual
            visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
            
            # Set visual
            trimesh_mesh.visual = visual
    else:
        trimesh_mesh = mesh
    
    # Save mesh
    trimesh_mesh.export(path, file_type=file_type)

def load_point_cloud(
    path: str,
    return_type: str = 'numpy'
) -> Union[np.ndarray, torch.Tensor]:
    """
    Load a 3D point cloud from a file.
    
    Args:
        path (str): Path to the point cloud file.
        return_type (str): Type of object to return ('numpy' or 'tensor').
        
    Returns:
        np.ndarray or torch.Tensor: Loaded point cloud.
    """
    # Check if file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Point cloud file not found: {path}")
    
    # Load point cloud
    points = np.loadtxt(path)
    
    # Return as numpy array
    if return_type == 'numpy':
        return points
    
    # Convert to tensor
    elif return_type == 'tensor':
        return torch.from_numpy(points).float()
    
    else:
        raise ValueError(f"Invalid return_type: {return_type}. Must be 'numpy' or 'tensor'.")

def save_point_cloud(
    points: Union[np.ndarray, torch.Tensor],
    path: str
) -> None:
    """
    Save a 3D point cloud to a file.
    
    Args:
        points (np.ndarray or torch.Tensor): Point cloud to save.
        path (str): Path to save the point cloud.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    # Convert torch tensor to numpy array if necessary
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    
    # Save point cloud
    np.savetxt(path, points)
