import torch
import numpy as np
import trimesh
from typing import Tuple, Union, Optional, List

def iou_3d(
    mesh1: Union[trimesh.Trimesh, str],
    mesh2: Union[trimesh.Trimesh, str],
    resolution: int = 64,
    return_voxels: bool = False
) -> Union[float, Tuple[float, np.ndarray, np.ndarray]]:
    """
    Compute the 3D Intersection over Union (IoU) between two meshes.
    
    Args:
        mesh1 (trimesh.Trimesh or str): First mesh or path to mesh file.
        mesh2 (trimesh.Trimesh or str): Second mesh or path to mesh file.
        resolution (int): Resolution of the voxel grid.
        return_voxels (bool): If True, return the voxelized meshes.
        
    Returns:
        float or Tuple[float, np.ndarray, np.ndarray]: IoU value, and optionally the voxelized meshes.
    """
    # Load meshes if paths are provided
    if isinstance(mesh1, str):
        mesh1 = trimesh.load(mesh1)
    if isinstance(mesh2, str):
        mesh2 = trimesh.load(mesh2)
    
    # Ensure meshes are watertight
    mesh1 = ensure_watertight(mesh1)
    mesh2 = ensure_watertight(mesh2)
    
    # Normalize meshes to have the same bounding box
    mesh1, mesh2 = normalize_meshes(mesh1, mesh2)
    
    # Voxelize meshes
    voxels1 = voxelize_mesh(mesh1, resolution)
    voxels2 = voxelize_mesh(mesh2, resolution)
    
    # Compute IoU
    intersection = np.logical_and(voxels1, voxels2).sum()
    union = np.logical_or(voxels1, voxels2).sum()
    
    iou = intersection / union if union > 0 else 0.0
    
    if return_voxels:
        return iou, voxels1, voxels2
    else:
        return iou

def ensure_watertight(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Ensure that a mesh is watertight (closed).
    
    Args:
        mesh (trimesh.Trimesh): Input mesh.
        
    Returns:
        trimesh.Trimesh: Watertight mesh.
    """
    # Check if mesh is already watertight
    if mesh.is_watertight:
        return mesh
    
    # Try to fix the mesh
    mesh.fill_holes()
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_infinite_values()
    mesh.fix_normals()
    
    # If still not watertight, try to create a convex hull
    if not mesh.is_watertight:
        try:
            mesh = mesh.convex_hull
        except:
            # If convex hull fails, just return the original mesh
            pass
    
    return mesh

def normalize_meshes(mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """
    Normalize meshes to have the same bounding box.
    
    Args:
        mesh1 (trimesh.Trimesh): First mesh.
        mesh2 (trimesh.Trimesh): Second mesh.
        
    Returns:
        Tuple[trimesh.Trimesh, trimesh.Trimesh]: Normalized meshes.
    """
    # Compute bounding boxes
    bbox1 = mesh1.bounding_box.bounds
    bbox2 = mesh2.bounding_box.bounds
    
    # Compute centers
    center1 = (bbox1[0] + bbox1[1]) / 2
    center2 = (bbox2[0] + bbox2[1]) / 2
    
    # Compute scales
    scale1 = np.max(bbox1[1] - bbox1[0])
    scale2 = np.max(bbox2[1] - bbox2[0])
    
    # Create normalized copies
    mesh1_norm = mesh1.copy()
    mesh2_norm = mesh2.copy()
    
    # Translate to origin
    mesh1_norm.apply_translation(-center1)
    mesh2_norm.apply_translation(-center2)
    
    # Scale to unit size
    mesh1_norm.apply_scale(1.0 / scale1)
    mesh2_norm.apply_scale(1.0 / scale2)
    
    return mesh1_norm, mesh2_norm

def voxelize_mesh(mesh: trimesh.Trimesh, resolution: int) -> np.ndarray:
    """
    Voxelize a mesh into a binary grid.
    
    Args:
        mesh (trimesh.Trimesh): Input mesh.
        resolution (int): Resolution of the voxel grid.
        
    Returns:
        np.ndarray: Binary voxel grid.
    """
    # Create a voxel grid
    voxels = np.zeros((resolution, resolution, resolution), dtype=bool)
    
    # Get mesh bounds
    bounds = mesh.bounds
    min_bound = bounds[0]
    max_bound = bounds[1]
    
    # Create grid points
    x = np.linspace(min_bound[0], max_bound[0], resolution)
    y = np.linspace(min_bound[1], max_bound[1], resolution)
    z = np.linspace(min_bound[2], max_bound[2], resolution)
    
    # Create meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
    
    # Check which points are inside the mesh
    inside = mesh.contains(points)
    
    # Reshape to voxel grid
    voxels = inside.reshape((resolution, resolution, resolution))
    
    return voxels
