import torch
import numpy as np
from typing import Tuple, Union, Optional

def chamfer_distance(
    x: Union[torch.Tensor, np.ndarray],
    y: Union[torch.Tensor, np.ndarray],
    bidirectional: bool = True,
    reduction: str = 'mean'
) -> Union[torch.Tensor, float]:
    """
    Compute the Chamfer distance between two point clouds.
    
    Args:
        x (torch.Tensor or np.ndarray): First point cloud of shape (B, N, 3) or (N, 3).
        y (torch.Tensor or np.ndarray): Second point cloud of shape (B, M, 3) or (M, 3).
        bidirectional (bool): If True, compute bidirectional Chamfer distance.
        reduction (str): Reduction method ('mean', 'sum', or 'none').
        
    Returns:
        torch.Tensor or float: Chamfer distance.
    """
    # Convert numpy arrays to torch tensors if necessary
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()
    
    # Add batch dimension if not present
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if y.dim() == 2:
        y = y.unsqueeze(0)
    
    # Check shapes
    assert x.dim() == 3 and y.dim() == 3, "Input point clouds must be of shape (B, N, 3) or (N, 3)"
    assert x.size(2) == 3 and y.size(2) == 3, "Point clouds must have 3D points"
    
    # Compute pairwise distances
    x_expanded = x.unsqueeze(2)  # (B, N, 1, 3)
    y_expanded = y.unsqueeze(1)  # (B, 1, M, 3)
    
    # Compute squared distances
    dist = torch.sum((x_expanded - y_expanded) ** 2, dim=3)  # (B, N, M)
    
    # Compute minimum distances
    x_to_y = torch.min(dist, dim=2)[0]  # (B, N)
    y_to_x = torch.min(dist, dim=1)[0]  # (B, M)
    
    # Compute Chamfer distance
    if bidirectional:
        # Bidirectional Chamfer distance
        if reduction == 'mean':
            chamfer = torch.mean(x_to_y, dim=1) + torch.mean(y_to_x, dim=1)  # (B,)
        elif reduction == 'sum':
            chamfer = torch.sum(x_to_y, dim=1) + torch.sum(y_to_x, dim=1)  # (B,)
        else:  # 'none'
            chamfer = torch.cat([x_to_y, y_to_x], dim=1)  # (B, N+M)
    else:
        # Unidirectional Chamfer distance (x to y)
        if reduction == 'mean':
            chamfer = torch.mean(x_to_y, dim=1)  # (B,)
        elif reduction == 'sum':
            chamfer = torch.sum(x_to_y, dim=1)  # (B,)
        else:  # 'none'
            chamfer = x_to_y  # (B, N)
    
    # Return mean over batch if needed
    if reduction == 'mean' and chamfer.dim() > 0:
        chamfer = torch.mean(chamfer)
    elif reduction == 'sum' and chamfer.dim() > 0:
        chamfer = torch.sum(chamfer)
    
    return chamfer
