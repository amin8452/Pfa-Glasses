import torch
import numpy as np
from typing import Tuple, Union, Optional
from scipy.optimize import linear_sum_assignment

def earth_movers_distance(
    x: Union[torch.Tensor, np.ndarray],
    y: Union[torch.Tensor, np.ndarray],
    reduction: str = 'mean'
) -> Union[torch.Tensor, float]:
    """
    Compute the Earth Mover's Distance (EMD) between two point clouds.
    
    Args:
        x (torch.Tensor or np.ndarray): First point cloud of shape (B, N, 3) or (N, 3).
        y (torch.Tensor or np.ndarray): Second point cloud of shape (B, N, 3) or (N, 3).
        reduction (str): Reduction method ('mean', 'sum', or 'none').
        
    Returns:
        torch.Tensor or float: Earth Mover's Distance.
    """
    # Convert torch tensors to numpy arrays if necessary
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    
    # Add batch dimension if not present
    if x.ndim == 2:
        x = np.expand_dims(x, 0)
    if y.ndim == 2:
        y = np.expand_dims(y, 0)
    
    # Check shapes
    assert x.ndim == 3 and y.ndim == 3, "Input point clouds must be of shape (B, N, 3) or (N, 3)"
    assert x.shape[2] == 3 and y.shape[2] == 3, "Point clouds must have 3D points"
    
    # EMD requires equal number of points
    if x.shape[1] != y.shape[1]:
        # Subsample the larger point cloud
        if x.shape[1] > y.shape[1]:
            indices = np.random.choice(x.shape[1], y.shape[1], replace=False)
            x = x[:, indices, :]
        else:
            indices = np.random.choice(y.shape[1], x.shape[1], replace=False)
            y = y[:, indices, :]
    
    batch_size = x.shape[0]
    num_points = x.shape[1]
    
    # Compute EMD for each batch
    emd_batch = np.zeros(batch_size)
    
    for b in range(batch_size):
        # Compute pairwise distances
        dist_matrix = np.zeros((num_points, num_points))
        for i in range(num_points):
            for j in range(num_points):
                dist_matrix[i, j] = np.sum((x[b, i] - y[b, j]) ** 2)
        
        # Solve the assignment problem
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        
        # Compute EMD
        emd_batch[b] = np.sum(dist_matrix[row_ind, col_ind]) / num_points
    
    # Apply reduction
    if reduction == 'mean':
        emd = np.mean(emd_batch)
    elif reduction == 'sum':
        emd = np.sum(emd_batch)
    else:  # 'none'
        emd = emd_batch
    
    # Convert back to torch tensor if inputs were torch tensors
    if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
        emd = torch.from_numpy(emd).float()
    
    return emd
