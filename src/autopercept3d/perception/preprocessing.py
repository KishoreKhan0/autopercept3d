from __future__ import annotations

import numpy as np


def filter_finite_points(points: np.ndarray) -> np.ndarray:
    """
    Keep only points whose x, y, z, intensity values are finite.

    Expected point format: [x, y, z, intensity]
    """
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("Expected points with shape (N, 4) or at least (N, 3).")

    valid_mask = np.isfinite(points).all(axis=1)
    return points[valid_mask]



def crop_roi(
    points: np.ndarray,
    x_range: tuple[float, float] = (0.0, 70.0),
    y_range: tuple[float, float] = (-40.0, 40.0),
    z_range: tuple[float, float] = (-3.0, 3.0),
) -> np.ndarray:
    """
    Crop points to a region of interest (ROI) useful for driving scenes.

    x: forward distance
    y: left/right distance
    z: height
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    mask = (
        (x >= x_range[0]) & (x <= x_range[1]) &
        (y >= y_range[0]) & (y <= y_range[1]) &
        (z >= z_range[0]) & (z <= z_range[1])
    )

    return points[mask]



def voxel_downsample(points: np.ndarray, voxel_size: float = 0.2) -> np.ndarray:
    """
    Very simple voxel downsampling using NumPy only.

    Idea:
    - divide 3D space into small cubes (voxels)
    - keep only one point per voxel

    This reduces the number of points and makes later processing faster.
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")

    if len(points) == 0:
        return points

    xyz = points[:, :3]

    # Map each point to an integer voxel coordinate.
    voxel_indices = np.floor(xyz / voxel_size).astype(np.int32)

    # Keep the first point from each unique voxel.
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)

    return points[unique_indices]
