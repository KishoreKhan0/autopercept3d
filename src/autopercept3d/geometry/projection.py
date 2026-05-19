from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from autopercept3d.geometry.kitti_transforms import (
    projection_matrix,
    transform_points,
    velo_to_rect_matrix,
)


@dataclass
class ProjectionResult:
    """
    Result of projecting LiDAR points into the camera image.

    pixels:
        Nx2 image coordinates [u, v].

    depths:
        N depth values in rectified camera coordinates.

    points_velodyne:
        Original Velodyne/LiDAR points that were successfully projected.

    points_rect_camera:
        Corresponding points in rectified camera coordinates.
    """

    pixels: np.ndarray
    depths: np.ndarray
    points_velodyne: np.ndarray
    points_rect_camera: np.ndarray


def project_velodyne_to_camera_image(
    points: np.ndarray,
    calibration: dict[str, np.ndarray],
    image_shape: tuple[int, int] | tuple[int, int, int],
    camera: str = "P2",
    min_depth: float = 0.1,
) -> ProjectionResult:
    """
    Project KITTI Velodyne/LiDAR points into a camera image.

    KITTI coordinate flow:
        Velodyne/LiDAR coordinates
            -> rectified camera coordinates
            -> image pixel coordinates

    For KITTI object detection:
        P2 is the left color camera projection matrix.

    Args:
        points:
            LiDAR points with shape (N, 4) or (N, 3).
            Only x, y, z are used.

        calibration:
            Dictionary loaded from KITTI calib file.

        image_shape:
            Image shape from NumPy image array.
            Example: (height, width, 3)

        camera:
            KITTI projection matrix key. Use "P2" for left color image.

        min_depth:
            Remove points too close or behind the camera.

    Returns:
        ProjectionResult containing only points that land inside the image.
    """
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("Expected points with shape (N, 4) or at least (N, 3).")

    if camera not in calibration:
        raise KeyError(f"Calibration does not contain camera projection matrix: {camera}")

    height = image_shape[0]
    width = image_shape[1]

    points_xyz = points[:, :3]

    velo_to_rect = velo_to_rect_matrix(calibration)
    points_rect = transform_points(points_xyz, velo_to_rect)

    # Keep only points in front of the camera.
    depth = points_rect[:, 2]
    front_mask = depth > min_depth

    points_rect_front = points_rect[front_mask]
    points_velo_front = points[front_mask]
    depth_front = depth[front_mask]

    if len(points_rect_front) == 0:
        empty_pixels = np.empty((0, 2), dtype=np.float64)
        return ProjectionResult(
            pixels=empty_pixels,
            depths=np.empty((0,), dtype=np.float64),
            points_velodyne=points_velo_front,
            points_rect_camera=points_rect_front,
        )

    p = projection_matrix(calibration[camera])

    ones = np.ones((points_rect_front.shape[0], 1), dtype=np.float64)
    points_rect_h = np.hstack([points_rect_front, ones])

    projected = (p @ points_rect_h.T).T

    u = projected[:, 0] / projected[:, 2]
    v = projected[:, 1] / projected[:, 2]

    image_mask = (
        (u >= 0.0)
        & (u < width)
        & (v >= 0.0)
        & (v < height)
        & np.isfinite(u)
        & np.isfinite(v)
    )

    pixels = np.stack([u[image_mask], v[image_mask]], axis=1)

    return ProjectionResult(
        pixels=pixels,
        depths=depth_front[image_mask],
        points_velodyne=points_velo_front[image_mask],
        points_rect_camera=points_rect_front[image_mask],
    )


def depth_to_normalized_values(depths: np.ndarray) -> np.ndarray:
    """
    Normalize depth values to [0, 1] for coloring.
    """
    if len(depths) == 0:
        return depths

    depth_min = np.percentile(depths, 2)
    depth_max = np.percentile(depths, 98)

    normalized = (depths - depth_min) / (depth_max - depth_min + 1e-8)
    return np.clip(normalized, 0.0, 1.0)
