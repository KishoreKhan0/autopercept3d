from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GroundRemovalResult:
    """
    Container for ground-removal output.

    ground_points:
        Points classified as road/ground.

    non_ground_points:
        Points classified as obstacles / objects / non-ground.

    plane_model:
        Plane equation [a, b, c, d] where:
            a*x + b*y + c*z + d = 0

    ground_mask:
        Boolean mask with shape (N,), True for ground points.
    """

    ground_points: np.ndarray
    non_ground_points: np.ndarray
    plane_model: np.ndarray
    ground_mask: np.ndarray


def fit_plane_from_three_points(points_xyz: np.ndarray) -> np.ndarray | None:
    """
    Fit a plane from exactly three 3D points.

    Returns:
        plane [a, b, c, d] normalized so that sqrt(a^2+b^2+c^2)=1.

    If the 3 points are almost collinear, returns None.
    """
    p1, p2, p3 = points_xyz

    v1 = p2 - p1
    v2 = p3 - p1

    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)

    if norm < 1e-8:
        return None

    normal = normal / norm
    d = -np.dot(normal, p1)

    plane = np.array([normal[0], normal[1], normal[2], d], dtype=np.float64)
    return plane


def point_to_plane_distances(points_xyz: np.ndarray, plane_model: np.ndarray) -> np.ndarray:
    """
    Compute perpendicular distance of each point to a plane.

    Plane model:
        a*x + b*y + c*z + d = 0
    """
    a, b, c, d = plane_model
    numerator = np.abs(points_xyz @ np.array([a, b, c]) + d)
    denominator = np.sqrt(a * a + b * b + c * c) + 1e-8
    return numerator / denominator


def ransac_ground_plane(
    points: np.ndarray,
    distance_threshold: float = 0.25,
    num_iterations: int = 120,
    random_seed: int = 42,
    max_ransac_points: int = 8000,
) -> GroundRemovalResult:
    """
    Remove the ground plane from a LiDAR point cloud using simple NumPy RANSAC.

    This implementation is intentionally dependency-light:
    it does not need Open3D, PCL, or scikit-learn.

    Steps:
        1. Use only x, y, z coordinates.
        2. Sample 3 random points many times.
        3. Fit a plane candidate.
        4. Count points close to the plane.
        5. Choose the plane with the most inliers.
        6. Split original points into ground and non-ground.

    Notes:
        KITTI Velodyne coordinate convention:
            x = forward
            y = left/right
            z = up/down

        For road scenes, the ground plane is usually close to horizontal,
        so its normal should be mostly aligned with the z axis.
    """
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("Expected points with shape (N, 4) or at least (N, 3).")

    if len(points) < 3:
        empty_mask = np.zeros(len(points), dtype=bool)
        return GroundRemovalResult(
            ground_points=points[empty_mask],
            non_ground_points=points,
            plane_model=np.array([0.0, 0.0, 1.0, 0.0]),
            ground_mask=empty_mask,
        )

    points_xyz = points[:, :3]

    # RANSAC works better if we mostly sample low points,
    # because the road/ground is usually below objects.
    z = points_xyz[:, 2]
    low_point_mask = z < np.percentile(z, 45)

    candidate_indices = np.where(low_point_mask)[0]

    if len(candidate_indices) < 3:
        candidate_indices = np.arange(len(points))

    # Limit candidate points for speed.
    rng = np.random.default_rng(random_seed)

    if len(candidate_indices) > max_ransac_points:
        candidate_indices = rng.choice(
            candidate_indices,
            size=max_ransac_points,
            replace=False,
        )

    best_plane = None
    best_inlier_count = -1

    for _ in range(num_iterations):
        sample_indices = rng.choice(candidate_indices, size=3, replace=False)
        sample_points = points_xyz[sample_indices]

        plane = fit_plane_from_three_points(sample_points)

        if plane is None:
            continue

        # Ground plane should be mostly horizontal.
        # Therefore, its normal should have a strong z component.
        normal_z_strength = abs(plane[2])
        if normal_z_strength < 0.75:
            continue

        distances = point_to_plane_distances(points_xyz, plane)
        inliers = distances < distance_threshold

        # Avoid treating high structures parallel to ground as ground.
        # Most road points are usually not far above the sensor origin.
        inliers = inliers & (points_xyz[:, 2] < 0.5)

        inlier_count = int(np.sum(inliers))

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_plane = plane

    if best_plane is None:
        # Safe fallback: simple height threshold.
        # This keeps the pipeline working even if RANSAC fails.
        ground_mask = points_xyz[:, 2] < np.percentile(points_xyz[:, 2], 25)
        best_plane = np.array([0.0, 0.0, 1.0, -np.median(points_xyz[ground_mask, 2])])
    else:
        distances = point_to_plane_distances(points_xyz, best_plane)
        ground_mask = (distances < distance_threshold) & (points_xyz[:, 2] < 0.5)

    ground_points = points[ground_mask]
    non_ground_points = points[~ground_mask]

    return GroundRemovalResult(
        ground_points=ground_points,
        non_ground_points=non_ground_points,
        plane_model=best_plane,
        ground_mask=ground_mask,
    )
