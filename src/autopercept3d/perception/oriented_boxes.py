from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OrientedBox3D:
    """
    PCA-based oriented 3D bounding box for one LiDAR cluster.

    Coordinate convention:
        x = forward
        y = left/right
        z = up/down

    The orientation is estimated in BEV using PCA on x,y cluster points.
    The box is vertical in z and oriented only around the z-axis.
    """

    cluster_id: int
    num_points: int
    center_xyz: np.ndarray
    size_lwh: np.ndarray
    yaw_rad: float
    corners_bev: np.ndarray
    min_z: float
    max_z: float


def _ensure_right_handed_axes(axes: np.ndarray) -> np.ndarray:
    """
    Make PCA axes deterministic and right-handed in 2D.
    """
    axis_0 = axes[:, 0]

    # Deterministic sign: main axis should generally point toward +x.
    if axis_0[0] < 0:
        axis_0 = -axis_0

    # Make second axis perpendicular and right-handed.
    axis_1 = np.array([-axis_0[1], axis_0[0]], dtype=np.float64)

    return np.column_stack([axis_0, axis_1])


def create_oriented_box_from_cluster(
    cluster_points: np.ndarray,
    cluster_id: int,
) -> OrientedBox3D:
    """
    Create a PCA-based oriented 3D box from one cluster.

    Steps:
        1. Use x,y points in BEV.
        2. Compute PCA direction.
        3. Rotate points into local box coordinates.
        4. Find min/max in local coordinates.
        5. Convert rectangle corners back to world coordinates.
        6. Use min/max z for vertical extent.

    This is not a perfect minimum-area rectangle, but it is a strong
    first oriented-box baseline and works well for elongated objects.
    """
    if cluster_points.ndim != 2 or cluster_points.shape[1] < 3:
        raise ValueError("Expected cluster points with shape (N, 4) or at least (N, 3).")

    if len(cluster_points) < 3:
        raise ValueError("At least 3 points are required for an oriented box.")

    xyz = cluster_points[:, :3]
    xy = xyz[:, :2]

    xy_mean = xy.mean(axis=0)
    centered_xy = xy - xy_mean

    covariance = np.cov(centered_xy.T)

    if not np.isfinite(covariance).all():
        axes = np.eye(2, dtype=np.float64)
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        axes = eigenvectors[:, order]
        axes = _ensure_right_handed_axes(axes)

    local_xy = centered_xy @ axes

    min_local = local_xy.min(axis=0)
    max_local = local_xy.max(axis=0)

    center_local = (min_local + max_local) / 2.0
    size_xy = max_local - min_local

    rectangle_local = np.array(
        [
            [min_local[0], min_local[1]],
            [max_local[0], min_local[1]],
            [max_local[0], max_local[1]],
            [min_local[0], max_local[1]],
            [min_local[0], min_local[1]],
        ],
        dtype=np.float64,
    )

    corners_bev = rectangle_local @ axes.T + xy_mean
    center_xy = center_local @ axes.T + xy_mean

    min_z = float(xyz[:, 2].min())
    max_z = float(xyz[:, 2].max())
    center_z = (min_z + max_z) / 2.0
    height_z = max_z - min_z

    yaw_rad = float(np.arctan2(axes[1, 0], axes[0, 0]))

    center_xyz = np.array([center_xy[0], center_xy[1], center_z], dtype=np.float64)

    # length = major PCA direction extent, width = minor direction extent, height = z extent
    size_lwh = np.array([size_xy[0], size_xy[1], height_z], dtype=np.float64)

    return OrientedBox3D(
        cluster_id=int(cluster_id),
        num_points=int(len(cluster_points)),
        center_xyz=center_xyz,
        size_lwh=size_lwh,
        yaw_rad=yaw_rad,
        corners_bev=corners_bev,
        min_z=min_z,
        max_z=max_z,
    )


def create_oriented_boxes(
    clustered_points: np.ndarray,
    clustered_labels: np.ndarray,
    min_points_per_cluster: int = 10,
) -> list[OrientedBox3D]:
    """
    Create one oriented 3D box for each cluster.
    """
    if len(clustered_points) == 0:
        return []

    boxes: list[OrientedBox3D] = []

    cluster_ids = sorted(set(clustered_labels.tolist()))

    for cluster_id in cluster_ids:
        if cluster_id == -1:
            continue

        cluster_points = clustered_points[clustered_labels == cluster_id]

        if len(cluster_points) < min_points_per_cluster:
            continue

        try:
            box = create_oriented_box_from_cluster(
                cluster_points=cluster_points,
                cluster_id=int(cluster_id),
            )
            boxes.append(box)
        except ValueError:
            continue

    return boxes


def oriented_box_area_bev(box: OrientedBox3D) -> float:
    """
    BEV footprint area of an oriented box.
    """
    length, width, _ = box.size_lwh
    return float(max(length, 0.0) * max(width, 0.0))


def summarize_oriented_boxes(boxes: list[OrientedBox3D]) -> list[dict]:
    """
    Convert oriented boxes to a list of printable/exportable dictionaries.
    """
    summaries = []

    for box in boxes:
        summaries.append(
            {
                "cluster_id": box.cluster_id,
                "num_points": box.num_points,
                "center_x": float(box.center_xyz[0]),
                "center_y": float(box.center_xyz[1]),
                "center_z": float(box.center_xyz[2]),
                "length": float(box.size_lwh[0]),
                "width": float(box.size_lwh[1]),
                "height": float(box.size_lwh[2]),
                "yaw_deg": float(np.degrees(box.yaw_rad)),
                "bev_area": oriented_box_area_bev(box),
            }
        )

    return summaries
