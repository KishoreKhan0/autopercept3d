from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AxisAlignedBox3D:
    """
    Simple 3D bounding box aligned with the LiDAR coordinate axes.

    This is the first bounding-box type for AutoPercept3D.

    Coordinate convention for KITTI Velodyne:
        x = forward
        y = left/right
        z = up/down

    The box is represented by:
        min_xyz: [min_x, min_y, min_z]
        max_xyz: [max_x, max_y, max_z]
        center_xyz: box center
        size_xyz: [length_x, width_y, height_z]
    """

    cluster_id: int
    num_points: int
    min_xyz: np.ndarray
    max_xyz: np.ndarray
    center_xyz: np.ndarray
    size_xyz: np.ndarray


def create_axis_aligned_boxes(
    clustered_points: np.ndarray,
    clustered_labels: np.ndarray,
    min_points_per_cluster: int = 10,
) -> list[AxisAlignedBox3D]:
    """
    Create one axis-aligned 3D bounding box for each cluster.

    This is a simple baseline:
        - find min x, y, z of each cluster
        - find max x, y, z of each cluster
        - create a box around that cluster

    Later we can improve this with oriented bounding boxes.
    """
    if len(clustered_points) == 0:
        return []

    boxes: list[AxisAlignedBox3D] = []

    cluster_ids = sorted(set(clustered_labels.tolist()))

    for cluster_id in cluster_ids:
        if cluster_id == -1:
            continue

        cluster_points = clustered_points[clustered_labels == cluster_id]

        if len(cluster_points) < min_points_per_cluster:
            continue

        xyz = cluster_points[:, :3]

        min_xyz = xyz.min(axis=0)
        max_xyz = xyz.max(axis=0)
        center_xyz = (min_xyz + max_xyz) / 2.0
        size_xyz = max_xyz - min_xyz

        box = AxisAlignedBox3D(
            cluster_id=int(cluster_id),
            num_points=int(len(cluster_points)),
            min_xyz=min_xyz,
            max_xyz=max_xyz,
            center_xyz=center_xyz,
            size_xyz=size_xyz,
        )

        boxes.append(box)

    return boxes


def get_bev_rectangle(box: AxisAlignedBox3D) -> np.ndarray:
    """
    Return the closed BEV rectangle corners of a 3D box.

    Output shape:
        (5, 2)

    The first corner is repeated at the end so it can be plotted directly.
    """
    min_x, min_y, _ = box.min_xyz
    max_x, max_y, _ = box.max_xyz

    return np.array(
        [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
            [min_x, min_y],
        ],
        dtype=np.float64,
    )


def get_3d_box_corners(box: AxisAlignedBox3D) -> np.ndarray:
    """
    Return the 8 corners of an axis-aligned 3D bounding box.

    Corner order:
        bottom face then top face
    """
    min_x, min_y, min_z = box.min_xyz
    max_x, max_y, max_z = box.max_xyz

    return np.array(
        [
            [min_x, min_y, min_z],
            [max_x, min_y, min_z],
            [max_x, max_y, min_z],
            [min_x, max_y, min_z],
            [min_x, min_y, max_z],
            [max_x, min_y, max_z],
            [max_x, max_y, max_z],
            [min_x, max_y, max_z],
        ],
        dtype=np.float64,
    )


def summarize_boxes(boxes: list[AxisAlignedBox3D]) -> list[dict]:
    """
    Convert boxes to a simple list of dictionaries for printing/exporting.
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
                "length_x": float(box.size_xyz[0]),
                "width_y": float(box.size_xyz[1]),
                "height_z": float(box.size_xyz[2]),
            }
        )

    return summaries
