from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN


@dataclass
class ClusteringResult:
    """
    Output of point-cloud clustering.

    labels:
        Cluster label for every input point.
        -1 means noise / unclustered point.

    clustered_points:
        Points that belong to a valid cluster.

    clustered_labels:
        Cluster labels for clustered_points only.

    num_clusters:
        Number of valid clusters.

    noise_points:
        Points labeled as noise by DBSCAN.
    """

    labels: np.ndarray
    clustered_points: np.ndarray
    clustered_labels: np.ndarray
    num_clusters: int
    noise_points: np.ndarray


def cluster_dbscan(
    points: np.ndarray,
    eps: float = 0.8,
    min_samples: int = 10,
    use_z: bool = True,
) -> ClusteringResult:
    """
    Cluster non-ground LiDAR points using DBSCAN.

    DBSCAN groups points based on distance.

    Important parameters:
        eps:
            Maximum distance between neighboring points in the same cluster.
            Larger eps => bigger clusters, possibly merged objects.
            Smaller eps => smaller clusters, possibly fragmented objects.

        min_samples:
            Minimum number of nearby points required to form a cluster.
            Higher min_samples removes more noise but may miss small objects.

        use_z:
            If True, cluster in full 3D using x, y, z.
            If False, cluster in BEV using only x, y.

    Returns:
        ClusteringResult
    """
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("Expected points with shape (N, 4) or at least (N, 3).")

    if len(points) == 0:
        empty_labels = np.empty((0,), dtype=np.int32)
        return ClusteringResult(
            labels=empty_labels,
            clustered_points=points,
            clustered_labels=empty_labels,
            num_clusters=0,
            noise_points=points,
        )

    if use_z:
        features = points[:, :3]
    else:
        features = points[:, :2]

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(features)

    valid_mask = labels != -1
    clustered_points = points[valid_mask]
    clustered_labels = labels[valid_mask]
    noise_points = points[~valid_mask]

    if len(clustered_labels) == 0:
        num_clusters = 0
    else:
        num_clusters = len(set(clustered_labels.tolist()))

    return ClusteringResult(
        labels=labels,
        clustered_points=clustered_points,
        clustered_labels=clustered_labels,
        num_clusters=num_clusters,
        noise_points=noise_points,
    )


def summarize_clusters(points: np.ndarray, labels: np.ndarray) -> list[dict]:
    """
    Compute simple statistics for each cluster.

    Returns one dictionary per cluster with:
        cluster_id
        num_points
        min_xyz
        max_xyz
        size_xyz
        center_xyz
    """
    summaries = []

    valid_cluster_ids = sorted(cluster_id for cluster_id in set(labels.tolist()) if cluster_id != -1)

    for cluster_id in valid_cluster_ids:
        cluster_points = points[labels == cluster_id]

        xyz = cluster_points[:, :3]
        min_xyz = xyz.min(axis=0)
        max_xyz = xyz.max(axis=0)
        size_xyz = max_xyz - min_xyz
        center_xyz = (min_xyz + max_xyz) / 2.0

        summaries.append(
            {
                "cluster_id": int(cluster_id),
                "num_points": int(len(cluster_points)),
                "min_xyz": min_xyz,
                "max_xyz": max_xyz,
                "size_xyz": size_xyz,
                "center_xyz": center_xyz,
            }
        )

    return summaries


def filter_clusters_by_size(
    points: np.ndarray,
    labels: np.ndarray,
    min_points: int = 20,
    max_points: int = 5000,
    min_height: float = 0.2,
    max_height: float = 4.0,
    max_length: float = 8.0,
    max_width: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove clusters that are too small, too large, or physically unrealistic.

    This is not object detection yet. It is only cleaning DBSCAN clusters
    so the next bounding-box step is less noisy.

    Returns:
        filtered_points, filtered_labels
    """
    if len(points) == 0:
        return points, labels

    keep_masks = []

    for summary in summarize_clusters(points, labels):
        cluster_id = summary["cluster_id"]
        num_points = summary["num_points"]
        size_x, size_y, size_z = summary["size_xyz"]

        keep = (
            num_points >= min_points
            and num_points <= max_points
            and size_z >= min_height
            and size_z <= max_height
            and size_x <= max_length
            and size_y <= max_width
        )

        if keep:
            keep_masks.append(labels == cluster_id)

    if not keep_masks:
        return np.empty((0, points.shape[1]), dtype=points.dtype), np.empty((0,), dtype=labels.dtype)

    keep_mask = np.logical_or.reduce(keep_masks)

    return points[keep_mask], labels[keep_mask]
