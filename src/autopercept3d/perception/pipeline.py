from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from autopercept3d.datasets.kitti import KITTIDataset
from autopercept3d.perception.bounding_boxes import AxisAlignedBox3D, create_axis_aligned_boxes
from autopercept3d.perception.clustering import ClusteringResult, cluster_dbscan, filter_clusters_by_size
from autopercept3d.perception.ground_removal import GroundRemovalResult, ransac_ground_plane
from autopercept3d.perception.preprocessing import (
    crop_roi,
    filter_finite_points,
    voxel_downsample,
)


@dataclass
class ClassicalPipelineConfig:
    """
    Configuration for the classical LiDAR perception pipeline.

    This is the MVP perception pipeline:
        ROI crop -> voxel downsample -> ground removal -> DBSCAN -> boxes
    """

    voxel_size: float = 0.2
    ground_distance_threshold: float = 0.25
    ground_iterations: int = 120
    dbscan_eps: float = 0.8
    dbscan_min_samples: int = 10

    roi_x_min: float = 0.0
    roi_x_max: float = 70.0
    roi_y_min: float = -40.0
    roi_y_max: float = 40.0
    roi_z_min: float = -3.0
    roi_z_max: float = 3.0


@dataclass
class PipelineFrameResult:
    """
    Complete output for one processed KITTI frame.
    """

    frame_id: str
    raw_points: np.ndarray
    finite_points: np.ndarray
    cropped_points: np.ndarray
    downsampled_points: np.ndarray
    ground_result: GroundRemovalResult
    clustering_result: ClusteringResult
    filtered_cluster_points: np.ndarray
    filtered_cluster_labels: np.ndarray
    boxes: list[AxisAlignedBox3D]
    metrics: dict
    timings_ms: dict


def run_classical_lidar_pipeline(
    dataset: KITTIDataset,
    frame_id: str,
    config: ClassicalPipelineConfig | None = None,
) -> PipelineFrameResult:
    """
    Run the full classical LiDAR perception pipeline on one KITTI frame.

    This function is important because it centralizes the project pipeline.
    Scripts and dashboards should call this instead of duplicating all steps.
    """
    if config is None:
        config = ClassicalPipelineConfig()

    timings_ms: dict[str, float] = {}

    t0 = perf_counter()
    raw_points = dataset.load_point_cloud(frame_id)
    timings_ms["load_point_cloud"] = (perf_counter() - t0) * 1000.0

    t0 = perf_counter()
    finite_points = filter_finite_points(raw_points)
    timings_ms["filter_finite"] = (perf_counter() - t0) * 1000.0

    t0 = perf_counter()
    cropped_points = crop_roi(
        finite_points,
        x_range=(config.roi_x_min, config.roi_x_max),
        y_range=(config.roi_y_min, config.roi_y_max),
        z_range=(config.roi_z_min, config.roi_z_max),
    )
    timings_ms["roi_crop"] = (perf_counter() - t0) * 1000.0

    t0 = perf_counter()
    downsampled_points = voxel_downsample(
        cropped_points,
        voxel_size=config.voxel_size,
    )
    timings_ms["voxel_downsample"] = (perf_counter() - t0) * 1000.0

    t0 = perf_counter()
    ground_result = ransac_ground_plane(
        downsampled_points,
        distance_threshold=config.ground_distance_threshold,
        num_iterations=config.ground_iterations,
    )
    timings_ms["ground_removal"] = (perf_counter() - t0) * 1000.0

    t0 = perf_counter()
    clustering_result = cluster_dbscan(
        ground_result.non_ground_points,
        eps=config.dbscan_eps,
        min_samples=config.dbscan_min_samples,
        use_z=True,
    )
    timings_ms["dbscan_clustering"] = (perf_counter() - t0) * 1000.0

    t0 = perf_counter()
    filtered_cluster_points, filtered_cluster_labels = filter_clusters_by_size(
        clustering_result.clustered_points,
        clustering_result.clustered_labels,
    )
    timings_ms["cluster_filtering"] = (perf_counter() - t0) * 1000.0

    t0 = perf_counter()
    boxes = create_axis_aligned_boxes(
        filtered_cluster_points,
        filtered_cluster_labels,
    )
    timings_ms["bounding_boxes"] = (perf_counter() - t0) * 1000.0

    total_time_ms = sum(timings_ms.values())
    timings_ms["total"] = total_time_ms

    metrics = {
        "frame_id": frame_id,
        "raw_points": int(len(raw_points)),
        "finite_points": int(len(finite_points)),
        "cropped_points": int(len(cropped_points)),
        "downsampled_points": int(len(downsampled_points)),
        "ground_points": int(len(ground_result.ground_points)),
        "non_ground_points": int(len(ground_result.non_ground_points)),
        "raw_dbscan_clusters": int(clustering_result.num_clusters),
        "dbscan_noise_points": int(len(clustering_result.noise_points)),
        "filtered_cluster_points": int(len(filtered_cluster_points)),
        "filtered_clusters": int(len(set(filtered_cluster_labels.tolist()))) if len(filtered_cluster_labels) else 0,
        "boxes": int(len(boxes)),
        "total_time_ms": float(total_time_ms),
        "approx_fps": float(1000.0 / total_time_ms) if total_time_ms > 0 else 0.0,
    }

    return PipelineFrameResult(
        frame_id=frame_id,
        raw_points=raw_points,
        finite_points=finite_points,
        cropped_points=cropped_points,
        downsampled_points=downsampled_points,
        ground_result=ground_result,
        clustering_result=clustering_result,
        filtered_cluster_points=filtered_cluster_points,
        filtered_cluster_labels=filtered_cluster_labels,
        boxes=boxes,
        metrics=metrics,
        timings_ms=timings_ms,
    )


def boxes_to_records(boxes: list[AxisAlignedBox3D]) -> list[dict]:
    """
    Convert box dataclasses into JSON/CSV-friendly dictionaries.
    """
    records = []

    for box in boxes:
        records.append(
            {
                "cluster_id": int(box.cluster_id),
                "num_points": int(box.num_points),
                "center_x": float(box.center_xyz[0]),
                "center_y": float(box.center_xyz[1]),
                "center_z": float(box.center_xyz[2]),
                "length_x": float(box.size_xyz[0]),
                "width_y": float(box.size_xyz[1]),
                "height_z": float(box.size_xyz[2]),
                "min_x": float(box.min_xyz[0]),
                "min_y": float(box.min_xyz[1]),
                "min_z": float(box.min_xyz[2]),
                "max_x": float(box.max_xyz[0]),
                "max_y": float(box.max_xyz[1]),
                "max_z": float(box.max_xyz[2]),
            }
        )

    return records
