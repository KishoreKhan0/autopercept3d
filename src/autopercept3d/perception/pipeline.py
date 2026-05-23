from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from autopercept3d.datasets.kitti import KITTIDataset
from autopercept3d.perception.bounding_boxes import AxisAlignedBox3D, create_axis_aligned_boxes
from autopercept3d.perception.clustering import cluster_dbscan, filter_clusters_by_size
from autopercept3d.perception.ground_removal import ransac_ground_plane
from autopercept3d.perception.oriented_boxes import (
    OrientedBox3D,
    create_oriented_boxes,
    summarize_oriented_boxes,
)
from autopercept3d.perception.preprocessing import (
    crop_roi,
    filter_finite_points,
    voxel_downsample,
)


@dataclass
class ClassicalPipelineConfig:
    """
    Configuration for the classical LiDAR perception pipeline.

    The pipeline remains intentionally classical:
        ROI crop -> voxel downsample -> RANSAC ground removal
        -> DBSCAN clustering -> 3D proposal boxes

    v2 addition:
        The pipeline now returns both axis-aligned boxes and PCA-oriented boxes.
    """

    voxel_size: float = 0.2
    ground_distance_threshold: float = 0.25
    ground_iterations: int = 120
    dbscan_eps: float = 0.8
    dbscan_min_samples: int = 10
    min_points_per_oriented_box: int = 10


@dataclass
class PipelineFrameResult:
    """
    Output of the full classical LiDAR perception pipeline for one frame.

    boxes:
        Original axis-aligned 3D boxes. Kept for backward compatibility.

    oriented_boxes:
        New PCA-oriented 3D boxes in BEV. These improve visual fit for
        elongated or angled objects.
    """

    frame_id: str
    raw_points: np.ndarray
    finite_points: np.ndarray
    cropped_points: np.ndarray
    downsampled_points: np.ndarray
    ground_result: Any
    clustering_result: Any
    filtered_cluster_points: np.ndarray
    filtered_cluster_labels: np.ndarray
    boxes: list[AxisAlignedBox3D]
    oriented_boxes: list[OrientedBox3D]
    metrics: dict
    timings_ms: dict


def _elapsed_ms(start_time: float) -> float:
    """
    Convert perf_counter delta to milliseconds.
    """
    return (perf_counter() - start_time) * 1000.0


def _run_ground_removal(points: np.ndarray, config: ClassicalPipelineConfig):
    """
    Run ground removal while staying compatible with possible older function signatures.
    """
    try:
        return ransac_ground_plane(
            points,
            distance_threshold=config.ground_distance_threshold,
            max_iterations=config.ground_iterations,
        )
    except TypeError:
        try:
            return ransac_ground_plane(
                points,
                distance_threshold=config.ground_distance_threshold,
                num_iterations=config.ground_iterations,
            )
        except TypeError:
            return ransac_ground_plane(
                points,
                distance_threshold=config.ground_distance_threshold,
            )


def _safe_len(value) -> int:
    """
    Safe len helper for optional arrays/lists.
    """
    if value is None:
        return 0
    return len(value)


def _axis_box_to_record(box: AxisAlignedBox3D) -> dict:
    """
    Convert an axis-aligned box to a JSON-friendly record.

    This helper is intentionally tolerant to small field-name differences,
    because earlier project files may use size_xyz while newer printouts use
    length/width/height names.
    """
    center = np.asarray(box.center_xyz, dtype=float)

    if hasattr(box, "size_xyz"):
        size = np.asarray(box.size_xyz, dtype=float)
    elif hasattr(box, "size_lwh"):
        size = np.asarray(box.size_lwh, dtype=float)
    elif hasattr(box, "min_xyz") and hasattr(box, "max_xyz"):
        size = np.asarray(box.max_xyz, dtype=float) - np.asarray(box.min_xyz, dtype=float)
    else:
        size = np.array([0.0, 0.0, 0.0], dtype=float)

    record = {
        "cluster_id": int(box.cluster_id),
        "num_points": int(getattr(box, "num_points", getattr(box, "points", 0))),
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "center_z": float(center[2]),
        "length": float(size[0]),
        "width": float(size[1]),
        "height": float(size[2]),
    }

    if hasattr(box, "min_xyz"):
        min_xyz = np.asarray(box.min_xyz, dtype=float)
        record.update(
            {
                "min_x": float(min_xyz[0]),
                "min_y": float(min_xyz[1]),
                "min_z": float(min_xyz[2]),
            }
        )

    if hasattr(box, "max_xyz"):
        max_xyz = np.asarray(box.max_xyz, dtype=float)
        record.update(
            {
                "max_x": float(max_xyz[0]),
                "max_y": float(max_xyz[1]),
                "max_z": float(max_xyz[2]),
            }
        )

    return record


def boxes_to_records(boxes: list[AxisAlignedBox3D]) -> list[dict]:
    """
    Convert axis-aligned boxes to JSON-friendly records.

    Kept under the same name for backward compatibility with existing scripts.
    """
    return [_axis_box_to_record(box) for box in boxes]


def oriented_boxes_to_records(boxes: list[OrientedBox3D]) -> list[dict]:
    """
    Convert PCA-oriented boxes to JSON-friendly records.
    """
    records = summarize_oriented_boxes(boxes)

    for record, box in zip(records, boxes):
        record["corners_bev"] = [
            [float(x), float(y)]
            for x, y in box.corners_bev
        ]
        record["yaw_rad"] = float(box.yaw_rad)

    return records


def run_classical_lidar_pipeline(
    dataset: KITTIDataset,
    frame_id: str,
    config: ClassicalPipelineConfig | None = None,
) -> PipelineFrameResult:
    """
    Run the full classical LiDAR perception pipeline for one KITTI frame.

    This is the reusable central pipeline used by scripts, dashboard, reports,
    and future extensions.
    """
    if config is None:
        config = ClassicalPipelineConfig()

    total_start = perf_counter()
    timings_ms: dict[str, float] = {}

    stage_start = perf_counter()
    raw_points = dataset.load_point_cloud(frame_id)
    timings_ms["load_point_cloud"] = _elapsed_ms(stage_start)

    stage_start = perf_counter()
    finite_points = filter_finite_points(raw_points)
    timings_ms["filter_finite"] = _elapsed_ms(stage_start)

    stage_start = perf_counter()
    cropped_points = crop_roi(finite_points)
    timings_ms["roi_crop"] = _elapsed_ms(stage_start)

    stage_start = perf_counter()
    downsampled_points = voxel_downsample(cropped_points, voxel_size=config.voxel_size)
    timings_ms["voxel_downsample"] = _elapsed_ms(stage_start)

    stage_start = perf_counter()
    ground_result = _run_ground_removal(downsampled_points, config)
    timings_ms["ground_removal"] = _elapsed_ms(stage_start)

    stage_start = perf_counter()
    clustering_result = cluster_dbscan(
        ground_result.non_ground_points,
        eps=config.dbscan_eps,
        min_samples=config.dbscan_min_samples,
        use_z=True,
    )
    timings_ms["dbscan_clustering"] = _elapsed_ms(stage_start)

    stage_start = perf_counter()
    filtered_cluster_points, filtered_cluster_labels = filter_clusters_by_size(
        clustering_result.clustered_points,
        clustering_result.clustered_labels,
    )
    timings_ms["cluster_filtering"] = _elapsed_ms(stage_start)

    stage_start = perf_counter()
    boxes = create_axis_aligned_boxes(
        filtered_cluster_points,
        filtered_cluster_labels,
    )
    timings_ms["bounding_boxes"] = _elapsed_ms(stage_start)

    stage_start = perf_counter()
    oriented_boxes = create_oriented_boxes(
        filtered_cluster_points,
        filtered_cluster_labels,
        min_points_per_cluster=config.min_points_per_oriented_box,
    )
    timings_ms["oriented_boxes"] = _elapsed_ms(stage_start)

    total_ms = _elapsed_ms(total_start)
    timings_ms["total"] = total_ms

    raw_cluster_count = int(getattr(clustering_result, "num_clusters", 0))
    noise_points = int(getattr(clustering_result, "num_noise_points", 0))

    if raw_cluster_count == 0 and hasattr(clustering_result, "clustered_labels"):
        labels = clustering_result.clustered_labels
        raw_cluster_count = len(set(labels.tolist()) - {-1})

    if noise_points == 0 and hasattr(clustering_result, "clustered_labels"):
        noise_points = int(np.sum(clustering_result.clustered_labels == -1))

    metrics = {
        "frame_id": frame_id,
        "raw_points": _safe_len(raw_points),
        "finite_points": _safe_len(finite_points),
        "cropped_points": _safe_len(cropped_points),
        "downsampled_points": _safe_len(downsampled_points),
        "ground_points": _safe_len(ground_result.ground_points),
        "non_ground_points": _safe_len(ground_result.non_ground_points),
        "raw_dbscan_clusters": raw_cluster_count,
        "dbscan_noise_points": noise_points,
        "filtered_cluster_points": _safe_len(filtered_cluster_points),
        "filtered_clusters": len(set(filtered_cluster_labels.tolist())) if len(filtered_cluster_labels) else 0,
        "boxes": len(boxes),
        "axis_aligned_boxes": len(boxes),
        "oriented_boxes": len(oriented_boxes),
        "total_time_ms": total_ms,
        "approx_fps": 1000.0 / total_ms if total_ms > 0.0 else 0.0,
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
        oriented_boxes=oriented_boxes,
        metrics=metrics,
        timings_ms=timings_ms,
    )
