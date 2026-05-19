from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autopercept3d.geometry.kitti_transforms import (
    bev_bottom_polygon_from_corners,
    kitti_label_corners_velodyne,
    parse_kitti_label,
)
from autopercept3d.perception.bounding_boxes import AxisAlignedBox3D, get_bev_rectangle
from autopercept3d.perception.pipeline import PipelineFrameResult


def sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    """
    Deterministically sample points for faster plotting.
    """
    if len(points) <= max_points:
        return points

    indices = np.linspace(0, len(points) - 1, max_points).astype(int)
    return points[indices]


def plot_points_bev(
    points: np.ndarray,
    label: str,
    max_points: int = 25000,
    point_size: float = 1.0,
    alpha: float = 0.7,
) -> None:
    """
    Plot LiDAR points in bird's-eye view.

    BEV axes:
        x = forward
        y = left/right
    """
    if len(points) == 0:
        return

    points = sample_points(points, max_points=max_points)

    plt.scatter(
        points[:, 0],
        points[:, 1],
        s=point_size,
        alpha=alpha,
        label=label,
    )


def plot_cluster_points_bev(
    points: np.ndarray,
    labels: np.ndarray,
    max_points: int = 25000,
) -> None:
    """
    Plot clustered points in BEV, colored by cluster id.
    """
    if len(points) == 0:
        return

    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points).astype(int)
        points = points[indices]
        labels = labels[indices]

    plt.scatter(
        points[:, 0],
        points[:, 1],
        c=labels,
        s=4,
        alpha=0.85,
        cmap="tab20",
        label="cluster points",
    )


def plot_proposal_boxes_bev(boxes: list[AxisAlignedBox3D], show_ids: bool = True) -> None:
    """
    Plot AutoPercept3D proposal boxes in BEV.
    """
    for index, box in enumerate(boxes):
        rectangle = get_bev_rectangle(box)

        plt.plot(
            rectangle[:, 0],
            rectangle[:, 1],
            linewidth=2.0,
            label="proposal box" if index == 0 else None,
        )

        if show_ids:
            plt.text(
                box.center_xyz[0],
                box.center_xyz[1],
                str(box.cluster_id),
                fontsize=8,
                ha="center",
                va="center",
            )


def plot_ground_truth_boxes_bev(
    labels: list[dict],
    calibration: dict[str, np.ndarray],
) -> None:
    """
    Plot KITTI ground-truth 3D boxes in LiDAR BEV coordinates.
    """
    first = True

    for raw_label in labels:
        label = parse_kitti_label(raw_label)

        if label.object_type == "DontCare":
            continue

        corners_velodyne = kitti_label_corners_velodyne(label, calibration)
        polygon = bev_bottom_polygon_from_corners(corners_velodyne)
        center = corners_velodyne[:, :2].mean(axis=0)

        plt.plot(
            polygon[:, 0],
            polygon[:, 1],
            linestyle="--",
            linewidth=2.5,
            label="KITTI ground truth" if first else None,
        )

        plt.text(
            center[0],
            center[1],
            label.object_type,
            fontsize=9,
            ha="center",
            va="center",
        )

        first = False


def set_bev_axes(
    title: str,
    xlim: tuple[float, float] = (-5.0, 70.0),
    ylim: tuple[float, float] = (-40.0, 40.0),
) -> None:
    """
    Apply consistent BEV axis formatting.
    """
    plt.title(title)
    plt.xlabel("x forward [m]")
    plt.ylabel("y left/right [m]")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.axis("equal")
    plt.grid(True, alpha=0.3)


def render_pipeline_bev_summary(
    result: PipelineFrameResult,
    labels: list[dict] | None = None,
    calibration: dict[str, np.ndarray] | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Render a clean portfolio-style BEV summary for one processed frame.

    Left:
        ground vs non-ground points

    Right:
        cluster points, proposal boxes, and optional KITTI ground truth boxes
    """
    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plot_points_bev(
        result.ground_result.ground_points,
        label="ground",
        point_size=1.0,
        alpha=0.25,
    )
    plot_points_bev(
        result.ground_result.non_ground_points,
        label="non-ground",
        point_size=1.0,
        alpha=0.75,
    )
    set_bev_axes(f"Ground Removal - Frame {result.frame_id}")
    plt.legend(markerscale=5)

    plt.subplot(1, 2, 2)
    plot_cluster_points_bev(
        result.filtered_cluster_points,
        result.filtered_cluster_labels,
    )
    plot_proposal_boxes_bev(result.boxes, show_ids=True)

    if labels is not None and calibration is not None:
        plot_ground_truth_boxes_bev(labels, calibration)

    title = (
        f"Object Proposals - Frame {result.frame_id} | "
        f"boxes={result.metrics['boxes']} | "
        f"fps={result.metrics['approx_fps']:.2f}"
    )
    set_bev_axes(title)
    plt.legend(markerscale=5)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Saved BEV summary to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def render_pipeline_bev_single(
    result: PipelineFrameResult,
    labels: list[dict] | None = None,
    calibration: dict[str, np.ndarray] | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Render a single clean BEV scene with proposal boxes and optional GT labels.
    """
    plt.figure(figsize=(10, 8))

    plot_cluster_points_bev(
        result.filtered_cluster_points,
        result.filtered_cluster_labels,
    )
    plot_proposal_boxes_bev(result.boxes, show_ids=True)

    if labels is not None and calibration is not None:
        plot_ground_truth_boxes_bev(labels, calibration)

    title = (
        f"AutoPercept3D BEV Scene - Frame {result.frame_id} | "
        f"boxes={result.metrics['boxes']} | "
        f"{result.metrics['total_time_ms']:.1f} ms"
    )
    set_bev_axes(title)
    plt.legend(markerscale=5)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Saved BEV scene to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
