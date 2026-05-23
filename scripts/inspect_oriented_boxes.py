import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autopercept3d.datasets.kitti import KITTIDataset
from autopercept3d.perception.bounding_boxes import (
    create_axis_aligned_boxes,
    get_bev_rectangle,
)
from autopercept3d.perception.clustering import cluster_dbscan, filter_clusters_by_size
from autopercept3d.perception.ground_removal import ransac_ground_plane
from autopercept3d.perception.oriented_boxes import (
    create_oriented_boxes,
    summarize_oriented_boxes,
)
from autopercept3d.perception.preprocessing import (
    crop_roi,
    filter_finite_points,
    voxel_downsample,
)


def plot_axis_aligned_boxes(points, labels, boxes, title: str) -> None:
    """
    Plot filtered clusters with old axis-aligned boxes.
    """
    if len(points) > 0:
        plt.scatter(
            points[:, 0],
            points[:, 1],
            c=labels,
            s=4,
            alpha=0.75,
            cmap="tab20",
        )

    for box in boxes:
        rectangle = get_bev_rectangle(box)
        plt.plot(rectangle[:, 0], rectangle[:, 1], linewidth=1.8)
        plt.text(
            box.center_xyz[0],
            box.center_xyz[1],
            str(box.cluster_id),
            fontsize=8,
            ha="center",
            va="center",
        )

    plt.title(title)
    plt.xlabel("x forward [m]")
    plt.ylabel("y left/right [m]")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)


def plot_oriented_boxes(points, labels, boxes, title: str) -> None:
    """
    Plot filtered clusters with PCA-oriented BEV boxes.
    """
    if len(points) > 0:
        plt.scatter(
            points[:, 0],
            points[:, 1],
            c=labels,
            s=4,
            alpha=0.75,
            cmap="tab20",
        )

    for box in boxes:
        corners = box.corners_bev
        plt.plot(corners[:, 0], corners[:, 1], linewidth=1.8)
        plt.text(
            box.center_xyz[0],
            box.center_xyz[1],
            str(box.cluster_id),
            fontsize=8,
            ha="center",
            va="center",
        )

        # Draw short orientation direction line from center.
        direction = np.array([np.cos(box.yaw_rad), np.sin(box.yaw_rad)])
        start = box.center_xyz[:2]
        end = start + direction * max(box.size_lwh[0] * 0.35, 0.5)
        plt.plot([start[0], end[0]], [start[1], end[1]], linewidth=1.5)

    plt.title(title)
    plt.xlabel("x forward [m]")
    plt.ylabel("y left/right [m]")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)


def print_oriented_box_summary(boxes, max_rows: int = 15) -> None:
    """
    Print largest oriented boxes.
    """
    summaries = summarize_oriented_boxes(boxes)
    summaries = sorted(summaries, key=lambda item: item["num_points"], reverse=True)

    print()
    print("Oriented 3D boxes:")
    print("cluster_id | points | center_x center_y center_z | length width height | yaw_deg")
    print("-" * 88)

    for item in summaries[:max_rows]:
        print(
            f"{item['cluster_id']:>9} | "
            f"{item['num_points']:>6} | "
            f"{item['center_x']:>8.2f} {item['center_y']:>8.2f} {item['center_z']:>8.2f} | "
            f"{item['length']:>6.2f} {item['width']:>6.2f} {item['height']:>6.2f} | "
            f"{item['yaw_deg']:>7.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect PCA-oriented 3D boxes for clustered KITTI LiDAR points."
    )

    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path to KITTI Object Detection dataset root.",
    )
    parser.add_argument(
        "--frame",
        type=str,
        default="000264",
        help="KITTI frame id.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.2,
        help="Voxel size in meters.",
    )
    parser.add_argument(
        "--ground-threshold",
        type=float,
        default=0.25,
        help="RANSAC ground-plane distance threshold in meters.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.8,
        help="DBSCAN eps in meters.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="DBSCAN min_samples.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save visualization.",
    )

    args = parser.parse_args()

    dataset = KITTIDataset(dataset_root=args.dataset_root)
    raw_points = dataset.load_point_cloud(args.frame)

    finite_points = filter_finite_points(raw_points)
    cropped_points = crop_roi(finite_points)
    downsampled_points = voxel_downsample(cropped_points, voxel_size=args.voxel_size)

    ground_result = ransac_ground_plane(
        downsampled_points,
        distance_threshold=args.ground_threshold,
    )

    clustering_result = cluster_dbscan(
        ground_result.non_ground_points,
        eps=args.eps,
        min_samples=args.min_samples,
        use_z=True,
    )

    filtered_points, filtered_labels = filter_clusters_by_size(
        clustering_result.clustered_points,
        clustering_result.clustered_labels,
    )

    axis_boxes = create_axis_aligned_boxes(filtered_points, filtered_labels)
    oriented_boxes = create_oriented_boxes(filtered_points, filtered_labels)

    print("=" * 60)
    print(f"Oriented box inspection for KITTI frame: {args.frame}")
    print("=" * 60)
    print(f"Raw points:               {len(raw_points):>8}")
    print(f"After ROI crop:           {len(cropped_points):>8}")
    print(f"After downsampling:       {len(downsampled_points):>8}")
    print(f"Non-ground points:        {len(ground_result.non_ground_points):>8}")
    print(f"Filtered cluster points:  {len(filtered_points):>8}")
    print(f"Axis-aligned boxes:       {len(axis_boxes):>8}")
    print(f"Oriented boxes:           {len(oriented_boxes):>8}")

    print_oriented_box_summary(oriented_boxes)

    plt.figure(figsize=(16, 7))

    plt.subplot(1, 2, 1)
    plot_axis_aligned_boxes(
        filtered_points,
        filtered_labels,
        axis_boxes,
        "Axis-aligned boxes",
    )

    plt.subplot(1, 2, 2)
    plot_oriented_boxes(
        filtered_points,
        filtered_labels,
        oriented_boxes,
        "PCA-oriented boxes",
    )

    plt.tight_layout()

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Saved oriented box comparison to: {save_path}")

    plt.show()


if __name__ == "__main__":
    main()
