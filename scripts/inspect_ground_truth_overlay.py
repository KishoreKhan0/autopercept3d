import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autopercept3d.datasets.kitti import KITTIDataset
from autopercept3d.geometry.kitti_transforms import (
    bev_bottom_polygon_from_corners,
    kitti_label_corners_velodyne,
    parse_kitti_label,
)
from autopercept3d.perception.bounding_boxes import (
    AxisAlignedBox3D,
    create_axis_aligned_boxes,
    get_bev_rectangle,
)
from autopercept3d.perception.clustering import cluster_dbscan, filter_clusters_by_size
from autopercept3d.perception.ground_removal import ransac_ground_plane
from autopercept3d.perception.preprocessing import (
    crop_roi,
    filter_finite_points,
    voxel_downsample,
)


def plot_predicted_boxes(
    clustered_points: np.ndarray,
    clustered_labels: np.ndarray,
    boxes: list[AxisAlignedBox3D],
) -> None:
    """
    Plot cluster-based object proposal boxes.
    """
    if len(clustered_points) > 0:
        plt.scatter(
            clustered_points[:, 0],
            clustered_points[:, 1],
            c=clustered_labels,
            s=4,
            alpha=0.65,
            cmap="tab20",
            label="cluster points",
        )

    for box in boxes:
        rectangle = get_bev_rectangle(box)
        plt.plot(
            rectangle[:, 0],
            rectangle[:, 1],
            linewidth=1.8,
            label="proposal box" if box == boxes[0] else None,
        )


def plot_kitti_ground_truth_labels(labels: list[dict], calibration: dict[str, np.ndarray]) -> None:
    """
    Plot KITTI ground-truth 3D boxes in LiDAR BEV coordinates.
    """
    first_label = True

    for raw_label in labels:
        label = parse_kitti_label(raw_label)

        # Ignore DontCare regions for this visualization.
        if label.object_type == "DontCare":
            continue

        corners_velodyne = kitti_label_corners_velodyne(label, calibration)
        polygon = bev_bottom_polygon_from_corners(corners_velodyne)

        plt.plot(
            polygon[:, 0],
            polygon[:, 1],
            linestyle="--",
            linewidth=2.8,
            label="KITTI ground truth" if first_label else None,
        )

        center = corners_velodyne[:, :2].mean(axis=0)
        plt.text(
            center[0],
            center[1],
            label.object_type,
            fontsize=9,
            ha="center",
            va="center",
        )

        first_label = False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay AutoPercept3D cluster boxes with KITTI ground-truth labels in BEV."
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
        default="000000",
        help="KITTI frame id, for example 000000.",
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
        help="DBSCAN neighborhood radius in meters.",
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
        help="Optional path to save visualization image.",
    )

    args = parser.parse_args()

    dataset = KITTIDataset(dataset_root=args.dataset_root)

    raw_points = dataset.load_point_cloud(args.frame)
    calibration = dataset.load_calibration(args.frame)
    labels = dataset.load_labels(args.frame)

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

    boxes = create_axis_aligned_boxes(filtered_points, filtered_labels)

    valid_gt_labels = [label for label in labels if label["type"] != "DontCare"]

    print("=" * 60)
    print(f"Ground-truth overlay for KITTI frame: {args.frame}")
    print("=" * 60)
    print(f"Raw points:              {len(raw_points):>8}")
    print(f"After ROI crop:          {len(cropped_points):>8}")
    print(f"After downsampling:      {len(downsampled_points):>8}")
    print(f"Non-ground points:       {len(ground_result.non_ground_points):>8}")
    print(f"Proposal boxes:          {len(boxes):>8}")
    print(f"KITTI GT labels:         {len(valid_gt_labels):>8}")

    if valid_gt_labels:
        print()
        print("KITTI labels in this frame:")
        for index, label in enumerate(valid_gt_labels, start=1):
            print(f"  {index}. {label['type']}")

    plt.figure(figsize=(10, 8))

    plot_predicted_boxes(filtered_points, filtered_labels, boxes)
    plot_kitti_ground_truth_labels(labels, calibration)

    plt.title(f"Proposal Boxes vs KITTI Ground Truth - Frame {args.frame}")
    plt.xlabel("x forward [m]")
    plt.ylabel("y left/right [m]")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Saved visualization to: {save_path}")

    plt.show()


if __name__ == "__main__":
    main()
