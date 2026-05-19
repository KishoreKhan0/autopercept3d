import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autopercept3d.datasets.kitti import KITTIDataset
from autopercept3d.perception.bounding_boxes import (
    AxisAlignedBox3D,
    create_axis_aligned_boxes,
    get_bev_rectangle,
    summarize_boxes,
)
from autopercept3d.perception.clustering import cluster_dbscan, filter_clusters_by_size
from autopercept3d.perception.ground_removal import ransac_ground_plane
from autopercept3d.perception.preprocessing import (
    crop_roi,
    filter_finite_points,
    voxel_downsample,
)


def plot_boxes_bev(
    clustered_points: np.ndarray,
    clustered_labels: np.ndarray,
    boxes: list[AxisAlignedBox3D],
    title: str,
) -> None:
    """
    Plot filtered DBSCAN clusters with axis-aligned BEV boxes.
    """
    if len(clustered_points) > 0:
        plt.scatter(
            clustered_points[:, 0],
            clustered_points[:, 1],
            c=clustered_labels,
            s=4,
            alpha=0.8,
            cmap="tab20",
        )

    for box in boxes:
        rectangle = get_bev_rectangle(box)
        plt.plot(rectangle[:, 0], rectangle[:, 1], linewidth=2)
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


def print_box_summary(boxes: list[AxisAlignedBox3D]) -> None:
    """
    Print bounding box summary table.
    """
    summaries = summarize_boxes(boxes)
    summaries = sorted(summaries, key=lambda item: item["num_points"], reverse=True)

    print()
    print("3D bounding boxes:")
    print("cluster_id | points | center_x center_y center_z | length width height")
    print("-" * 75)

    for item in summaries:
        print(
            f"{item['cluster_id']:>9} | "
            f"{item['num_points']:>6} | "
            f"{item['center_x']:>8.2f} {item['center_y']:>8.2f} {item['center_z']:>8.2f} | "
            f"{item['length_x']:>6.2f} {item['width_y']:>6.2f} {item['height_z']:>6.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create simple 3D bounding boxes around clustered KITTI LiDAR points."
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
        help="Voxel size in meters before ground removal and clustering.",
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
        help="DBSCAN minimum nearby points to form a cluster.",
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

    boxes = create_axis_aligned_boxes(
        filtered_points,
        filtered_labels,
        min_points_per_cluster=10,
    )

    print("=" * 60)
    print(f"3D bounding boxes for KITTI frame: {args.frame}")
    print("=" * 60)
    print(f"Raw points:               {len(raw_points):>8}")
    print(f"After ROI crop:           {len(cropped_points):>8}")
    print(f"After downsampling:       {len(downsampled_points):>8}")
    print(f"Non-ground points:        {len(ground_result.non_ground_points):>8}")
    print(f"Filtered cluster points:  {len(filtered_points):>8}")
    print(f"Number of boxes:          {len(boxes):>8}")

    print_box_summary(boxes)

    plt.figure(figsize=(10, 8))
    plot_boxes_bev(
        filtered_points,
        filtered_labels,
        boxes,
        f"Cluster-based 3D Boxes - KITTI Frame {args.frame}",
    )
    plt.tight_layout()

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Saved visualization to: {save_path}")

    plt.show()


if __name__ == "__main__":
    main()
