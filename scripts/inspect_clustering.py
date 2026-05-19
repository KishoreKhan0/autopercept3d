import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autopercept3d.datasets.kitti import KITTIDataset
from autopercept3d.perception.clustering import (
    cluster_dbscan,
    filter_clusters_by_size,
    summarize_clusters,
)
from autopercept3d.perception.ground_removal import ransac_ground_plane
from autopercept3d.perception.preprocessing import (
    crop_roi,
    filter_finite_points,
    voxel_downsample,
)


def plot_clustered_bev(
    clustered_points: np.ndarray,
    clustered_labels: np.ndarray,
    noise_points: np.ndarray,
    title: str,
    max_noise_points: int = 6000,
) -> None:
    """
    Plot DBSCAN clusters in bird's-eye-view.

    Noise points are shown lightly.
    Clustered points are colored by cluster id.
    """
    if len(noise_points) > max_noise_points:
        idx = np.linspace(0, len(noise_points) - 1, max_noise_points).astype(int)
        noise_points = noise_points[idx]

    if len(noise_points) > 0:
        plt.scatter(
            noise_points[:, 0],
            noise_points[:, 1],
            s=1,
            alpha=0.15,
            label="noise",
        )

    if len(clustered_points) > 0:
        plt.scatter(
            clustered_points[:, 0],
            clustered_points[:, 1],
            c=clustered_labels,
            s=4,
            alpha=0.9,
            cmap="tab20",
            label="clusters",
        )

    plt.title(title)
    plt.xlabel("x forward [m]")
    plt.ylabel("y left/right [m]")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)


def print_cluster_summary(clustered_points: np.ndarray, clustered_labels: np.ndarray, max_rows: int = 12) -> None:
    """
    Print a compact summary of the largest clusters.
    """
    summaries = summarize_clusters(clustered_points, clustered_labels)
    summaries = sorted(summaries, key=lambda item: item["num_points"], reverse=True)

    print()
    print("Largest clusters:")
    print("cluster_id | points | center_x center_y center_z | size_x size_y size_z")
    print("-" * 72)

    for summary in summaries[:max_rows]:
        center = summary["center_xyz"]
        size = summary["size_xyz"]

        print(
            f"{summary['cluster_id']:>9} | "
            f"{summary['num_points']:>6} | "
            f"{center[0]:>8.2f} {center[1]:>8.2f} {center[2]:>8.2f} | "
            f"{size[0]:>6.2f} {size[1]:>6.2f} {size[2]:>6.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster non-ground KITTI LiDAR points using DBSCAN."
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
        "--bev-only",
        action="store_true",
        help="Cluster using only x,y instead of x,y,z.",
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

    non_ground_points = ground_result.non_ground_points

    clustering_result = cluster_dbscan(
        non_ground_points,
        eps=args.eps,
        min_samples=args.min_samples,
        use_z=not args.bev_only,
    )

    filtered_points, filtered_labels = filter_clusters_by_size(
        clustering_result.clustered_points,
        clustering_result.clustered_labels,
    )

    print("=" * 60)
    print(f"DBSCAN clustering for KITTI frame: {args.frame}")
    print("=" * 60)
    print(f"Raw points:               {len(raw_points):>8}")
    print(f"After ROI crop:           {len(cropped_points):>8}")
    print(f"After downsampling:       {len(downsampled_points):>8}")
    print(f"Non-ground points:        {len(non_ground_points):>8}")
    print(f"DBSCAN eps:               {args.eps}")
    print(f"DBSCAN min_samples:       {args.min_samples}")
    print(f"Clustering mode:          {'BEV x,y only' if args.bev_only else '3D x,y,z'}")
    print(f"Raw DBSCAN clusters:      {clustering_result.num_clusters:>8}")
    print(f"DBSCAN noise points:      {len(clustering_result.noise_points):>8}")
    print(f"Filtered cluster points:  {len(filtered_points):>8}")

    if len(filtered_labels) > 0:
        print(f"Filtered clusters:        {len(set(filtered_labels.tolist())):>8}")
    else:
        print(f"Filtered clusters:        {0:>8}")

    print_cluster_summary(filtered_points, filtered_labels)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plot_clustered_bev(
        clustering_result.clustered_points,
        clustering_result.clustered_labels,
        clustering_result.noise_points,
        "Raw DBSCAN clusters",
    )

    plt.subplot(1, 2, 2)
    plot_clustered_bev(
        filtered_points,
        filtered_labels,
        np.empty((0, non_ground_points.shape[1]), dtype=non_ground_points.dtype),
        "Filtered clusters",
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
