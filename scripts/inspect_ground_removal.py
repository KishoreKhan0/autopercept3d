import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autopercept3d.datasets.kitti import KITTIDataset
from autopercept3d.perception.ground_removal import ransac_ground_plane
from autopercept3d.perception.preprocessing import (
    crop_roi,
    filter_finite_points,
    voxel_downsample,
)


def plot_bev(
    points: np.ndarray,
    title: str,
    color_by_height: bool = True,
    max_points: int = 20000,
) -> None:
    """
    Draw bird's-eye-view LiDAR points.
    """
    if len(points) == 0:
        plt.title(title)
        plt.xlabel("x forward [m]")
        plt.ylabel("y left/right [m]")
        plt.grid(True, alpha=0.3)
        return

    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points).astype(int)
        points = points[indices]

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    if color_by_height:
        plt.scatter(x, y, c=z, s=1, alpha=0.8)
    else:
        plt.scatter(x, y, s=1, alpha=0.8)

    plt.title(title)
    plt.xlabel("x forward [m]")
    plt.ylabel("y left/right [m]")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)


def plot_ground_vs_nonground(
    ground_points: np.ndarray,
    non_ground_points: np.ndarray,
    title: str,
    max_points_each: int = 15000,
) -> None:
    """
    Draw ground and non-ground points together in BEV.

    Ground points are plotted smaller/lighter.
    Non-ground points are plotted more strongly.
    """
    if len(ground_points) > max_points_each:
        idx = np.linspace(0, len(ground_points) - 1, max_points_each).astype(int)
        ground_points = ground_points[idx]

    if len(non_ground_points) > max_points_each:
        idx = np.linspace(0, len(non_ground_points) - 1, max_points_each).astype(int)
        non_ground_points = non_ground_points[idx]

    if len(ground_points) > 0:
        plt.scatter(
            ground_points[:, 0],
            ground_points[:, 1],
            s=1,
            alpha=0.25,
            label="ground",
        )

    if len(non_ground_points) > 0:
        plt.scatter(
            non_ground_points[:, 0],
            non_ground_points[:, 1],
            s=2,
            alpha=0.8,
            label="non-ground",
        )

    plt.title(title)
    plt.xlabel("x forward [m]")
    plt.ylabel("y left/right [m]")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend(markerscale=5)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run simple RANSAC ground removal on one KITTI LiDAR frame."
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
        help="Voxel size in meters for downsampling before ground removal.",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.25,
        help="Maximum point-to-plane distance in meters for ground inliers.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=120,
        help="Number of RANSAC iterations.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save the visualization image.",
    )

    args = parser.parse_args()

    dataset = KITTIDataset(dataset_root=args.dataset_root)
    raw_points = dataset.load_point_cloud(args.frame)

    finite_points = filter_finite_points(raw_points)
    cropped_points = crop_roi(finite_points)
    downsampled_points = voxel_downsample(cropped_points, voxel_size=args.voxel_size)

    result = ransac_ground_plane(
        downsampled_points,
        distance_threshold=args.distance_threshold,
        num_iterations=args.iterations,
    )

    print("=" * 60)
    print(f"Ground removal for KITTI frame: {args.frame}")
    print("=" * 60)
    print(f"Raw points:              {len(raw_points):>8}")
    print(f"After ROI crop:          {len(cropped_points):>8}")
    print(f"After downsampling:      {len(downsampled_points):>8}")
    print(f"Ground points:           {len(result.ground_points):>8}")
    print(f"Non-ground points:       {len(result.non_ground_points):>8}")

    ground_ratio = 100.0 * len(result.ground_points) / max(len(downsampled_points), 1)
    non_ground_ratio = 100.0 * len(result.non_ground_points) / max(len(downsampled_points), 1)

    print(f"Ground ratio:            {ground_ratio:>7.2f}%")
    print(f"Non-ground ratio:        {non_ground_ratio:>7.2f}%")
    print(
        "Plane model [a, b, c, d]: "
        f"{np.array2string(result.plane_model, precision=4)}"
    )

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plot_bev(cropped_points, "Cropped ROI")

    plt.subplot(1, 3, 2)
    plot_bev(downsampled_points, "Downsampled ROI")

    plt.subplot(1, 3, 3)
    plot_ground_vs_nonground(
        result.ground_points,
        result.non_ground_points,
        "Ground vs Non-ground",
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
