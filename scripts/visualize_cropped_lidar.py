"""
Visualize a cropped KITTI LiDAR point cloud using Matplotlib.

This script is part of AutoPercept3D Step 3.
It loads one KITTI LiDAR frame, crops it to a useful driving region,
and visualizes both a 3D point cloud and a bird's-eye-view plot.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autopercept3d.datasets.kitti import KITTIDataset


def crop_point_cloud(
    points: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> np.ndarray:
    """
    Crop KITTI LiDAR points to a useful driving region.

    KITTI LiDAR point format:
        x = forward direction in meters
        y = left/right direction in meters
        z = vertical direction in meters
        intensity = laser return intensity
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    mask = (
        (x >= x_min)
        & (x <= x_max)
        & (y >= y_min)
        & (y <= y_max)
        & (z >= z_min)
        & (z <= z_max)
    )

    return points[mask]


def downsample_for_display(points: np.ndarray, max_points: int) -> np.ndarray:
    """Randomly downsample points only for faster plotting."""
    if points.shape[0] <= max_points:
        return points

    indices = np.random.choice(points.shape[0], size=max_points, replace=False)
    return points[indices]


def plot_cropped_3d(points: np.ndarray, frame_id: str, save_path: str | None = None) -> None:
    """Show a 3D scatter plot of the cropped LiDAR points."""
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    intensity = points[:, 3]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        x,
        y,
        z,
        c=intensity,
        cmap="viridis",
        s=0.5,
        alpha=0.8,
    )

    ax.set_title(f"AutoPercept3D - Cropped KITTI LiDAR Frame {frame_id}")
    ax.set_xlabel("x forward [m]")
    ax.set_ylabel("y left/right [m]")
    ax.set_zlabel("z up/down [m]")

    ax.set_xlim(0, 70)
    ax.set_ylim(-40, 40)
    ax.set_zlim(-3, 3)

    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.65, pad=0.1)
    colorbar.set_label("LiDAR intensity")

    plt.tight_layout()

    if save_path:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)
        print(f"Saved 3D cropped visualization to: {output_path}")

    plt.show()


def plot_bev(points: np.ndarray, frame_id: str, save_path: str | None = None) -> None:
    """
    Show a bird's-eye-view plot.

    BEV means looking at the LiDAR scene from above:
        horizontal axis = y left/right
        vertical axis = x forward
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(
        y,
        x,
        c=z,
        cmap="viridis",
        s=0.5,
        alpha=0.8,
    )

    plt.title(f"AutoPercept3D - BEV Cropped LiDAR Frame {frame_id}")
    plt.xlabel("y left/right [m]")
    plt.ylabel("x forward [m]")
    plt.xlim(-40, 40)
    plt.ylim(0, 70)
    plt.grid(True, linewidth=0.3)
    plt.gca().set_aspect("equal", adjustable="box")

    colorbar = plt.colorbar(scatter)
    colorbar.set_label("height z [m]")

    plt.tight_layout()

    if save_path:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)
        print(f"Saved BEV visualization to: {output_path}")

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize cropped KITTI LiDAR point cloud."
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

    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=70.0)
    parser.add_argument("--y-min", type=float, default=-40.0)
    parser.add_argument("--y-max", type=float, default=40.0)
    parser.add_argument("--z-min", type=float, default=-3.0)
    parser.add_argument("--z-max", type=float, default=3.0)

    parser.add_argument(
        "--max-points",
        type=int,
        default=20000,
        help="Maximum number of points to draw for speed.",
    )

    parser.add_argument(
        "--view",
        choices=["3d", "bev"],
        default="bev",
        help="Choose 3D view or bird's-eye view.",
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

    cropped_points = crop_point_cloud(
        points=raw_points,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        z_min=args.z_min,
        z_max=args.z_max,
    )

    display_points = downsample_for_display(cropped_points, args.max_points)

    print("=" * 60)
    print(f"Cropped KITTI LiDAR frame: {args.frame}")
    print("=" * 60)
    print(f"Raw point cloud shape: {raw_points.shape}")
    print(f"Cropped point cloud shape: {cropped_points.shape}")
    print(f"Drawing point cloud shape: {display_points.shape}")
    print()
    print("Crop region:")
    print(f"  x forward:    {args.x_min} to {args.x_max} m")
    print(f"  y left/right: {args.y_min} to {args.y_max} m")
    print(f"  z height:     {args.z_min} to {args.z_max} m")
    print()

    if args.view == "3d":
        plot_cropped_3d(display_points, args.frame, args.save)
    else:
        plot_bev(display_points, args.frame, args.save)


if __name__ == "__main__":
    main()
