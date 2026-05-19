import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autopercept3d.datasets.kitti import KITTIDataset


def sample_points(points: np.ndarray, max_points: int, seed: int = 42) -> np.ndarray:
    """
    Matplotlib becomes slow if we draw all 100k+ LiDAR points.
    So for visualization only, we randomly sample a smaller number of points.
    The original point cloud is not modified.
    """
    if points.shape[0] <= max_points:
        return points

    rng = np.random.default_rng(seed)
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[indices]


def set_axes_equal(ax) -> None:
    """
    Make 3D axes use roughly equal scale, so the point cloud is not visually distorted.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_lidar(points: np.ndarray, frame_id: str, max_points: int, save_path: str | None) -> None:
    sampled = sample_points(points, max_points=max_points)

    xyz = sampled[:, :3]
    intensity = sampled[:, 3]

    # KITTI LiDAR coordinate system:
    # x = forward, y = left/right, z = up/down
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        x,
        y,
        z,
        c=intensity,
        s=0.5,
        cmap="viridis",
        alpha=0.8,
    )

    ax.set_title(f"AutoPercept3D - Raw KITTI LiDAR Frame {frame_id}")
    ax.set_xlabel("x forward [m]")
    ax.set_ylabel("y left/right [m]")
    ax.set_zlabel("z up/down [m]")

    fig.colorbar(scatter, ax=ax, shrink=0.6, label="LiDAR intensity")

    # A useful initial driving-scene view.
    ax.view_init(elev=20, azim=-120)
    set_axes_equal(ax)

    plt.tight_layout()

    if save_path:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)
        print(f"Saved visualization to: {output_path}")

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a raw KITTI LiDAR frame using Matplotlib."
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
        "--max-points",
        type=int,
        default=20000,
        help="Maximum number of LiDAR points to draw. Lower is faster.",
    )

    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save the visualization as an image.",
    )

    args = parser.parse_args()

    dataset = KITTIDataset(dataset_root=args.dataset_root)
    points = dataset.load_point_cloud(args.frame)

    print("=" * 60)
    print(f"Visualizing raw KITTI LiDAR frame: {args.frame}")
    print("=" * 60)
    print(f"Full point cloud shape: {points.shape}")
    print(f"Drawing up to {args.max_points} points for speed")
    print("Point format: x, y, z, intensity")
    print()
    print("Matplotlib window controls:")
    print("  Left mouse drag = rotate")
    print("  Mouse wheel     = zoom")
    print("  Close window    = finish")
    print()

    visualize_lidar(
        points=points,
        frame_id=args.frame,
        max_points=args.max_points,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
