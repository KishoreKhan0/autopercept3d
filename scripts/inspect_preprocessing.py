import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autopercept3d.datasets.kitti import KITTIDataset
from autopercept3d.perception.preprocessing import (
    crop_roi,
    filter_finite_points,
    voxel_downsample,
)


def plot_bev(points: np.ndarray, title: str, max_points: int = 15000) -> None:
    """
    Draw a simple bird's-eye-view (BEV) scatter plot.

    BEV uses:
    - x on horizontal axis (forward)
    - y on vertical axis (left/right)
    """
    if len(points) == 0:
        plt.title(title)
        plt.xlabel("x (forward, m)")
        plt.ylabel("y (left/right, m)")
        plt.grid(True, alpha=0.3)
        return

    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points).astype(int)
        points = points[indices]

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    plt.scatter(x, y, c=z, s=1, alpha=0.8)
    plt.title(title)
    plt.xlabel("x (forward, m)")
    plt.ylabel("y (left/right, m)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect ROI cropping and voxel downsampling on a KITTI frame."
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
        help="Voxel size in meters for downsampling.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save the comparison image.",
    )
    args = parser.parse_args()

    dataset = KITTIDataset(dataset_root=args.dataset_root)
    raw_points = dataset.load_point_cloud(args.frame)

    finite_points = filter_finite_points(raw_points)
    cropped_points = crop_roi(finite_points)
    downsampled_points = voxel_downsample(cropped_points, voxel_size=args.voxel_size)

    print("=" * 60)
    print(f"Preprocessing inspection for KITTI frame: {args.frame}")
    print("=" * 60)
    print(f"Raw points:         {len(raw_points):>8}")
    print(f"Finite points:      {len(finite_points):>8}")
    print(f"After ROI crop:     {len(cropped_points):>8}")
    print(f"After downsampling: {len(downsampled_points):>8}")
    print(f"Voxel size:         {args.voxel_size} m")

    reduction_crop = 100.0 * (1.0 - len(cropped_points) / max(len(finite_points), 1))
    reduction_voxel = 100.0 * (1.0 - len(downsampled_points) / max(len(cropped_points), 1))

    print(f"Crop reduction:     {reduction_crop:>7.2f}%")
    print(f"Voxel reduction:    {reduction_voxel:>7.2f}%")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plot_bev(cropped_points, "Cropped ROI")

    plt.subplot(1, 2, 2)
    plot_bev(downsampled_points, f"Downsampled (voxel={args.voxel_size} m)")

    plt.tight_layout()

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Saved comparison image to: {save_path}")

    plt.show()


if __name__ == "__main__":
    main()
