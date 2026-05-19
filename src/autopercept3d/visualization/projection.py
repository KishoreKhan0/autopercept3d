from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from autopercept3d.geometry.projection import ProjectionResult, depth_to_normalized_values
from autopercept3d.visualization.camera import parse_2d_bbox


def draw_projected_lidar_on_image(
    image: np.ndarray,
    projection: ProjectionResult,
    labels: list[dict] | None = None,
    title: str = "LiDAR projected into camera image",
    max_points: int = 25000,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Draw camera image with projected LiDAR points.

    Points are colored by depth:
        near/far values are mapped through Matplotlib's color map.
    """
    pixels = projection.pixels
    depths = projection.depths

    if len(pixels) > max_points:
        indices = np.linspace(0, len(pixels) - 1, max_points).astype(int)
        pixels = pixels[indices]
        depths = depths[indices]

    plt.figure(figsize=(13, 6))
    plt.imshow(image)

    if len(pixels) > 0:
        colors = depth_to_normalized_values(depths)
        scatter = plt.scatter(
            pixels[:, 0],
            pixels[:, 1],
            c=colors,
            s=1.0,
            alpha=0.75,
            cmap="turbo",
        )
        colorbar = plt.colorbar(scatter, fraction=0.025, pad=0.01)
        colorbar.set_label("Normalized depth")

    if labels is not None:
        for label in labels:
            object_type = label["type"]

            if object_type == "DontCare":
                continue

            left, top, right, bottom = parse_2d_bbox(label)
            rect = Rectangle(
                (left, top),
                right - left,
                bottom - top,
                fill=False,
                linewidth=2,
            )
            plt.gca().add_patch(rect)
            plt.text(
                left,
                max(top - 5, 0),
                object_type,
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )

    plt.title(f"{title} | projected points={len(projection.pixels)}")
    plt.axis("off")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Saved LiDAR projection image to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
