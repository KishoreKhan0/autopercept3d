from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def parse_2d_bbox(label: dict) -> tuple[float, float, float, float]:
    """
    Parse KITTI 2D bounding box from a label dictionary.

    KITTI bbox format:
        left, top, right, bottom
    """
    left, top, right, bottom = [float(value) for value in label["bbox_2d"]]
    return left, top, right, bottom


def draw_kitti_2d_labels(
    image: np.ndarray,
    labels: list[dict],
    title: str = "KITTI Camera Image with 2D Labels",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Draw KITTI camera image with 2D label boxes.

    This uses KITTI's provided 2D bounding boxes from label_2.
    It does not project LiDAR yet. Projection comes in a later step.
    """
    plt.figure(figsize=(12, 5))
    plt.imshow(image)
    ax = plt.gca()

    valid_count = 0

    for label in labels:
        object_type = label["type"]

        if object_type == "DontCare":
            continue

        left, top, right, bottom = parse_2d_bbox(label)
        width = right - left
        height = bottom - top

        rect = Rectangle(
            (left, top),
            width,
            height,
            fill=False,
            linewidth=2,
        )

        ax.add_patch(rect)
        ax.text(
            left,
            max(top - 5, 0),
            object_type,
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

        valid_count += 1

    plt.title(f"{title} | labels={valid_count}")
    plt.axis("off")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Saved camera label image to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def draw_camera_panel(
    ax,
    image: np.ndarray,
    labels: list[dict],
    title: str = "Camera image + KITTI labels",
) -> None:
    """
    Draw camera image with 2D labels into an existing Matplotlib axis.

    Used by combined frame summary visualizations.
    """
    ax.imshow(image)

    valid_count = 0

    for label in labels:
        object_type = label["type"]

        if object_type == "DontCare":
            continue

        left, top, right, bottom = parse_2d_bbox(label)
        width = right - left
        height = bottom - top

        rect = Rectangle(
            (left, top),
            width,
            height,
            fill=False,
            linewidth=2,
        )

        ax.add_patch(rect)
        ax.text(
            left,
            max(top - 5, 0),
            object_type,
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

        valid_count += 1

    ax.set_title(f"{title} | labels={valid_count}")
    ax.axis("off")
