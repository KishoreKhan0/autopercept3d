from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from autopercept3d.geometry.kitti_transforms import (
    kitti_label_corners_camera,
    parse_kitti_label,
    projection_matrix,
    transform_points,
    velo_to_rect_matrix,
)
from autopercept3d.perception.bounding_boxes import AxisAlignedBox3D, get_3d_box_corners


BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
]


def project_rect_camera_points_to_image(
    points_rect_camera: np.ndarray,
    calibration: dict[str, np.ndarray],
    camera: str = "P2",
) -> tuple[np.ndarray, np.ndarray]:
    """Project rectified camera-frame 3D points into image pixel coordinates."""
    if len(points_rect_camera) == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)

    p = projection_matrix(calibration[camera])

    ones = np.ones((points_rect_camera.shape[0], 1), dtype=np.float64)
    points_h = np.hstack([points_rect_camera, ones])

    projected = (p @ points_h.T).T
    depths = points_rect_camera[:, 2]

    pixels = np.empty((points_rect_camera.shape[0], 2), dtype=np.float64)
    pixels[:, 0] = projected[:, 0] / (projected[:, 2] + 1e-8)
    pixels[:, 1] = projected[:, 1] / (projected[:, 2] + 1e-8)

    return pixels, depths


def project_velodyne_box_to_image(
    box: AxisAlignedBox3D,
    calibration: dict[str, np.ndarray],
    camera: str = "P2",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project an AutoPercept3D proposal box from Velodyne coordinates
    into the camera image.
    """
    corners_velodyne = get_3d_box_corners(box)
    corners_rect = transform_points(corners_velodyne, velo_to_rect_matrix(calibration))
    return project_rect_camera_points_to_image(corners_rect, calibration, camera=camera)


def project_kitti_label_3d_box_to_image(
    raw_label: dict,
    calibration: dict[str, np.ndarray],
    camera: str = "P2",
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Project a KITTI ground-truth 3D box into the camera image.
    KITTI labels are already in rectified camera coordinates.
    """
    label = parse_kitti_label(raw_label)
    corners_rect = kitti_label_corners_camera(label)
    pixels, depths = project_rect_camera_points_to_image(corners_rect, calibration, camera=camera)
    return pixels, depths, label.object_type


def projected_box_overlaps_image(
    pixels: np.ndarray,
    image_width: int,
    image_height: int,
) -> bool:
    """Return True if the projected box has any overlap with the image rectangle."""
    x = pixels[:, 0]
    y = pixels[:, 1]

    return (
        x.max() >= 0
        and x.min() <= image_width
        and y.max() >= 0
        and y.min() <= image_height
    )


def projected_box_is_reasonable(
    pixels: np.ndarray,
    depths: np.ndarray,
    image_width: int,
    image_height: int,
    min_depth: float = 0.1,
    max_extent_ratio: float = 1.25,
) -> bool:
    """
    Filter projected boxes before drawing.

    Some cluster boxes are not real objects. When projected into the camera,
    they can become enormous and force Matplotlib to zoom out. This function
    rejects those boxes.
    """
    if len(pixels) != 8 or len(depths) != 8:
        return False

    if not np.isfinite(pixels).all():
        return False

    if not np.all(depths > min_depth):
        return False

    if not projected_box_overlaps_image(pixels, image_width, image_height):
        return False

    pixel_width = float(pixels[:, 0].max() - pixels[:, 0].min())
    pixel_height = float(pixels[:, 1].max() - pixels[:, 1].min())

    if pixel_width > image_width * max_extent_ratio:
        return False

    if pixel_height > image_height * max_extent_ratio:
        return False

    return True


def draw_projected_3d_box(
    ax: Axes,
    pixels: np.ndarray,
    color: str,
    linewidth: float = 2.0,
    linestyle: str = "-",
    label: str | None = None,
) -> None:
    """Draw projected 3D box edges onto an image axis."""
    for edge_index, (start, end) in enumerate(BOX_EDGES):
        ax.plot(
            [pixels[start, 0], pixels[end, 0]],
            [pixels[start, 1], pixels[end, 1]],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label if edge_index == 0 else None,
            clip_on=True,
        )


def draw_camera_3d_box_overlay(
    image: np.ndarray,
    calibration: dict[str, np.ndarray],
    proposal_boxes: list[AxisAlignedBox3D],
    labels: list[dict] | None = None,
    camera: str = "P2",
    max_proposals: int = 25,
    title: str = "Projected 3D boxes",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Draw proposal and KITTI GT 3D boxes on the camera image.

    V2 intentionally locks the axes to image coordinates AFTER drawing.
    This prevents invalid/noisy proposal boxes from shrinking the image.
    """
    image_height = int(image.shape[0])
    image_width = int(image.shape[1])

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.imshow(image)

    drawn_proposals = 0
    skipped_proposals = 0

    for box in proposal_boxes[:max_proposals]:
        pixels, depths = project_velodyne_box_to_image(box, calibration, camera=camera)

        if not projected_box_is_reasonable(
            pixels,
            depths,
            image_width=image_width,
            image_height=image_height,
            max_extent_ratio=1.25,
        ):
            skipped_proposals += 1
            continue

        draw_projected_3d_box(
            ax,
            pixels,
            color="tab:blue",
            linewidth=1.5,
            linestyle="-",
            label="proposal 3D box" if drawn_proposals == 0 else None,
        )

        center = pixels.mean(axis=0)
        if 0 <= center[0] < image_width and 0 <= center[1] < image_height:
            ax.text(
                center[0],
                center[1],
                str(box.cluster_id),
                fontsize=8,
                color="tab:blue",
                bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"},
            )

        drawn_proposals += 1

    drawn_gt = 0

    if labels is not None:
        for raw_label in labels:
            if raw_label["type"] == "DontCare":
                continue

            pixels, depths, object_type = project_kitti_label_3d_box_to_image(
                raw_label,
                calibration,
                camera=camera,
            )

            if not projected_box_is_reasonable(
                pixels,
                depths,
                image_width=image_width,
                image_height=image_height,
                max_extent_ratio=3.0,
            ):
                continue

            draw_projected_3d_box(
                ax,
                pixels,
                color="tab:red",
                linewidth=2.5,
                linestyle="--",
                label="KITTI GT 3D box" if drawn_gt == 0 else None,
            )

            center = pixels.mean(axis=0)
            if 0 <= center[0] < image_width and 0 <= center[1] < image_height:
                ax.text(
                    center[0],
                    center[1],
                    object_type,
                    fontsize=9,
                    color="tab:red",
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                )

            drawn_gt += 1

    ax.set_title(
        f"{title} | proposals drawn={drawn_proposals} | "
        f"proposals skipped={skipped_proposals} | GT drawn={drawn_gt}"
    )

    # The critical line: force axes back to the original camera image.
    ax.set_xlim(0, image_width)
    ax.set_ylim(image_height, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    if drawn_proposals > 0 or drawn_gt > 0:
        ax.legend(loc="upper right")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Saved fixed camera 3D box overlay to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
