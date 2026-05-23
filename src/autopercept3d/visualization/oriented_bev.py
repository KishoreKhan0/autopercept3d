from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from autopercept3d.perception.oriented_boxes import OrientedBox3D


def plot_oriented_boxes_bev(
    boxes: list[OrientedBox3D],
    show_ids: bool = True,
    linewidth: float = 2.0,
    draw_heading: bool = True,
) -> None:
    """
    Plot PCA-oriented 3D boxes in bird's-eye view.

    These boxes are oriented in x/y using PCA and vertical in z.
    Only the BEV footprint is drawn here.

    Args:
        boxes:
            List of OrientedBox3D objects.

        show_ids:
            If True, draw the cluster id at the box center.

        linewidth:
            Line width for box edges.

        draw_heading:
            If True, draw a short line showing the estimated yaw direction.
    """
    if not boxes:
        return

    for box in boxes:
        corners = box.corners_bev

        plt.plot(
            corners[:, 0],
            corners[:, 1],
            linewidth=linewidth,
            label="oriented box" if box == boxes[0] else None,
        )

        center = box.center_xyz[:2]

        if show_ids:
            plt.text(
                center[0],
                center[1],
                str(box.cluster_id),
                fontsize=8,
                ha="center",
                va="center",
            )

        if draw_heading:
            direction = np.array(
                [np.cos(box.yaw_rad), np.sin(box.yaw_rad)],
                dtype=np.float64,
            )
            heading_length = max(float(box.size_lwh[0]) * 0.35, 0.5)
            end = center + direction * heading_length

            plt.plot(
                [center[0], end[0]],
                [center[1], end[1]],
                linewidth=max(linewidth * 0.75, 1.0),
                alpha=0.9,
            )


def plot_axis_or_oriented_boxes_bev(
    axis_boxes,
    oriented_boxes: list[OrientedBox3D],
    box_type: str = "axis",
    show_ids: bool = True,
) -> None:
    """
    Convenience function used by dashboard scripts.

    box_type values:
        "axis"      -> draw existing axis-aligned boxes
        "oriented"  -> draw PCA-oriented boxes

    This function imports the existing axis-aligned renderer lazily to avoid
    circular imports.
    """
    if box_type == "oriented":
        plot_oriented_boxes_bev(oriented_boxes, show_ids=show_ids)
        return

    from autopercept3d.visualization.bev import plot_proposal_boxes_bev

    plot_proposal_boxes_bev(axis_boxes, show_ids=show_ids)
