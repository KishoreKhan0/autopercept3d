from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def oriented_box_corners_bev(center_xy, size_lw, yaw_rad):
    """
    Return 4 BEV corners for an oriented box in x-forward / y-left coordinates.
    """
    cx, cy = float(center_xy[0]), float(center_xy[1])
    length, width = float(size_lw[0]), float(size_lw[1])

    dx = length / 2.0
    dy = width / 2.0

    local = np.array(
        [
            [dx, dy],
            [dx, -dy],
            [-dx, -dy],
            [-dx, dy],
        ],
        dtype=float,
    )

    c = np.cos(yaw_rad)
    s = np.sin(yaw_rad)
    rot = np.array([[c, -s], [s, c]], dtype=float)

    world = local @ rot.T
    world[:, 0] += cx
    world[:, 1] += cy
    return world


def axis_box_corners_bev(center_xy, size_lw):
    """
    Return 4 BEV corners for an axis-aligned box.
    """
    cx, cy = float(center_xy[0]), float(center_xy[1])
    length, width = float(size_lw[0]), float(size_lw[1])

    dx = length / 2.0
    dy = width / 2.0

    return np.array(
        [
            [cx + dx, cy + dy],
            [cx + dx, cy - dy],
            [cx - dx, cy - dy],
            [cx - dx, cy + dy],
        ],
        dtype=float,
    )


def _rows_by_frame(rows: list[dict]) -> dict[str, list[dict]]:
    index: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        index[str(row["frame_id"])].append(row)

    for frame_rows in index.values():
        frame_rows.sort(key=lambda item: (int(item["track_id"]), int(item["frame_index"])))

    return dict(index)


def _rows_by_track(rows: list[dict]) -> dict[int, list[dict]]:
    index: dict[int, list[dict]] = defaultdict(list)

    for row in rows:
        index[int(row["track_id"])].append(row)

    for track_rows in index.values():
        track_rows.sort(key=lambda item: int(item["frame_index"]))

    return dict(index)


def _history_until_frame(track_rows: list[dict], frame_index: int, history_length: int) -> list[dict]:
    valid = [row for row in track_rows if int(row["frame_index"]) <= frame_index]
    return valid[-history_length:]


def _proposal_rows_from_result(result, box_type: str) -> list[dict]:
    rows: list[dict] = []

    if box_type == "oriented":
        boxes = getattr(result, "oriented_boxes", [])
        for box in boxes:
            rows.append(
                {
                    "center_x": float(box.center_xyz[0]),
                    "center_y": float(box.center_xyz[1]),
                    "length": float(box.size_lwh[0]),
                    "width": float(box.size_lwh[1]),
                    "yaw_rad": float(box.yaw_rad),
                }
            )
    else:
        boxes = getattr(result, "boxes", [])
        for box in boxes:
            if hasattr(box, "size_xyz"):
                size = box.size_xyz
            elif hasattr(box, "size_lwh"):
                size = box.size_lwh
            else:
                size = np.asarray(box.max_xyz) - np.asarray(box.min_xyz)

            rows.append(
                {
                    "center_x": float(box.center_xyz[0]),
                    "center_y": float(box.center_xyz[1]),
                    "length": float(size[0]),
                    "width": float(size[1]),
                    "yaw_rad": 0.0,
                }
            )

    return rows


def render_bev_tracklet_frame(
    result,
    stable_rows: list[dict],
    frame_id: str,
    frame_index: int,
    output_path,
    box_type: str = "oriented",
    history_length: int = 6,
    top_k_tracks: int = 25,
    x_limits=(0.0, 70.0),
    y_limits=(-25.0, 35.0),
):
    """
    Render one BEV frame with current proposal boxes plus stable tracklet trails.
    """
    stable_rows = list(stable_rows)
    rows_by_track = _rows_by_track(stable_rows)
    rows_by_frame = _rows_by_frame(stable_rows)

    track_lengths = sorted(
        ((track_id, len(track_rows)) for track_id, track_rows in rows_by_track.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    selected_track_ids = {track_id for track_id, _ in track_lengths[:top_k_tracks]}
    current_rows = [
        row
        for row in rows_by_frame.get(frame_id, [])
        if int(row["track_id"]) in selected_track_ids
    ]

    proposal_rows = _proposal_rows_from_result(result=result, box_type=box_type)

    plt.figure(figsize=(10, 8))

    # Draw current proposals lightly in the background.
    for row in proposal_rows:
        if box_type == "oriented":
            corners = oriented_box_corners_bev(
                center_xy=[row["center_x"], row["center_y"]],
                size_lw=[row["length"], row["width"]],
                yaw_rad=row["yaw_rad"],
            )
        else:
            corners = axis_box_corners_bev(
                center_xy=[row["center_x"], row["center_y"]],
                size_lw=[row["length"], row["width"]],
            )

        closed = np.vstack([corners, corners[0]])
        plt.plot(closed[:, 0], closed[:, 1], linewidth=0.7, alpha=0.20)

    # Plot stable track history.
    plotted_tracks = 0

    for row in current_rows:
        track_id = int(row["track_id"])
        track_rows = rows_by_track[track_id]
        history = _history_until_frame(
            track_rows=track_rows,
            frame_index=frame_index,
            history_length=history_length,
        )

        xs = [float(item["center_x"]) for item in history]
        ys = [float(item["center_y"]) for item in history]

        if len(xs) >= 2:
            plt.plot(xs, ys, marker="o", linewidth=2.0, markersize=4)

        plt.scatter([xs[0]], [ys[0]], marker="s", s=28)
        plt.text(xs[-1], ys[-1], str(track_id), fontsize=9)

        plotted_tracks += 1

    plt.title(
        f"AutoPercept3D BEV stable tracklets - frame {frame_id} | "
        f"visible stable tracks={plotted_tracks}"
    )
    plt.xlabel("x forward [m]")
    plt.ylabel("y left/right [m]")
    plt.xlim(*x_limits)
    plt.ylim(*y_limits)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
