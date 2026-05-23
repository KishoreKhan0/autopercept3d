from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class TrackFilterConfig:
    """
    Configuration for post-processing simple tracking results.

    The first tracker intentionally assigns IDs to every proposal. This filter
    keeps only more useful trajectories for reporting and visualization.
    """

    min_track_length: int = 3
    top_k_tracks: int = 40
    x_min: float = 0.0
    x_max: float = 80.0
    y_abs_max: float = 40.0
    min_length: float = 0.3
    max_length: float = 10.0
    min_width: float = 0.2
    max_width: float = 6.0
    min_height: float = 0.2
    max_height: float = 4.0


def read_track_csv(path: Path) -> list[dict]:
    """
    Read tracker CSV records.
    """
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict]) -> None:
    """
    Write dictionaries to CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    """
    Write JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def group_tracks(rows: list[dict]) -> dict[int, list[dict]]:
    """
    Group CSV rows by track ID.
    """
    grouped: dict[int, list[dict]] = {}

    for row in rows:
        track_id = int(row["track_id"])
        grouped.setdefault(track_id, []).append(row)

    for track_rows in grouped.values():
        track_rows.sort(key=lambda item: item["frame_id"])

    return grouped


def _float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except ValueError:
        return default


def row_passes_spatial_filter(row: dict, config: TrackFilterConfig) -> bool:
    """
    Filter one detection row by region and object-size sanity limits.
    """
    x = _float(row, "center_x")
    y = _float(row, "center_y")

    length = _float(row, "length")
    width = _float(row, "width")
    height = _float(row, "height")

    if x < config.x_min or x > config.x_max:
        return False

    if abs(y) > config.y_abs_max:
        return False

    if length < config.min_length or length > config.max_length:
        return False

    if width < config.min_width or width > config.max_width:
        return False

    if height < config.min_height or height > config.max_height:
        return False

    return True


def track_passes_filter(track_rows: list[dict], config: TrackFilterConfig) -> bool:
    """
    Keep tracks that are long enough and mostly pass spatial/object-size limits.
    """
    if len(track_rows) < config.min_track_length:
        return False

    passed_rows = [
        row
        for row in track_rows
        if row_passes_spatial_filter(row, config)
    ]

    # Require most detections in the trajectory to pass.
    return len(passed_rows) >= config.min_track_length


def filter_tracks(
    rows: list[dict],
    config: TrackFilterConfig,
) -> tuple[list[dict], dict[int, list[dict]]]:
    """
    Filter tracks and return flattened rows plus grouped kept tracks.
    """
    grouped = group_tracks(rows)

    kept_tracks: dict[int, list[dict]] = {}

    for track_id, track_rows in grouped.items():
        if track_passes_filter(track_rows, config):
            kept_tracks[track_id] = track_rows

    sorted_tracks = sorted(
        kept_tracks.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    )

    if config.top_k_tracks > 0:
        sorted_tracks = sorted_tracks[: config.top_k_tracks]

    final_tracks = dict(sorted_tracks)

    filtered_rows: list[dict] = []

    for track_id, track_rows in final_tracks.items():
        for row in track_rows:
            row = dict(row)
            row["filtered_track_length"] = len(track_rows)
            filtered_rows.append(row)

    filtered_rows.sort(key=lambda row: (int(row["track_id"]), row["frame_id"]))

    return filtered_rows, final_tracks


def summarize_filtered_tracks(
    raw_rows: list[dict],
    filtered_rows: list[dict],
    filtered_tracks: dict[int, list[dict]],
) -> dict:
    """
    Build summary statistics for filtered tracks.
    """
    raw_tracks = group_tracks(raw_rows)

    raw_lengths = [len(rows) for rows in raw_tracks.values()]
    filtered_lengths = [len(rows) for rows in filtered_tracks.values()]

    if filtered_lengths:
        longest = max(filtered_lengths)
        mean_length = float(np.mean(filtered_lengths))
        median_length = float(np.median(filtered_lengths))
    else:
        longest = 0
        mean_length = 0.0
        median_length = 0.0

    return {
        "raw_records": len(raw_rows),
        "raw_tracks": len(raw_tracks),
        "raw_longest_track_length": max(raw_lengths) if raw_lengths else 0,
        "filtered_records": len(filtered_rows),
        "filtered_tracks": len(filtered_tracks),
        "filtered_longest_track_length": longest,
        "filtered_mean_track_length": mean_length,
        "filtered_median_track_length": median_length,
    }


def make_track_length_table(filtered_tracks: dict[int, list[dict]]) -> list[dict]:
    """
    Create a per-track summary table sorted by track length.
    """
    rows: list[dict] = []

    for track_id, track_rows in filtered_tracks.items():
        xs = [_float(row, "center_x") for row in track_rows]
        ys = [_float(row, "center_y") for row in track_rows]

        rows.append(
            {
                "track_id": track_id,
                "track_length": len(track_rows),
                "first_frame": track_rows[0]["frame_id"],
                "last_frame": track_rows[-1]["frame_id"],
                "start_x": xs[0],
                "start_y": ys[0],
                "end_x": xs[-1],
                "end_y": ys[-1],
                "displacement_m": float(np.linalg.norm([xs[-1] - xs[0], ys[-1] - ys[0]])),
            }
        )

    rows.sort(key=lambda item: item["track_length"], reverse=True)
    return rows


def plot_filtered_tracks_bev(
    filtered_tracks: dict[int, list[dict]],
    output_path: Path,
    title: str = "Filtered BEV tracks",
) -> None:
    """
    Plot filtered track center trajectories in BEV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 8))

    for track_id, track_rows in filtered_tracks.items():
        xs = [_float(row, "center_x") for row in track_rows]
        ys = [_float(row, "center_y") for row in track_rows]

        plt.plot(xs, ys, marker="o", linewidth=1.8, markersize=4)

        # Label at the final point.
        plt.text(xs[-1], ys[-1], str(track_id), fontsize=8)

        # Add a small start marker.
        plt.scatter([xs[0]], [ys[0]], marker="s", s=25)

    plt.title(f"{title} | tracks={len(filtered_tracks)}")
    plt.xlabel("x forward [m]")
    plt.ylabel("y left/right [m]")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()


def save_filtered_tracking_report(
    raw_tracks_csv: Path,
    output_dir: Path,
    config: TrackFilterConfig,
) -> dict:
    """
    Load raw tracks.csv, filter trajectories, save CSV/JSON/plot outputs.
    """
    raw_rows = read_track_csv(raw_tracks_csv)
    filtered_rows, filtered_tracks = filter_tracks(raw_rows, config)

    summary = summarize_filtered_tracks(raw_rows, filtered_rows, filtered_tracks)
    track_table = make_track_length_table(filtered_tracks)

    output_dir.mkdir(parents=True, exist_ok=True)

    filtered_csv = output_dir / "filtered_tracks.csv"
    track_table_csv = output_dir / "filtered_track_summary.csv"
    summary_json = output_dir / "filtered_tracking_summary.json"
    plot_path = output_dir / "filtered_tracks_bev.png"

    write_csv(filtered_csv, filtered_rows)
    write_csv(track_table_csv, track_table)

    payload = {
        "config": {
            "min_track_length": config.min_track_length,
            "top_k_tracks": config.top_k_tracks,
            "x_min": config.x_min,
            "x_max": config.x_max,
            "y_abs_max": config.y_abs_max,
            "min_length": config.min_length,
            "max_length": config.max_length,
            "min_width": config.min_width,
            "max_width": config.max_width,
            "min_height": config.min_height,
            "max_height": config.max_height,
        },
        "summary": summary,
        "tracks": track_table,
    }

    write_json(summary_json, payload)

    plot_filtered_tracks_bev(
        filtered_tracks,
        output_path=plot_path,
        title=f"Filtered BEV tracks, min length {config.min_track_length}",
    )

    return {
        "summary": summary,
        "filtered_tracks_csv": str(filtered_csv),
        "filtered_track_summary_csv": str(track_table_csv),
        "summary_json": str(summary_json),
        "plot_path": str(plot_path),
    }
