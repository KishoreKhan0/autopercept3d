from __future__ import annotations

from pathlib import Path
import json
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd


def load_optional_csv(path_like: str | Path) -> pd.DataFrame | None:
    path = Path(path_like)
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_optional_json(path_like: str | Path) -> dict | None:
    path = Path(path_like)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_available_frame_ids(*frames_dfs: pd.DataFrame | None) -> list[int]:
    frame_ids: set[int] = set()
    for df in frames_dfs:
        if df is None or df.empty or "frame_id" not in df.columns:
            continue
        frame_ids.update(int(value) for value in df["frame_id"].dropna().unique().tolist())
    return sorted(frame_ids)


def summarize_tracklets(tracklets_df: pd.DataFrame | None) -> dict:
    if tracklets_df is None or tracklets_df.empty:
        return {
            "records": 0,
            "tracks": 0,
            "frames": 0,
            "mean_track_length": 0.0,
            "mean_score": 0.0,
            "mean_points": 0.0,
        }

    grouped = tracklets_df.groupby("track_id")
    return {
        "records": int(len(tracklets_df)),
        "tracks": int(tracklets_df["track_id"].nunique()),
        "frames": int(tracklets_df["frame_id"].nunique()),
        "mean_track_length": float(grouped.size().mean()),
        "mean_score": float(tracklets_df["score"].mean()) if "score" in tracklets_df.columns else 0.0,
        "mean_points": float(tracklets_df["num_points"].mean()) if "num_points" in tracklets_df.columns else 0.0,
    }


def compute_track_metrics_from_rows(tracklets_df: pd.DataFrame) -> pd.DataFrame:
    if tracklets_df.empty:
        return pd.DataFrame(
            columns=[
                "track_id",
                "track_length",
                "first_frame_id",
                "last_frame_id",
                "path_length",
                "net_displacement",
                "straightness_ratio",
                "mean_score",
                "mean_points",
            ]
        )

    rows: list[dict] = []

    for track_id, group in tracklets_df.groupby("track_id"):
        group = group.sort_values("frame_id").reset_index(drop=True)

        x = group["center_x"].astype(float).to_numpy()
        y = group["center_y"].astype(float).to_numpy()

        if len(group) > 1:
            dx = np.diff(x)
            dy = np.diff(y)
            step_dist = np.sqrt(dx ** 2 + dy ** 2)
            path_length = float(step_dist.sum())
            max_step_distance = float(step_dist.max())
            mean_step_distance = float(step_dist.mean())
        else:
            path_length = 0.0
            max_step_distance = 0.0
            mean_step_distance = 0.0

        net_displacement = float(np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2))
        straightness_ratio = float(net_displacement / path_length) if path_length > 1e-9 else 0.0

        rows.append(
            {
                "track_id": int(track_id),
                "track_length": int(len(group)),
                "first_frame_id": int(group["frame_id"].iloc[0]),
                "last_frame_id": int(group["frame_id"].iloc[-1]),
                "path_length": path_length,
                "net_displacement": net_displacement,
                "mean_step_distance": mean_step_distance,
                "max_step_distance": max_step_distance,
                "straightness_ratio": straightness_ratio,
                "mean_score": float(group["score"].mean()) if "score" in group.columns else 0.0,
                "mean_points": float(group["num_points"].mean()) if "num_points" in group.columns else 0.0,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["track_length", "mean_score", "path_length"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def get_active_track_ids(tracklets_df: pd.DataFrame, selected_frame: int) -> list[int]:
    if tracklets_df.empty:
        return []
    active = tracklets_df.loc[tracklets_df["frame_id"] == selected_frame, "track_id"]
    return sorted(int(value) for value in active.dropna().unique().tolist())


def get_visible_history_rows(
    tracklets_df: pd.DataFrame,
    selected_frame: int,
    history_length: int = 6,
    top_k: int = 25,
    ranking_metrics_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if tracklets_df.empty:
        return tracklets_df.iloc[0:0].copy()

    active_track_ids = get_active_track_ids(tracklets_df, selected_frame)
    if not active_track_ids:
        return tracklets_df.iloc[0:0].copy()

    active_set = set(active_track_ids)

    if ranking_metrics_df is not None and not ranking_metrics_df.empty and "track_id" in ranking_metrics_df.columns:
        ranked = ranking_metrics_df[ranking_metrics_df["track_id"].isin(active_set)].copy()
        if "quality_score" in ranked.columns:
            ranked = ranked.sort_values(["quality_score", "track_length"], ascending=[False, False])
        elif "track_length" in ranked.columns:
            ranked = ranked.sort_values(["track_length", "mean_score"], ascending=[False, False])
        selected_ids = ranked["track_id"].astype(int).head(top_k).tolist()
    else:
        selected_ids = active_track_ids[:top_k]

    min_frame = selected_frame - max(history_length - 1, 0)

    visible = tracklets_df[
        (tracklets_df["track_id"].isin(selected_ids))
        & (tracklets_df["frame_id"] >= min_frame)
        & (tracklets_df["frame_id"] <= selected_frame)
    ].copy()

    return visible.sort_values(["track_id", "frame_id"]).reset_index(drop=True)


def merge_active_track_table(
    visible_rows_df: pd.DataFrame,
    selected_frame: int,
    metrics_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if visible_rows_df.empty:
        return pd.DataFrame()

    current_rows = visible_rows_df[visible_rows_df["frame_id"] == selected_frame].copy()

    base_columns = [
        "track_id",
        "frame_id",
        "center_x",
        "center_y",
        "speed_m_per_frame",
        "score",
        "num_points",
        "final_tracklet_length",
    ]
    available_base = [column for column in base_columns if column in current_rows.columns]

    table_df = current_rows[available_base].copy()

    if metrics_df is not None and not metrics_df.empty and "track_id" in metrics_df.columns:
        keep_columns = [
            "track_id",
            "track_length",
            "path_length",
            "net_displacement",
            "straightness_ratio",
            "mean_score",
            "mean_points",
            "quality_score",
            "rejection_reasons",
            "passes_curation",
        ]
        available_keep = [column for column in keep_columns if column in metrics_df.columns]
        table_df = table_df.merge(
            metrics_df[available_keep],
            on="track_id",
            how="left",
        )

    if "quality_score" in table_df.columns:
        table_df = table_df.sort_values(
            ["quality_score", "score", "track_id"],
            ascending=[False, False, True],
        )
    else:
        table_df = table_df.sort_values(
            ["score", "track_id"],
            ascending=[False, True],
        )

    return table_df.reset_index(drop=True)


def _oriented_box_corners(
    center_x: float,
    center_y: float,
    length: float,
    width: float,
    yaw_rad: float,
) -> np.ndarray:
    half_l = max(length, 1e-6) / 2.0
    half_w = max(width, 1e-6) / 2.0

    local = np.array(
        [
            [ half_l,  half_w],
            [ half_l, -half_w],
            [-half_l, -half_w],
            [-half_l,  half_w],
        ],
        dtype=float,
    )

    c = float(np.cos(yaw_rad))
    s = float(np.sin(yaw_rad))
    rotation = np.array([[c, -s], [s, c]], dtype=float)

    corners = local @ rotation.T
    corners[:, 0] += float(center_x)
    corners[:, 1] += float(center_y)
    return corners


def make_tracklet_bev_figure(
    visible_rows_df: pd.DataFrame,
    selected_frame: int,
    title: str,
    show_boxes: bool = True,
):
    fig = plt.figure(figsize=(11, 8))
    ax = plt.gca()

    if visible_rows_df.empty:
        ax.set_title(f"{title} | no visible tracks")
        ax.set_xlabel("x forward [m]")
        ax.set_ylabel("y left/right [m]")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    plotted = 0

    for track_id, group in visible_rows_df.groupby("track_id"):
        group = group.sort_values("frame_id")

        x = group["center_x"].astype(float).to_numpy()
        y = group["center_y"].astype(float).to_numpy()

        ax.plot(x, y, marker="o", linewidth=2.0, markersize=4)

        start_row = group.iloc[0]
        end_row = group.iloc[-1]

        ax.scatter([start_row["center_x"]], [start_row["center_y"]], marker="s", s=45)
        ax.scatter([end_row["center_x"]], [end_row["center_y"]], marker="o", s=35)
        ax.text(float(end_row["center_x"]), float(end_row["center_y"]), str(int(track_id)), fontsize=8)

        if show_boxes and {"length", "width", "yaw_rad"}.issubset(group.columns):
            for _, row in group.iterrows():
                corners = _oriented_box_corners(
                    center_x=float(row["center_x"]),
                    center_y=float(row["center_y"]),
                    length=float(row["length"]),
                    width=float(row["width"]),
                    yaw_rad=float(row["yaw_rad"]),
                )
                is_current = int(row["frame_id"]) == int(selected_frame)
                patch = Polygon(
                    corners,
                    closed=True,
                    fill=False,
                    linewidth=1.4 if is_current else 0.8,
                    alpha=0.55 if is_current else 0.18,
                )
                ax.add_patch(patch)

        plotted += 1

    ax.set_title(f"{title} | tracks plotted={plotted}")
    ax.set_xlabel("x forward [m]")
    ax.set_ylabel("y left/right [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def make_rejection_bar_figure(curation_summary: dict | None):
    fig = plt.figure(figsize=(8, 4))
    ax = plt.gca()

    counts = {}
    if curation_summary is not None:
        counts = curation_summary.get("rejection_counts", {}) or {}

    if counts:
        names = list(counts.keys())
        values = [counts[name] for name in names]
        ax.bar(names, values)
        ax.set_xticklabels(names, rotation=30, ha="right")
    else:
        ax.text(0.5, 0.5, "No rejection counts available", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_title("Tracklet curation rejection counts")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def find_replay_frame_image(frames_dir: str | Path, selected_frame: int) -> Path | None:
    frames_path = Path(frames_dir)
    if not frames_path.exists():
        return None

    candidates: list[Path] = []
    patterns = [
        f"*{selected_frame:06d}*.png",
        f"*{selected_frame:06d}*.jpg",
        f"*{selected_frame:06d}*.jpeg",
        f"*{selected_frame}*.png",
        f"*{selected_frame}*.jpg",
        f"*{selected_frame}*.jpeg",
    ]

    for pattern in patterns:
        candidates.extend(sorted(frames_path.glob(pattern)))

    return candidates[0] if candidates else None
