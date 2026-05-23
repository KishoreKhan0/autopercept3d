from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


def _find_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True, default=None):
    for name in candidates:
        if name in df.columns:
            return name
    if required:
        raise KeyError(f"Could not find any of these columns: {list(candidates)}")
    return default


@dataclass
class TrackletAnalysisArtifacts:
    track_metrics: pd.DataFrame
    frame_metrics: pd.DataFrame
    summary: dict


def analyze_tracklets_dataframe(tracklets_df: pd.DataFrame) -> TrackletAnalysisArtifacts:
    if tracklets_df.empty:
        empty = pd.DataFrame()
        return TrackletAnalysisArtifacts(
            track_metrics=empty,
            frame_metrics=empty,
            summary={
                "num_records": 0,
                "num_tracks": 0,
            },
        )

    track_id_col = _find_column(tracklets_df, ["track_id"])
    frame_id_col = _find_column(tracklets_df, ["frame_id"])
    frame_index_col = _find_column(tracklets_df, ["frame_index"])
    x_col = _find_column(tracklets_df, ["center_x", "x"])
    y_col = _find_column(tracklets_df, ["center_y", "y"])
    score_col = _find_column(tracklets_df, ["score", "detection_score"], required=False)
    points_col = _find_column(tracklets_df, ["num_points", "points", "detection_points"], required=False)

    df = tracklets_df.copy()
    df[track_id_col] = df[track_id_col].astype(int)
    df[frame_index_col] = df[frame_index_col].astype(int)
    df[x_col] = df[x_col].astype(float)
    df[y_col] = df[y_col].astype(float)

    if score_col is not None:
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    if points_col is not None:
        df[points_col] = pd.to_numeric(df[points_col], errors="coerce")

    df = df.sort_values([track_id_col, frame_index_col]).reset_index(drop=True)

    rows = []
    for track_id, group in df.groupby(track_id_col):
        group = group.sort_values(frame_index_col).reset_index(drop=True)

        xs = group[x_col].to_numpy(dtype=float)
        ys = group[y_col].to_numpy(dtype=float)
        frame_indices = group[frame_index_col].to_numpy(dtype=int)

        if len(group) >= 2:
            dx = np.diff(xs)
            dy = np.diff(ys)
            step_distances = np.sqrt(dx * dx + dy * dy)
            path_length = float(step_distances.sum())
            mean_step_distance = float(step_distances.mean())
            max_step_distance = float(step_distances.max())
        else:
            path_length = 0.0
            mean_step_distance = 0.0
            max_step_distance = 0.0

        net_dx = float(xs[-1] - xs[0])
        net_dy = float(ys[-1] - ys[0])
        net_displacement = float(np.sqrt(net_dx * net_dx + net_dy * net_dy))

        frame_span = int(frame_indices[-1] - frame_indices[0] + 1)
        track_length = int(len(group))

        row = {
            "track_id": int(track_id),
            "track_length": track_length,
            "first_frame_id": str(group[frame_id_col].iloc[0]),
            "last_frame_id": str(group[frame_id_col].iloc[-1]),
            "first_frame_index": int(frame_indices[0]),
            "last_frame_index": int(frame_indices[-1]),
            "frame_span": frame_span,
            "start_x": float(xs[0]),
            "start_y": float(ys[0]),
            "end_x": float(xs[-1]),
            "end_y": float(ys[-1]),
            "net_displacement": net_displacement,
            "path_length": path_length,
            "mean_step_distance": mean_step_distance,
            "max_step_distance": max_step_distance,
            "straightness_ratio": float(net_displacement / path_length) if path_length > 1e-9 else 0.0,
        }

        if score_col is not None:
            scores = group[score_col].dropna()
            row["mean_score"] = float(scores.mean()) if not scores.empty else np.nan
            row["min_score"] = float(scores.min()) if not scores.empty else np.nan
            row["max_score"] = float(scores.max()) if not scores.empty else np.nan

        if points_col is not None:
            pts = group[points_col].dropna()
            row["mean_points"] = float(pts.mean()) if not pts.empty else np.nan
            row["min_points"] = float(pts.min()) if not pts.empty else np.nan
            row["max_points"] = float(pts.max()) if not pts.empty else np.nan

        rows.append(row)

    track_metrics = pd.DataFrame(rows).sort_values(
        ["track_length", "net_displacement"],
        ascending=[False, False],
    ).reset_index(drop=True)

    frame_rows = []
    for frame_id, group in df.groupby(frame_id_col):
        group = group.sort_values(frame_index_col)
        frame_rows.append(
            {
                "frame_id": str(frame_id),
                "frame_index": int(group[frame_index_col].iloc[0]),
                "active_tracks": int(group[track_id_col].nunique()),
                "detections": int(len(group)),
                "mean_x": float(group[x_col].mean()),
                "mean_y": float(group[y_col].mean()),
            }
        )

    frame_metrics = pd.DataFrame(frame_rows).sort_values("frame_index").reset_index(drop=True)

    summary = {
        "num_records": int(len(df)),
        "num_tracks": int(track_metrics["track_id"].nunique()),
        "mean_track_length": float(track_metrics["track_length"].mean()),
        "median_track_length": float(track_metrics["track_length"].median()),
        "max_track_length": int(track_metrics["track_length"].max()),
        "mean_path_length": float(track_metrics["path_length"].mean()),
        "mean_net_displacement": float(track_metrics["net_displacement"].mean()),
        "mean_straightness_ratio": float(track_metrics["straightness_ratio"].mean()),
        "max_active_tracks_per_frame": int(frame_metrics["active_tracks"].max()),
        "mean_active_tracks_per_frame": float(frame_metrics["active_tracks"].mean()),
    }

    if "mean_score" in track_metrics.columns:
        summary["mean_track_score"] = float(track_metrics["mean_score"].mean())

    if "mean_points" in track_metrics.columns:
        summary["mean_track_points"] = float(track_metrics["mean_points"].mean())

    return TrackletAnalysisArtifacts(
        track_metrics=track_metrics,
        frame_metrics=frame_metrics,
        summary=summary,
    )
