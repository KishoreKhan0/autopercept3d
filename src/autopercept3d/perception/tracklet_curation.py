from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from autopercept3d.perception.tracklet_metrics import analyze_tracklets_dataframe


@dataclass
class TrackletCurationConfig:
    """
    Thresholds for selecting high-quality stable tracklets.

    The goal is not to delete the original stable tracklets. The goal is to
    create a stricter subset for cleaner final reporting, dashboard display,
    and demo assets.
    """

    min_track_length: int = 4
    min_mean_score: float = 0.60
    min_mean_points: float = 40.0
    min_straightness_ratio: float = 0.50
    max_step_distance: float = 8.0
    min_net_displacement: float = 1.0
    top_k_plot: int = 25


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def add_quality_columns(
    track_metrics: pd.DataFrame,
    config: TrackletCurationConfig,
) -> pd.DataFrame:
    """
    Add pass/fail columns, rejection reasons, and an interpretable quality score.
    """
    metrics = track_metrics.copy()

    if metrics.empty:
        metrics["passes_curation"] = []
        metrics["rejection_reasons"] = []
        metrics["quality_score"] = []
        return metrics

    required_columns = [
        "track_id",
        "track_length",
        "mean_score",
        "mean_points",
        "straightness_ratio",
        "max_step_distance",
        "net_displacement",
    ]

    missing = [column for column in required_columns if column not in metrics.columns]
    if missing:
        raise KeyError(f"Missing required curation metric columns: {missing}")

    passes = []
    reasons = []
    quality_scores = []

    for _, row in metrics.iterrows():
        row_reasons: list[str] = []

        track_length = _safe_float(row["track_length"])
        mean_score = _safe_float(row["mean_score"])
        mean_points = _safe_float(row["mean_points"])
        straightness = _safe_float(row["straightness_ratio"])
        max_step = _safe_float(row["max_step_distance"])
        displacement = _safe_float(row["net_displacement"])

        if track_length < config.min_track_length:
            row_reasons.append("short_track")

        if mean_score < config.min_mean_score:
            row_reasons.append("low_score")

        if mean_points < config.min_mean_points:
            row_reasons.append("low_point_support")

        if straightness < config.min_straightness_ratio:
            row_reasons.append("jittery_track")

        if max_step > config.max_step_distance:
            row_reasons.append("large_frame_jump")

        if displacement < config.min_net_displacement:
            row_reasons.append("low_displacement")

        passes.append(len(row_reasons) == 0)
        reasons.append(";".join(row_reasons) if row_reasons else "passed")

        # Score is only for ranking/display. The actual curation decision is
        # still made using the explicit thresholds above.
        length_score = min(track_length / max(config.min_track_length + 3, 1), 1.0)
        detection_score = min(max(mean_score, 0.0), 1.0)
        point_score = min(mean_points / 150.0, 1.0)
        straight_score = min(max(straightness, 0.0), 1.0)
        jump_score = 1.0 - min(max_step / max(config.max_step_distance, 1e-6), 1.0)

        quality_score = (
            0.25 * length_score
            + 0.25 * detection_score
            + 0.20 * point_score
            + 0.20 * straight_score
            + 0.10 * jump_score
        )
        quality_scores.append(float(quality_score))

    metrics["passes_curation"] = passes
    metrics["rejection_reasons"] = reasons
    metrics["quality_score"] = quality_scores

    metrics = metrics.sort_values(
        ["passes_curation", "quality_score", "track_length", "net_displacement"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    return metrics


def filter_tracklet_rows(
    tracklets_df: pd.DataFrame,
    curated_track_ids: set[int],
) -> pd.DataFrame:
    """
    Return only rows whose track_id passed curation.
    """
    if tracklets_df.empty or not curated_track_ids:
        return tracklets_df.iloc[0:0].copy()

    df = tracklets_df.copy()
    df["track_id"] = df["track_id"].astype(int)
    curated = df[df["track_id"].isin(curated_track_ids)].copy()
    curated = curated.sort_values(["track_id", "frame_index"]).reset_index(drop=True)
    return curated


def summarize_curation(
    all_metrics: pd.DataFrame,
    curated_metrics: pd.DataFrame,
    rejected_metrics: pd.DataFrame,
    curated_rows: pd.DataFrame,
    config: TrackletCurationConfig,
) -> dict:
    """
    Build curation summary for JSON export.
    """
    def metric_mean(df: pd.DataFrame, column: str) -> float:
        if df.empty or column not in df.columns:
            return 0.0
        return float(df[column].mean())

    def metric_max(df: pd.DataFrame, column: str) -> float:
        if df.empty or column not in df.columns:
            return 0.0
        return float(df[column].max())

    rejection_counts: dict[str, int] = {}

    if not rejected_metrics.empty and "rejection_reasons" in rejected_metrics.columns:
        for reason_string in rejected_metrics["rejection_reasons"].astype(str):
            for reason in reason_string.split(";"):
                if reason and reason != "passed":
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    return {
        "config": asdict(config),
        "input_tracks": int(len(all_metrics)),
        "curated_tracks": int(len(curated_metrics)),
        "rejected_tracks": int(len(rejected_metrics)),
        "curated_records": int(len(curated_rows)),
        "curated_ratio": float(len(curated_metrics) / len(all_metrics)) if len(all_metrics) else 0.0,
        "all_mean_track_length": metric_mean(all_metrics, "track_length"),
        "curated_mean_track_length": metric_mean(curated_metrics, "track_length"),
        "all_max_track_length": metric_max(all_metrics, "track_length"),
        "curated_max_track_length": metric_max(curated_metrics, "track_length"),
        "all_mean_score": metric_mean(all_metrics, "mean_score"),
        "curated_mean_score": metric_mean(curated_metrics, "mean_score"),
        "all_mean_points": metric_mean(all_metrics, "mean_points"),
        "curated_mean_points": metric_mean(curated_metrics, "mean_points"),
        "all_mean_straightness": metric_mean(all_metrics, "straightness_ratio"),
        "curated_mean_straightness": metric_mean(curated_metrics, "straightness_ratio"),
        "all_mean_path_length": metric_mean(all_metrics, "path_length"),
        "curated_mean_path_length": metric_mean(curated_metrics, "path_length"),
        "rejection_counts": rejection_counts,
    }


def plot_curated_tracklets_bev(
    curated_rows: pd.DataFrame,
    output_path: Path,
    top_k: int = 25,
) -> None:
    """
    Plot curated tracklet trajectories in BEV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 8))

    if curated_rows.empty:
        plt.title("Curated BEV tracklets | tracks=0")
        plt.xlabel("x forward [m]")
        plt.ylabel("y left/right [m]")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=180)
        plt.close()
        return

    df = curated_rows.copy()
    df["track_id"] = df["track_id"].astype(int)
    df["frame_index"] = df["frame_index"].astype(int)

    track_lengths = (
        df.groupby("track_id")
        .size()
        .sort_values(ascending=False)
    )

    selected_track_ids = list(track_lengths.head(top_k).index)

    plotted = 0

    for track_id in selected_track_ids:
        group = df[df["track_id"] == track_id].sort_values("frame_index")

        xs = group["center_x"].astype(float).to_numpy()
        ys = group["center_y"].astype(float).to_numpy()

        if len(xs) < 2:
            continue

        plt.plot(xs, ys, marker="o", linewidth=2.0, markersize=4)
        plt.scatter([xs[0]], [ys[0]], marker="s", s=30)
        plt.text(xs[-1], ys[-1], str(track_id), fontsize=8)

        plotted += 1

    plt.title(f"Curated high-quality BEV tracklets | tracks plotted={plotted}")
    plt.xlabel("x forward [m]")
    plt.ylabel("y left/right [m]")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_curation_quality_scatter(
    metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot curation status in track length vs score space.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))

    if not metrics.empty:
        passed = metrics[metrics["passes_curation"] == True]
        rejected = metrics[metrics["passes_curation"] == False]

        if not rejected.empty:
            plt.scatter(
                rejected["track_length"],
                rejected["mean_score"],
                marker="x",
                label="rejected",
            )

        if not passed.empty:
            plt.scatter(
                passed["track_length"],
                passed["mean_score"],
                marker="o",
                label="curated",
            )

    plt.title("Tracklet curation: length vs mean score")
    plt.xlabel("Track length [detections]")
    plt.ylabel("Mean detection score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_curated_length_histogram(
    curated_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot curated track length distribution.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    if not curated_metrics.empty:
        plt.hist(
            curated_metrics["track_length"],
            bins=min(10, max(4, len(curated_metrics))),
        )

    plt.title("Curated Tracklet Length Distribution")
    plt.xlabel("Track length [detections]")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_json(path: Path, payload: dict) -> None:
    """
    Write JSON with clean formatting.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def curate_tracklets(
    stable_tracklets_csv: Path,
    output_dir: Path,
    config: TrackletCurationConfig,
) -> dict:
    """
    Curate stable tracklets and export stricter high-quality reports.
    """
    tracklets_df = pd.read_csv(stable_tracklets_csv)

    artifacts = analyze_tracklets_dataframe(tracklets_df)
    all_metrics = add_quality_columns(artifacts.track_metrics, config)

    curated_metrics = all_metrics[all_metrics["passes_curation"] == True].copy()
    rejected_metrics = all_metrics[all_metrics["passes_curation"] == False].copy()

    curated_track_ids = set(curated_metrics["track_id"].astype(int).tolist())
    curated_rows = filter_tracklet_rows(tracklets_df, curated_track_ids)

    summary = summarize_curation(
        all_metrics=all_metrics,
        curated_metrics=curated_metrics,
        rejected_metrics=rejected_metrics,
        curated_rows=curated_rows,
        config=config,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    curated_tracklets_path = output_dir / "curated_tracklets.csv"
    curated_metrics_path = output_dir / "curated_track_metrics.csv"
    rejected_metrics_path = output_dir / "rejected_track_metrics.csv"
    all_metrics_path = output_dir / "all_track_metrics_with_quality.csv"
    summary_path = output_dir / "curation_summary.json"

    curated_rows.to_csv(curated_tracklets_path, index=False)
    curated_metrics.to_csv(curated_metrics_path, index=False)
    rejected_metrics.to_csv(rejected_metrics_path, index=False)
    all_metrics.to_csv(all_metrics_path, index=False)
    write_json(summary_path, summary)

    bev_plot_path = output_dir / "curated_tracklets_bev.png"
    quality_plot_path = output_dir / "curation_quality_scatter.png"
    length_hist_path = output_dir / "curated_track_length_hist.png"

    plot_curated_tracklets_bev(
        curated_rows=curated_rows,
        output_path=bev_plot_path,
        top_k=config.top_k_plot,
    )

    plot_curation_quality_scatter(
        metrics=all_metrics,
        output_path=quality_plot_path,
    )

    plot_curated_length_histogram(
        curated_metrics=curated_metrics,
        output_path=length_hist_path,
    )

    return {
        "summary": summary,
        "curated_tracklets_csv": str(curated_tracklets_path),
        "curated_track_metrics_csv": str(curated_metrics_path),
        "rejected_track_metrics_csv": str(rejected_metrics_path),
        "all_track_metrics_with_quality_csv": str(all_metrics_path),
        "summary_json": str(summary_path),
        "curated_bev_plot": str(bev_plot_path),
        "quality_scatter_plot": str(quality_plot_path),
        "curated_length_hist": str(length_hist_path),
    }
