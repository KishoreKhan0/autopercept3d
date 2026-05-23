from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from autopercept3d.perception.tracklet_metrics import analyze_tracklets_dataframe


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def make_histogram(values, title: str, xlabel: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=min(15, max(5, len(values))))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def make_line_plot(x, y, title: str, xlabel: str, ylabel: str, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def make_scatter_plot(x, y, title: str, xlabel: str, ylabel: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze AutoPercept3D stable tracklet outputs and export quantitative plots."
    )
    parser.add_argument("--tracklets-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="tracklet_analysis")
    args = parser.parse_args()

    tracklets_csv = Path(args.tracklets_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(tracklets_csv)
    artifacts = analyze_tracklets_dataframe(df)

    track_metrics = artifacts.track_metrics
    frame_metrics = artifacts.frame_metrics
    summary = artifacts.summary

    track_metrics_path = output_dir / "track_metrics.csv"
    frame_metrics_path = output_dir / "frame_metrics.csv"
    summary_path = output_dir / "tracklet_analysis_summary.json"

    track_metrics.to_csv(track_metrics_path, index=False)
    frame_metrics.to_csv(frame_metrics_path, index=False)
    write_json(summary_path, summary)

    if not track_metrics.empty:
        make_histogram(
            track_metrics["track_length"],
            title="Stable Tracklet Length Distribution",
            xlabel="Track length [detections]",
            output_path=output_dir / "track_length_hist.png",
        )

        make_histogram(
            track_metrics["path_length"],
            title="Stable Tracklet Path Length Distribution",
            xlabel="Path length [m]",
            output_path=output_dir / "path_length_hist.png",
        )

        make_scatter_plot(
            track_metrics["track_length"],
            track_metrics["net_displacement"],
            title="Track Length vs Net Displacement",
            xlabel="Track length [detections]",
            ylabel="Net displacement [m]",
            output_path=output_dir / "track_length_vs_displacement.png",
        )

        make_scatter_plot(
            track_metrics["path_length"],
            track_metrics["straightness_ratio"],
            title="Path Length vs Straightness Ratio",
            xlabel="Path length [m]",
            ylabel="Straightness ratio",
            output_path=output_dir / "path_length_vs_straightness.png",
        )

        if "mean_score" in track_metrics.columns:
            make_histogram(
                track_metrics["mean_score"].dropna(),
                title="Mean Detection Score per Tracklet",
                xlabel="Mean score",
                output_path=output_dir / "mean_score_hist.png",
            )

        if "mean_points" in track_metrics.columns:
            make_histogram(
                track_metrics["mean_points"].dropna(),
                title="Mean Points per Tracklet",
                xlabel="Mean points",
                output_path=output_dir / "mean_points_hist.png",
            )

    if not frame_metrics.empty:
        make_line_plot(
            frame_metrics["frame_id"],
            frame_metrics["active_tracks"],
            title="Active Stable Tracklets per Frame",
            xlabel="Frame ID",
            ylabel="Active stable tracklets",
            output_path=output_dir / "active_tracklets_per_frame.png",
        )

    print("=" * 60)
    print("AutoPercept3D tracklet analysis")
    print("=" * 60)
    print(f"Input CSV:           {tracklets_csv}")
    print(f"Output directory:    {output_dir}")
    print()
    print("Summary")
    print("-" * 60)
    for key, value in summary.items():
        print(f"{key:28s}: {value}")
    print()
    print("Saved outputs:")
    print(f"  Track metrics CSV:       {track_metrics_path}")
    print(f"  Frame metrics CSV:       {frame_metrics_path}")
    print(f"  Summary JSON:            {summary_path}")
    print(f"  Track length hist:       {output_dir / 'track_length_hist.png'}")
    print(f"  Path length hist:        {output_dir / 'path_length_hist.png'}")
    print(f"  Length vs displacement:  {output_dir / 'track_length_vs_displacement.png'}")
    print(f"  Path vs straightness:    {output_dir / 'path_length_vs_straightness.png'}")
    if "mean_score" in track_metrics.columns:
        print(f"  Mean score hist:         {output_dir / 'mean_score_hist.png'}")
    if "mean_points" in track_metrics.columns:
        print(f"  Mean points hist:        {output_dir / 'mean_points_hist.png'}")
    print(f"  Active tracks plot:      {output_dir / 'active_tracklets_per_frame.png'}")


if __name__ == "__main__":
    main()
