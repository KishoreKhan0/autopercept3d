from __future__ import annotations

import argparse
from pathlib import Path

from autopercept3d.perception.tracklet_curation import (
    TrackletCurationConfig,
    curate_tracklets,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a stricter high-quality subset from AutoPercept3D stable tracklets."
    )

    parser.add_argument(
        "--tracklets-csv",
        type=str,
        required=True,
        help="Path to stable_tracklets.csv from render_tracklet_replay.py or run_tracking_sequence.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tracklet_curation",
        help="Output directory for curated tracklet reports.",
    )
    parser.add_argument(
        "--min-track-length",
        type=int,
        default=4,
        help="Minimum detections required for a curated tracklet.",
    )
    parser.add_argument(
        "--min-mean-score",
        type=float,
        default=0.60,
        help="Minimum average proposal quality score.",
    )
    parser.add_argument(
        "--min-mean-points",
        type=float,
        default=40.0,
        help="Minimum average number of LiDAR points per tracklet detection.",
    )
    parser.add_argument(
        "--min-straightness-ratio",
        type=float,
        default=0.50,
        help="Minimum net displacement / path length ratio.",
    )
    parser.add_argument(
        "--max-step-distance",
        type=float,
        default=8.0,
        help="Maximum allowed single-frame BEV jump in meters.",
    )
    parser.add_argument(
        "--min-net-displacement",
        type=float,
        default=1.0,
        help="Minimum net BEV displacement in meters.",
    )
    parser.add_argument(
        "--top-k-plot",
        type=int,
        default=25,
        help="Maximum curated tracklets to draw in the BEV plot.",
    )

    args = parser.parse_args()

    config = TrackletCurationConfig(
        min_track_length=args.min_track_length,
        min_mean_score=args.min_mean_score,
        min_mean_points=args.min_mean_points,
        min_straightness_ratio=args.min_straightness_ratio,
        max_step_distance=args.max_step_distance,
        min_net_displacement=args.min_net_displacement,
        top_k_plot=args.top_k_plot,
    )

    print("=" * 60)
    print("AutoPercept3D high-quality tracklet curation")
    print("=" * 60)
    print(f"Input stable tracklets:   {args.tracklets_csv}")
    print(f"Output directory:         {args.output_dir}")
    print()
    print("Curation thresholds")
    print("-" * 60)
    print(f"min_track_length:         {config.min_track_length}")
    print(f"min_mean_score:           {config.min_mean_score}")
    print(f"min_mean_points:          {config.min_mean_points}")
    print(f"min_straightness_ratio:   {config.min_straightness_ratio}")
    print(f"max_step_distance:        {config.max_step_distance}")
    print(f"min_net_displacement:     {config.min_net_displacement}")
    print()

    result = curate_tracklets(
        stable_tracklets_csv=Path(args.tracklets_csv),
        output_dir=Path(args.output_dir),
        config=config,
    )

    summary = result["summary"]

    print("=" * 60)
    print("Curation summary")
    print("=" * 60)
    print(f"Input tracks:             {summary['input_tracks']}")
    print(f"Curated tracks:           {summary['curated_tracks']}")
    print(f"Rejected tracks:          {summary['rejected_tracks']}")
    print(f"Curated records:          {summary['curated_records']}")
    print(f"Curated ratio:            {summary['curated_ratio']:.3f}")
    print()
    print(f"All mean track length:    {summary['all_mean_track_length']:.2f}")
    print(f"Curated mean length:      {summary['curated_mean_track_length']:.2f}")
    print(f"All mean score:           {summary['all_mean_score']:.3f}")
    print(f"Curated mean score:       {summary['curated_mean_score']:.3f}")
    print(f"All mean points:          {summary['all_mean_points']:.2f}")
    print(f"Curated mean points:      {summary['curated_mean_points']:.2f}")
    print(f"All mean straightness:    {summary['all_mean_straightness']:.3f}")
    print(f"Curated mean straightness:{summary['curated_mean_straightness']:.3f}")
    print()

    print("Rejection counts")
    print("-" * 60)
    for reason, count in summary["rejection_counts"].items():
        print(f"{reason:24s}: {count}")

    print()
    print("Saved outputs:")
    print(f"  Curated tracklets CSV:          {result['curated_tracklets_csv']}")
    print(f"  Curated track metrics CSV:      {result['curated_track_metrics_csv']}")
    print(f"  Rejected track metrics CSV:     {result['rejected_track_metrics_csv']}")
    print(f"  All quality metrics CSV:        {result['all_track_metrics_with_quality_csv']}")
    print(f"  Summary JSON:                   {result['summary_json']}")
    print(f"  Curated BEV plot:               {result['curated_bev_plot']}")
    print(f"  Quality scatter plot:           {result['quality_scatter_plot']}")
    print(f"  Curated length histogram:       {result['curated_length_hist']}")


if __name__ == "__main__":
    main()
