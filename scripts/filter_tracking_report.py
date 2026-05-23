import argparse
from pathlib import Path

from autopercept3d.perception.track_filtering import (
    TrackFilterConfig,
    save_filtered_tracking_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter noisy AutoPercept3D tracking results and create cleaner BEV trajectory reports."
    )

    parser.add_argument(
        "--tracks-csv",
        type=str,
        required=True,
        help="Path to raw tracks.csv created by run_tracking_sequence.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tracking_reports_filtered",
        help="Output directory for filtered tracking reports.",
    )
    parser.add_argument(
        "--min-track-length",
        type=int,
        default=3,
        help="Minimum number of frames required to keep a track.",
    )
    parser.add_argument(
        "--top-k-tracks",
        type=int,
        default=40,
        help="Keep only the top K longest tracks. Use 0 to keep all passing tracks.",
    )
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=80.0)
    parser.add_argument("--y-abs-max", type=float, default=40.0)

    parser.add_argument("--min-length", type=float, default=0.3)
    parser.add_argument("--max-length", type=float, default=10.0)
    parser.add_argument("--min-width", type=float, default=0.2)
    parser.add_argument("--max-width", type=float, default=6.0)
    parser.add_argument("--min-height", type=float, default=0.2)
    parser.add_argument("--max-height", type=float, default=4.0)

    args = parser.parse_args()

    config = TrackFilterConfig(
        min_track_length=args.min_track_length,
        top_k_tracks=args.top_k_tracks,
        x_min=args.x_min,
        x_max=args.x_max,
        y_abs_max=args.y_abs_max,
        min_length=args.min_length,
        max_length=args.max_length,
        min_width=args.min_width,
        max_width=args.max_width,
        min_height=args.min_height,
        max_height=args.max_height,
    )

    print("=" * 60)
    print("AutoPercept3D filtered tracking report")
    print("=" * 60)
    print(f"Input tracks:       {args.tracks_csv}")
    print(f"Output directory:   {args.output_dir}")
    print(f"Min track length:   {args.min_track_length}")
    print(f"Top-K tracks:       {args.top_k_tracks}")
    print()

    result = save_filtered_tracking_report(
        raw_tracks_csv=Path(args.tracks_csv),
        output_dir=Path(args.output_dir),
        config=config,
    )

    summary = result["summary"]

    print()
    print("=" * 60)
    print("Filtered tracking summary")
    print("=" * 60)
    print(f"Raw records:                  {summary['raw_records']}")
    print(f"Raw tracks:                   {summary['raw_tracks']}")
    print(f"Raw longest track length:     {summary['raw_longest_track_length']}")
    print(f"Filtered records:             {summary['filtered_records']}")
    print(f"Filtered tracks:              {summary['filtered_tracks']}")
    print(f"Filtered longest track length:{summary['filtered_longest_track_length']:>6}")
    print(f"Filtered mean track length:   {summary['filtered_mean_track_length']:.2f}")
    print(f"Filtered median track length: {summary['filtered_median_track_length']:.2f}")

    print()
    print("Saved outputs:")
    print(f"  Filtered track CSV:       {result['filtered_tracks_csv']}")
    print(f"  Track summary CSV:        {result['filtered_track_summary_csv']}")
    print(f"  Summary JSON:             {result['summary_json']}")
    print(f"  Filtered BEV plot:        {result['plot_path']}")


if __name__ == "__main__":
    main()
