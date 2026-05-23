import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from autopercept3d.datasets.kitti import KITTIDataset
from autopercept3d.perception.pipeline import ClassicalPipelineConfig, run_classical_lidar_pipeline
from autopercept3d.perception.tracking import (
    StableTrackletTracker,
    detections_from_axis_aligned_boxes,
    detections_from_oriented_boxes,
    filter_detections,
    summarize_tracks,
    tracks_to_record_dicts,
    tracks_to_summary_rows,
)


def list_frame_ids(dataset: KITTIDataset, start_index: int, max_frames: int) -> list[str]:
    frame_ids = sorted(path.stem for path in dataset.velodyne_dir.glob("*.bin"))
    return frame_ids[start_index : start_index + max_frames]


def write_csv(path: Path, rows: list[dict]) -> None:
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
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def plot_stable_tracklets(
    track_summary_rows: list[dict],
    stable_rows: list[dict],
    output_path: Path,
    top_k: int = 25,
) -> int:
    if not stable_rows:
        return 0

    selected_track_ids = [
        int(row["track_id"])
        for row in track_summary_rows[:top_k]
    ]

    rows_by_track: dict[int, list[dict]] = {}

    for row in stable_rows:
        track_id = int(row["track_id"])

        if track_id not in selected_track_ids:
            continue

        rows_by_track.setdefault(track_id, []).append(row)

    for rows in rows_by_track.values():
        rows.sort(key=lambda item: int(item["frame_index"]))

    plt.figure(figsize=(11, 8))

    plotted = 0

    for track_id, rows in rows_by_track.items():
        if len(rows) < 2:
            continue

        xs = [float(row["center_x"]) for row in rows]
        ys = [float(row["center_y"]) for row in rows]

        plt.plot(xs, ys, marker="o", linewidth=2.0, markersize=4)
        plt.scatter([xs[0]], [ys[0]], marker="s", s=30)
        plt.text(xs[-1], ys[-1], str(track_id), fontsize=8)

        plotted += 1

    plt.title(f"AutoPercept3D stable BEV tracklets | tracks plotted={plotted}")
    plt.xlabel("x forward [m]")
    plt.ylabel("y left/right [m]")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.show()

    return plotted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run stable BEV tracklet mining over AutoPercept3D proposal boxes."
    )

    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--start-index", type=int, default=250)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--box-type", choices=["oriented", "axis"], default="oriented")

    parser.add_argument("--max-match-distance", type=float, default=4.5)
    parser.add_argument("--max-missed", type=int, default=2)
    parser.add_argument("--min-stable-hits", type=int, default=3)
    parser.add_argument("--velocity-smoothing", type=float, default=0.55)

    parser.add_argument("--min-detection-points", type=int, default=18)
    parser.add_argument("--min-detection-score", type=float, default=0.20)

    parser.add_argument("--voxel-size", type=float, default=0.2)
    parser.add_argument("--ground-threshold", type=float, default=0.25)
    parser.add_argument("--ground-iterations", type=int, default=120)
    parser.add_argument("--eps", type=float, default=0.8)
    parser.add_argument("--min-samples", type=int, default=10)

    parser.add_argument("--plot-top-k-tracks", type=int, default=25)
    parser.add_argument("--output-dir", type=str, default="tracklet_reports")

    args = parser.parse_args()

    dataset = KITTIDataset(dataset_root=args.dataset_root)

    config = ClassicalPipelineConfig(
        voxel_size=args.voxel_size,
        ground_distance_threshold=args.ground_threshold,
        ground_iterations=args.ground_iterations,
        dbscan_eps=args.eps,
        dbscan_min_samples=args.min_samples,
    )

    frame_ids = list_frame_ids(dataset, args.start_index, args.max_frames)

    if not frame_ids:
        raise RuntimeError("No frames found. Check dataset root and frame range.")

    tracker = StableTrackletTracker(
        max_match_distance=args.max_match_distance,
        max_missed=args.max_missed,
        min_stable_hits=args.min_stable_hits,
        velocity_smoothing=args.velocity_smoothing,
    )

    print("=" * 60)
    print("AutoPercept3D stable BEV tracklet mining")
    print("=" * 60)
    print(f"Frames:               {frame_ids[0]} to {frame_ids[-1]}")
    print(f"Count:                {len(frame_ids)}")
    print(f"Box type:             {args.box_type}")
    print(f"Max match distance:   {args.max_match_distance:.2f} m")
    print(f"Max missed:           {args.max_missed}")
    print(f"Min stable hits:      {args.min_stable_hits}")
    print(f"Min detection points: {args.min_detection_points}")
    print(f"Min detection score:  {args.min_detection_score:.2f}")
    print()

    per_frame_rows: list[dict] = []

    for frame_index, frame_id in enumerate(frame_ids):
        print(f"[{frame_index + 1}/{len(frame_ids)}] Processing frame {frame_id}...")

        result = run_classical_lidar_pipeline(
            dataset=dataset,
            frame_id=frame_id,
            config=config,
        )

        if args.box_type == "oriented":
            raw_detections = detections_from_oriented_boxes(
                frame_id=frame_id,
                boxes=result.oriented_boxes,
                frame_index=frame_index,
            )
        else:
            raw_detections = detections_from_axis_aligned_boxes(
                frame_id=frame_id,
                boxes=result.boxes,
                frame_index=frame_index,
            )

        detections = filter_detections(
            raw_detections,
            min_points=args.min_detection_points,
            min_score=args.min_detection_score,
        )

        matched_records = tracker.update(frame_id=frame_id, detections=detections)

        per_frame_rows.append(
            {
                "frame_id": frame_id,
                "frame_index": frame_index,
                "raw_detections": len(raw_detections),
                "filtered_detections": len(detections),
                "matched_records": len(matched_records),
                "active_tracks": len(tracker.active_tracks),
                "tracks_created": tracker.next_track_id - 1,
                "pipeline_time_ms": result.metrics["total_time_ms"],
                "approx_fps": result.metrics["approx_fps"],
            }
        )

    tracker.finish()

    all_tracks = tracker.all_tracks()
    stable_tracks = tracker.stable_tracks(
        min_hits=args.min_stable_hits,
        min_mean_score=args.min_detection_score,
    )

    all_rows = tracks_to_record_dicts(all_tracks, stable=False)
    stable_rows = tracks_to_record_dicts(stable_tracks, stable=True)
    stable_summary_rows = tracks_to_summary_rows(stable_tracks)

    all_summary = summarize_tracks(all_tracks)
    stable_summary = summarize_tracks(stable_tracks)

    output_dir = Path(args.output_dir)

    all_tracks_csv = output_dir / "all_tracklets.csv"
    stable_tracks_csv = output_dir / "stable_tracklets.csv"
    stable_summary_csv = output_dir / "stable_tracklet_summary.csv"
    frames_csv = output_dir / "tracklet_frames.csv"
    summary_json = output_dir / "tracklet_summary.json"
    plot_path = output_dir / "stable_tracklets_bev.png"

    write_csv(all_tracks_csv, all_rows)
    write_csv(stable_tracks_csv, stable_rows)
    write_csv(stable_summary_csv, stable_summary_rows)
    write_csv(frames_csv, per_frame_rows)

    plotted_tracks = plot_stable_tracklets(
        track_summary_rows=stable_summary_rows,
        stable_rows=stable_rows,
        output_path=plot_path,
        top_k=args.plot_top_k_tracks,
    )

    payload = {
        "frames": frame_ids,
        "box_type": args.box_type,
        "module": "stable_tracklet_mining",
        "max_match_distance": args.max_match_distance,
        "max_missed": args.max_missed,
        "min_stable_hits": args.min_stable_hits,
        "velocity_smoothing": args.velocity_smoothing,
        "min_detection_points": args.min_detection_points,
        "min_detection_score": args.min_detection_score,
        "all_tracklets": all_summary,
        "stable_tracklets": stable_summary,
        "plotted_tracks": plotted_tracks,
        "per_frame": per_frame_rows,
    }

    write_json(summary_json, payload)

    print()
    print("=" * 60)
    print("Tracklet mining summary")
    print("=" * 60)
    print(f"All assigned detections:       {all_summary['assigned_detections']}")
    print(f"All tracklets:                 {all_summary['unique_tracks']}")
    print(f"All longest tracklet:          {all_summary['longest_tracklet']}")
    print(f"All mean tracklet length:      {all_summary['mean_tracklet_length']:.2f}")
    print()
    print(f"Stable assigned detections:    {stable_summary['assigned_detections']}")
    print(f"Stable tracklets:              {stable_summary['unique_tracks']}")
    print(f"Stable longest tracklet:       {stable_summary['longest_tracklet']}")
    print(f"Stable mean tracklet length:   {stable_summary['mean_tracklet_length']:.2f}")
    print(f"Plotted stable tracklets:      {plotted_tracks}")

    print()
    print("Saved outputs:")
    print(f"  All tracklets CSV:       {all_tracks_csv}")
    print(f"  Stable tracklets CSV:    {stable_tracks_csv}")
    print(f"  Tracklet summary CSV:    {stable_summary_csv}")
    print(f"  Per-frame CSV:           {frames_csv}")
    print(f"  Summary JSON:            {summary_json}")
    print(f"  Stable BEV plot:         {plot_path}")


if __name__ == "__main__":
    main()
