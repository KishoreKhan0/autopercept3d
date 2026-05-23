import argparse
import csv
import json
from pathlib import Path

from autopercept3d.datasets.kitti import KITTIDataset
from autopercept3d.perception.pipeline import ClassicalPipelineConfig, run_classical_lidar_pipeline
from autopercept3d.perception.tracking import (
    StableTrackletTracker,
    detections_from_axis_aligned_boxes,
    detections_from_oriented_boxes,
    filter_detections,
    tracks_to_record_dicts,
    tracks_to_summary_rows,
)
from autopercept3d.visualization.tracklet_viz import render_bev_tracklet_frame


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render per-frame BEV replay images for stable AutoPercept3D tracklets."
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

    parser.add_argument("--history-length", type=int, default=6)
    parser.add_argument("--top-k-tracks", type=int, default=25)
    parser.add_argument("--output-dir", type=str, default="tracklet_replay")

    args = parser.parse_args()

    dataset = KITTIDataset(dataset_root=args.dataset_root)
    frame_ids = list_frame_ids(dataset, args.start_index, args.max_frames)

    if not frame_ids:
        raise RuntimeError("No frames found for the requested range.")

    config = ClassicalPipelineConfig(
        voxel_size=args.voxel_size,
        ground_distance_threshold=args.ground_threshold,
        ground_iterations=args.ground_iterations,
        dbscan_eps=args.eps,
        dbscan_min_samples=args.min_samples,
    )

    tracker = StableTrackletTracker(
        max_match_distance=args.max_match_distance,
        max_missed=args.max_missed,
        min_stable_hits=args.min_stable_hits,
        velocity_smoothing=args.velocity_smoothing,
    )

    per_frame_results = []

    print("=" * 60)
    print("AutoPercept3D stable tracklet replay export")
    print("=" * 60)
    print(f"Frames:               {frame_ids[0]} to {frame_ids[-1]}")
    print(f"Count:                {len(frame_ids)}")
    print(f"Box type:             {args.box_type}")
    print(f"History length:       {args.history_length}")
    print(f"Top-K tracks:         {args.top_k_tracks}")
    print()

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

        tracker.update(frame_id=frame_id, detections=detections)

        per_frame_results.append(
            {
                "frame_id": frame_id,
                "frame_index": frame_index,
                "result": result,
                "raw_detections": len(raw_detections),
                "filtered_detections": len(detections),
            }
        )

    tracker.finish()

    stable_tracks = tracker.stable_tracks(
        min_hits=args.min_stable_hits,
        min_mean_score=args.min_detection_score,
    )

    stable_rows = tracks_to_record_dicts(stable_tracks, stable=True)
    stable_summary_rows = tracks_to_summary_rows(stable_tracks)

    output_dir = Path(args.output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    rendered_rows = []

    for item in per_frame_results:
        frame_id = item["frame_id"]
        frame_index = item["frame_index"]
        result = item["result"]

        output_path = frames_dir / f"tracklets_bev_{frame_id}.png"

        render_bev_tracklet_frame(
            result=result,
            stable_rows=stable_rows,
            frame_id=frame_id,
            frame_index=frame_index,
            output_path=output_path,
            box_type=args.box_type,
            history_length=args.history_length,
            top_k_tracks=args.top_k_tracks,
        )

        rendered_rows.append(
            {
                "frame_id": frame_id,
                "frame_index": frame_index,
                "image_path": str(output_path),
                "raw_detections": item["raw_detections"],
                "filtered_detections": item["filtered_detections"],
            }
        )

    summary_payload = {
        "module": "stable_tracklet_replay",
        "frames": frame_ids,
        "box_type": args.box_type,
        "history_length": args.history_length,
        "top_k_tracks": args.top_k_tracks,
        "num_stable_tracklets": len(stable_tracks),
        "rendered_frames": len(rendered_rows),
    }

    write_csv(output_dir / "stable_tracklets.csv", stable_rows)
    write_csv(output_dir / "stable_tracklet_summary.csv", stable_summary_rows)
    write_csv(output_dir / "rendered_frames.csv", rendered_rows)
    write_json(output_dir / "replay_summary.json", summary_payload)

    print()
    print("=" * 60)
    print("Replay export summary")
    print("=" * 60)
    print(f"Stable tracklets:     {len(stable_tracks)}")
    print(f"Rendered frames:      {len(rendered_rows)}")
    print(f"Frames directory:     {frames_dir}")
    print()
    print("Saved outputs:")
    print(f"  Stable tracklets CSV:    {output_dir / 'stable_tracklets.csv'}")
    print(f"  Tracklet summary CSV:    {output_dir / 'stable_tracklet_summary.csv'}")
    print(f"  Rendered frames CSV:     {output_dir / 'rendered_frames.csv'}")
    print(f"  Replay summary JSON:     {output_dir / 'replay_summary.json'}")
    print(f"  Frame images dir:        {frames_dir}")


if __name__ == "__main__":
    main()
