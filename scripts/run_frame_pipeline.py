import argparse
import csv
import json
from pathlib import Path

from autopercept3d.datasets.kitti import KITTIDataset
from autopercept3d.perception.pipeline import (
    ClassicalPipelineConfig,
    boxes_to_records,
    run_classical_lidar_pipeline,
)


def save_metrics_csv(path: Path, metrics: dict, timings_ms: dict) -> None:
    """
    Save one-row frame metrics as CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    row = dict(metrics)

    for key, value in timings_ms.items():
        row[f"time_{key}_ms"] = value

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def save_result_json(path: Path, frame_id: str, metrics: dict, timings_ms: dict, boxes: list[dict]) -> None:
    """
    Save frame result as JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "frame_id": frame_id,
        "metrics": metrics,
        "timings_ms": timings_ms,
        "boxes": boxes,
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def print_table(title: str, items: dict) -> None:
    print()
    print(title)
    print("-" * len(title))

    for key, value in items.items():
        if isinstance(value, float):
            print(f"{key:<28}: {value:>10.2f}")
        else:
            print(f"{key:<28}: {value:>10}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the AutoPercept3D classical LiDAR pipeline on one KITTI frame."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path to KITTI Object Detection dataset root.",
    )
    parser.add_argument(
        "--frame",
        type=str,
        default="000000",
        help="KITTI frame id, for example 000000.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.2,
        help="Voxel size in meters.",
    )
    parser.add_argument(
        "--ground-threshold",
        type=float,
        default=0.25,
        help="RANSAC ground distance threshold in meters.",
    )
    parser.add_argument(
        "--ground-iterations",
        type=int,
        default=120,
        help="Number of RANSAC iterations.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.8,
        help="DBSCAN eps in meters.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="DBSCAN min_samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory where JSON and CSV results are saved.",
    )

    args = parser.parse_args()

    config = ClassicalPipelineConfig(
        voxel_size=args.voxel_size,
        ground_distance_threshold=args.ground_threshold,
        ground_iterations=args.ground_iterations,
        dbscan_eps=args.eps,
        dbscan_min_samples=args.min_samples,
    )

    dataset = KITTIDataset(dataset_root=args.dataset_root)
    result = run_classical_lidar_pipeline(dataset, args.frame, config)

    boxes = boxes_to_records(result.boxes)

    output_dir = Path(args.output_dir)
    json_path = output_dir / f"frame_{args.frame}_result.json"
    csv_path = output_dir / f"frame_{args.frame}_metrics.csv"

    save_result_json(
        json_path,
        frame_id=args.frame,
        metrics=result.metrics,
        timings_ms=result.timings_ms,
        boxes=boxes,
    )

    save_metrics_csv(
        csv_path,
        metrics=result.metrics,
        timings_ms=result.timings_ms,
    )

    print("=" * 60)
    print(f"AutoPercept3D classical LiDAR pipeline - frame {args.frame}")
    print("=" * 60)

    print_table("Metrics", result.metrics)
    print_table("Timings in milliseconds", result.timings_ms)

    print()
    print(f"Saved JSON result: {json_path}")
    print(f"Saved CSV metrics: {csv_path}")

    print()
    print("First boxes:")
    print("cluster_id | points | center_x center_y center_z | length width height")
    print("-" * 75)

    for box in boxes[:10]:
        print(
            f"{box['cluster_id']:>9} | "
            f"{box['num_points']:>6} | "
            f"{box['center_x']:>8.2f} {box['center_y']:>8.2f} {box['center_z']:>8.2f} | "
            f"{box['length_x']:>6.2f} {box['width_y']:>6.2f} {box['height_z']:>6.2f}"
        )


if __name__ == "__main__":
    main()
