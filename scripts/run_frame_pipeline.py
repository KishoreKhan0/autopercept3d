import argparse
import csv
import json
from pathlib import Path

from autopercept3d.datasets.kitti import KITTIDataset
from autopercept3d.perception.pipeline import (
    ClassicalPipelineConfig,
    boxes_to_records,
    oriented_boxes_to_records,
    run_classical_lidar_pipeline,
)


def write_metrics_csv(path: Path, metrics: dict, timings_ms: dict) -> None:
    """
    Write one-row metrics CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    row = dict(metrics)

    for key, value in timings_ms.items():
        row[f"time_{key}_ms"] = value

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_result_json(path: Path, result) -> None:
    """
    Write full frame result JSON with both axis-aligned and oriented boxes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "frame_id": result.frame_id,
        "metrics": result.metrics,
        "timings_ms": result.timings_ms,
        "axis_aligned_boxes": boxes_to_records(result.boxes),
        "oriented_boxes": oriented_boxes_to_records(result.oriented_boxes),
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def print_metrics(metrics: dict, timings_ms: dict) -> None:
    print()
    print("Metrics")
    print("-------")

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:<28}: {value:>10.2f}")
        else:
            print(f"{key:<28}: {str(value):>10}")

    print()
    print("Timings in milliseconds")
    print("-----------------------")

    for key, value in timings_ms.items():
        print(f"{key:<28}: {value:>10.2f}")


def print_axis_box_preview(box_records: list[dict], max_rows: int = 10) -> None:
    print()
    print("First axis-aligned boxes:")
    print("cluster_id | points | center_x center_y center_z | length width height")
    print("-" * 75)

    for item in box_records[:max_rows]:
        print(
            f"{item['cluster_id']:>9} | "
            f"{item['num_points']:>6} | "
            f"{item['center_x']:>8.2f} {item['center_y']:>8.2f} {item['center_z']:>8.2f} | "
            f"{item['length']:>6.2f} {item['width']:>6.2f} {item['height']:>6.2f}"
        )


def print_oriented_box_preview(box_records: list[dict], max_rows: int = 10) -> None:
    print()
    print("First oriented boxes:")
    print("cluster_id | points | center_x center_y center_z | length width height | yaw_deg")
    print("-" * 88)

    for item in box_records[:max_rows]:
        print(
            f"{item['cluster_id']:>9} | "
            f"{item['num_points']:>6} | "
            f"{item['center_x']:>8.2f} {item['center_y']:>8.2f} {item['center_z']:>8.2f} | "
            f"{item['length']:>6.2f} {item['width']:>6.2f} {item['height']:>6.2f} | "
            f"{item['yaw_deg']:>7.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AutoPercept3D classical LiDAR pipeline for one KITTI frame."
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
        help="Output directory.",
    )

    args = parser.parse_args()

    dataset = KITTIDataset(dataset_root=args.dataset_root)

    config = ClassicalPipelineConfig(
        voxel_size=args.voxel_size,
        ground_distance_threshold=args.ground_threshold,
        ground_iterations=args.ground_iterations,
        dbscan_eps=args.eps,
        dbscan_min_samples=args.min_samples,
    )

    print("=" * 60)
    print(f"AutoPercept3D classical LiDAR pipeline - frame {args.frame}")
    print("=" * 60)

    result = run_classical_lidar_pipeline(
        dataset=dataset,
        frame_id=args.frame,
        config=config,
    )

    output_dir = Path(args.output_dir)

    json_path = output_dir / f"frame_{args.frame}_result.json"
    csv_path = output_dir / f"frame_{args.frame}_metrics.csv"

    write_result_json(json_path, result)
    write_metrics_csv(csv_path, result.metrics, result.timings_ms)

    print_metrics(result.metrics, result.timings_ms)

    print()
    print(f"Saved JSON result: {json_path}")
    print(f"Saved CSV metrics: {csv_path}")

    axis_records = boxes_to_records(result.boxes)
    oriented_records = oriented_boxes_to_records(result.oriented_boxes)

    print_axis_box_preview(axis_records)
    print_oriented_box_preview(oriented_records)


if __name__ == "__main__":
    main()
