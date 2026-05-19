import argparse
from pathlib import Path

from autopercept3d.datasets.kitti import KITTIDataset
from autopercept3d.evaluation.batch import (
    list_kitti_frame_ids,
    run_batch,
    save_batch_plots,
    summarize_batch,
    write_metrics_csv,
)
from autopercept3d.perception.pipeline import ClassicalPipelineConfig


def print_summary(summary: dict) -> None:
    print()
    print("=" * 60)
    print("Batch summary")
    print("=" * 60)

    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key:<28}: {value:>10.2f}")
        else:
            print(f"{key:<28}: {value:>10}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AutoPercept3D classical LiDAR pipeline over multiple KITTI frames."
    )

    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path to KITTI Object Detection dataset root.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index in sorted KITTI frame list.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=10,
        help="Number of frames to process.",
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
        default="reports",
        help="Output directory for batch reports.",
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

    frame_ids = list_kitti_frame_ids(
        dataset,
        start_index=args.start_index,
        max_frames=args.max_frames,
    )

    if not frame_ids:
        raise RuntimeError("No KITTI frames found. Check dataset root and velodyne folder.")

    print("=" * 60)
    print("AutoPercept3D batch pipeline")
    print("=" * 60)
    print(f"Dataset root: {dataset.dataset_root}")
    print(f"Frames:       {frame_ids[0]} to {frame_ids[-1]}")
    print(f"Count:        {len(frame_ids)}")
    print()

    rows = run_batch(dataset, frame_ids, config)

    output_dir = Path(args.output_dir)
    csv_path = output_dir / "batch_metrics.csv"

    write_metrics_csv(rows, csv_path)
    save_batch_plots(rows, output_dir)

    summary = summarize_batch(rows)
    print_summary(summary)

    print()
    print("Saved outputs:")
    print(f"  CSV metrics:        {csv_path}")
    print(f"  Timing plot:        {output_dir / 'timing_total_ms.png'}")
    print(f"  FPS plot:           {output_dir / 'approx_fps.png'}")
    print(f"  Boxes plot:         {output_dir / 'boxes_per_frame.png'}")
    print(f"  Clusters plot:      {output_dir / 'clusters_per_frame.png'}")


if __name__ == "__main__":
    main()
