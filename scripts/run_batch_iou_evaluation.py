import argparse
from pathlib import Path

from autopercept3d.datasets.kitti import KITTIDataset
from autopercept3d.evaluation.batch_iou import (
    list_frame_ids,
    run_batch_iou_evaluation,
    save_batch_iou_plots,
    summarize_batch_iou,
    write_batch_iou_csv,
)
from autopercept3d.perception.pipeline import ClassicalPipelineConfig


def print_summary(summary: dict) -> None:
    print()
    print("=" * 60)
    print("Batch BEV IoU summary")
    print("=" * 60)

    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key:<28}: {value:>10.3f}")
        else:
            print(f"{key:<28}: {value:>10}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run simple BEV IoU evaluation over multiple KITTI frames."
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
        help="Number of frames to evaluate.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.10,
        help="BEV IoU threshold for greedy matching.",
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
        default="reports_iou",
        help="Output directory for IoU reports.",
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

    frame_ids = list_frame_ids(
        dataset=dataset,
        start_index=args.start_index,
        max_frames=args.max_frames,
    )

    if not frame_ids:
        raise RuntimeError("No KITTI frames found. Check dataset root and velodyne folder.")

    print("=" * 60)
    print("AutoPercept3D batch BEV IoU evaluation")
    print("=" * 60)
    print("Note: this is a lightweight AABB BEV metric, not official KITTI evaluation.")
    print(f"Dataset root:  {dataset.dataset_root}")
    print(f"Frames:        {frame_ids[0]} to {frame_ids[-1]}")
    print(f"Count:         {len(frame_ids)}")
    print(f"IoU threshold: {args.iou_threshold}")
    print()

    rows = run_batch_iou_evaluation(
        dataset=dataset,
        frame_ids=frame_ids,
        config=config,
        iou_threshold=args.iou_threshold,
    )

    output_dir = Path(args.output_dir)
    csv_path = output_dir / "batch_iou_metrics.csv"

    write_batch_iou_csv(rows, csv_path)
    save_batch_iou_plots(rows, output_dir)

    summary = summarize_batch_iou(rows)
    print_summary(summary)

    print()
    print("Saved outputs:")
    print(f"  CSV metrics:           {csv_path}")
    print(f"  Precision plot:        {output_dir / 'precision_per_frame.png'}")
    print(f"  Recall plot:           {output_dir / 'recall_per_frame.png'}")
    print(f"  Mean IoU plot:         {output_dir / 'mean_iou_per_frame.png'}")
    print(f"  Matches plot:          {output_dir / 'matches_per_frame.png'}")
    print(f"  Ground-truth plot:     {output_dir / 'ground_truth_per_frame.png'}")


if __name__ == "__main__":
    main()
