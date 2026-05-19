import argparse

from autopercept3d.datasets.kitti import KITTIDataset
from autopercept3d.perception.pipeline import (
    ClassicalPipelineConfig,
    run_classical_lidar_pipeline,
)
from autopercept3d.visualization.bev import (
    render_pipeline_bev_single,
    render_pipeline_bev_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a clean BEV visualization from the AutoPercept3D pipeline."
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
        "--view",
        type=str,
        choices=["summary", "single"],
        default="summary",
        help="Visualization layout.",
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
        "--no-ground-truth",
        action="store_true",
        help="Do not overlay KITTI ground-truth labels.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save visualization image.",
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

    result = run_classical_lidar_pipeline(dataset, args.frame, config)

    labels = None
    calibration = None

    if not args.no_ground_truth:
        labels = dataset.load_labels(args.frame)
        calibration = dataset.load_calibration(args.frame)

    print("=" * 60)
    print(f"Rendering BEV scene for KITTI frame: {args.frame}")
    print("=" * 60)
    print(f"Raw points:             {result.metrics['raw_points']}")
    print(f"Downsampled points:     {result.metrics['downsampled_points']}")
    print(f"Ground points:          {result.metrics['ground_points']}")
    print(f"Non-ground points:      {result.metrics['non_ground_points']}")
    print(f"Filtered clusters:      {result.metrics['filtered_clusters']}")
    print(f"Proposal boxes:         {result.metrics['boxes']}")
    print(f"Total time:             {result.metrics['total_time_ms']:.2f} ms")
    print(f"Approx FPS:             {result.metrics['approx_fps']:.2f}")

    if args.view == "summary":
        render_pipeline_bev_summary(
            result,
            labels=labels,
            calibration=calibration,
            save_path=args.save,
            show=True,
        )
    else:
        render_pipeline_bev_single(
            result,
            labels=labels,
            calibration=calibration,
            save_path=args.save,
            show=True,
        )


if __name__ == "__main__":
    main()
