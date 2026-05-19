import argparse

from autopercept3d.datasets.kitti import KITTIDataset
from autopercept3d.perception.pipeline import (
    ClassicalPipelineConfig,
    run_classical_lidar_pipeline,
)
from autopercept3d.visualization.camera_boxes import draw_camera_3d_box_overlay


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project AutoPercept3D proposal boxes and KITTI 3D labels onto the camera image."
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
        "--camera",
        type=str,
        default="P2",
        help="KITTI camera projection matrix. P2 is left color camera.",
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
        "--max-proposals",
        type=int,
        default=25,
        help="Maximum proposal boxes to draw.",
    )
    parser.add_argument(
        "--no-ground-truth",
        action="store_true",
        help="Do not draw KITTI ground-truth 3D boxes.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save the overlay image.",
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

    image = dataset.load_image(args.frame)
    calibration = dataset.load_calibration(args.frame)
    labels = dataset.load_labels(args.frame)

    result = run_classical_lidar_pipeline(dataset, args.frame, config)

    print("=" * 60)
    print(f"Camera 3D box projection for KITTI frame: {args.frame}")
    print("=" * 60)
    print(f"Image shape:          {image.shape}")
    print(f"Proposal boxes:       {len(result.boxes)}")
    print(f"KITTI labels:         {len([label for label in labels if label['type'] != 'DontCare'])}")
    print(f"Camera matrix:        {args.camera}")
    print(f"Total pipeline time:  {result.metrics['total_time_ms']:.2f} ms")
    print("Renderer:             camera_boxes")

    draw_camera_3d_box_overlay(
        image=image,
        calibration=calibration,
        proposal_boxes=result.boxes,
        labels=None if args.no_ground_truth else labels,
        camera=args.camera,
        max_proposals=args.max_proposals,
        title=f"Camera 3D box overlay - frame {args.frame}",
        save_path=args.save,
        show=True,
    )


if __name__ == "__main__":
    main()
