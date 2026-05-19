import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from autopercept3d.datasets.kitti import KITTIDataset
from autopercept3d.perception.pipeline import (
    ClassicalPipelineConfig,
    run_classical_lidar_pipeline,
)
from autopercept3d.visualization.bev import (
    plot_cluster_points_bev,
    plot_ground_truth_boxes_bev,
    plot_proposal_boxes_bev,
    set_bev_axes,
)
from autopercept3d.visualization.camera import draw_camera_panel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a camera + BEV summary for one KITTI frame."
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

    image = dataset.load_image(args.frame)
    labels = dataset.load_labels(args.frame)
    calibration = dataset.load_calibration(args.frame)

    result = run_classical_lidar_pipeline(dataset, args.frame, config)

    print("=" * 60)
    print(f"Rendering camera + BEV summary for KITTI frame: {args.frame}")
    print("=" * 60)
    print(f"Image shape:            {image.shape}")
    print(f"KITTI labels:           {len([label for label in labels if label['type'] != 'DontCare'])}")
    print(f"Proposal boxes:         {result.metrics['boxes']}")
    print(f"Filtered clusters:      {result.metrics['filtered_clusters']}")
    print(f"Total time:             {result.metrics['total_time_ms']:.2f} ms")
    print(f"Approx FPS:             {result.metrics['approx_fps']:.2f}")

    plt.figure(figsize=(16, 9))

    ax1 = plt.subplot(2, 1, 1)
    draw_camera_panel(
        ax1,
        image=image,
        labels=labels,
        title=f"Camera image - frame {args.frame}",
    )

    plt.subplot(2, 1, 2)
    plot_cluster_points_bev(
        result.filtered_cluster_points,
        result.filtered_cluster_labels,
    )
    plot_proposal_boxes_bev(result.boxes, show_ids=True)
    plot_ground_truth_boxes_bev(labels, calibration)

    bev_title = (
        f"BEV proposals vs KITTI GT - frame {args.frame} | "
        f"boxes={result.metrics['boxes']} | "
        f"{result.metrics['total_time_ms']:.1f} ms"
    )
    set_bev_axes(bev_title, xlim=(-5.0, 40.0), ylim=(-25.0, 25.0))
    plt.legend(markerscale=5)

    plt.tight_layout()

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Saved frame summary to: {save_path}")

    plt.show()


if __name__ == "__main__":
    main()
