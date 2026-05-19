import argparse

from autopercept3d.datasets.kitti import KITTIDataset
from autopercept3d.geometry.projection import project_velodyne_to_camera_image
from autopercept3d.perception.preprocessing import crop_roi, filter_finite_points
from autopercept3d.visualization.projection import draw_projected_lidar_on_image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project KITTI LiDAR points into the left camera image."
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
        help="KITTI camera projection matrix. Use P2 for left color camera.",
    )
    parser.add_argument(
        "--crop-roi",
        action="store_true",
        help="Project only points inside the driving ROI.",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Do not draw KITTI 2D labels.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save the projection image.",
    )

    args = parser.parse_args()

    dataset = KITTIDataset(dataset_root=args.dataset_root)

    image = dataset.load_image(args.frame)
    points = dataset.load_point_cloud(args.frame)
    calibration = dataset.load_calibration(args.frame)
    labels = dataset.load_labels(args.frame)

    points = filter_finite_points(points)

    if args.crop_roi:
        points = crop_roi(points)

    projection = project_velodyne_to_camera_image(
        points=points,
        calibration=calibration,
        image_shape=image.shape,
        camera=args.camera,
    )

    print("=" * 60)
    print(f"LiDAR projection for KITTI frame: {args.frame}")
    print("=" * 60)
    print(f"Image shape:              {image.shape}")
    print(f"Input LiDAR points:       {len(points)}")
    print(f"Projected image points:   {len(projection.pixels)}")
    print(f"Camera matrix:            {args.camera}")

    if len(projection.depths) > 0:
        print(f"Min projected depth:      {projection.depths.min():.2f} m")
        print(f"Max projected depth:      {projection.depths.max():.2f} m")

    draw_projected_lidar_on_image(
        image=image,
        projection=projection,
        labels=None if args.no_labels else labels,
        title=f"KITTI LiDAR projection - frame {args.frame}",
        save_path=args.save,
        show=True,
    )


if __name__ == "__main__":
    main()
