from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from autopercept3d.datasets.kitti import KITTIDataset
from autopercept3d.geometry.projection import project_velodyne_to_camera_image
from autopercept3d.perception.pipeline import (
    ClassicalPipelineConfig,
    boxes_to_records,
    run_classical_lidar_pipeline,
)
from autopercept3d.perception.preprocessing import crop_roi, filter_finite_points
from autopercept3d.visualization.bev import (
    render_pipeline_bev_single,
    render_pipeline_bev_summary,
)
from autopercept3d.visualization.camera_boxes import draw_camera_3d_box_overlay
from autopercept3d.visualization.projection import draw_projected_lidar_on_image


def write_metrics_csv(path: Path, metrics: dict, timings_ms: dict) -> None:
    """
    Save one-row metrics CSV for the demo frame.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    row = dict(metrics)

    for key, value in timings_ms.items():
        row[f"time_{key}_ms"] = value

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_result_json(path: Path, frame_id: str, metrics: dict, timings_ms: dict, boxes: list[dict]) -> None:
    """
    Save compact JSON result for the demo frame.
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


def write_demo_summary_markdown(
    path: Path,
    frame_id: str,
    metrics: dict,
    output_dir: Path,
) -> None:
    """
    Save a small Markdown summary that can be copied into README later.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    text = f"""# AutoPercept3D Demo Summary

Demo frame: `{frame_id}`

## Pipeline Metrics

| Metric | Value |
|---|---:|
| Raw LiDAR points | {metrics['raw_points']} |
| Cropped points | {metrics['cropped_points']} |
| Downsampled points | {metrics['downsampled_points']} |
| Ground points | {metrics['ground_points']} |
| Non-ground points | {metrics['non_ground_points']} |
| Filtered clusters | {metrics['filtered_clusters']} |
| Proposal boxes | {metrics['boxes']} |
| Total runtime | {metrics['total_time_ms']:.2f} ms |
| Approx FPS | {metrics['approx_fps']:.2f} |

## Generated Assets

```text
{output_dir / f'bev_summary_{frame_id}.png'}
{output_dir / f'bev_scene_{frame_id}.png'}
{output_dir / f'lidar_projection_{frame_id}.png'}
{output_dir / f'camera_3d_boxes_{frame_id}.png'}
{output_dir / f'frame_{frame_id}_result.json'}
{output_dir / f'frame_{frame_id}_metrics.csv'}
```

## Description

This demo shows a classical LiDAR object-proposal pipeline on a KITTI frame:

```text
LiDAR loading
    -> ROI crop
    -> voxel downsampling
    -> RANSAC ground removal
    -> DBSCAN clustering
    -> axis-aligned 3D proposal boxes
    -> BEV and camera-space visualization
```

The generated boxes are cluster-based object proposals, not deep-learning detections.
"""

    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export portfolio/demo assets for one AutoPercept3D KITTI frame."
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
        "--output-dir",
        type=str,
        default="demo_assets",
        help="Directory where demo assets will be saved.",
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
        "--max-camera-boxes",
        type=int,
        default=25,
        help="Maximum proposal boxes to draw on camera 3D overlay.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = KITTIDataset(dataset_root=args.dataset_root)

    config = ClassicalPipelineConfig(
        voxel_size=args.voxel_size,
        ground_distance_threshold=args.ground_threshold,
        ground_iterations=args.ground_iterations,
        dbscan_eps=args.eps,
        dbscan_min_samples=args.min_samples,
    )

    print("=" * 60)
    print(f"Exporting AutoPercept3D demo assets for frame {args.frame}")
    print("=" * 60)

    image = dataset.load_image(args.frame)
    labels = dataset.load_labels(args.frame)
    calibration = dataset.load_calibration(args.frame)

    result = run_classical_lidar_pipeline(
        dataset=dataset,
        frame_id=args.frame,
        config=config,
    )

    boxes = boxes_to_records(result.boxes)

    json_path = output_dir / f"frame_{args.frame}_result.json"
    csv_path = output_dir / f"frame_{args.frame}_metrics.csv"
    summary_path = output_dir / f"demo_summary_{args.frame}.md"

    write_result_json(
        json_path,
        frame_id=args.frame,
        metrics=result.metrics,
        timings_ms=result.timings_ms,
        boxes=boxes,
    )

    write_metrics_csv(
        csv_path,
        metrics=result.metrics,
        timings_ms=result.timings_ms,
    )

    print()
    print("Saving BEV summary...")
    render_pipeline_bev_summary(
        result,
        labels=labels,
        calibration=calibration,
        save_path=output_dir / f"bev_summary_{args.frame}.png",
        show=False,
    )

    print("Saving BEV single scene...")
    render_pipeline_bev_single(
        result,
        labels=labels,
        calibration=calibration,
        save_path=output_dir / f"bev_scene_{args.frame}.png",
        show=False,
    )

    print("Saving LiDAR projection...")
    finite_points = filter_finite_points(result.raw_points)
    cropped_points = crop_roi(finite_points)

    projection = project_velodyne_to_camera_image(
        points=cropped_points,
        calibration=calibration,
        image_shape=image.shape,
        camera="P2",
    )

    draw_projected_lidar_on_image(
        image=image,
        projection=projection,
        labels=labels,
        title=f"KITTI LiDAR projection - frame {args.frame}",
        save_path=output_dir / f"lidar_projection_{args.frame}.png",
        show=False,
    )

    print("Saving camera 3D box overlay...")
    draw_camera_3d_box_overlay(
        image=image,
        calibration=calibration,
        proposal_boxes=result.boxes,
        labels=labels,
        camera="P2",
        max_proposals=args.max_camera_boxes,
        title=f"Camera 3D box overlay - frame {args.frame}",
        save_path=output_dir / f"camera_3d_boxes_{args.frame}.png",
        show=False,
    )

    write_demo_summary_markdown(
        summary_path,
        frame_id=args.frame,
        metrics=result.metrics,
        output_dir=output_dir,
    )

    print()
    print("=" * 60)
    print("Demo export complete")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Metrics CSV:      {csv_path}")
    print(f"Result JSON:      {json_path}")
    print(f"Summary Markdown: {summary_path}")
    print()
    print("Main images:")
    print(f"  {output_dir / f'bev_summary_{args.frame}.png'}")
    print(f"  {output_dir / f'bev_scene_{args.frame}.png'}")
    print(f"  {output_dir / f'lidar_projection_{args.frame}.png'}")
    print(f"  {output_dir / f'camera_3d_boxes_{args.frame}.png'}")


if __name__ == "__main__":
    main()
