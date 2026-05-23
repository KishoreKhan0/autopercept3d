# AutoPercept3D Demo Summary

Demo frame: `000264`

## Pipeline Metrics

| Metric | Value |
|---|---:|
| Raw LiDAR points | 101481 |
| Cropped points | 54571 |
| Downsampled points | 10055 |
| Ground points | 3611 |
| Non-ground points | 6444 |
| Filtered clusters | 34 |
| Proposal boxes | 34 |
| Total runtime | 146.19 ms |
| Approx FPS | 6.84 |

## Generated Assets

```text
showcase_export_250_269\frame_demo\bev_summary_000264.png
showcase_export_250_269\frame_demo\bev_scene_000264.png
showcase_export_250_269\frame_demo\lidar_projection_000264.png
showcase_export_250_269\frame_demo\camera_3d_boxes_000264.png
showcase_export_250_269\frame_demo\frame_000264_result.json
showcase_export_250_269\frame_demo\frame_000264_metrics.csv
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
