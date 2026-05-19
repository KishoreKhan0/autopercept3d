from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

from autopercept3d.datasets.kitti import KITTIDataset
from autopercept3d.perception.pipeline import (
    ClassicalPipelineConfig,
    run_classical_lidar_pipeline,
)


def list_kitti_frame_ids(
    dataset: KITTIDataset,
    start_index: int = 0,
    max_frames: int = 10,
) -> list[str]:
    """
    List KITTI frame ids from the Velodyne folder.

    Example returned ids:
        ["000000", "000001", "000002"]
    """
    frame_ids = sorted(path.stem for path in dataset.velodyne_dir.glob("*.bin"))

    if start_index < 0:
        raise ValueError("start_index must be >= 0")

    if max_frames <= 0:
        raise ValueError("max_frames must be > 0")

    return frame_ids[start_index : start_index + max_frames]


def run_batch(
    dataset: KITTIDataset,
    frame_ids: list[str],
    config: ClassicalPipelineConfig,
) -> list[dict]:
    """
    Run the classical LiDAR pipeline over multiple KITTI frames.

    Returns one metrics dictionary per frame.
    """
    all_metrics: list[dict] = []

    for index, frame_id in enumerate(frame_ids, start=1):
        print(f"[{index}/{len(frame_ids)}] Processing frame {frame_id}...")

        try:
            result = run_classical_lidar_pipeline(dataset, frame_id, config)
            row = dict(result.metrics)

            for key, value in result.timings_ms.items():
                row[f"time_{key}_ms"] = float(value)

            row["status"] = "ok"
            row["error"] = ""

        except Exception as exc:
            row = {
                "frame_id": frame_id,
                "status": "error",
                "error": str(exc),
            }
            print(f"  ERROR: {exc}")

        all_metrics.append(row)

    return all_metrics


def write_metrics_csv(rows: list[dict], output_path: Path) -> None:
    """
    Write batch metrics to a CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_batch(rows: list[dict]) -> dict:
    """
    Compute compact summary statistics for successful frames.
    """
    ok_rows = [row for row in rows if row.get("status") == "ok"]

    if not ok_rows:
        return {
            "frames_total": len(rows),
            "frames_ok": 0,
            "frames_error": len(rows),
        }

    def values(key: str) -> list[float]:
        return [float(row[key]) for row in ok_rows if key in row and row[key] != ""]

    total_times = values("total_time_ms")
    fps_values = values("approx_fps")
    boxes_values = values("boxes")
    cluster_values = values("filtered_clusters")
    non_ground_values = values("non_ground_points")

    return {
        "frames_total": len(rows),
        "frames_ok": len(ok_rows),
        "frames_error": len(rows) - len(ok_rows),
        "avg_total_time_ms": mean(total_times) if total_times else 0.0,
        "avg_fps": mean(fps_values) if fps_values else 0.0,
        "avg_boxes": mean(boxes_values) if boxes_values else 0.0,
        "avg_filtered_clusters": mean(cluster_values) if cluster_values else 0.0,
        "avg_non_ground_points": mean(non_ground_values) if non_ground_values else 0.0,
    }


def plot_metric_over_frames(
    rows: list[dict],
    metric_name: str,
    output_path: Path,
    title: str,
    y_label: str,
) -> None:
    """
    Save a simple line plot for one metric across frames.
    """
    ok_rows = [row for row in rows if row.get("status") == "ok" and metric_name in row]

    if not ok_rows:
        return

    frame_ids = [row["frame_id"] for row in ok_rows]
    values = [float(row[metric_name]) for row in ok_rows]

    plt.figure(figsize=(10, 5))
    plt.plot(frame_ids, values, marker="o")
    plt.title(title)
    plt.xlabel("Frame ID")
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_batch_plots(rows: list[dict], output_dir: Path) -> None:
    """
    Save standard batch evaluation plots.
    """
    plot_metric_over_frames(
        rows,
        metric_name="total_time_ms",
        output_path=output_dir / "timing_total_ms.png",
        title="Total Processing Time per Frame",
        y_label="Time [ms]",
    )

    plot_metric_over_frames(
        rows,
        metric_name="approx_fps",
        output_path=output_dir / "approx_fps.png",
        title="Approximate FPS per Frame",
        y_label="FPS",
    )

    plot_metric_over_frames(
        rows,
        metric_name="boxes",
        output_path=output_dir / "boxes_per_frame.png",
        title="Generated 3D Boxes per Frame",
        y_label="Number of boxes",
    )

    plot_metric_over_frames(
        rows,
        metric_name="filtered_clusters",
        output_path=output_dir / "clusters_per_frame.png",
        title="Filtered Clusters per Frame",
        y_label="Number of clusters",
    )
