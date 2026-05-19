from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

from autopercept3d.datasets.kitti import KITTIDataset
from autopercept3d.evaluation.matching import evaluate_bev_proposals
from autopercept3d.perception.pipeline import (
    ClassicalPipelineConfig,
    run_classical_lidar_pipeline,
)


def list_frame_ids(
    dataset: KITTIDataset,
    start_index: int = 0,
    max_frames: int = 10,
) -> list[str]:
    """
    List KITTI frame ids from the Velodyne folder.
    """
    frame_ids = sorted(path.stem for path in dataset.velodyne_dir.glob("*.bin"))

    if start_index < 0:
        raise ValueError("start_index must be >= 0")

    if max_frames <= 0:
        raise ValueError("max_frames must be > 0")

    return frame_ids[start_index : start_index + max_frames]


def run_batch_iou_evaluation(
    dataset: KITTIDataset,
    frame_ids: list[str],
    config: ClassicalPipelineConfig,
    iou_threshold: float = 0.10,
) -> list[dict]:
    """
    Run the full pipeline and simple BEV IoU evaluation for multiple frames.

    This is NOT official KITTI evaluation.
    It is an internal proposal-quality check.
    """
    rows: list[dict] = []

    for index, frame_id in enumerate(frame_ids, start=1):
        print(f"[{index}/{len(frame_ids)}] Evaluating frame {frame_id}...")

        try:
            labels = dataset.load_labels(frame_id)
            calibration = dataset.load_calibration(frame_id)

            pipeline_result = run_classical_lidar_pipeline(
                dataset=dataset,
                frame_id=frame_id,
                config=config,
            )

            eval_result = evaluate_bev_proposals(
                proposal_boxes_3d=pipeline_result.boxes,
                raw_labels=labels,
                calibration=calibration,
                iou_threshold=iou_threshold,
            )

            row = {
                "frame_id": frame_id,
                "status": "ok",
                "error": "",
                "num_proposals": len(eval_result.proposal_boxes),
                "num_ground_truth": len(eval_result.ground_truth_boxes),
                "num_matches": len(eval_result.matches),
                "precision": eval_result.precision,
                "recall": eval_result.recall,
                "mean_iou": eval_result.mean_iou,
                "raw_points": pipeline_result.metrics["raw_points"],
                "cropped_points": pipeline_result.metrics["cropped_points"],
                "downsampled_points": pipeline_result.metrics["downsampled_points"],
                "non_ground_points": pipeline_result.metrics["non_ground_points"],
                "filtered_clusters": pipeline_result.metrics["filtered_clusters"],
                "boxes": pipeline_result.metrics["boxes"],
                "total_time_ms": pipeline_result.metrics["total_time_ms"],
                "approx_fps": pipeline_result.metrics["approx_fps"],
            }

        except Exception as exc:
            row = {
                "frame_id": frame_id,
                "status": "error",
                "error": str(exc),
            }
            print(f"  ERROR: {exc}")

        rows.append(row)

    return rows


def write_batch_iou_csv(rows: list[dict], output_path: Path) -> None:
    """
    Save batch IoU evaluation rows as CSV.
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


def summarize_batch_iou(rows: list[dict]) -> dict:
    """
    Summarize successful batch evaluation rows.
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

    return {
        "frames_total": len(rows),
        "frames_ok": len(ok_rows),
        "frames_error": len(rows) - len(ok_rows),
        "avg_precision": mean(values("precision")) if values("precision") else 0.0,
        "avg_recall": mean(values("recall")) if values("recall") else 0.0,
        "avg_mean_iou": mean(values("mean_iou")) if values("mean_iou") else 0.0,
        "avg_matches": mean(values("num_matches")) if values("num_matches") else 0.0,
        "avg_ground_truth": mean(values("num_ground_truth")) if values("num_ground_truth") else 0.0,
        "avg_proposals": mean(values("num_proposals")) if values("num_proposals") else 0.0,
        "avg_total_time_ms": mean(values("total_time_ms")) if values("total_time_ms") else 0.0,
        "avg_fps": mean(values("approx_fps")) if values("approx_fps") else 0.0,
    }


def plot_metric(
    rows: list[dict],
    metric_name: str,
    output_path: Path,
    title: str,
    y_label: str,
) -> None:
    """
    Save a simple metric-over-frame plot.
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


def save_batch_iou_plots(rows: list[dict], output_dir: Path) -> None:
    """
    Save standard batch IoU evaluation plots.
    """
    plot_metric(
        rows,
        metric_name="precision",
        output_path=output_dir / "precision_per_frame.png",
        title="BEV Proposal Precision per Frame",
        y_label="Precision",
    )

    plot_metric(
        rows,
        metric_name="recall",
        output_path=output_dir / "recall_per_frame.png",
        title="BEV Proposal Recall per Frame",
        y_label="Recall",
    )

    plot_metric(
        rows,
        metric_name="mean_iou",
        output_path=output_dir / "mean_iou_per_frame.png",
        title="Mean Matched BEV IoU per Frame",
        y_label="Mean IoU",
    )

    plot_metric(
        rows,
        metric_name="num_matches",
        output_path=output_dir / "matches_per_frame.png",
        title="Matched Proposals per Frame",
        y_label="Matches",
    )

    plot_metric(
        rows,
        metric_name="num_ground_truth",
        output_path=output_dir / "ground_truth_per_frame.png",
        title="Ground-Truth Objects per Frame",
        y_label="GT objects",
    )
