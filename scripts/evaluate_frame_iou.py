import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from autopercept3d.datasets.kitti import KITTIDataset
from autopercept3d.evaluation.matching import (
    aabb_iou,
    evaluate_bev_proposals,
    evaluation_to_dict,
)
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


def save_evaluation_json(path: Path, payload: dict) -> None:
    """
    Save evaluation output as JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def print_evaluation_table(payload: dict) -> None:
    """
    Print compact frame evaluation result.
    """
    print()
    print("BEV proposal evaluation")
    print("-" * 60)
    print(f"Proposals:       {payload['num_proposals']}")
    print(f"Ground truth:    {payload['num_ground_truth']}")
    print(f"Matches:         {payload['num_matches']}")
    print(f"Precision:       {payload['precision']:.3f}")
    print(f"Recall:          {payload['recall']:.3f}")
    print(f"Mean IoU:        {payload['mean_iou']:.3f}")

    print()
    print("Matches:")
    print("proposal_id | gt_id | gt_label | IoU")
    print("-" * 42)

    for match in payload["matches"]:
        print(
            f"{match['proposal_id']:>11} | "
            f"{match['gt_id']:>5} | "
            f"{match['gt_label']:<8} | "
            f"{match['iou']:.3f}"
        )


def plot_evaluation_bev(result, labels, calibration, eval_result, save_path: Path | None) -> None:
    """
    Plot BEV proposals and ground truth for evaluation inspection.
    """
    plt.figure(figsize=(10, 8))

    plot_cluster_points_bev(
        result.filtered_cluster_points,
        result.filtered_cluster_labels,
    )

    plot_proposal_boxes_bev(result.boxes, show_ids=True)
    plot_ground_truth_boxes_bev(labels, calibration)

    title = (
        f"BEV proposal evaluation - frame {result.frame_id} | "
        f"matches={len(eval_result.matches)} | "
        f"recall={eval_result.recall:.2f} | "
        f"mean IoU={eval_result.mean_iou:.2f}"
    )

    set_bev_axes(title)
    plt.legend(markerscale=5)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Saved evaluation plot to: {save_path}")

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate AutoPercept3D proposal boxes against KITTI labels using simple BEV IoU."
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
        help="KITTI frame id.",
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
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save evaluation JSON.",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Optional path to save BEV evaluation plot.",
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

    labels = dataset.load_labels(args.frame)
    calibration = dataset.load_calibration(args.frame)

    result = run_classical_lidar_pipeline(
        dataset=dataset,
        frame_id=args.frame,
        config=config,
    )

    eval_result = evaluate_bev_proposals(
        proposal_boxes_3d=result.boxes,
        raw_labels=labels,
        calibration=calibration,
        iou_threshold=args.iou_threshold,
    )

    payload = evaluation_to_dict(eval_result)
    payload["frame_id"] = args.frame
    payload["iou_threshold"] = args.iou_threshold
    payload["pipeline_metrics"] = result.metrics

    print("=" * 60)
    print(f"BEV IoU evaluation for KITTI frame: {args.frame}")
    print("=" * 60)
    print("Note: this is a lightweight AABB BEV metric, not official KITTI evaluation.")

    print_evaluation_table(payload)

    if args.output_json:
        save_evaluation_json(Path(args.output_json), payload)
        print(f"Saved evaluation JSON to: {args.output_json}")

    if args.save_plot:
        plot_evaluation_bev(
            result=result,
            labels=labels,
            calibration=calibration,
            eval_result=eval_result,
            save_path=Path(args.save_plot),
        )


if __name__ == "__main__":
    main()
