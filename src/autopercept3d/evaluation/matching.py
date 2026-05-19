from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from autopercept3d.geometry.kitti_transforms import (
    kitti_label_corners_velodyne,
    parse_kitti_label,
)
from autopercept3d.perception.bounding_boxes import AxisAlignedBox3D


@dataclass
class BEVBox:
    """
    Axis-aligned bird's-eye-view box.

    This is used for simple proposal evaluation.

    Important:
        This is NOT official KITTI 3D evaluation.
        It is a lightweight internal metric for debugging our proposal quality.
    """

    box_id: int | str
    label: str
    min_x: float
    min_y: float
    max_x: float
    max_y: float


@dataclass
class BEVMatch:
    """
    A matched proposal/ground-truth pair.
    """

    proposal_id: int
    gt_id: int
    gt_label: str
    iou: float


@dataclass
class BEVEvaluationResult:
    """
    Simple BEV proposal evaluation result.
    """

    proposal_boxes: list[BEVBox]
    ground_truth_boxes: list[BEVBox]
    matches: list[BEVMatch]
    unmatched_proposal_ids: list[int]
    unmatched_gt_ids: list[int]
    precision: float
    recall: float
    mean_iou: float


def proposal_to_bev_box(box: AxisAlignedBox3D) -> BEVBox:
    """
    Convert an AutoPercept3D proposal box to a BEV AABB.
    """
    return BEVBox(
        box_id=box.cluster_id,
        label="proposal",
        min_x=float(box.min_xyz[0]),
        min_y=float(box.min_xyz[1]),
        max_x=float(box.max_xyz[0]),
        max_y=float(box.max_xyz[1]),
    )


def kitti_label_to_bev_box(
    raw_label: dict,
    calibration: dict[str, np.ndarray],
    gt_id: int,
) -> BEVBox | None:
    """
    Convert one KITTI 3D ground-truth label to a BEV axis-aligned box.

    KITTI 3D boxes are oriented. For this simple metric, we convert the
    oriented BEV polygon into an axis-aligned bounding rectangle.
    """
    label = parse_kitti_label(raw_label)

    if label.object_type == "DontCare":
        return None

    corners_velodyne = kitti_label_corners_velodyne(label, calibration)
    xy = corners_velodyne[:, :2]

    return BEVBox(
        box_id=gt_id,
        label=label.object_type,
        min_x=float(xy[:, 0].min()),
        min_y=float(xy[:, 1].min()),
        max_x=float(xy[:, 0].max()),
        max_y=float(xy[:, 1].max()),
    )


def aabb_area(box: BEVBox) -> float:
    """
    Area of a BEV axis-aligned box.
    """
    width = max(0.0, box.max_x - box.min_x)
    height = max(0.0, box.max_y - box.min_y)
    return width * height


def aabb_iou(box_a: BEVBox, box_b: BEVBox) -> float:
    """
    Intersection-over-union for two BEV axis-aligned boxes.
    """
    inter_min_x = max(box_a.min_x, box_b.min_x)
    inter_min_y = max(box_a.min_y, box_b.min_y)
    inter_max_x = min(box_a.max_x, box_b.max_x)
    inter_max_y = min(box_a.max_y, box_b.max_y)

    inter_width = max(0.0, inter_max_x - inter_min_x)
    inter_height = max(0.0, inter_max_y - inter_min_y)
    intersection = inter_width * inter_height

    union = aabb_area(box_a) + aabb_area(box_b) - intersection

    if union <= 0.0:
        return 0.0

    return intersection / union


def compute_iou_matrix(
    proposal_boxes: list[BEVBox],
    ground_truth_boxes: list[BEVBox],
) -> np.ndarray:
    """
    Compute IoU matrix with shape:
        number of proposals x number of ground-truth boxes
    """
    matrix = np.zeros((len(proposal_boxes), len(ground_truth_boxes)), dtype=np.float64)

    for proposal_index, proposal in enumerate(proposal_boxes):
        for gt_index, gt in enumerate(ground_truth_boxes):
            matrix[proposal_index, gt_index] = aabb_iou(proposal, gt)

    return matrix


def greedy_match_iou(
    proposal_boxes: list[BEVBox],
    ground_truth_boxes: list[BEVBox],
    iou_threshold: float = 0.10,
) -> list[BEVMatch]:
    """
    Greedy proposal-to-ground-truth matching.

    Steps:
        1. Compute all pairwise IoUs.
        2. Pick the highest IoU pair.
        3. Remove that proposal and ground-truth box.
        4. Repeat until best IoU is below threshold.

    Low default threshold:
        Our current boxes are simple cluster proposals, not refined detector boxes.
        A low threshold is useful for checking whether a proposal roughly covers
        an object.
    """
    if not proposal_boxes or not ground_truth_boxes:
        return []

    iou_matrix = compute_iou_matrix(proposal_boxes, ground_truth_boxes)

    used_proposals: set[int] = set()
    used_gt: set[int] = set()
    matches: list[BEVMatch] = []

    while True:
        best_iou = -1.0
        best_pair: tuple[int, int] | None = None

        for proposal_index in range(len(proposal_boxes)):
            if proposal_index in used_proposals:
                continue

            for gt_index in range(len(ground_truth_boxes)):
                if gt_index in used_gt:
                    continue

                iou = float(iou_matrix[proposal_index, gt_index])

                if iou > best_iou:
                    best_iou = iou
                    best_pair = (proposal_index, gt_index)

        if best_pair is None or best_iou < iou_threshold:
            break

        proposal_index, gt_index = best_pair
        used_proposals.add(proposal_index)
        used_gt.add(gt_index)

        proposal = proposal_boxes[proposal_index]
        gt = ground_truth_boxes[gt_index]

        matches.append(
            BEVMatch(
                proposal_id=int(proposal.box_id),
                gt_id=int(gt.box_id),
                gt_label=gt.label,
                iou=best_iou,
            )
        )

    return matches


def evaluate_bev_proposals(
    proposal_boxes_3d: list[AxisAlignedBox3D],
    raw_labels: list[dict],
    calibration: dict[str, np.ndarray],
    iou_threshold: float = 0.10,
) -> BEVEvaluationResult:
    """
    Evaluate proposal boxes against KITTI ground truth using simple BEV AABB IoU.

    This is a debug/evaluation helper, not official KITTI evaluation.
    """
    proposal_boxes = [proposal_to_bev_box(box) for box in proposal_boxes_3d]

    ground_truth_boxes: list[BEVBox] = []

    gt_id = 0
    for raw_label in raw_labels:
        gt_box = kitti_label_to_bev_box(raw_label, calibration, gt_id=gt_id)

        if gt_box is not None:
            ground_truth_boxes.append(gt_box)
            gt_id += 1

    matches = greedy_match_iou(
        proposal_boxes,
        ground_truth_boxes,
        iou_threshold=iou_threshold,
    )

    matched_proposal_ids = {match.proposal_id for match in matches}
    matched_gt_ids = {match.gt_id for match in matches}

    unmatched_proposal_ids = [
        int(box.box_id)
        for box in proposal_boxes
        if int(box.box_id) not in matched_proposal_ids
    ]

    unmatched_gt_ids = [
        int(box.box_id)
        for box in ground_truth_boxes
        if int(box.box_id) not in matched_gt_ids
    ]

    precision = len(matches) / max(len(proposal_boxes), 1)
    recall = len(matches) / max(len(ground_truth_boxes), 1)
    mean_iou = float(np.mean([match.iou for match in matches])) if matches else 0.0

    return BEVEvaluationResult(
        proposal_boxes=proposal_boxes,
        ground_truth_boxes=ground_truth_boxes,
        matches=matches,
        unmatched_proposal_ids=unmatched_proposal_ids,
        unmatched_gt_ids=unmatched_gt_ids,
        precision=precision,
        recall=recall,
        mean_iou=mean_iou,
    )


def evaluation_to_dict(result: BEVEvaluationResult) -> dict:
    """
    Convert evaluation result to JSON-friendly dictionary.
    """
    return {
        "num_proposals": len(result.proposal_boxes),
        "num_ground_truth": len(result.ground_truth_boxes),
        "num_matches": len(result.matches),
        "precision": result.precision,
        "recall": result.recall,
        "mean_iou": result.mean_iou,
        "matches": [
            {
                "proposal_id": match.proposal_id,
                "gt_id": match.gt_id,
                "gt_label": match.gt_label,
                "iou": match.iou,
            }
            for match in result.matches
        ],
        "unmatched_proposal_ids": result.unmatched_proposal_ids,
        "unmatched_gt_ids": result.unmatched_gt_ids,
    }
