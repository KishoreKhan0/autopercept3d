from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrackDetection:
    """
    One object proposal converted into a tracker-friendly detection.
    """

    frame_id: str
    frame_index: int
    detection_id: int
    center_xyz: np.ndarray
    size_lwh: np.ndarray
    yaw_rad: float = 0.0
    score: float = 1.0
    num_points: int = 0
    source: str = "proposal"


@dataclass
class TrackRecord:
    """
    One stored detection assignment inside a tracklet.
    """

    frame_id: str
    frame_index: int
    track_id: int
    detection_id: int
    center_x: float
    center_y: float
    center_z: float
    velocity_x: float
    velocity_y: float
    speed_m_per_frame: float
    length: float
    width: float
    height: float
    yaw_rad: float
    yaw_deg: float
    score: float
    num_points: int
    age: int
    hits: int
    missed: int
    tracklet_length: int
    stable: bool


@dataclass
class TrackState:
    """
    Internal state for one active BEV tracklet.
    """

    track_id: int
    last_detection: TrackDetection
    center_xy: np.ndarray
    velocity_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    age: int = 1
    hits: int = 1
    missed: int = 0
    records: list[TrackRecord] = field(default_factory=list)

    def predict_xy(self) -> np.ndarray:
        return self.center_xy + self.velocity_xy * max(self.missed + 1, 1)

    def append_record(self, detection: TrackDetection, stable: bool = False) -> None:
        speed = float(np.linalg.norm(self.velocity_xy))
        length, width, height = detection.size_lwh
        x, y, z = detection.center_xyz

        self.records.append(
            TrackRecord(
                frame_id=detection.frame_id,
                frame_index=detection.frame_index,
                track_id=self.track_id,
                detection_id=detection.detection_id,
                center_x=float(x),
                center_y=float(y),
                center_z=float(z),
                velocity_x=float(self.velocity_xy[0]),
                velocity_y=float(self.velocity_xy[1]),
                speed_m_per_frame=speed,
                length=float(length),
                width=float(width),
                height=float(height),
                yaw_rad=float(detection.yaw_rad),
                yaw_deg=float(np.degrees(detection.yaw_rad)),
                score=float(detection.score),
                num_points=int(detection.num_points),
                age=int(self.age),
                hits=int(self.hits),
                missed=int(self.missed),
                tracklet_length=int(self.hits),
                stable=bool(stable),
            )
        )

    def update(self, detection: TrackDetection, velocity_smoothing: float = 0.55) -> None:
        new_center_xy = detection.center_xyz[:2].astype(float)
        dt = max(detection.frame_index - self.last_detection.frame_index, 1)
        observed_velocity = (new_center_xy - self.center_xy) / float(dt)

        self.velocity_xy = (
            velocity_smoothing * self.velocity_xy
            + (1.0 - velocity_smoothing) * observed_velocity
        )

        self.center_xy = new_center_xy
        self.last_detection = detection
        self.age += dt
        self.hits += 1
        self.missed = 0
        self.append_record(detection)

    def mark_missed(self) -> None:
        self.age += 1
        self.missed += 1


def _angle_difference(a: float, b: float) -> float:
    """
    Smallest absolute angular difference in radians.
    """
    diff = (a - b + np.pi) % (2.0 * np.pi) - np.pi
    return float(abs(diff))


def proposal_quality_score(num_points: int, size_lwh: np.ndarray) -> float:
    """
    Deterministic objectness-like score in [0, 1].

    This is not a classifier. It suppresses tiny/unstable fragments and favors
    clusters with enough LiDAR support and plausible 3D dimensions.
    """
    length, width, height = [float(v) for v in size_lwh]

    point_score = min(max(num_points, 0) / 140.0, 1.0)

    length_score = 1.0 if 0.5 <= length <= 8.0 else 0.0
    width_score = 1.0 if 0.25 <= width <= 4.5 else 0.0
    height_score = 1.0 if 0.35 <= height <= 4.0 else 0.0
    shape_score = (length_score + width_score + height_score) / 3.0

    volume = max(length * width * height, 0.0)
    volume_score = min(volume / 16.0, 1.0)

    elongation = max(length, width) / max(min(length, width), 1e-6)
    elongation_score = 1.0 if elongation <= 8.0 else 0.4

    return float(
        0.40 * point_score
        + 0.35 * shape_score
        + 0.15 * volume_score
        + 0.10 * elongation_score
    )


def detection_passes_filter(
    detection: TrackDetection,
    min_points: int = 18,
    min_score: float = 0.20,
    x_min: float = 0.0,
    x_max: float = 80.0,
    y_abs_max: float = 35.0,
    min_length: float = 0.35,
    max_length: float = 9.0,
    min_width: float = 0.20,
    max_width: float = 5.0,
    min_height: float = 0.25,
    max_height: float = 4.5,
) -> bool:
    x, y = detection.center_xyz[:2]
    length, width, height = detection.size_lwh

    if detection.num_points < min_points:
        return False

    if detection.score < min_score:
        return False

    if x < x_min or x > x_max:
        return False

    if abs(y) > y_abs_max:
        return False

    if not (min_length <= length <= max_length):
        return False

    if not (min_width <= width <= max_width):
        return False

    if not (min_height <= height <= max_height):
        return False

    return True


def filter_detections(
    detections: list[TrackDetection],
    min_points: int = 18,
    min_score: float = 0.20,
) -> list[TrackDetection]:
    return [
        detection
        for detection in detections
        if detection_passes_filter(
            detection,
            min_points=min_points,
            min_score=min_score,
        )
    ]


class StableTrackletTracker:
    """
    Stable BEV tracklet miner over LiDAR proposal boxes.

    This is designed for the current AutoPercept3D KITTI-object setup where
    proposals are noisy and class-free. It does not pretend to be an official
    tracking benchmark. It mines stable short tracklets by combining:

        - pre-tracking proposal filtering
        - constant-velocity prediction
        - distance + size + yaw association cost
        - tracklet finalization
        - stable-tracklet export only

    This is much stricter than plotting every raw proposal ID.
    """

    def __init__(
        self,
        max_match_distance: float = 4.5,
        max_missed: int = 2,
        min_stable_hits: int = 3,
        velocity_smoothing: float = 0.55,
        max_size_ratio: float = 3.5,
        yaw_weight: float = 0.08,
        size_weight: float = 0.20,
    ) -> None:
        self.max_match_distance = float(max_match_distance)
        self.max_missed = int(max_missed)
        self.min_stable_hits = int(min_stable_hits)
        self.velocity_smoothing = float(velocity_smoothing)
        self.max_size_ratio = float(max_size_ratio)
        self.yaw_weight = float(yaw_weight)
        self.size_weight = float(size_weight)

        self.next_track_id = 1
        self.active_tracks: dict[int, TrackState] = {}
        self.finished_tracks: list[TrackState] = []

    def _new_track(self, detection: TrackDetection) -> TrackState:
        track = TrackState(
            track_id=self.next_track_id,
            last_detection=detection,
            center_xy=detection.center_xyz[:2].astype(float),
        )
        track.append_record(detection)
        self.active_tracks[track.track_id] = track
        self.next_track_id += 1
        return track

    def _size_ratio_ok(self, track: TrackState, detection: TrackDetection) -> bool:
        old_size = np.maximum(track.last_detection.size_lwh, 1e-6)
        new_size = np.maximum(detection.size_lwh, 1e-6)

        ratio = np.maximum(old_size / new_size, new_size / old_size)
        return bool(np.all(ratio <= self.max_size_ratio))

    def _association_cost(self, track: TrackState, detection: TrackDetection) -> float | None:
        predicted_xy = track.predict_xy()
        detection_xy = detection.center_xyz[:2]

        distance = float(np.linalg.norm(predicted_xy - detection_xy))

        if distance > self.max_match_distance:
            return None

        if not self._size_ratio_ok(track, detection):
            return None

        old_size = np.maximum(track.last_detection.size_lwh, 1e-6)
        new_size = np.maximum(detection.size_lwh, 1e-6)
        size_cost = float(np.mean(np.abs(np.log(new_size / old_size))))

        yaw_cost = _angle_difference(track.last_detection.yaw_rad, detection.yaw_rad) / np.pi

        # Lower detection score should make association slightly less attractive.
        score_bonus = 0.10 * detection.score

        cost = (
            distance / self.max_match_distance
            + self.size_weight * size_cost
            + self.yaw_weight * yaw_cost
            - score_bonus
        )

        return float(cost)

    def update(self, frame_id: str, detections: list[TrackDetection]) -> list[TrackRecord]:
        if not self.active_tracks:
            for detection in detections:
                self._new_track(detection)
            return []

        candidate_pairs: list[tuple[float, int, int]] = []

        track_ids = list(self.active_tracks.keys())

        for track_id in track_ids:
            track = self.active_tracks[track_id]

            for det_index, detection in enumerate(detections):
                cost = self._association_cost(track, detection)

                if cost is not None:
                    candidate_pairs.append((cost, track_id, det_index))

        candidate_pairs.sort(key=lambda item: item[0])

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        current_records: list[TrackRecord] = []

        for _, track_id, det_index in candidate_pairs:
            if track_id in matched_tracks or det_index in matched_detections:
                continue

            track = self.active_tracks[track_id]
            detection = detections[det_index]

            track.update(
                detection=detection,
                velocity_smoothing=self.velocity_smoothing,
            )

            matched_tracks.add(track_id)
            matched_detections.add(det_index)
            current_records.append(track.records[-1])

        tracks_to_finish: list[int] = []

        for track_id, track in self.active_tracks.items():
            if track_id in matched_tracks:
                continue

            track.mark_missed()

            if track.missed > self.max_missed:
                tracks_to_finish.append(track_id)

        for track_id in tracks_to_finish:
            self.finished_tracks.append(self.active_tracks.pop(track_id))

        for det_index, detection in enumerate(detections):
            if det_index in matched_detections:
                continue

            self._new_track(detection)

        return current_records

    def finish(self) -> None:
        for track in self.active_tracks.values():
            self.finished_tracks.append(track)
        self.active_tracks = {}

    def all_tracks(self) -> list[TrackState]:
        return list(self.finished_tracks) + list(self.active_tracks.values())

    def stable_tracks(
        self,
        min_hits: int | None = None,
        max_mean_speed: float = 10.0,
        min_mean_score: float = 0.25,
    ) -> list[TrackState]:
        if min_hits is None:
            min_hits = self.min_stable_hits

        stable: list[TrackState] = []

        for track in self.all_tracks():
            if track.hits < min_hits:
                continue

            speeds = [record.speed_m_per_frame for record in track.records[1:]]
            mean_speed = float(np.mean(speeds)) if speeds else 0.0

            scores = [record.score for record in track.records]
            mean_score = float(np.mean(scores)) if scores else 0.0

            if mean_speed > max_mean_speed:
                continue

            if mean_score < min_mean_score:
                continue

            stable.append(track)

        stable.sort(key=lambda item: item.hits, reverse=True)
        return stable


# Backward-compatible aliases.
ConstantVelocityTracker = StableTrackletTracker
NearestNeighborTracker = StableTrackletTracker


def detections_from_oriented_boxes(
    frame_id: str,
    boxes,
    frame_index: int = 0,
) -> list[TrackDetection]:
    detections: list[TrackDetection] = []

    for box in boxes:
        size_lwh = np.asarray(box.size_lwh, dtype=float)
        num_points = int(getattr(box, "num_points", 0))
        score = proposal_quality_score(num_points=num_points, size_lwh=size_lwh)

        detections.append(
            TrackDetection(
                frame_id=frame_id,
                frame_index=frame_index,
                detection_id=int(box.cluster_id),
                center_xyz=np.asarray(box.center_xyz, dtype=float),
                size_lwh=size_lwh,
                yaw_rad=float(box.yaw_rad),
                score=score,
                num_points=num_points,
                source="oriented_box",
            )
        )

    return detections


def detections_from_axis_aligned_boxes(
    frame_id: str,
    boxes,
    frame_index: int = 0,
) -> list[TrackDetection]:
    detections: list[TrackDetection] = []

    for box in boxes:
        center = np.asarray(box.center_xyz, dtype=float)

        if hasattr(box, "size_xyz"):
            size = np.asarray(box.size_xyz, dtype=float)
        elif hasattr(box, "size_lwh"):
            size = np.asarray(box.size_lwh, dtype=float)
        elif hasattr(box, "min_xyz") and hasattr(box, "max_xyz"):
            size = np.asarray(box.max_xyz, dtype=float) - np.asarray(box.min_xyz, dtype=float)
        else:
            size = np.zeros(3, dtype=float)

        num_points = int(getattr(box, "num_points", getattr(box, "points", 0)))
        score = proposal_quality_score(num_points=num_points, size_lwh=size)

        detections.append(
            TrackDetection(
                frame_id=frame_id,
                frame_index=frame_index,
                detection_id=int(box.cluster_id),
                center_xyz=center,
                size_lwh=size,
                yaw_rad=0.0,
                score=score,
                num_points=num_points,
                source="axis_aligned_box",
            )
        )

    return detections


def _record_to_dict(record: TrackRecord, stable: bool) -> dict:
    return {
        "frame_id": record.frame_id,
        "frame_index": record.frame_index,
        "track_id": record.track_id,
        "detection_id": record.detection_id,
        "center_x": record.center_x,
        "center_y": record.center_y,
        "center_z": record.center_z,
        "velocity_x": record.velocity_x,
        "velocity_y": record.velocity_y,
        "speed_m_per_frame": record.speed_m_per_frame,
        "length": record.length,
        "width": record.width,
        "height": record.height,
        "yaw_rad": record.yaw_rad,
        "yaw_deg": record.yaw_deg,
        "score": record.score,
        "num_points": record.num_points,
        "age": record.age,
        "hits": record.hits,
        "missed": record.missed,
        "tracklet_length": record.tracklet_length,
        "stable": stable,
    }


def track_records_to_dicts(records: list[TrackRecord]) -> list[dict]:
    return [_record_to_dict(record, stable=record.stable) for record in records]


def tracks_to_record_dicts(tracks: list[TrackState], stable: bool) -> list[dict]:
    rows: list[dict] = []

    for track in tracks:
        for record in track.records:
            row = _record_to_dict(record, stable=stable)
            row["final_tracklet_length"] = track.hits
            rows.append(row)

    rows.sort(key=lambda row: (int(row["track_id"]), int(row["frame_index"])))
    return rows


def tracks_to_summary_rows(tracks: list[TrackState]) -> list[dict]:
    rows: list[dict] = []

    for track in tracks:
        if not track.records:
            continue

        xs = [record.center_x for record in track.records]
        ys = [record.center_y for record in track.records]
        speeds = [record.speed_m_per_frame for record in track.records[1:]]
        scores = [record.score for record in track.records]

        rows.append(
            {
                "track_id": track.track_id,
                "tracklet_length": track.hits,
                "first_frame": track.records[0].frame_id,
                "last_frame": track.records[-1].frame_id,
                "start_x": xs[0],
                "start_y": ys[0],
                "end_x": xs[-1],
                "end_y": ys[-1],
                "displacement_m": float(np.linalg.norm([xs[-1] - xs[0], ys[-1] - ys[0]])),
                "mean_speed_m_per_frame": float(np.mean(speeds)) if speeds else 0.0,
                "mean_score": float(np.mean(scores)) if scores else 0.0,
            }
        )

    rows.sort(key=lambda row: row["tracklet_length"], reverse=True)
    return rows


def summarize_tracks(tracks_or_records) -> dict:
    """
    Summarize either TrackState objects or TrackRecord objects.

    This stays backward-compatible with earlier scripts.
    """
    if not tracks_or_records:
        return {
            "assigned_detections": 0,
            "unique_tracks": 0,
            "stable_tracks": 0,
            "longest_tracklet": 0,
            "mean_tracklet_length": 0.0,
        }

    first = tracks_or_records[0]

    if isinstance(first, TrackState):
        tracks = tracks_or_records
        lengths = [track.hits for track in tracks]

        return {
            "assigned_detections": int(sum(lengths)),
            "unique_tracks": int(len(tracks)),
            "stable_tracks": int(len(tracks)),
            "longest_tracklet": int(max(lengths)) if lengths else 0,
            "mean_tracklet_length": float(np.mean(lengths)) if lengths else 0.0,
        }

    track_lengths: dict[int, int] = {}

    for record in tracks_or_records:
        track_lengths[record.track_id] = track_lengths.get(record.track_id, 0) + 1

    lengths = list(track_lengths.values())

    return {
        "assigned_detections": len(tracks_or_records),
        "unique_tracks": len(track_lengths),
        "stable_tracks": 0,
        "longest_tracklet": int(max(lengths)) if lengths else 0,
        "mean_tracklet_length": float(np.mean(lengths)) if lengths else 0.0,
    }
