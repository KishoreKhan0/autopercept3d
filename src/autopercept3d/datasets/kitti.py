from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


class KITTIDataset:
    """
    Minimal KITTI Object Detection dataset loader.

    Expected folder structure:

    dataset_root/
        training/
            image_2/
            velodyne/
            calib/
            label_2/
    """

    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root).expanduser()
        self.training_dir = self.dataset_root / "training"

        self.image_dir = self.training_dir / "image_2"
        self.velodyne_dir = self.training_dir / "velodyne"
        self.calib_dir = self.training_dir / "calib"
        self.label_dir = self.training_dir / "label_2"

        self._check_structure()

    def _check_structure(self) -> None:
        required_dirs = [
            self.training_dir,
            self.image_dir,
            self.velodyne_dir,
            self.calib_dir,
            self.label_dir,
        ]

        for directory in required_dirs:
            if not directory.exists():
                raise FileNotFoundError(f"Missing required KITTI directory: {directory}")

    def load_image(self, frame_id: str) -> np.ndarray:
        image_path = self.image_dir / f"{frame_id}.png"

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        # OpenCV loads images as BGR. Convert to RGB for normal visualization later.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def load_point_cloud(self, frame_id: str) -> np.ndarray:
        point_cloud_path = self.velodyne_dir / f"{frame_id}.bin"

        if not point_cloud_path.exists():
            raise FileNotFoundError(f"Point cloud not found: {point_cloud_path}")

        # KITTI Velodyne files store float32 values:
        # x, y, z, intensity
        points = np.fromfile(str(point_cloud_path), dtype=np.float32)

        if points.size % 4 != 0:
            raise ValueError(
                f"Invalid KITTI point cloud file. Expected values divisible by 4, "
                f"got {points.size} values in {point_cloud_path}"
            )

        points = points.reshape(-1, 4)

        return points

    def load_calibration(self, frame_id: str) -> Dict[str, np.ndarray]:
        calib_path = self.calib_dir / f"{frame_id}.txt"

        if not calib_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {calib_path}")

        calibration = {}

        with open(calib_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                key, value = line.split(":", 1)
                numbers = np.array([float(x) for x in value.strip().split()])

                calibration[key] = numbers

        return calibration

    def load_labels(self, frame_id: str) -> List[Dict[str, object]]:
        label_path = self.label_dir / f"{frame_id}.txt"

        if not label_path.exists():
            raise FileNotFoundError(f"Label file not found: {label_path}")

        labels = []

        with open(label_path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()

                if not parts:
                    continue

                if len(parts) < 15:
                    raise ValueError(
                        f"Invalid KITTI label line in {label_path}: {line}"
                    )

                label = {
                    "type": parts[0],
                    "truncated": float(parts[1]),
                    "occluded": int(parts[2]),
                    "alpha": float(parts[3]),
                    "bbox_2d": [float(x) for x in parts[4:8]],
                    "dimensions_hwl": [float(x) for x in parts[8:11]],
                    "location_xyz_camera": [float(x) for x in parts[11:14]],
                    "rotation_y": float(parts[14]),
                }

                labels.append(label)

        return labels

    def inspect_frame(self, frame_id: str) -> None:
        image = self.load_image(frame_id)
        points = self.load_point_cloud(frame_id)
        calibration = self.load_calibration(frame_id)
        labels = self.load_labels(frame_id)

        print("=" * 60)
        print(f"KITTI frame: {frame_id}")
        print("=" * 60)

        print(f"Dataset root: {self.dataset_root}")
        print(f"Image shape: {image.shape}")
        print(f"Point cloud shape: {points.shape}")
        print(f"Number of LiDAR points: {points.shape[0]}")
        print("Point format: x, y, z, intensity")

        print()
        print("First 5 LiDAR points:")
        print(points[:5])

        print()
        print("Calibration keys:")
        for key in calibration.keys():
            print(f"  - {key}: {calibration[key].shape}")

        print()
        print(f"Number of labels: {len(labels)}")

        for index, label in enumerate(labels):
            print(f"  {index + 1}. {label['type']}")
