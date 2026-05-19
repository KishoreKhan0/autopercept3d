from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KITTIObjectLabel:
    """
    Parsed KITTI object label.

    KITTI label format:
        type truncated occluded alpha bbox_left bbox_top bbox_right bbox_bottom
        height width length x y z rotation_y

    Important:
        KITTI 3D object locations are in the rectified camera coordinate frame.
        The location is the bottom center of the 3D object box.
    """

    object_type: str
    truncated: float
    occluded: int
    alpha: float
    bbox_2d: np.ndarray
    dimensions_hwl: np.ndarray
    location_xyz_camera: np.ndarray
    rotation_y: float


def parse_kitti_label(label: dict) -> KITTIObjectLabel:
    """
    Convert the simple dictionary returned by KITTIDataset.load_labels()
    into numeric KITTIObjectLabel fields.
    """
    return KITTIObjectLabel(
        object_type=label["type"],
        truncated=float(label["truncated"]),
        occluded=int(label["occluded"]),
        alpha=float(label["alpha"]),
        bbox_2d=np.array(label["bbox_2d"], dtype=np.float64),
        dimensions_hwl=np.array(label["dimensions_hwl"], dtype=np.float64),
        location_xyz_camera=np.array(label["location_xyz_camera"], dtype=np.float64),
        rotation_y=float(label["rotation_y"]),
    )


def projection_matrix(values: np.ndarray) -> np.ndarray:
    """
    Convert a KITTI P0/P1/P2/P3 flat 12-value array to a 3x4 matrix.
    """
    return values.reshape(3, 4)


def rectification_matrix(values: np.ndarray) -> np.ndarray:
    """
    Convert KITTI R0_rect flat 9-value array to a 4x4 homogeneous matrix.
    """
    r3 = values.reshape(3, 3)
    r4 = np.eye(4, dtype=np.float64)
    r4[:3, :3] = r3
    return r4


def transform_matrix_3x4(values: np.ndarray) -> np.ndarray:
    """
    Convert KITTI Tr_velo_to_cam flat 12-value array to a 4x4 homogeneous matrix.
    """
    t = values.reshape(3, 4)
    t4 = np.eye(4, dtype=np.float64)
    t4[:3, :4] = t
    return t4


def velo_to_rect_matrix(calibration: dict[str, np.ndarray]) -> np.ndarray:
    """
    Build transform matrix from Velodyne/LiDAR coordinates to rectified camera coordinates.

    KITTI calibration gives:
        Tr_velo_to_cam: Velodyne -> reference camera
        R0_rect: reference camera -> rectified camera

    So:
        point_rect = R0_rect * Tr_velo_to_cam * point_velo
    """
    tr_velo_to_cam = transform_matrix_3x4(calibration["Tr_velo_to_cam"])
    r0_rect = rectification_matrix(calibration["R0_rect"])

    return r0_rect @ tr_velo_to_cam


def rect_to_velo_matrix(calibration: dict[str, np.ndarray]) -> np.ndarray:
    """
    Build inverse transform from rectified camera coordinates to Velodyne/LiDAR coordinates.
    """
    return np.linalg.inv(velo_to_rect_matrix(calibration))


def transform_points(points_xyz: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    """
    Apply a 4x4 homogeneous transform to Nx3 points.
    """
    if len(points_xyz) == 0:
        return points_xyz

    ones = np.ones((points_xyz.shape[0], 1), dtype=np.float64)
    points_h = np.hstack([points_xyz[:, :3], ones])

    transformed_h = (transform_4x4 @ points_h.T).T
    return transformed_h[:, :3]


def kitti_label_corners_camera(label: KITTIObjectLabel) -> np.ndarray:
    """
    Compute the 8 corners of a KITTI 3D object label in rectified camera coordinates.

    KITTI camera coordinate convention:
        x = right
        y = down
        z = forward

    Label dimensions are:
        h, w, l

    Label location is:
        bottom center of the object in camera coordinates
    """
    h, w, l = label.dimensions_hwl
    x, y, z = label.location_xyz_camera
    ry = label.rotation_y

    # 3D box corners around object origin before rotation.
    # y=0 is bottom, y=-h is top because camera y points downward.
    x_corners = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    z_corners = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])

    corners = np.vstack([x_corners, y_corners, z_corners])

    rotation = np.array(
        [
            [np.cos(ry), 0.0, np.sin(ry)],
            [0.0, 1.0, 0.0],
            [-np.sin(ry), 0.0, np.cos(ry)],
        ],
        dtype=np.float64,
    )

    corners_rotated = rotation @ corners
    corners_translated = corners_rotated + np.array([[x], [y], [z]], dtype=np.float64)

    return corners_translated.T


def kitti_label_corners_velodyne(
    label: KITTIObjectLabel,
    calibration: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Compute KITTI label 3D box corners in Velodyne/LiDAR coordinates.
    """
    corners_camera = kitti_label_corners_camera(label)
    camera_to_velo = rect_to_velo_matrix(calibration)

    return transform_points(corners_camera, camera_to_velo)


def bev_bottom_polygon_from_corners(corners_velodyne: np.ndarray) -> np.ndarray:
    """
    Return the bottom-face BEV polygon from 8 Velodyne box corners.

    Output shape:
        (5, 2)

    The first point is repeated at the end for direct plotting.
    """
    bottom = corners_velodyne[:4, :2]
    return np.vstack([bottom, bottom[0]])
