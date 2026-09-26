#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COLMAP model I/O utilities.
==========================
Read/write COLMAP sparse models (cameras/images/points3D) and pose conversion
helpers. Ref: aerial-megadepth/data_generation/ges_utils.py.

Requires hloc.utils.read_write_model:
    pip install git+https://github.com/cvg/Hierarchical-Localization.git
"""
import os
import numpy as np

try:
    from hloc.utils.read_write_model import (
        Camera, Image, Point3D,
        read_model, write_model,
        rotmat2qvec, qvec2rotmat,
    )
except ImportError:
    # Fallback when hloc is not installed
    Camera = Image = Point3D = None
    read_model = write_model = None

    def rotmat2qvec(R):
        Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
        K = np.array([
            [Rxx - Ryy - Rzz, 2 * (Rxy + Ryz), 2 * (Rxz - Ryx), 1],
            [2 * (Rxy - Ryz), Ryy - Rxx - Rzz, 2 * (Ryx + Rxz), 1],
            [2 * (Rxz + Ryz), 2 * (Ryx - Rxz), Rzz - Rxx - Ryy, 1],
            [2 * (Ryx + Ryz), 2 * (Ryz - Rxz), 2 * (Rxy - Rxx), 1],
        ])
        K = K[K[:, 3].argsort()]
        qvec = np.sqrt(K[3]) * K[:3]
        if qvec[0] < 0:
            qvec = -qvec
        return qvec

    def qvec2rotmat(qvec):
        w, x, y, z = qvec
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
        ])


def colmap_raw_pose_to_RT(image_pose: list) -> np.ndarray:
    """COLMAP images.txt raw pose [qvec, tvec] -> 4x4 w2c matrix.

    Ref: aerial-megadepth preprocess_aerialmegadepth.py
    """
    qvec = np.array(image_pose[:4])
    qvec = qvec / np.linalg.norm(qvec)
    R = qvec2rotmat(qvec)
    t = np.array(image_pose[4:7])
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t
    return pose


def colmap_raw_pose_to_principal_axis(image_pose: list) -> np.ndarray:
    """Extract the camera principal axis (z-axis direction)."""
    qvec = np.array(image_pose[:4])
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec
    return np.float32([
        2 * x * z - 2 * y * w,
        2 * y * z + 2 * x * w,
        1 - 2 * x * x - 2 * y * y,
    ])


def make_colmap_camera(cam_id: int, width: int, height: int,
                       focal: float, cx: float = None, cy: float = None,
                       model: str = "SIMPLE_PINHOLE") -> "Camera":
    """Create a COLMAP Camera object."""
    if cx is None:
        cx = width / 2
    if cy is None:
        cy = height / 2
    if Camera is None:
        raise ImportError("hloc is required: pip install git+https://github.com/cvg/Hierarchical-Localization.git")
    params = np.array([focal, cx, cy]) if model == "SIMPLE_PINHOLE" else np.array([focal, focal, cx, cy])
    return Camera(id=cam_id, model=model, width=width, height=height, params=params)


def make_colmap_image(image_id: int, c2w: np.ndarray, cam_id: int,
                      name: str) -> "Image":
    """Create a COLMAP Image object from a c2w matrix."""
    if Image is None:
        raise ImportError("hloc is required")
    w2c = np.linalg.inv(c2w)
    return Image(
        id=image_id,
        qvec=rotmat2qvec(w2c[:3, :3]),
        tvec=w2c[:3, 3],
        camera_id=cam_id,
        name=name,
        xys=np.empty([0, 2]),
        point3D_ids=np.empty(0),
    )


def write_empty_model(cameras: dict, images: dict, output_dir: str,
                      ext: str = ".bin"):
    """Write an empty COLMAP model (no points)."""
    if write_model is None:
        raise ImportError("hloc is required")
    points3D = {}
    os.makedirs(output_dir, exist_ok=True)
    write_model(cameras, images, points3D, output_dir, ext)


def fov_to_focal(fov_deg: float, image_size_px: int) -> float:
    """FOV (degrees) -> focal length (pixels)."""
    return image_size_px / (2 * np.tan(np.radians(fov_deg) / 2))


def focal_to_fov(focal_px: float, image_size_px: int) -> float:
    """Focal length (pixels) -> FOV (degrees)."""
    return 2 * np.degrees(np.arctan(image_size_px / 2 / focal_px))
