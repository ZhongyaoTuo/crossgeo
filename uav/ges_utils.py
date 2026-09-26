#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Earth Studio JSON -> COLMAP model conversion.
====================================================
Ref: aerial-megadepth/data_generation/ges_utils.py

CrossGeo uses the EDS world frame (X=South, Y=Down, Z=East), whereas
aerial-megadepth uses ENU. The ENU->EDS conversion is:
    X_eds = -North,  Y_eds = -Up,  Z_eds = East

Flow:
    1. Parse GE Studio 3D Camera Tracking JSON (Global frame)
    2. GPS -> ENU (pymap3d), then ENU -> EDS
    3. ECEF rotation -> ENU rotation -> EDS rotation
    4. Write COLMAP empty model (known poses, no points)
"""
import os
import json
import numpy as np
from pymap3d.enu import geodetic2enu
from scipy.spatial.transform import Rotation

import sys
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.colmap_io import (
    fov_to_focal, make_colmap_camera, make_colmap_image, write_empty_model,
)


def rot_ecef2enu(lat: float, lon: float) -> np.ndarray:
    """ECEF -> ENU rotation matrix."""
    lamb = np.deg2rad(lon)
    phi = np.deg2rad(lat)
    sL, sP = np.sin(lamb), np.sin(phi)
    cL, cP = np.cos(lamb), np.cos(phi)
    return np.array([
        [     -sL,       cL,  0],
        [-sP * cL, -sP * sL, cP],
        [ cP * cL,  cP * sL, sP],
    ])


def enu_to_eds(east: float, north: float, up: float) -> np.ndarray:
    """ENU -> EDS world frame: X=South, Y=Down, Z=East."""
    return np.array([-north, -up, east])


def json_to_empty_colmap_model(json_file: str, ref_sfm_empty: str,
                               max_num_images: int = 200) -> dict:
    """Convert GE Studio JSON to a COLMAP empty model in EDS frame.

    Ref: aerial-megadepth ges_utils.py:json_to_empty_colmap_model

    Args:
        json_file: path to GE Studio 3D Camera Tracking JSON.
        ref_sfm_empty: output directory for the COLMAP empty model.
        max_num_images: max number of frames to process.

    Returns:
        {"scene_name", "lat0", "lon0", "alt0", "num_cameras"}
    """
    with open(json_file, "rb") as f:
        raw = json.load(f)

    scene_name = raw["name"]
    w = raw["width"]
    h = raw["height"]

    lat0 = raw["cameraFrames"][0]["coordinate"]["latitude"]
    lon0 = raw["cameraFrames"][0]["coordinate"]["longitude"]
    alt0 = raw["cameraFrames"][0]["coordinate"]["altitude"]

    print(f"Scene: {scene_name}  GPS: ({lat0}, {lon0}, {alt0})")

    rot = rot_ecef2enu(lat0, lon0)

    images = {}
    cameras = {}
    cam_id = 1
    poses = []

    for i, frame in enumerate(raw["cameraFrames"]):
        if max_num_images is not None and i > max_num_images:
            break

        e, n, u = geodetic2enu(
            frame["coordinate"]["latitude"],
            frame["coordinate"]["longitude"],
            frame["coordinate"]["altitude"],
            lat0, lon0, alt0,
        )
        # ENU -> EDS
        pos_eds = enu_to_eds(e, n, u)

        rx = frame["rotation"]["x"]
        ry = frame["rotation"]["y"]
        rz = frame["rotation"]["z"]
        R_ecef = Rotation.from_euler("XYZ", [rx, ry, rz], degrees=True).as_matrix()

        # ECEF -> ENU rotation, then ENU -> EDS rotation
        R_enu = rot @ R_ecef
        # EDS rotation: [X=South, Y=Down, Z=East] = [[0,-1,0],[0,0,-1],[1,0,0]] @ ENU
        R_enu_to_eds = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        R_eds = R_enu_to_eds @ R_enu

        c2w = np.eye(4)
        c2w[:3, :3] = R_eds
        c2w[:3, 3] = pos_eds

        fov_v = frame["fovVertical"]
        fl = fov_to_focal(fov_v, h)
        cx, cy = w / 2, h / 2

        cameras[cam_id] = make_colmap_camera(cam_id, w, h, fl, cx, cy)

        img_name = f"{scene_name}_{i:03d}.jpeg"
        images[i + 1] = make_colmap_image(i + 1, c2w, cam_id, img_name)
        cam_id += 1
        poses.append(c2w)

    print(f">>> CAMERAS: {len(cameras)}, IMAGES: {len(images)}")
    write_empty_model(cameras, images, ref_sfm_empty)

    return {
        "scene_name": scene_name,
        "lat0": lat0, "lon0": lon0, "alt0": alt0,
        "num_cameras": len(cameras),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GE Studio JSON -> COLMAP (EDS)")
    parser.add_argument("--json", type=str, required=True, help="GE tracking JSON")
    parser.add_argument("--output", type=str, required=True, help="COLMAP empty model dir")
    parser.add_argument("--max_images", type=int, default=200)
    args = parser.parse_args()

    result = json_to_empty_colmap_model(args.json, args.output, args.max_images)
    print(f"Done: {result}")