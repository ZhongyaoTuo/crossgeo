#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract frames from Google Earth Studio .mp4 and copy JSON metadata.
====================================================================
Ref: aerial-megadepth/data_generation/datasets_preprocess/preprocess_ge.py

Flow:
    1. Read .mp4 video from downloaded_data/
    2. Extract every frame to footage/ as .jpeg
    3. Copy the GE 3D Camera Tracking JSON alongside

Usage:
    python uav/preprocess_ge.py --data_root data/uav --scene_id 0005
"""
import os
import shutil
import argparse
import cv2


def extract_frames_and_metadata(data_root: str, scene_id: str,
                                downloaded_dir: str = None,
                                output_dir: str = None) -> dict:
    """Extract frames from .mp4 and copy JSON metadata.

    Ref: aerial-megadepth preprocess_ge.py:extract_frames_and_copy_json

    Args:
        data_root: root directory (contains downloaded_data/ and data/).
        scene_id: scene name (e.g. "0005").
        downloaded_dir: directory with .mp4 and .json (default: data_root/downloaded_data).
        output_dir: output directory (default: data_root/data/scene_id).

    Returns:
        {"num_frames", "footage_dir", "json_path"}
    """
    if downloaded_dir is None:
        downloaded_dir = os.path.join(data_root, "downloaded_data")
    if output_dir is None:
        output_dir = os.path.join(data_root, "data", scene_id)

    mp4_path = os.path.join(downloaded_dir, f"{scene_id}.mp4")
    if not os.path.exists(mp4_path):
        raise FileNotFoundError(f"Video not found: {mp4_path}")

    footage_dir = os.path.join(output_dir, "footage")
    os.makedirs(footage_dir, exist_ok=True)

    vidcap = cv2.VideoCapture(mp4_path)
    success, image = vidcap.read()
    count = 0
    while success:
        frame_path = os.path.join(footage_dir, f"{scene_id}_{count:03d}.jpeg")
        cv2.imwrite(frame_path, image)
        success, image = vidcap.read()
        count += 1

    print(f"Extracted {count} frames to {footage_dir}")

    json_src = os.path.join(downloaded_dir, f"{scene_id}.json")
    json_dst = os.path.join(output_dir, f"{scene_id}.json")
    if os.path.exists(json_src):
        shutil.copy(json_src, json_dst)
        print(f"Copied metadata: {json_dst}")
    else:
        print(f"Warning: JSON not found for {scene_id}")

    return {"num_frames": count, "footage_dir": footage_dir, "json_path": json_dst}


def process_ge_scene(data_root: str, scene_id: str) -> dict:
    """Full GE scene processing: extract frames + build COLMAP model.

    Args:
        data_root: root directory.
        scene_id: scene name.

    Returns:
        {"frames", "colmap_empty", "origin_gps"}
    """
    import sys
    sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from uav.ges_utils import json_to_empty_colmap_model

    result = extract_frames_and_metadata(data_root, scene_id)

    json_path = result["json_path"]
    colmap_empty = os.path.join(data_root, "data", scene_id, "sfm_empty")
    colmap_result = json_to_empty_colmap_model(json_path, colmap_empty)

    return {
        "frames": result,
        "colmap_empty": colmap_empty,
        "origin_gps": (colmap_result["lat0"], colmap_result["lon0"], colmap_result["alt0"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract GE Studio frames + metadata")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--scene_id", type=str, required=True)
    args = parser.parse_args()

    result = process_ge_scene(args.data_root, args.scene_id)
    print(f"Done: {result}")