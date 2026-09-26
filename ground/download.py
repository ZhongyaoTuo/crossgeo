#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ground-level data download via Google Street View API.
=====================================================
CrossGeo (paper Sec 2.1): Ground images come from Google Street View.

**We do NOT redistribute Street View images or depth.** We release only the
**pano IDs** (and associated metadata: heading, pitch, lat, lon). Users must
download RGB and depth themselves via the official Google Street View API.

This script provides helper functions for downloading RGB panoramas and depth
profiles using the official Google Street View Static API and the unofficial
depth endpoint. For full API documentation, see:
    https://developers.google.com/maps/documentation/streetview

Usage:
    # 1. Load the released pano ID list
    # 2. Download RGB + depth for each pano using this script
    python ground/download.py --pano_list data/pano_ids/scene_0005.json --output_dir data/ground/0005
"""
import os
import json
import requests
import numpy as np

import sys
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import GOOGLE_STREETVIEW_API_KEY, STREETVIEW_SIZE


def load_pano_list(path: str) -> list:
    """Load the released pano ID list.

    Each entry: {"pano_id": str, "lat": float, "lon": float,
                 "heading": float, "pitch": float, "date": str}

    Args:
        path: path to the pano ID JSON file.

    Returns:
        List of pano metadata dicts.
    """
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "panos" in data:
        return data["panos"]
    return list(data.values())


def download_panorama_rgb(pano_id: str, heading: float = 0, pitch: float = 0,
                          fov: int = 90, size: tuple = STREETVIEW_SIZE,
                          api_key: str = None) -> np.ndarray:
    """Download a Street View RGB image via the official Static API.

    API: https://maps.googleapis.com/maps/api/streetview
         ?pano=...&heading=...&fov=...&pitch=...&size=...&key=...

    Args:
        pano_id: panorama ID.
        heading: heading in degrees (0 = North, 90 = East).
        pitch: pitch in degrees (0 = level).
        fov: field of view in degrees.
        size: image size (width, height).
        api_key: Google Street View API key.

    Returns:
        Image (numpy.ndarray, BGR) or None.
    """
    api_key = api_key or GOOGLE_STREETVIEW_API_KEY
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_STREETVIEW_API_KEY is required. "
            "Get one at: https://console.cloud.google.com/google/maps-apis"
        )

    url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "pano": pano_id,
        "heading": heading,
        "fov": fov,
        "pitch": pitch,
        "size": f"{size[0]}x{size[1]}",
        "key": api_key,
        "return_error_code": "true",
    }
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code != 200:
        print(f"RGB download failed (pano={pano_id}): HTTP {resp.status_code}")
        return None

    import cv2
    arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
    return cv2.imdecode(arr, 1)


def download_panorama_depth(pano_id: str, api_key: str = None) -> bytes:
    """Download Street View depth profile (unofficial endpoint).

    The depth is returned in Google's CDM (Cube Depth Map) format.
    Use ``ground/cdm_decoder.py`` to decode it into a per-pixel depth map.

    NOTE: This endpoint is unofficial and may require special access.
    Alternative: use the Google Maps JavaScript API's StreetViewPanoramaData
    to extract depth via browser automation.

    Args:
        pano_id: panorama ID.
        api_key: API key.

    Returns:
        Raw depth bytes (CDM format), or None.
    """
    api_key = api_key or GOOGLE_STREETVIEW_API_KEY
    if not api_key:
        raise EnvironmentError("GOOGLE_STREETVIEW_API_KEY is required")

    url = "https://maps.googleapis.com/maps/api/streetview"
    params = {"pano": pano_id, "key": api_key, "output": "depth"}
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.content
        print(f"Depth download failed (pano={pano_id}): HTTP {resp.status_code}")
    except Exception as e:
        print(f"Depth download error (pano={pano_id}): {e}")
    return None


def download_scene(pano_list_path: str, output_dir: str,
                   headings: list = None,
                   api_key: str = None) -> dict:
    """Download RGB + depth for all panos in a scene.

    Args:
        pano_list_path: path to the released pano ID JSON.
        output_dir: output directory for RGB and depth files.
        headings: list of headings to download (default: use pano metadata).
        api_key: API key.

    Returns:
        Download statistics.
    """
    import cv2
    panos = load_pano_list(pano_list_path)
    os.makedirs(output_dir, exist_ok=True)

    num_rgb, num_depth = 0, 0
    for pano in panos:
        pano_id = pano["pano_id"]
        h_list = headings if headings else [pano.get("heading", 0)]

        for heading in h_list:
            img = download_panorama_rgb(
                pano_id, heading, pano.get("pitch", 0), api_key=api_key
            )
            if img is not None:
                fname = f"{pano_id}_h{int(heading):03d}.jpg"
                cv2.imwrite(os.path.join(output_dir, fname), img)
                num_rgb += 1

        depth_raw = download_panorama_depth(pano_id, api_key=api_key)
        if depth_raw is not None:
            dname = f"{pano_id}_depth.bin"
            with open(os.path.join(output_dir, dname), "wb") as f:
                f.write(depth_raw)
            num_depth += 1

    print(f"Downloaded {num_rgb} RGB images, {num_depth} depth maps")
    return {"num_rgb": num_rgb, "num_depth": num_depth, "output_dir": output_dir}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Download Google Street View RGB + depth from pano IDs"
    )
    parser.add_argument("--pano_list", type=str, required=True,
                        help="path to released pano ID JSON")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="output directory")
    parser.add_argument("--headings", type=float, nargs="*", default=None,
                        help="headings to download (default: use pano metadata)")
    args = parser.parse_args()

    result = download_scene(args.pano_list, args.output_dir, args.headings)
    print(f"Result: {result}")