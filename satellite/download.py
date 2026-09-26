#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Satellite RGB tile download from Google Maps.
=============================================
The RGB download logic is based on:
    https://github.com/andolg/satellite-imagery-downloader

NOTE: The original project uses WGS84 / Web Mercator (EPSG:3857) for tile
addressing. CrossGeo uses the EDS world frame (X=South, Y=Down, Z=East) for
poses. The downloaded RGB tile is in pixel space; the associated camera pose
(see ``satellite/pose.py``) converts WGS84 GPS to EDS.

CrossGeo (paper Sec 2.1): download a 500m x 500m Google Maps satellite tile
centered at the scene GPS anchor, at zoom 19 (~0.5 m/px).
"""
import os
import cv2
import threading
import requests
import numpy as np
from datetime import datetime

import sys
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    SATELLITE_TILE_SIZE_M, SATELLITE_ZOOM, SATELLITE_TILE_URL,
    SATELLITE_TILE_PX, HTTP_HEADERS,
)
from utils.geo import compute_tile_bounds, latlon_to_meters_per_pixel


def download_tile(url: str, headers: dict, channels: int = 3) -> np.ndarray:
    """Download a single map tile (ref: andolg/satellite-imagery-downloader)."""
    response = requests.get(url, headers=headers, timeout=30)
    arr = np.asarray(bytearray(response.content), dtype=np.uint8)
    if channels == 3:
        return cv2.imdecode(arr, 1)
    return cv2.imdecode(arr, -1)


def project_with_scale(lat: float, lon: float, scale: int):
    """Web Mercator projection (ref: andolg/satellite-imagery-downloader).

    https://developers.google.com/maps/documentation/javascript/examples/map-coordinates
    """
    siny = np.sin(lat * np.pi / 180)
    siny = min(max(siny, -0.9999), 0.9999)
    x = scale * (0.5 + lon / 360)
    y = scale * (0.5 - np.log((1 + siny) / (1 - siny)) / (4 * np.pi))
    return x, y


def download_region(lat1: float, lon1: float, lat2: float, lon2: float,
                    zoom: int = SATELLITE_ZOOM,
                    url: str = SATELLITE_TILE_URL,
                    headers: dict = HTTP_HEADERS,
                    tile_size: int = SATELLITE_TILE_PX,
                    channels: int = 3) -> np.ndarray:
    """Download and stitch a rectangular satellite region.

    Ref: andolg/satellite-imagery-downloader (image_downloading.py:download_image)

    Args:
        lat1, lon1: top-left corner (lat, lon).
        lat2, lon2: bottom-right corner (lat, lon).
        zoom: zoom level.
        url: tile URL template with {x} {y} {z}.
        headers: HTTP headers.
        tile_size: pixel size of a single tile.
        channels: 3 = BGR, 4 = BGRA.

    Returns:
        Stitched image (numpy.ndarray).
    """
    scale = 1 << zoom
    tl_proj_x, tl_proj_y = project_with_scale(lat1, lon1, scale)
    br_proj_x, br_proj_y = project_with_scale(lat2, lon2, scale)

    tl_pixel_x = int(tl_proj_x * tile_size)
    tl_pixel_y = int(tl_proj_y * tile_size)
    br_pixel_x = int(br_proj_x * tile_size)
    br_pixel_y = int(br_proj_y * tile_size)

    tl_tile_x, tl_tile_y = int(tl_proj_x), int(tl_proj_y)
    br_tile_x, br_tile_y = int(br_proj_x), int(br_proj_y)

    img_w = abs(tl_pixel_x - br_pixel_x)
    img_h = br_pixel_y - tl_pixel_y
    img = np.zeros((img_h, img_w, channels), np.uint8)

    def build_row(tile_y):
        for tile_x in range(tl_tile_x, br_tile_x + 1):
            tile = download_tile(url.format(x=tile_x, y=tile_y, z=zoom), headers, channels)
            if tile is None:
                continue
            tl_rel_x = tile_x * tile_size - tl_pixel_x
            tl_rel_y = tile_y * tile_size - tl_pixel_y
            br_rel_x = tl_rel_x + tile_size
            br_rel_y = tl_rel_y + tile_size

            img_x_l = max(0, tl_rel_x)
            img_x_r = min(img_w + 1, br_rel_x)
            img_y_l = max(0, tl_rel_y)
            img_y_r = min(img_h + 1, br_rel_y)

            cr_x_l = max(0, -tl_rel_x)
            cr_x_r = tile_size + min(0, img_w - br_rel_x)
            cr_y_l = max(0, -tl_rel_y)
            cr_y_r = tile_size + min(0, img_h - br_rel_y)

            img[img_y_l:img_y_r, img_x_l:img_x_r] = tile[cr_y_l:cr_y_r, cr_x_l:cr_x_r]

    threads = []
    for tile_y in range(tl_tile_y, br_tile_y + 1):
        t = threading.Thread(target=build_row, args=[tile_y])
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    return img


def download_satellite_tile(lat_center: float, lon_center: float,
                            size_meters: float = SATELLITE_TILE_SIZE_M,
                            zoom: int = SATELLITE_ZOOM,
                            output_dir: str = None,
                            scene_id: str = None) -> dict:
    """Download a 500m x 500m satellite tile centered at a GPS anchor.

    CrossGeo main entry for satellite RGB download.
    The tile URL and Mercator projection follow andolg/satellite-imagery-downloader.
    The WGS84 GPS is converted to EDS world coordinates in satellite/pose.py.

    Args:
        lat_center, lon_center: scene GPS center.
        size_meters: tile edge length in meters (default 500 m).
        zoom: zoom level.
        output_dir: output directory.
        scene_id: scene ID (for naming).

    Returns:
        Dict with image path and metadata.
    """
    (lat_top, lon_left), (lat_bottom, lon_right) = compute_tile_bounds(
        lat_center, lon_center, size_meters
    )

    img = download_region(lat_top, lon_left, lat_bottom, lon_right, zoom=zoom)
    mpp = latlon_to_meters_per_pixel(lat_center, zoom)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    name = f"{scene_id or 'scene'}_sat_{timestamp}.png"

    meta = {
        "scene_id": scene_id,
        "lat_center": lat_center,
        "lon_center": lon_center,
        "size_meters": size_meters,
        "zoom": zoom,
        "meters_per_pixel": mpp,
        "image_size": [img.shape[1], img.shape[0]],
        "bounds": {
            "lat_top": lat_top, "lon_left": lon_left,
            "lat_bottom": lat_bottom, "lon_right": lon_right,
        },
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        img_path = os.path.join(output_dir, name)
        cv2.imwrite(img_path, img)
        meta["image_path"] = img_path
        import json
        meta_path = img_path.replace(".png", ".json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    return meta


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download CrossGeo satellite RGB tile")
    parser.add_argument("--lat", type=float, required=True, help="scene center latitude")
    parser.add_argument("--lon", type=float, required=True, help="scene center longitude")
    parser.add_argument("--size", type=float, default=SATELLITE_TILE_SIZE_M, help="coverage (meters)")
    parser.add_argument("--zoom", type=int, default=SATELLITE_ZOOM, help="zoom level")
    parser.add_argument("--output", type=str, default="data/satellite", help="output directory")
    parser.add_argument("--scene_id", type=str, default=None, help="scene ID")
    args = parser.parse_args()

    meta = download_satellite_tile(
        args.lat, args.lon, args.size, args.zoom, args.output, args.scene_id
    )
    print(f"Satellite RGB saved: {meta.get('image_path', 'not saved')}")
    print(f"Resolution: {meta['meters_per_pixel']:.4f} m/px, size: {meta['image_size']}")