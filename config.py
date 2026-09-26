#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CrossGeo dataset collection configuration.
=========================================
Centralized configuration for the CrossGeo tri-view (satellite / UAV / ground)
dataset collection pipeline (paper arXiv:2605.07978).

Sensitive values (API keys) should be injected via environment variables or a
``.env`` file — never hard-coded.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 1. Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.getenv("CROSSGEO_DATA_ROOT", PROJECT_ROOT / "data"))

SATELLITE_DIR = DATA_ROOT / "satellite"
UAV_DIR = DATA_ROOT / "uav"
GROUND_DIR = DATA_ROOT / "ground"
SCENES_DIR = DATA_ROOT / "scenes"

OUTPUT_ROOT = Path(os.getenv("CROSSGEO_OUTPUT_ROOT", DATA_ROOT / "processed"))
TRIVIEW_DIR = OUTPUT_ROOT / "triview"

# ============================================================
# 2. API keys (read from environment — do NOT hard-code)
# ============================================================
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
GOOGLE_STREETVIEW_API_KEY = os.getenv("GOOGLE_STREETVIEW_API_KEY", GOOGLE_MAPS_API_KEY)
BAIDU_MAPS_AK = os.getenv("BAIDU_MAPS_AK", "")

# ============================================================
# 3. Satellite collection (paper Sec 2.1)
#    500m x 500m tiles, modeled as a virtual pinhole camera looking
#    straight down (FOV 5 deg, altitude 5726 m, resolution 1024x1024).
#    Coverage: 2 * 5726 * tan(2.5 deg) ~= 500 m.
# ============================================================
SATELLITE_TILE_SIZE_M = 500
SATELLITE_ZOOM = 19                  # ~0.5 m/px at most latitudes
SATELLITE_TILE_URL = "https://mt.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
SATELLITE_TILE_PX = 256

SATELLITE_VIRTUAL_ALTITUDE_M = 5726.0
SATELLITE_FOV_DEG = 5.0
SATELLITE_RESOLUTION = (1024, 1024)

# ============================================================
# 4. UAV collection (paper Sec 2.1, same as AerialMegaDepth)
# ============================================================
UAV_ALTITUDE_RANGE_M = (30, 120)
UAV_PITCH_RANGE_DEG = (0, 90)
UAV_HIGH_PITCH_OVERSAMPLE = True     # oversample 60-90 deg (low-pitch + high-alt stretches pixels)
UAV_HIGH_PITCH_THRESHOLD_DEG = 60

GE_RENDER_FPS = 30
GE_RENDER_RESOLUTION = (1920, 1080)

# ============================================================
# 5. Ground (Street View) collection
# ============================================================
STREETVIEW_HEADING_LIST = [0, 60, 120, 180, 240, 300]
STREETVIEW_FOV_DEG = 90
STREETVIEW_RADIUS_M = 50
STREETVIEW_SIZE = (640, 640)

# ============================================================
# 6. Depth (paper Sec 2.1 — Metric depth)
# ============================================================
DEPTHANY_V3_CKPT = os.getenv("DEPTHANY_V3_CKPT", "")
PRIOR_DEPTH_ANYTHING_CKPT = os.getenv("PDA_CKPT", "")
GROUND_DEPTH_PEARSON_THRESHOLD = 0.3  # discard samples below this Pearson correlation

COLMAP_MAX_IMAGE_SIZE = 2000
COLMAP_PATCHMATCH_CACHE = 32

# ============================================================
# 7. Tri-view pairing (paper Sec 2.1)
# ============================================================
VOXEL_SIZE_M = 1.0
TRIVIEW_VIEWS_PER_MODALITY = 2
TRIVIEW_MIN_OVERLAP = 0.0

# ============================================================
# 8. Train / val / test split (paper Sec 2.1)
# ============================================================
NUM_TRAIN_SCENES = 75
NUM_VAL_SCENES = 5
NUM_TEST_SCENES = 5
TOTAL_SCENES = 85

# ============================================================
# 9. World coordinate system (EDS) (paper Sec 2.1 — Pose recovery)
#    X -> South, Y -> Down, Z -> East
# ============================================================
WORLD_AXIS = {
    "x": "south",
    "y": "down",
    "z": "east",
}

# ============================================================
# 10. HTTP headers (ref: satellite-imagery-downloader)
# ============================================================
HTTP_HEADERS = {
    "cache-control": "max-age=0",
    "sec-ch-ua": '" Not A;Brand";v="99", "Chromium";v="99", "Google Chrome";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "none",
    "sec-fetch-user": "?1",
    "upgrade-in-requests": "1",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/99.0.4844.82 Safari/537.36",
}


def validate_api_keys():
    """Raise EnvironmentError if required API keys are missing."""
    missing = []
    if not GOOGLE_MAPS_API_KEY:
        missing.append("GOOGLE_MAPS_API_KEY")
    if not GOOGLE_STREETVIEW_API_KEY:
        missing.append("GOOGLE_STREETVIEW_API_KEY")
    if missing:
        raise EnvironmentError(
            f"Missing required API keys: {missing}. "
            f"Set them via environment variables or a .env file."
        )
