#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CrossGeo dataset collection pipeline.
=====================================
Main entry point for collecting the CrossGeo tri-view (satellite / UAV / ground)
dataset (paper arXiv:2605.07978).

Stages:
    1. Satellite: download RGB (Google Maps) + pose + depth projection
    2. UAV:       Google Earth Studio render -> COLMAP MVS depth
    3. Ground:    Street View pano IDs -> RGB + depth download -> depth refinement
    4. Pairing:   tri-view pairing + train/val/test split

Usage:
    python pipeline.py --scene_config data/scenes/example.json
    python pipeline.py --scene_config data/scenes/example.json --steps satellite ground
"""
import os
import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    OUTPUT_ROOT, SATELLITE_RESOLUTION,
)


def step_satellite(scene: dict, output_root: str) -> dict:
    """Stage 1: satellite RGB download + pose + depth."""
    from satellite.download import download_satellite_tile

    print(f"\n{'='*60}")
    print(f"[Stage 1] Satellite: {scene['scene_id']}")
    print(f"{'='*60}")

    out_dir = os.path.join(output_root, scene["scene_id"], "satellite")
    meta = download_satellite_tile(
        lat_center=scene["lat_center"],
        lon_center=scene["lon_center"],
        output_dir=out_dir,
        scene_id=scene["scene_id"],
    )
    return {"satellite": meta}


def step_uav(scene: dict, output_root: str) -> dict:
    """Stage 2: UAV data collection (Google Earth Studio)."""
    from uav.preprocess_ge import process_ge_scene

    print(f"\n{'='*60}")
    print(f"[Stage 2] UAV: {scene['scene_id']}")
    print(f"{'='*60}")

    data_root = os.path.join(output_root, scene["scene_id"], "uav")
    result = process_ge_scene(data_root, scene["scene_id"])
    return {"uav": result}


def step_ground(scene: dict, output_root: str) -> dict:
    """Stage 3: ground data (Street View pano IDs -> download)."""
    from ground.download import download_scene

    print(f"\n{'='*60}")
    print(f"[Stage 3] Ground: {scene['scene_id']}")
    print(f"{'='*60}")

    out_dir = os.path.join(output_root, scene["scene_id"], "ground")
    pano_list = scene.get("pano_list_path")
    if pano_list:
        result = download_scene(pano_list, out_dir)
    else:
        print("No pano_list_path provided, skipping ground download.")
        result = {"status": "skipped"}
    return {"ground": result}


def step_depth(scene: dict, output_root: str) -> dict:
    """Stage 4: depth acquisition for all modalities."""
    print(f"\n{'='*60}")
    print(f"[Stage 4] Depth: {scene['scene_id']}")
    print(f"{'='*60}")

    results = {}
    scene_dir = os.path.join(output_root, scene["scene_id"])

    # UAV depth (COLMAP MVS)
    try:
        from uav.colmap_mvs import process_uav_depth
        uav_dir = os.path.join(scene_dir, "uav", "data", scene["scene_id"])
        colmap_empty = os.path.join(uav_dir, "sfm_empty")
        if os.path.exists(colmap_empty):
            print(">>> UAV depth (COLMAP MVS)...")
            results["uav"] = process_uav_depth(uav_dir, colmap_empty)
    except Exception as e:
        print(f"UAV depth failed: {e}")

    # Ground depth refinement (CDM + DA3 + PDA)
    try:
        from ground.depth_refine import process_scene
        ground_dir = os.path.join(scene_dir, "ground")
        if os.path.exists(ground_dir):
            print(">>> Ground depth refinement...")
            results["ground"] = process_scene(ground_dir)
    except Exception as e:
        print(f"Ground depth failed: {e}")

    # Satellite depth (point cloud projection)
    try:
        from satellite.depth import process_satellite_depth
        uav_result = results.get("uav", {})
        if uav_result.get("fused_ply") and os.path.exists(uav_result["fused_ply"]):
            print(">>> Satellite depth (point cloud projection)...")
            sat_depth_path = os.path.join(scene_dir, "satellite", "depth.tiff")
            w, h = SATELLITE_RESOLUTION
            results["satellite"] = process_satellite_depth(
                uav_result["fused_ply"], sat_depth_path, w, h,
            )
    except Exception as e:
        print(f"Satellite depth failed: {e}")

    return results


def step_pairing(scene: dict, output_root: str) -> dict:
    """Stage 5: tri-view pairing."""
    from utils.pairing.tri_view_pairing import generate_triview_samples

    print(f"\n{'='*60}")
    print(f"[Stage 5] Tri-view pairing: {scene['scene_id']}")
    print(f"{'='*60}")

    scene_dir = os.path.join(output_root, scene["scene_id"])
    output_dir = os.path.join(scene_dir, "triview")
    samples = generate_triview_samples(scene_dir, output_dir)
    return {"triview_samples": samples}


def run_pipeline(scene_config: str, output_root: str = None,
                 steps: list = None) -> dict:
    """Run the full collection pipeline."""
    if output_root is None:
        output_root = str(OUTPUT_ROOT)

    with open(scene_config) as f:
        scene = json.load(f)

    print(f"\n{'#'*60}")
    print(f"# CrossGeo Dataset Collection")
    print(f"# Scene: {scene['scene_id']}")
    print(f"# GPS:  ({scene['lat_center']}, {scene['lon_center']})")
    print(f"{'#'*60}")

    all_steps = ["satellite", "uav", "ground", "depth", "pairing"]
    if steps is None:
        steps = all_steps

    results = {}
    if "satellite" in steps:
        results.update(step_satellite(scene, output_root))
    if "uav" in steps:
        results.update(step_uav(scene, output_root))
    if "ground" in steps:
        results.update(step_ground(scene, output_root))
    if "depth" in steps:
        results.update(step_depth(scene, output_root))
    if "pairing" in steps:
        results.update(step_pairing(scene, output_root))

    result_path = os.path.join(output_root, scene["scene_id"], "pipeline_result.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n# Done: {scene['scene_id']} -> {result_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="CrossGeo dataset collection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --scene_config data/scenes/example.json
  python pipeline.py --scene_config data/scenes/example.json --steps satellite ground
  python pipeline.py --split --scenes_dir data/scenes
        """,
    )
    parser.add_argument("--scene_config", type=str, help="path to scene config JSON")
    parser.add_argument("--output", type=str, default=None, help="output root directory")
    parser.add_argument("--steps", type=str, nargs="*", default=None,
                        help="stages: satellite uav ground depth pairing")
    parser.add_argument("--split", action="store_true", help="run dataset split")
    parser.add_argument("--scenes_dir", type=str, help="scenes directory (for split)")
    args = parser.parse_args()

    if args.split:
        if not args.scenes_dir:
            parser.error("--split requires --scenes_dir")
        from utils.pairing.scene_split import create_scene_split
        create_scene_split(args.scenes_dir, args.output or "data/split")
    elif args.scene_config:
        run_pipeline(args.scene_config, args.output, args.steps)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
