#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV depth via COLMAP multi-view stereo.
======================================
Ref: aerial-megadepth/data_generation/do_colmap_localization.py

CrossGeo (paper Sec 2.1 -- Metric depth):
    "UAV depth is recovered by COLMAP multi-view stereo over a denser
     auxiliary capture of each scene."

Flow:
    1. hloc triangulation (SuperPoint + SuperGlue) -> sparse SfM
    2. COLMAP patch_match_stereo -> dense depth maps
    3. COLMAP stereo_fusion -> fused point cloud
    4. Export depth maps to h5

Usage:
    python uav/colmap_mvs.py --scene_dir data/uav/data/0005 --colmap_empty data/uav/data/0005/sfm_empty
"""
import os
import subprocess
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import COLMAP_MAX_IMAGE_SIZE, COLMAP_PATCHMATCH_CACHE


def run_hloc_triangulation(colmap_empty_dir: str, image_dir: str,
                           output_dir: str, num_ref_pairs: int = 50) -> str:
    """Triangulate sparse point cloud with hloc.

    Ref: aerial-megadepth do_colmap_localization.py:run_hloc_triangulation

    Args:
        colmap_empty_dir: COLMAP empty model (known poses, no points).
        image_dir: image directory.
        output_dir: output directory.
        num_ref_pairs: matching pairs per image.

    Returns:
        Path to triangulated SfM model.
    """
    from hloc import extract_features, match_features, pairs_from_poses, triangulation

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    ref_sfm_empty = Path(colmap_empty_dir)
    ref_pairs = output_dir / f"pairs-db-dist{num_ref_pairs}.txt"

    pairs_from_poses.main(ref_sfm_empty, ref_pairs, num_ref_pairs, rotation_threshold=75)

    fconf = extract_features.confs["superpoint_max"]
    mconf = match_features.confs["superglue"]
    ref_sfm = output_dir / "sfm_superpoint+superglue"

    with open(ref_pairs, "r") as f:
        ref_images_list = set()
        for line in f:
            ref_images_list.add(line.split()[0])
            ref_images_list.add(line.split()[1])
    ref_images_list = list(ref_images_list)

    ffile = extract_features.main(
        conf=fconf, image_dir=Path(image_dir),
        export_dir=output_dir, image_list=ref_images_list,
    )
    mfile = match_features.main(
        conf=mconf, pairs=ref_pairs,
        features=fconf["output"], export_dir=output_dir,
    )
    triangulation.main(ref_sfm, ref_sfm_empty, Path(image_dir), ref_pairs, ffile, mfile)
    return str(ref_sfm)


def run_colmap_mvs(sparse_sfm: str, dense_sfm: str, image_dir: str,
                   gpu_idx: int = 0) -> str:
    """Run COLMAP MVS to recover dense depth.

    Ref: aerial-megadepth do_colmap_localization.py:run_mvs

    Returns:
        Path to fused point cloud.
    """
    colmap_path = "colmap"

    subprocess.run([
        colmap_path, "image_undistorter",
        "--image_path", str(image_dir),
        "--input_path", str(sparse_sfm),
        "--output_path", str(dense_sfm),
    ], check=True)

    subprocess.run([
        colmap_path, "patch_match_stereo",
        "--workspace_path", str(dense_sfm),
        "--PatchMatchStereo.cache_size", str(COLMAP_PATCHMATCH_CACHE),
        "--PatchMatchStereo.gpu_index", str(gpu_idx),
        "--PatchMatchStereo.max_image_size", str(COLMAP_MAX_IMAGE_SIZE),
    ], check=True)

    fused_path = os.path.join(str(dense_sfm), "fused.ply")
    subprocess.run([
        colmap_path, "stereo_fusion",
        "--workspace_path", str(dense_sfm),
        "--output_path", fused_path,
    ], check=True)
    return fused_path


def export_depth_maps(dense_sfm: str, output_dir: str) -> int:
    """Export COLMAP depth maps to h5.

    Ref: aerial-megadepth do_colmap_localization.py:depth_to_h5
    """
    import h5py
    from hloc.utils.read_write_dense import read_array
    from tqdm import tqdm

    depths_path = os.path.join(dense_sfm, "stereo", "depth_maps")
    images_path = os.path.join(dense_sfm, "images")
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for image_name in tqdm(sorted(os.listdir(images_path))):
        depth_path = os.path.join(depths_path, image_name + ".geometric.bin")
        if not os.path.exists(depth_path):
            continue
        depth_map = read_array(depth_path)
        h5_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + ".h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("depth", data=depth_map, compression="gzip", compression_opts=9)
        count += 1
    return count


def process_uav_depth(scene_dir: str, colmap_empty_dir: str,
                      output_dir: str = None, gpu_idx: int = 0) -> dict:
    """Full UAV depth pipeline: hloc triangulation + COLMAP MVS + export.

    Args:
        scene_dir: scene directory (contains footage/).
        colmap_empty_dir: COLMAP empty model (known poses).
        output_dir: depth output directory.
        gpu_idx: GPU index.

    Returns:
        {"sparse_sfm", "dense_sfm", "fused_ply", "num_depths"}
    """
    if output_dir is None:
        output_dir = os.path.join(scene_dir, "depths")

    image_dir = os.path.join(scene_dir, "footage")
    sfm_output = os.path.join(scene_dir, "sfm_output")

    print(">>> hloc triangulation...")
    ref_sfm = run_hloc_triangulation(colmap_empty_dir, image_dir, sfm_output)

    print(">>> COLMAP MVS...")
    dense_sfm = os.path.join(sfm_output, "dense")
    fused_ply = run_colmap_mvs(ref_sfm, dense_sfm, image_dir, gpu_idx)

    print(">>> Exporting depth maps...")
    num_depths = export_depth_maps(dense_sfm, output_dir)

    print(f"UAV depth done: {num_depths} depth maps")
    return {
        "sparse_sfm": ref_sfm,
        "dense_sfm": dense_sfm,
        "fused_ply": fused_ply,
        "num_depths": num_depths,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="UAV depth via COLMAP MVS")
    parser.add_argument("--scene_dir", type=str, required=True)
    parser.add_argument("--colmap_empty", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    result = process_uav_depth(args.scene_dir, args.colmap_empty, args.output_dir, args.gpu)
    print(f"Result: {result}")