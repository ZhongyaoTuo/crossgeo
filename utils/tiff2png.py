#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depth TIFF to colored PNG visualization.
=========================================
Convert depth maps (.tiff) to color-mapped PNG images for quick inspection.

Usage:
    python utils/tiff2png.py --pair_dir path/to/pair_76 [--output_dir out/]
"""
import os
import glob
import argparse
import numpy as np
import cv2
import tifffile


def depth_to_colormap(depth, min_depth=None, max_depth=None, percentile_clip=2.0):
    """Convert a depth map to a BGR color image (blue=near, red=far).

    Args:
        depth: 2D array of depth values (meters).
        min_depth: minimum depth (if None, computed from data).
        max_depth: maximum depth (if None, computed from data).
        percentile_clip: clip percentile to exclude outliers.

    Returns:
        (H, W, 3) BGR uint8 image.
    """
    valid_mask = (depth > 0) & np.isfinite(depth)
    valid_depths = depth[valid_mask]

    if len(valid_depths) == 0:
        return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

    if min_depth is None or max_depth is None:
        low_perc = np.percentile(valid_depths, percentile_clip)
        high_perc = np.percentile(valid_depths, 100 - percentile_clip)
        min_depth = low_perc
        max_depth = high_perc
        if min_depth == max_depth:
            max_depth = min_depth + 1e-6

    norm = (depth - min_depth) / (max_depth - min_depth)
    norm = np.clip(norm, 0.0, 1.0)

    # Color map: low (near) -> blue (0,0,255), high (far) -> red (255,0,0)
    b = (1.0 - norm) * 255
    g = np.zeros_like(norm) * 255
    r = norm * 255

    color_map = np.stack([b, g, r], axis=-1).astype(np.uint8)
    color_map[~valid_mask] = [0, 0, 0]

    return color_map


def process_pair_directory(pair_dir, output_dir=None, percentile_clip=2.0):
    """Process all *_depth.tiff files in a pair directory.

    Args:
        pair_dir: directory containing depth TIFFs.
        output_dir: output directory (default: same as input).
        percentile_clip: percentile clip for depth range.
    """
    if not os.path.isdir(pair_dir):
        raise FileNotFoundError(f"Directory does not exist: {pair_dir}")

    depth_files = glob.glob(os.path.join(pair_dir, "*_depth.tiff"))
    if not depth_files:
        print(f"Warning: no *_depth.tiff files found in {pair_dir}")
        return

    if output_dir is None:
        output_dir = pair_dir
    else:
        os.makedirs(output_dir, exist_ok=True)

    for depth_path in depth_files:
        print(f"Processing: {depth_path}")
        depth = tifffile.imread(depth_path)
        if depth is None:
            print(f"  Cannot read file, skipping")
            continue

        color_img = depth_to_colormap(depth, percentile_clip=percentile_clip)

        base_name = os.path.basename(depth_path)
        out_name = base_name.replace(".tiff", "_color.png").replace(".tif", "_color.png")
        out_path = os.path.join(output_dir, out_name)

        cv2.imwrite(out_path, color_img)
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert depth TIFFs to colored PNG visualizations"
    )
    parser.add_argument("--pair_dir", type=str, required=True,
                        help="directory containing *_depth.tiff files")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="output directory (default: same as input)")
    parser.add_argument("--clip", type=float, default=2.0,
                        help="percentile clip for depth range (default: 2.0)")
    args = parser.parse_args()

    process_pair_directory(args.pair_dir, args.output_dir, args.clip)
    print("All depth maps converted.")
