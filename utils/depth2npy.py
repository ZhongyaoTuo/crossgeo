#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert depth TIFFs to NPY format.
=================================
Traverses pair_* folders and converts all *_depth.tiff files to .npy for
faster loading. Optionally reconstructs a point cloud from the .npy depths.

Usage:
    python utils/depth2npy.py --dataset_dir path/to/scene_pair [--delete] [--reconstruct]
"""
import os
import argparse
import numpy as np
import tifffile
from tqdm import tqdm


def convert_depth_to_npy(root_dir, delete_original=False):
    """Convert all *_depth.tiff files under pair_* folders to .npy.

    Args:
        root_dir: root directory containing pair_* folders.
        delete_original: if True, delete the original TIFF after conversion.
    """
    folders = [d for d in os.listdir(root_dir)
               if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("pair_")]

    print(f"[*] Found {len(folders)} folders, converting depth maps...")

    count = 0
    for folder in tqdm(folders, desc="Progress"):
        folder_path = os.path.join(root_dir, folder)

        # Match all tiff depth files (ground/uav _depth.tiff and satellite _satellite_depth.tiff)
        depth_files = [f for f in os.listdir(folder_path)
                       if (f.endswith('.tiff') or f.endswith('.tif')) and 'depth' in f]

        for f in depth_files:
            tiff_path = os.path.join(folder_path, f)
            npy_path = os.path.splitext(tiff_path)[0] + ".npy"

            try:
                depth_data = tifffile.imread(tiff_path).astype(np.float32)
                np.save(npy_path, depth_data)

                if delete_original:
                    os.remove(tiff_path)

                count += 1
            except Exception as e:
                print(f"\n[!] Conversion failed {tiff_path}: {e}")

    print(f"\n[OK] Done! Converted {count} depth files.")


def unified_reconstruction_from_npy(folder_path,
                                    output_name="reconstruction_from_npy.ply"):
    """Reconstruct a point cloud directly from .npy depth maps."""
    import cv2
    import open3d as o3d

    npy_configs = [f for f in os.listdir(folder_path) if f.endswith('_rgb.npy')]
    all_points, all_colors = [], []

    for npy_name in npy_configs:
        prefix = npy_name.replace('_rgb.npy', '')
        meta = np.load(os.path.join(folder_path, npy_name), allow_pickle=True).item()
        K, c2w = meta['intrinsics'], meta['c2w']

        rgb = None
        for ext in ['.jpg', '.png', '.JPG']:
            path = os.path.join(folder_path,
                                 f"{prefix}{ext}" if "satellite" in prefix
                                 else f"{prefix}_rgb{ext}")
            if os.path.exists(path):
                rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                break

        depth_npy_path = os.path.join(folder_path, f"{prefix}_depth.npy")
        if not os.path.exists(depth_npy_path):
            continue
        depth = np.load(depth_npy_path)

        if rgb.shape[0] != depth.shape[0]:
            rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]))

        mask = depth > 0
        v, u = np.where(mask)
        u, v, z = u[::4], v[::4], depth[v[::4], u[::4]]

        x_c = (u - K[0, 2]) * z / K[0, 0]
        y_c = (v - K[1, 2]) * z / K[1, 1]
        pts_world = (np.stack([x_c, y_c, z], axis=-1) @ c2w[:3, :3].T) + c2w[:3, 3]

        all_points.append(pts_world)
        all_colors.append(rgb[v, u] / 255.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
    o3d.io.write_point_cloud(os.path.join(folder_path, output_name), pcd)
    print(f"[OK] Reconstruction complete (loaded from NPY)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert depth TIFFs to NPY format"
    )
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="root directory containing pair_* folders")
    parser.add_argument("--delete", action="store_true",
                        help="delete original TIFF after conversion")
    parser.add_argument("--reconstruct", action="store_true",
                        help="test reconstruction from NPY (first pair)")
    args = parser.parse_args()

    convert_depth_to_npy(args.dataset_dir, delete_original=args.delete)

    if args.reconstruct:
        import glob
        pairs = sorted(glob.glob(os.path.join(args.dataset_dir, "pair_*")))
        if pairs:
            unified_reconstruction_from_npy(pairs[0])
