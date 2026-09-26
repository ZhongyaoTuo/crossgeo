#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge ground panorama + UAV + satellite point clouds from a pair folder.
=====================================================================
Reads existing files in a pair folder (no dependency on original XML / depth
directories) and merges them into a single colored point cloud.

Data sources (read from the pair folder):
  - Ground panorama : {prefix}_pano_rgb.npy   (intrinsics + pose)
                      {prefix}_pano_rgb.jpg
                      {prefix}_pano_depth.(npy|tiff)
  - UAV             : uav_*_rgb.npy + uav_*_rgb.jpg + uav_*_depth.(npy|tiff)
  - Satellite       : {prefix}_satellite_rgb.npy + {prefix}_satellite.png
                      + {prefix}_satellite_depth.(npy|tiff)

All poses are stored as c2w in the EDS world frame (East-Down-South, Y down):
  UAV / satellite        -> standard pinhole back-projection:
                              pts_cam = [(u-cx)z/f, (v-cy)z/f, z]
                              pts_eds = pts_cam @ R^T + t
  Ground panorama        -> spherical projection (note +sin(phi) in y):
                              theta = (u-cx)/fx,  phi = (v-cy)/fy
                              dir = [cos(phi)sin(theta), sin(phi), cos(phi)cos(theta)]
                              pts_eds = (dir * z) @ R^T + t

Usage:
    python utils/ply_show.py <pair_dir> [--out merged_rgb.ply] [--step 2] \
        [--voxel 0.1] [--seg] [--include-pinhole]
"""
import os
import sys
import glob
import argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import cv2
import open3d as o3d

# UAV depth horizontal shift (px): 0 for new pair_resize (full-frame, no offset);
# 199 for old pair_resizev3 data. Override with --uav-shift.
UAV_DEPTH_SHIFT_PX = 0

# ============ Basic utilities ============

def load_pose(npy_path):
    """Load *_rgb.npy; return {'intrinsics': K, 'c2w': 4x4, 'raw_data': ...}."""
    return np.load(npy_path, allow_pickle=True).item()


def load_depth(pair_dir, prefix):
    """Load *_depth.npy first, then *_depth.tiff (lazy tifffile import)."""
    npy_path = os.path.join(pair_dir, f"{prefix}_depth.npy")
    if os.path.exists(npy_path):
        return np.load(npy_path)
    tiff_path = os.path.join(pair_dir, f"{prefix}_depth.tiff")
    if os.path.exists(tiff_path):
        import tifffile
        return tifffile.imread(tiff_path).astype(np.float32)
    return None


def load_rgb(pair_dir, prefix):
    """Load an RGB image (cv2, BGR -> RGB float32).

    Supports two naming conventions:
      {prefix}_rgb.jpg   -- UAV / ground / pano (e.g. uav_1_rgb.jpg)
      {prefix}.png       -- satellite (e.g. ground_1_satellite.png)
    Returns None if not found.
    """
    patterns = (f"{prefix}.", f"{prefix}_rgb.")
    for fname in sorted(os.listdir(pair_dir)):
        lower = fname.lower()
        if lower.endswith(('.jpg', '.jpeg', '.png')) and any(
                lower.startswith(p.lower()) for p in patterns):
            img = cv2.imread(os.path.join(pair_dir, fname))
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return None


def sample_rgb(rgb, u, v, depth_shape):
    """Sample RGB at depth coords (u, v), auto-scaling if resolutions differ.

    E.g. satellite depth 1024x1024 -> satellite RGB 2279x2279, or
    panorama depth 512x256 -> panorama RGB 4096x2048.
    """
    if rgb is None:
        return None
    su = rgb.shape[1] / depth_shape[1]
    sv = rgb.shape[0] / depth_shape[0]
    uu = np.clip((u * su).astype(int), 0, rgb.shape[1] - 1)
    vv = np.clip((v * sv).astype(int), 0, rgb.shape[0] - 1)
    return rgb[vv, uu]


def project_pinhole(depth, K, c2w, step=2, dmin=0.5, dmax=120.0):
    """Standard pinhole back-projection (OpenCV: x right, y down, z forward).

    Used for UAV / satellite / ground pinhole.
    """
    mask = (depth > dmin) & (depth < dmax)
    vy, ux = np.where(mask)
    u, v = ux[::step], vy[::step]
    z = depth[v, u]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pts_cam = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=-1)
    pts_eds = (pts_cam @ c2w[:3, :3].T) + c2w[:3, 3]
    return pts_eds, (u, v)


def project_pano(depth, K_full, c2w, step=1, dmin=0.5, dmax=135.0):
    """Equirectangular panorama spherical back-projection.

    K_full is the intrinsics of the original panorama (e.g. 4096x2048); the
    depth map is usually downsampled (e.g. 512x256), so K is scaled by
    (depth_width / orig_width).
    Convention: theta=(u-cx)/fx, phi=(v-cy)/fy,
                dir=[cos(phi)sin(theta), sin(phi), cos(phi)cos(theta)].
    """
    h, w = depth.shape
    scale = w / (2.0 * K_full[0, 2])   # orig width = 2*cx
    K = K_full.copy()
    K[0, 0] *= scale
    K[1, 1] *= scale
    K[0, 2] *= scale
    K[1, 2] *= scale

    mask = (depth > dmin) & (depth < dmax)
    vy, ux = np.where(mask)
    u, v = ux[::step], vy[::step]
    z = depth[v, u]

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    theta = (u - cx) / fx
    phi = (v - cy) / fy
    cp = np.cos(phi)
    dir_cam = np.stack([cp * np.sin(theta), np.sin(phi), cp * np.cos(theta)], axis=-1)
    pts_eds = (dir_cam * z[:, None]) @ c2w[:3, :3].T + c2w[:3, 3]
    return pts_eds, (u, v)


# ============ Per-class point cloud construction ============

def collect_ground_pano(pair_dir, step):
    """Collect ground panorama point clouds: {prefix}_pano_rgb.npy."""
    results = []
    for npy in sorted(glob.glob(os.path.join(pair_dir, "*_pano_rgb.npy"))):
        prefix = os.path.basename(npy).replace("_pano_rgb.npy", "")
        meta = load_pose(npy)
        depth = load_depth(pair_dir, f"{prefix}_pano")
        if depth is None:
            print(f"  [skip] {prefix}: panorama depth missing")
            continue
        pts, (u, v) = project_pano(depth, meta['intrinsics'], meta['c2w'], step=step)
        if len(pts) == 0:
            continue
        rgb = load_rgb(pair_dir, f"{prefix}_pano")
        cols = sample_rgb(rgb, u, v, depth.shape)
        if cols is None:
            cols = np.full((len(pts), 3), 0.8, dtype=np.float32)
        seg = np.tile(np.array([1.0, 0.0, 0.0]), (len(pts), 1))
        results.append((pts, cols, seg))
        print(f"  [ground pano] {prefix}: {len(pts):,} points")
    return results


def collect_uav(pair_dir, step):
    """Collect UAV point clouds: uav_*_rgb.npy (exclude satellite).

    Uses the depth saved in the pair folder, and shifts the depth content
    right by UAV_DEPTH_SHIFT_PX pixels (to fix pair_resizev3 offset cropping),
    dropping the left columns.
    """
    results = []
    for npy in sorted(glob.glob(os.path.join(pair_dir, "uav_*_rgb.npy"))):
        prefix = os.path.basename(npy).replace("_rgb.npy", "")
        meta = load_pose(npy)
        depth = load_depth(pair_dir, prefix)
        if depth is None:
            print(f"  [skip] {prefix}: depth missing")
            continue
        if depth.shape[1] > UAV_DEPTH_SHIFT_PX:
            shifted = np.zeros_like(depth)
            shifted[:, UAV_DEPTH_SHIFT_PX:] = depth[:, :depth.shape[1] - UAV_DEPTH_SHIFT_PX]
            depth = shifted
        pts, (u, v) = project_pinhole(depth, meta['intrinsics'], meta['c2w'],
                                      step=step, dmin=0.5, dmax=120.0)
        if len(pts) == 0:
            continue
        rgb = load_rgb(pair_dir, prefix)
        cols = sample_rgb(rgb, u, v, depth.shape)
        if cols is None:
            cols = np.full((len(pts), 3), 0.8, dtype=np.float32)
        seg = np.tile(np.array([0.0, 0.0, 1.0]), (len(pts), 1))
        results.append((pts, cols, seg))
        print(f"  [uav] {prefix}: {len(pts):,} points")
    return results


def collect_satellite(pair_dir, step):
    """Collect satellite point clouds: {prefix}_satellite_rgb.npy.

    RGB ({prefix}_satellite.png) may have a different resolution than depth;
    sampling auto-scales.
    """
    results = []
    for npy in sorted(glob.glob(os.path.join(pair_dir, "*_satellite_rgb.npy"))):
        prefix = os.path.basename(npy).replace("_satellite_rgb.npy", "")
        meta = load_pose(npy)
        depth = load_depth(pair_dir, f"{prefix}_satellite")
        if depth is None:
            print(f"  [skip] {prefix}: satellite depth missing")
            continue
        # Satellite depth is Z-Buffer rendered, range ~4000-7000m
        pts, (u, v) = project_pinhole(depth, meta['intrinsics'], meta['c2w'],
                                      step=step, dmin=4000.0, dmax=7000.0)
        if len(pts) == 0:
            continue
        rgb = load_rgb(pair_dir, f"{prefix}_satellite")
        cols = sample_rgb(rgb, u, v, depth.shape)
        if cols is None:
            cols = np.full((len(pts), 3), 0.7, dtype=np.float32)  # gray if no RGB
        seg = np.tile(np.array([0.0, 1.0, 0.0]), (len(pts), 1))
        results.append((pts, cols, seg))
        print(f"  [satellite] {prefix}: {len(pts):,} points")
    return results


def collect_ground_pinhole(pair_dir, step):
    """(Optional) Ground pinhole four views: ground_*_rgb.npy.

    Same station as pano; disabled by default.
    """
    results = []
    for npy in sorted(glob.glob(os.path.join(pair_dir, "ground_*_rgb.npy"))):
        prefix = os.path.basename(npy).replace("_rgb.npy", "")
        if "pano" in prefix or "satellite" in prefix:
            continue
        meta = load_pose(npy)
        depth = load_depth(pair_dir, prefix)
        if depth is None:
            continue
        pts, (u, v) = project_pinhole(depth, meta['intrinsics'], meta['c2w'],
                                      step=step, dmin=0.5, dmax=40.0)
        if len(pts) == 0:
            continue
        rgb = load_rgb(pair_dir, prefix)
        cols = sample_rgb(rgb, u, v, depth.shape)
        if cols is None:
            cols = np.full((len(pts), 3), 0.8, dtype=np.float32)
        seg = np.tile(np.array([1.0, 0.0, 0.0]), (len(pts), 1))
        results.append((pts, cols, seg))
        print(f"  [ground pinhole] {prefix}: {len(pts):,} points")
    return results


# ============ Main flow ============

def build_pair_cloud(pair_dir, step, include_pinhole):
    all_pts, all_rgb, all_seg = [], [], []
    for pts, cols, seg in collect_ground_pano(pair_dir, step):
        all_pts.append(pts); all_rgb.append(cols); all_seg.append(seg)
    for pts, cols, seg in collect_uav(pair_dir, step):
        all_pts.append(pts); all_rgb.append(cols); all_seg.append(seg)
    for pts, cols, seg in collect_satellite(pair_dir, step):
        all_pts.append(pts); all_rgb.append(cols); all_seg.append(seg)
    if include_pinhole:
        for pts, cols, seg in collect_ground_pinhole(pair_dir, step):
            all_pts.append(pts); all_rgb.append(cols); all_seg.append(seg)

    if not all_pts:
        print("[!] No point clouds generated")
        return None, None, None

    pts = np.vstack(all_pts)
    rgb = np.vstack(all_rgb)
    seg = np.vstack(all_seg)
    return pts, rgb, seg


def save_pcd(path, pts, cols):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    o3d.io.write_point_cloud(path, pcd)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Merge ground panorama + UAV + satellite point clouds from a pair folder"
    )
    parser.add_argument("pair_dir", nargs="?", default=None,
                        help="pair folder path")
    parser.add_argument("--out", default=None,
                        help="output PLY path (default: pair_dir/merged_rgb.ply)")
    parser.add_argument("--step", type=int, default=2,
                        help="depth sampling step (larger = fewer points, default 2)")
    parser.add_argument("--voxel", type=float, default=None,
                        help="optional voxel downsample size (meters)")
    parser.add_argument("--seg", action="store_true",
                        help="also output segmentation-colored cloud (ground red / uav blue / sat green)")
    parser.add_argument("--include-pinhole", action="store_true",
                        help="also merge ground pinhole four views")
    parser.add_argument("--uav-shift", type=int, default=0,
                        help="UAV depth horizontal shift in pixels (fix old offset crop; default 0)")
    args = parser.parse_args()

    if args.pair_dir is None:
        parser.error("pair_dir is required")

    global UAV_DEPTH_SHIFT_PX
    UAV_DEPTH_SHIFT_PX = args.uav_shift

    pair_dir = os.path.normpath(args.pair_dir)
    if not os.path.isdir(pair_dir):
        print(f"[ERROR] Directory does not exist: {pair_dir}")
        sys.exit(1)

    print(f"[*] Processing {pair_dir}")
    pts, rgb, seg = build_pair_cloud(pair_dir, args.step, args.include_pinhole)
    if pts is None:
        sys.exit(1)

    if args.voxel:
        print(f"[*] Voxel downsample (voxel={args.voxel}m)...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        pcd = pcd.voxel_down_sample(args.voxel)
        pts, rgb = np.asarray(pcd.points), np.asarray(pcd.colors)

    out = args.out or os.path.join(pair_dir, "merged_rgb.ply")
    save_pcd(out, pts, rgb)
    print(f"[OK] RGB cloud saved: {out}")

    if args.seg:
        seg_out = (args.out.replace(".ply", "_seg.ply") if args.out
                   else os.path.join(pair_dir, "merged_seg.ply"))
        save_pcd(seg_out, pts, seg)
        print(f"[OK] Segmentation cloud saved: {seg_out}")

    lo, hi = pts.min(0), pts.max(0)
    print(f"[*] Total points: {len(pts):,}")
    print(f"    Bounding box: X[{lo[0]:.1f},{hi[0]:.1f}] "
          f"Y[{lo[1]:.1f},{hi[1]:.1f}] Z[{lo[2]:.1f},{hi[2]:.1f}]")


if __name__ == "__main__":
    main()
