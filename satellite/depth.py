"""
Satellite depth generation for CrossGeo.
========================================
For each ground image in a pair, render the co-located satellite depth map
by projecting the UAV fused point cloud through a virtual top-down camera.

Flow (ref: map_depthv2.py):
  1. Traverse multi_dataset area_multi/sub/pair_*
  2. Group by (area, sub); build one global point cloud per group (from UAV depth)
  3. Render satellite depth for every ground image -> ground_XX_satellite_depth.tiff

Usage:
    python satellite/depth.py --multi_base <multi_dataset> --geo_base <geojsons>
"""
import os
import argparse
import numpy as np
import cv2
import tifffile
import xml.etree.ElementTree as ET
from pyproj import Transformer
from tqdm import tqdm
from collections import defaultdict

# ========== Parameters ==========
SATELLITE_ALTITUDE = 5726.0
FOV_DEG = 5.0
SAT_W, SAT_H = 1024, 1024

# ================= Coordinate helpers (consistent with map_depthv2.py) =================

def get_ecef_to_enu_rot(lat, lon):
    phi, lam = np.radians(lat), np.radians(lon)
    s_phi, c_phi = np.sin(phi), np.cos(phi)
    s_lam, c_lam = np.sin(lam), np.cos(lam)
    return np.array([
        [-s_lam,           c_lam,          0],
        [-s_phi * c_lam, -s_phi * s_lam, c_phi],
        [ c_phi * c_lam,  c_phi * s_lam, s_phi]
    ])

def parse_metashape_params(xml_path):
    """Parse Metashape XML; return poses, intrinsics, and coordinate transform params."""
    tree = ET.parse(xml_path)
    chunk = tree.getroot().find('chunk')
    transform_node = chunk.find('transform')
    chunk_R = np.fromstring(transform_node.find('rotation').text, sep=' ').reshape(3, 3)
    chunk_t_ecef = np.fromstring(transform_node.find('translation').text, sep=' ').reshape(3, 1)
    chunk_scale = float(transform_node.find('scale').text)

    transformer = Transformer.from_crs("EPSG:4978", "EPSG:4326")
    center_lat, center_lon, _ = transformer.transform(
        chunk_t_ecef[0].item(), chunk_t_ecef[1].item(), chunk_t_ecef[2].item()
    )
    R_ecef2enu = get_ecef_to_enu_rot(center_lat, center_lon)

    sensors = chunk.findall('sensors/sensor')
    sensor = next((s for s in sensors if s.find('calibration') is not None), None)
    if sensor is None:
        raise ValueError("No sensor with <calibration> found in XML")
    xml_res = sensor.find('resolution')
    xml_w, xml_h = int(xml_res.get('width')), int(xml_res.get('height'))
    f_xml = float(sensor.find('calibration/f').text)

    camera_poses = {}
    for cam in chunk.find('cameras').findall('camera'):
        if cam.find('transform') is not None:
            camera_poses[cam.get('label')] = np.fromstring(
                cam.find('transform').text, sep=' '
            ).reshape(4, 4)

    return camera_poses, f_xml, xml_w, xml_h, chunk_scale, chunk_R, R_ecef2enu, chunk_t_ecef


def build_global_point_cloud(camera_poses, f_xml, xml_w, xml_h,
                              chunk_scale, chunk_R, R_ecef2enu, depth_dir):
    """Build a global point cloud from all UAV depth maps (ENU frame)."""
    all_pts = []
    R_combined = R_ecef2enu @ chunk_R

    for label, T_c2chunk in tqdm(camera_poses.items(), desc="  Loading UAV depth"):
        d_path = os.path.join(depth_dir, label + ".tif")
        if not os.path.exists(d_path):
            d_path += "f"
        if not os.path.exists(d_path):
            continue

        depth = tifffile.imread(d_path).astype(np.float32)
        h_d, w_d = depth.shape

        sx, sy = w_d / xml_w, h_d / xml_h
        fx, fy = f_xml * sx, f_xml * sy
        cx, cy = (xml_w / 2) * sx, (xml_h / 2) * sy

        mask = (depth > 1.0) & (depth < 150.0)
        if not np.any(mask):
            continue

        # Subsample every 10th point to save memory
        z = depth[mask][::10]
        v, u = np.where(mask)
        u, v = u[::10], v[::10]

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        pts_cam = np.stack((x, y, z), axis=0)

        t_scaled = T_c2chunk[:3, 3].reshape(3, 1) * chunk_scale
        pts_chunk = (T_c2chunk[:3, :3] @ pts_cam) + t_scaled
        pts_enu = (R_combined @ pts_chunk).T
        all_pts.append(pts_enu.astype(np.float32))

    return np.vstack(all_pts)


def render_satellite_depth(global_pts, sat_enu_pos, img_size):
    """Z-Buffer render a satellite depth map."""
    img_w, img_h = img_size
    f_pixel = (img_w / 2.0) / np.tan(np.radians(FOV_DEG / 2.0))

    # Satellite camera pose: CamX=East, CamY=South, CamZ=Down
    R_w2c = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])

    pts_rel = global_pts - sat_enu_pos
    pts_cam = pts_rel @ R_w2c.T

    z = pts_cam[:, 2]
    mask = (z > 4000) & (z < 7000)
    pts_cam, z = pts_cam[mask], z[mask]

    u = (pts_cam[:, 0] * f_pixel / z + img_w / 2).astype(np.int32)
    v = (pts_cam[:, 1] * f_pixel / z + img_h / 2).astype(np.int32)

    valid = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    u, v, z = u[valid], v[valid], z[valid]

    depth_map = np.zeros((img_h, img_w), dtype=np.float32)
    if len(z) > 0:
        indices = np.argsort(-z)
        depth_map[v[indices], u[indices]] = z[indices]

    return depth_map


def save_depth_viz(depth_map, save_path):
    """Save a depth visualization preview (colormap)."""
    if not np.any(depth_map > 0):
        return
    viz = depth_map.copy()
    valid_mask = viz > 0
    viz[valid_mask] = (viz[valid_mask] - viz[valid_mask].min()) / \
                      (viz[valid_mask].max() - viz[valid_mask].min())
    viz_img = (viz * 255).astype(np.uint8)
    viz_colored = cv2.applyColorMap(255 - viz_img, cv2.COLORMAP_JET)
    cv2.imwrite(save_path, viz_colored)


def process_group(area, sub, pair_dirs, geo_to_ecef, geo_base):
    """Process one (area, sub) group:
      1. Build one global point cloud
      2. Render satellite depth for every ground image in the group's pairs
    """
    tag = f"{area}_{sub}"
    xml_path = os.path.join(geo_base, area, f"{tag}.xml")
    depth_dir = os.path.join(geo_base, area, f"{tag}_depth")

    if not os.path.exists(xml_path) or not os.path.isdir(depth_dir):
        print(f"  [!] Skipping {tag}: missing XML or depth directory")
        return 0, 0

    print(f"\n{'='*50}")
    print(f"  Processing: {tag} ({len(pair_dirs)} pairs)")
    print(f"{'='*50}")

    # 1. Parse XML
    camera_poses, f_xml, xml_w, xml_h, chunk_scale, chunk_R, R_ecef2enu, chunk_t_ecef = \
        parse_metashape_params(xml_path)
    origin_ecef = chunk_t_ecef.flatten()

    # 2. Build global point cloud
    print(f"  [*] Building global point cloud ({len(camera_poses)} cameras)...")
    global_pcd = build_global_point_cloud(
        camera_poses, f_xml, xml_w, xml_h, chunk_scale, chunk_R, R_ecef2enu, depth_dir
    )
    print(f"      Point cloud size: {len(global_pcd):,}")

    # 3. Iterate over all pairs' ground images
    total_pairs = 0
    total_depths = 0
    for pair_dir in tqdm(pair_dirs, desc="  Rendering satellite depth"):
        # Find all ground_XX_rgb.npy (exclude pano and already-generated satellite)
        npy_files = sorted([
            f for f in os.listdir(pair_dir)
            if f.startswith("ground_") and f.endswith("_rgb.npy")
               and "pano" not in f and "_satellite" not in f
        ])
        if not npy_files:
            continue

        rendered_any = False
        for npy_name in npy_files:
            npy_path = os.path.join(pair_dir, npy_name)
            try:
                data = np.load(npy_path, allow_pickle=True).item()
                raw_data = data['raw_data']
                lat, lon = raw_data[3], raw_data[4]
            except Exception:
                continue

            stem = npy_name.replace("_rgb.npy", "")
            out_name = f"{stem}_satellite_depth.tiff"
            out_path = os.path.join(pair_dir, out_name)
            if os.path.exists(out_path):
                continue  # skip if already exists

            # Compute satellite position in ENU
            sat_ecef = np.array(geo_to_ecef.transform(lat, lon, SATELLITE_ALTITUDE))
            sat_enu_pos = R_ecef2enu @ (sat_ecef - origin_ecef)

            # Render depth
            sat_depth = render_satellite_depth(global_pcd, sat_enu_pos, (SAT_W, SAT_H))
            tifffile.imwrite(out_path, sat_depth.astype(np.float32))

            # Visualization preview
            viz_path = out_path.replace(".tiff", "_viz.png")
            save_depth_viz(sat_depth, viz_path)

            total_depths += 1
            rendered_any = True

        if rendered_any:
            total_pairs += 1

    return total_pairs, total_depths


def main(multi_base, geo_base):
    if not os.path.isdir(multi_base):
        print(f"[!] Directory does not exist: {multi_base}")
        return

    geo_to_ecef = Transformer.from_crs("EPSG:4326", "EPSG:4978")

    # 1. Scan multi_dataset, group pair paths by (area, sub)
    groups = defaultdict(list)  # (area, sub) -> [pair_dir, ...]

    area_dirs = sorted(os.listdir(multi_base))
    for area in area_dirs:
        area_path = os.path.join(multi_base, area)
        if not os.path.isdir(area_path):
            continue
        # area may be "1001_multi", extract numeric part
        area_num = area.replace("_multi", "")
        sub_dirs = sorted(os.listdir(area_path))
        for sub in sub_dirs:
            sub_path = os.path.join(area_path, sub)
            if not os.path.isdir(sub_path):
                continue
            pair_dirs = [
                os.path.join(sub_path, d) for d in sorted(os.listdir(sub_path))
                if d.startswith("pair_") and os.path.isdir(os.path.join(sub_path, d))
            ]
            if pair_dirs:
                groups[(area_num, sub)].extend(pair_dirs)

    print(f"[*] Found {len(groups)} (area, sub) groups")
    total_pairs_total = 0
    total_depths_total = 0

    # 2. Process by group
    for (area, sub), pair_dirs in sorted(groups.items()):
        p, d = process_group(area, sub, pair_dirs, geo_to_ecef, geo_base)
        total_pairs_total += p
        total_depths_total += d

    print(f"\n{'='*50}")
    print(f"[OK] Done!")
    print(f"    Pairs processed: {total_pairs_total}")
    print(f"    Depth maps generated: {total_depths_total}")


def process_satellite_depth(fused_ply: str, output_path: str,
                            width: int = 1024, height: int = 1024,
                            sat_enu_pos: np.ndarray = None) -> dict:
    """Single-scene satellite depth from UAV fused point cloud.

    Wrapper for pipeline.py: reads the UAV fused point cloud, renders a
    satellite depth map via Z-Buffer (render_satellite_depth), and saves
    as TIFF.

    Args:
        fused_ply: path to UAV COLMAP fused point cloud (.ply).
        output_path: output depth TIFF path.
        width, height: satellite image size.
        sat_enu_pos: satellite ENU position (default: origin overhead).

    Returns:
        {"depth_path", "num_valid_pixels"}
    """
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(fused_ply)
    global_pts = np.asarray(pcd.points)

    if sat_enu_pos is None:
        sat_enu_pos = np.array([0.0, 0.0, SATELLITE_ALTITUDE])

    depth_map = render_satellite_depth(global_pts, sat_enu_pos, (width, height))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tifffile.imwrite(output_path, depth_map)

    num_valid = int(np.count_nonzero(depth_map > 0))
    print(f"Satellite depth: {num_valid} valid pixels -> {output_path}")
    return {"depth_path": output_path, "num_valid_pixels": num_valid}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate satellite depth maps for all ground images"
    )
    parser.add_argument("--multi_base", type=str, required=True,
                        help="Root directory of multi_dataset (area_multi/sub/pair_*)")
    parser.add_argument("--geo_base", type=str, required=True,
                        help="Root directory of geojsons (contains area/{tag}.xml and depth)")
    args = parser.parse_args()
    main(args.multi_base, args.geo_base)
