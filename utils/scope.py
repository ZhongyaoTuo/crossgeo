#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV dense coverage polygon generator.
=====================================
Builds a bird's-eye-view density map from UAV depth maps and extracts a
dense coverage polygon (in lat/lng) via morphological filtering and contour
approximation. Useful for determining the valid UAV observation area.

Usage:
    python utils/scope.py --xml path/to/scene.xml \
        --depth_dir <uav_depth> --out_json scope.json --out_img bev_map.png
"""
import json
import argparse
import numpy as np
import cv2
import os
import xml.etree.ElementTree as ET
from pyproj import Transformer
from tqdm import tqdm
import tifffile
from scipy.spatial.transform import Rotation as R


class UAVDensePolygonGenerator:
    def __init__(self):
        self.geo_to_ecef = Transformer.from_crs("EPSG:4326", "EPSG:4978")
        self.ecef_to_geo = Transformer.from_crs("EPSG:4978", "EPSG:4326")
        self.origin_ecef = None
        self.R_ecef2enu = None
        self.uav_R_combined = None
        self.uav_poses = {}
        self.uav_intr = {}
        self.chunk_scale = 1.0

    def setup_reference(self, xml_path):
        """Parse the reference Metashape XML for poses and intrinsics."""
        print(f"[*] Parsing reference XML: {xml_path}")
        tree = ET.parse(xml_path)
        chunk = tree.getroot().find('chunk')
        trans_node = chunk.find('transform')

        chunk_R = np.fromstring(trans_node.find('rotation').text, sep=' ').reshape(3, 3)
        self.origin_ecef = np.fromstring(trans_node.find('translation').text, sep=' ').flatten()
        self.chunk_scale = float(trans_node.find('scale').text)

        lat, lon, _ = self.ecef_to_geo.transform(
            self.origin_ecef[0], self.origin_ecef[1], self.origin_ecef[2])
        phi, lam = np.radians(lat), np.radians(lon)
        self.R_ecef2enu = np.array([
            [-np.sin(lam),                   np.cos(lam),                  0],
            [-np.sin(phi)*np.cos(lam), -np.sin(phi)*np.sin(lam), np.cos(phi)],
            [ np.cos(phi)*np.cos(lam),  np.cos(phi)*np.sin(lam), np.sin(phi)]
        ])

        self.uav_R_combined = self.R_ecef2enu @ chunk_R
        self.uav_poses = {
            cam.get('label'): np.fromstring(cam.find('transform').text, sep=' ').reshape(4, 4)
            for cam in chunk.find('cameras').findall('camera')
            if cam.find('transform') is not None
        }

        sensors = chunk.findall('sensors/sensor')
        sensor = next((s for s in sensors if s.find('calibration') is not None), None)
        if sensor is None:
            raise ValueError("No sensor with <calibration> found in XML")
        cal = sensor.find('calibration')
        cx_elem = cal.find('cx')
        cy_elem = cal.find('cy')
        cx_offset = float(cx_elem.text) if cx_elem is not None else 0.0
        cy_offset = float(cy_elem.text) if cy_elem is not None else 0.0
        self.uav_intr = {
            'f': float(cal.find('f').text),
            'cx': (int(sensor.find('resolution').get('width')) / 2.0) + cx_offset,
            'cy': (int(sensor.find('resolution').get('height')) / 2.0) + cy_offset,
            'w': int(sensor.find('resolution').get('width')),
            'h': int(sensor.find('resolution').get('height'))
        }

    def generate_dense_polygon(self, depth_dir, output_json, output_debug_img,
                               grid_res=1.0, min_pts_per_grid=8):
        """Generate a dense coverage polygon from UAV depth maps.

        Args:
            depth_dir: directory of UAV depth TIFFs.
            output_json: output polygon JSON path.
            output_debug_img: output BEV debug image path.
            grid_res: meters per pixel in the BEV grid.
            min_pts_per_grid: density threshold; grids with fewer points are dropped.
        """
        all_enu_xy = []
        labels = sorted(self.uav_poses.keys())

        print(f"[*] Step 1: extracting point cloud (small step for density)...")
        for label in tqdm(labels):
            d_path = os.path.join(depth_dir, label + ".tif")
            if not os.path.exists(d_path):
                continue

            depth_img = tifffile.imread(d_path)
            h_d, w_d = depth_img.shape
            sx, sy = w_d / self.uav_intr['w'], h_d / self.uav_intr['h']
            fx, fy = self.uav_intr['f'] * sx, self.uav_intr['f'] * sy
            cx, cy = self.uav_intr['cx'] * sx, self.uav_intr['cy'] * sy

            step = 20
            mask = (depth_img[::step, ::step] > 0) & (depth_img[::step, ::step] < 65535)
            z = depth_img[::step, ::step][mask]
            v_idx, u_idx = np.where(mask)
            u, v = u_idx * step, v_idx * step

            pts_cam = np.stack(((u - cx) * z / fx, (v - cy) * z / fy, z), axis=0)
            T = self.uav_poses[label]
            pts_chunk = (T[:3, :3] @ pts_cam) + T[:3, 3].reshape(3, 1) * self.chunk_scale
            pts_enu = (self.uav_R_combined @ pts_chunk).T

            all_enu_xy.append(pts_enu[:, :2])

        if not all_enu_xy:
            return
        all_enu_xy = np.vstack(all_enu_xy)

        # --- Step 2: density heat map ---
        x_min, y_min = np.min(all_enu_xy, axis=0) - 5
        x_max, y_max = np.max(all_enu_xy, axis=0) + 5
        w_px = int((x_max - x_min) / grid_res)
        h_px = int((y_max - y_min) / grid_res)

        count_map = np.zeros((h_px, w_px), dtype=np.float32)
        px_x = ((all_enu_xy[:, 0] - x_min) / grid_res).astype(int)
        px_y = ((all_enu_xy[:, 1] - y_min) / grid_res).astype(int)

        np.add.at(count_map, (np.clip(px_y, 0, h_px - 1),
                              np.clip(px_x, 0, w_px - 1)), 1)

        bev_binary = (count_map >= min_pts_per_grid).astype(np.uint8) * 255

        # --- Step 3: morphological filtering ---
        # Opening: remove isolated noise
        kernel_open = np.ones((5, 5), np.uint8)
        bev_binary = cv2.morphologyEx(bev_binary, cv2.MORPH_OPEN, kernel_open)

        # Closing: fill small holes
        kernel_close = np.ones((15, 15), np.uint8)
        bev_binary = cv2.morphologyEx(bev_binary, cv2.MORPH_CLOSE, kernel_close)

        # Erosion: shrink boundary to remove sparse edges
        kernel_erode = np.ones((5, 5), np.uint8)
        bev_binary = cv2.erode(bev_binary, kernel_erode, iterations=2)

        # --- Step 4: extract contour ---
        contours, _ = cv2.findContours(bev_binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("[!] No remaining area after filtering; "
                  "lower min_pts_per_grid or iterations")
            return

        max_cnt = max(contours, key=cv2.contourArea)
        epsilon = 0.006 * cv2.arcLength(max_cnt, True)
        approx_poly = cv2.approxPolyDP(max_cnt, epsilon, True)

        # --- Step 5: visualization ---
        debug_img = cv2.cvtColor(bev_binary, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(debug_img, [approx_poly], -1, (0, 255, 0), 2)
        cv2.imwrite(output_debug_img, debug_img)

        # --- Step 6: convert back to lat/lng ---
        final_poly = []
        for pt in approx_poly:
            gx, gy = pt[0][0], pt[0][1]
            ex, ey = gx * grid_res + x_min, gy * grid_res + y_min
            p_ecef = self.R_ecef2enu.T @ np.array([ex, ey, 0]) + self.origin_ecef
            lat, lon, _ = self.ecef_to_geo.transform(p_ecef[0], p_ecef[1], p_ecef[2])
            final_poly.append({"lat": lat, "lng": lon})

        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(final_poly, f, indent=2)
        print(f"[*] Dense polygon saved, vertex count: {len(final_poly)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a dense UAV coverage polygon from depth maps"
    )
    parser.add_argument("--xml", type=str, required=True, help="Metashape XML path")
    parser.add_argument("--depth_dir", type=str, required=True,
                        help="UAV depth directory")
    parser.add_argument("--out_json", type=str, required=True,
                        help="output polygon JSON path")
    parser.add_argument("--out_img", type=str, required=True,
                        help="output BEV debug image path")
    parser.add_argument("--grid_res", type=float, default=2.0,
                        help="meters per grid cell (default: 2.0)")
    parser.add_argument("--min_pts", type=int, default=1,
                        help="min points per grid cell (default: 1)")
    args = parser.parse_args()

    gen = UAVDensePolygonGenerator()
    gen.setup_reference(args.xml)
    gen.generate_dense_polygon(args.depth_dir, args.out_json, args.out_img,
                               grid_res=args.grid_res, min_pts_per_grid=args.min_pts)
