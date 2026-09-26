#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Panorama to pinhole camera conversion.
======================================
Splits a Google Street View equirectangular panorama into four pinhole
views (front/right/back/left) with associated depth maps.

Usage:
    python ground/pano2pinhole.py --xml path/to/scene.xml \
        --rgb_dir <pano_rgb> --processed_dir <pano_depth> --output_dir <pinhole_out>
"""
import os
import json
import argparse
import numpy as np
import cv2
import tifffile
import xml.etree.ElementTree as ET
from pyproj import Transformer
from tqdm import tqdm


class PinholeDatasetExporter:
    def __init__(self, xml_path, fov=90, output_size=(1024, 1024)):
        """
        Args:
            fov: horizontal field of view (degrees).
            output_size: output image size (h, w).
        """
        self.fov = fov
        self.out_h, self.out_w = output_size

        # Geographic reference (ECEF -> ENU)
        self.geo_to_ecef = Transformer.from_crs("EPSG:4326", "EPSG:4978")
        tree = ET.parse(xml_path)
        self.origin_ecef = np.fromstring(tree.find('.//translation').text, sep=' ').flatten()

        # Precompute pinhole rays (90 deg FOV)
        half_fov_rad = np.radians(self.fov / 2)
        focal_length = (self.out_w / 2.0) / np.tan(half_fov_rad)

        # Normalized plane coordinates
        x, y = np.meshgrid(np.arange(self.out_w), np.arange(self.out_h))
        self.x_cam = (x - self.out_w / 2.0) / focal_length
        self.y_cam = (y - self.out_h / 2.0) / focal_length
        self.z_cam = np.ones_like(self.x_cam)

        # Precompute ray lengths (to convert spherical depth to planar depth)
        self.ray_lengths = np.sqrt(self.x_cam**2 + self.y_cam**2 + self.z_cam**2)
        self.cos_theta = 1.0 / self.ray_lengths

    def get_pano_map(self, view_yaw_deg, pano_w, pano_h):
        """Build the remap matrix for a given view yaw."""
        yaw_rad = np.radians(view_yaw_deg)
        Ry = np.array([
            [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
            [0, 1, 0],
            [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
        ])

        directions_cam = np.stack([self.x_cam, self.y_cam, self.z_cam], axis=-1)
        directions_world = directions_cam @ Ry.T

        xw, yw, zw = directions_world[..., 0], directions_world[..., 1], directions_world[..., 2]
        r = np.sqrt(xw**2 + yw**2 + zw**2)

        # Map to spherical coordinates
        phi = np.degrees(np.arcsin(-yw / r))  # negate Y to match image coords
        theta = np.degrees(np.arctan2(xw, zw))
        theta = (theta + 360) % 360

        # Map to panorama pixel coordinates
        u = (theta / 360.0) * (pano_w - 1)
        v = ((90.0 - phi) / 180.0) * (pano_h - 1)
        return u.astype(np.float32), v.astype(np.float32)

    def export(self, rgb_dir, processed_dir, output_root, count=9999):
        """Export four pinhole views per panorama station."""
        fixed_views = {
            'front': 0,
            'right': 90,
            'back': 180,
            'left': 270
        }

        param_files = sorted([f for f in os.listdir(processed_dir)
                              if f.endswith('_params.json')])[:count]

        for pf in tqdm(param_files):
            with open(os.path.join(processed_dir, pf), 'r', encoding='utf-8') as fj:
                m = json.load(fj)

            base_id = pf.replace('_params.json', '')
            station_dir = os.path.join(output_root, base_id)
            os.makedirs(station_dir, exist_ok=True)

            pano_rgb = cv2.imread(os.path.join(rgb_dir, m['rgb_file']))
            depth_raw = tifffile.imread(os.path.join(processed_dir, m['depth_file']))
            if pano_rgb is None or depth_raw is None:
                continue

            ph, pw = pano_rgb.shape[:2]
            pano_depth = cv2.resize(depth_raw, (pw, ph), interpolation=cv2.INTER_NEAREST)

            for dir_name, yaw_val in fixed_views.items():
                u, v = self.get_pano_map(yaw_val, pw, ph)

                # RGB resample: nearest-neighbor
                pin_rgb = cv2.remap(pano_rgb, u, v, cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_WRAP)

                # Depth resample: nearest-neighbor
                spherical_depth = cv2.remap(pano_depth, u, v, cv2.INTER_NEAREST,
                                            borderMode=cv2.BORDER_WRAP)

                # Spherical distance -> planar depth (Z-Buffer)
                pin_dep = spherical_depth * self.cos_theta
                pin_dep = np.maximum(pin_dep, 0.0)

                # Compute actual heading
                pano_heading_deg = np.degrees(m['heading'])
                real_yaw_deg = (pano_heading_deg + yaw_val) % 360.0

                rgb_filename = f"{dir_name}_rgb.jpg"
                depth_filename = f"{dir_name}_depth.tiff"
                cv2.imwrite(os.path.join(station_dir, rgb_filename), pin_rgb)
                tifffile.imwrite(os.path.join(station_dir, depth_filename),
                                 pin_dep.astype(np.float32))

                pinhole_meta = m.copy()
                pinhole_meta.update({
                    "heading": np.radians(real_yaw_deg),
                    "depth_file": depth_filename,
                    "rgb_file": rgb_filename,
                    "view_direction": dir_name,
                    "fov": self.fov,
                    "width": self.out_w,
                    "height": self.out_h,
                    "interpolation": "nearest"
                })

                with open(os.path.join(station_dir, f"{dir_name}_pose.json"), 'w') as f_out:
                    json.dump(pinhole_meta, f_out, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert panoramas to pinhole views (front/right/back/left)"
    )
    parser.add_argument("--xml", type=str, required=True,
                        help="Metashape XML path (for geographic reference)")
    parser.add_argument("--rgb_dir", type=str, required=True,
                        help="panorama RGB directory")
    parser.add_argument("--processed_dir", type=str, required=True,
                        help="processed panorama depth directory (with *_params.json)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="output pinhole directory")
    parser.add_argument("--fov", type=float, default=90.0,
                        help="horizontal FOV in degrees (default: 90)")
    parser.add_argument("--size", type=int, default=1024,
                        help="output image size (default: 1024)")
    args = parser.parse_args()

    exporter = PinholeDatasetExporter(
        xml_path=args.xml,
        fov=args.fov,
        output_size=(args.size, args.size)
    )

    exporter.export(
        rgb_dir=args.rgb_dir,
        processed_dir=args.processed_dir,
        output_root=args.output_dir
    )
