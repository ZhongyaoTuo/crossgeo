#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tri-view pairing for CrossGeo.
=============================
Find satellite / UAV / ground quadruplets by mutual voxel overlap in the EDS
world frame (X=South, Y=Down, Z=East).

CrossGeo (paper Sec 2.1 -- Metric depth and tri-view pairing): pairs are
formed by voxel overlap scoring; the highest-scoring six-image tuples
(two views per modality) become the tri-view samples, requiring all
C(6,2)=15 pairwise overlaps to be non-empty.

Each saved pair contains:
  - 2 ground panoramas (+ optional pinhole views in the overlap region)
  - 2 UAV images
  - 2 satellite tiles (co-located with ground stations)
with poses and depths in the EDS frame.
"""
import os
import json
import numpy as np
import cv2
import tifffile
import xml.etree.ElementTree as ET
from pyproj import Transformer
from tqdm import tqdm
import shutil
import open3d as o3d
import glob
from scipy.spatial.transform import Rotation as R_rot


class MutualOverlapMatcher:
    def __init__(self, xml_path, voxel_size=1.0):
        self.v_size = voxel_size
        self.geo_to_ecef = Transformer.from_crs("EPSG:4326", "EPSG:4978")
        self.ecef_to_geo = Transformer.from_crs("EPSG:4978", "EPSG:4326")

        # ENU -> EDS
        self.colmap_to_opencv = np.array([
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.R_enu_to_eds = self.colmap_to_opencv[:3, :3]

        print(f"[*] Initializing overlap matcher (EDS, voxel={voxel_size}m, panorama pairing)...")
        tree = ET.parse(xml_path)
        chunk = tree.getroot().find('chunk')
        trans_node = chunk.find('transform')

        self.origin_ecef = np.fromstring(trans_node.find('translation').text, sep=' ').flatten()
        self.chunk_scale = float(trans_node.find('scale').text)
        chunk_R = np.fromstring(trans_node.find('rotation').text, sep=' ').reshape(3, 3)

        lat_ref, lon_ref, alt_ref = self.ecef_to_geo.transform(*self.origin_ecef[:3])
        phi, lam = np.radians(lat_ref), np.radians(lon_ref)
        self.R_ecef2enu = np.array([
            [-np.sin(lam),                   np.cos(lam),                  0],
            [-np.sin(phi)*np.cos(lam), -np.sin(phi)*np.sin(lam), np.cos(phi)],
            [ np.cos(phi)*np.cos(lam),  np.cos(phi)*np.sin(lam), np.sin(phi)]
        ])

        self.uav_R_global = self.R_ecef2enu @ chunk_R
        self.uav_poses = {}
        for cam in chunk.find('cameras').findall('camera'):
            label = cam.get('label')
            trans = cam.find('transform')
            if trans is not None:
                self.uav_poses[label] = np.fromstring(trans.text, sep=' ').reshape(4, 4)

        # UAV calibration
        sensor = chunk.find('sensors/sensor')
        cal = sensor.find('calibration')
        res = sensor.find('resolution')
        self.original_width = int(res.get('width'))
        self.original_height = int(res.get('height'))
        self.uav_intr_original = {
            'f': float(cal.find('f').text),
            'cx': float(cal.find('cx').text) + self.original_width / 2.0,
            'cy': float(cal.find('cy').text) + self.original_height / 2.0,
        }

        self.uav_data = []           # UAV point cloud / voxels (for pairing)
        self.ground_data = []        # panorama point cloud / voxels (for pairing)
        self.ground_panoramas = {}   # station_id -> (rgb_path, depth_path)
        self.ground_original_params = {}
        self.ground_pinhole_views = {}   # station_id -> {view_name: {...}}

    def load_image_robust(self, path_no_ext):
        for ext in ['.jpg', '.JPG', '.jpeg', '.png']:
            path = path_no_ext + ext
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    return path, img
        return None, None

    def get_voxels(self, pts):
        if pts.size == 0:
            return set()
        return set(map(tuple, np.floor(pts / self.v_size).astype(int)))

    def unified_projection(self, depth, K, R_c2w, t_c2w, m_json=None, step=8,
                           is_uav=False, use_formula=False):
        d_max = 120.0 if is_uav else 40.0
        mask = (depth > 0.5) & (depth < d_max)
        v_idx, u_idx = np.where(mask)
        u, v = u_idx[::step], v_idx[::step]
        z = depth[v, u]

        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        pts_cam = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=-1)

        if use_formula and m_json is not None:
            yaw_rad = m_json['heading'] + np.pi
            R_h = np.array([[np.cos(yaw_rad), 0, np.sin(yaw_rad)],
                            [0, 1, 0],
                            [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]])
            pts_rot = pts_cam @ R_h.T
            pts_enu_local = np.stack([pts_rot[:, 0], pts_rot[:, 2], -pts_rot[:, 1]], axis=-1)
            ecef = self.geo_to_ecef.transform(m_json['latitude'], m_json['longitude'], m_json['elevation'])
            t_enu = self.R_ecef2enu @ (np.array(ecef) - self.origin_ecef[:3])
            p_enu = pts_enu_local + t_enu
            p_eds = p_enu @ self.R_enu_to_eds.T
            return p_eds, (u, v)
        else:
            pts_eds = (pts_cam @ R_c2w.T) + t_c2w
            return pts_eds, (u, v)

    def calculate_ground_pose_matrix_pinhole(self, m):
        """Pinhole pose: heading + pi (consistent with pano2pinhole output)."""
        yaw_rad = m['heading'] + np.pi
        R_h = np.array([[np.cos(yaw_rad), 0, np.sin(yaw_rad)],
                        [0, 1, 0],
                        [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]])
        M_axis = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        R_enu = M_axis @ R_h
        ecef = self.geo_to_ecef.transform(m['latitude'], m['longitude'], m['elevation'])
        t_enu = self.R_ecef2enu @ (np.array(ecef) - self.origin_ecef[:3])
        R_eds = self.R_enu_to_eds @ R_enu
        t_eds = self.R_enu_to_eds @ t_enu
        return R_eds, t_eds

    def calculate_ground_pose_matrix_pano(self, m):
        """Panorama pose: heading without + pi (correct version)."""
        r = R_rot.from_euler('YXZ', [m['heading'], m.get('pitch', 0.0), m.get('roll', 0.0)], degrees=False)
        R_scipy = r.as_matrix()
        R_enu = R_scipy[[0, 2, 1], :]
        ecef = self.geo_to_ecef.transform(m['latitude'], m['longitude'], m['elevation'])
        t_enu = self.R_ecef2enu @ (np.array(ecef) - self.origin_ecef[:3])
        R_eds = self.R_enu_to_eds @ R_enu
        t_eds = self.R_enu_to_eds @ t_enu
        return R_eds, t_eds

    def calculate_uav_pose_matrix(self, T):
        R_enu = self.uav_R_global @ T[:3, :3]
        t_enu = self.uav_R_global @ (T[:3, 3] * self.chunk_scale)
        R_eds = self.R_enu_to_eds @ R_enu
        t_eds = self.R_enu_to_eds @ t_enu
        return R_eds, t_eds

    def matrix_to_euler(self, R):
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        return np.degrees(np.array([x, y, z]))

    # ------------------------------------------------------------------
    #  Panorama -> ENU point cloud
    # ------------------------------------------------------------------
    def pano_to_pointcloud(self, m, rgb_path, depth_path, step=8):
        rgb = cv2.imread(rgb_path)
        depth = tifffile.imread(depth_path).astype(np.float32)
        if rgb is None:
            return np.empty((0, 3)), None, None

        ph, pw = rgb.shape[:2]
        dh, dw = depth.shape
        if dh != ph or dw != pw:
            depth = cv2.resize(depth, (pw, ph), interpolation=cv2.INTER_NEAREST)

        u_grid, v_grid = np.meshgrid(np.arange(pw), np.arange(ph))
        theta = 2.0 * np.pi * u_grid / pw - np.pi
        phi_angle = np.pi / 2.0 - np.pi * v_grid / ph
        cos_phi = np.cos(phi_angle)
        dirs = np.stack([
            cos_phi * np.sin(theta),
            np.sin(phi_angle),
            cos_phi * np.cos(theta)
        ], axis=-1)

        r = R_rot.from_euler('YXZ', [m['heading'], m.get('pitch', 0.0), m.get('roll', 0.0)], degrees=False)
        dirs_rot = dirs @ r.as_matrix().T

        mask = (depth > 0.5) & (depth < 40.0)
        if not np.any(mask):
            return np.empty((0, 3)), None, None

        v_idx, u_idx = np.where(mask)
        u_s = u_idx[::step]
        v_s = v_idx[::step]
        d_vals = depth[v_s, u_s]
        dirs_valid = dirs_rot[v_s, u_s]

        pts_world = dirs_valid * d_vals[:, np.newaxis]
        pts_enu = np.stack([pts_world[:, 0], pts_world[:, 2], pts_world[:, 1]], axis=-1)

        ecef = np.array(self.geo_to_ecef.transform(m['latitude'], m['longitude'], m['elevation']))
        t_enu = self.R_ecef2enu @ (ecef - self.origin_ecef[:3])
        pts_enu = pts_enu + t_enu

        colors = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)[v_s, u_s] / 255.0
        return pts_enu, colors, (u_s, v_s)

    # ------------------------------------------------------------------
    #  UAV data collection (fixed: full-resolution depth + correct intrinsics scaling)
    # ------------------------------------------------------------------
    def collect_uav_data(self, rgb_dir, depth_dir):
        print("[*] Processing UAV data (EDS)...")
        for label, T in tqdm(self.uav_poses.items()):
            depth_path = os.path.join(depth_dir, label + ".tif")
            if not os.path.exists(depth_path):
                depth_path += "f"
            rgb_path_base = os.path.join(rgb_dir, label)
            rgb_path, _ = self.load_image_robust(rgb_path_base)
            if not os.path.exists(depth_path) or rgb_path is None:
                continue

            d_img = tifffile.imread(depth_path)
            depth_h, depth_w = d_img.shape

            # Depth resolution scaling relative to original
            sx = depth_w / self.original_width
            sy = depth_h / self.original_height
            fx_d = self.uav_intr_original['f'] * sx
            fy_d = self.uav_intr_original['f'] * sy
            cx_d = self.uav_intr_original['cx'] * sx
            cy_d = self.uav_intr_original['cy'] * sy
            K_depth = np.array([[fx_d, 0, cx_d], [0, fy_d, cy_d], [0, 0, 1]], dtype=np.float32)

            # Full-resolution depth projection -> correct voxels
            R_eds, t_eds = self.calculate_uav_pose_matrix(T)
            pts, _ = self.unified_projection(d_img, K_depth, R_eds, t_eds, step=8, is_uav=True)

            # Center crop + resize to 512x512 (for saving)
            crop_size = min(depth_h, depth_w)
            left_d = (depth_w - crop_size) // 2
            top_d = (depth_h - crop_size) // 2
            d_cropped = d_img[top_d:top_d+crop_size, left_d:left_d+crop_size]
            d_processed = cv2.resize(d_cropped, (512, 512), interpolation=cv2.INTER_NEAREST)

            scale_512 = 512.0 / crop_size
            f_512 = fx_d * scale_512
            cx_512 = (cx_d - left_d) * scale_512
            cy_512 = (cy_d - top_d) * scale_512
            K_512 = np.array([[f_512, 0, cx_512], [0, f_512, cy_512], [0, 0, 1]], dtype=np.float32)

            ecef_pos = self.origin_ecef[:3] + self.R_ecef2enu.T @ (self.R_enu_to_eds.T @ t_eds)
            lat, lon, alt = self.ecef_to_geo.transform(*ecef_pos)
            eulers = self.matrix_to_euler(R_eds)
            raw_array = np.array([eulers[1], eulers[0], eulers[2], lat, lon, alt])

            self.uav_data.append({
                'label': label, 'voxels': self.get_voxels(pts), 'center': t_eds,
                'rgb_src': rgb_path, 'depth_src': depth_path, 'depth_processed': d_processed,
                'K': K_512, 'R_c2w': R_eds, 't_c2w': t_eds,
                'raw_data': raw_array, 'type': 'uav'
            })
        print(f"    UAV: {len(self.uav_data)} frames")

    # ------------------------------------------------------------------
    #  Ground data collection (panoramas primary + pinhole views secondary)
    # ------------------------------------------------------------------
    def collect_ground_data(self, pinhole_dir, pano_rgb_dir, pano_depth_dir):
        self.pano_rgb_root = pano_rgb_dir
        self.pano_depth_root = pano_depth_dir
        print("[*] Processing ground panorama data (EDS)...")

        # Scan _params.json
        param_files = sorted([f for f in os.listdir(pano_depth_dir) if f.endswith('_params.json')])
        print(f"    Found {len(param_files)} stations")

        # pinhole subdirectories
        pinhole_subs = set()
        if os.path.isdir(pinhole_dir):
            area_prefix = os.path.basename(os.path.normpath(pinhole_dir))
            all_dirs = [d for d in os.listdir(pinhole_dir) if os.path.isdir(os.path.join(pinhole_dir, d))]
            pinhole_subs = set(d for d in all_dirs if d.startswith(area_prefix))

        for pf in tqdm(param_files, desc="panoramas"):
            with open(os.path.join(pano_depth_dir, pf), 'r') as f:
                m = json.load(f)
            sid = pf.replace('_params.json', '')
            self.ground_original_params[sid] = m

            # Find RGB
            rgb_path = os.path.join(pano_rgb_dir, m.get('rgb_file', ''))
            if not os.path.exists(rgb_path):
                alt = m.get('rgb_file', '').replace('_', '_ ')
                rgb_path = os.path.join(pano_rgb_dir, alt)
            if not os.path.exists(rgb_path):
                for ft in os.listdir(pano_rgb_dir):
                    if ft.startswith(sid) and ft.lower().endswith(('.jpg', '.jpeg', '.png')):
                        rgb_path = os.path.join(pano_rgb_dir, ft)
                        break

            # Find depth
            depth_path = os.path.join(pano_depth_dir, m.get('depth_file', ''))
            if not os.path.exists(depth_path):
                depth_path = depth_path.replace('.tiff', '.tif')
            if not os.path.exists(depth_path):
                depth_path = os.path.join(pano_depth_dir, f"{sid}_depth.tiff")
            if not os.path.exists(depth_path):
                depth_path = os.path.join(pano_depth_dir, f"{sid}_depth.tif")
            if not os.path.exists(depth_path):
                for c in glob.glob(os.path.join(pano_depth_dir, f"{sid}*.*")):
                    fn = os.path.basename(c).lower()
                    if fn.endswith(('.tif', '.tiff')) and 'depth' in fn:
                        depth_path = c
                        break
            if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                continue

            self.ground_panoramas[sid] = (rgb_path, depth_path)

            # ---- Panorama point cloud (for pairing) ----
            pts_enu, _, _ = self.pano_to_pointcloud(m, rgb_path, depth_path, step=8)
            if len(pts_enu) == 0:
                continue
            pts_eds = pts_enu @ self.R_enu_to_eds.T
            center = np.median(pts_eds, axis=0)

            h, w = cv2.imread(rgb_path).shape[:2]
            fx = w / (2 * np.pi)
            fy = h / np.pi
            K = np.array([[fx, 0, w/2], [0, fy, h/2], [0, 0, 1]], dtype=np.float32)

            R_eds, t_eds = self.calculate_ground_pose_matrix_pano(m)
            raw_array = np.array([m.get('pitch', 0.0), m.get('roll', 0.0), np.degrees(m['heading']),
                                  m['latitude'], m['longitude'], m['elevation']])

            self.ground_data.append({
                'station': sid, 'voxels': self.get_voxels(pts_eds), 'center': center,
                'pano_rgb_path': rgb_path, 'pano_depth_path': depth_path,
                'K': K, 'R_c2w': R_eds, 't_c2w': t_eds,
                'pano_params': m, 'raw_data': raw_array, 'type': 'ground_pano'
            })

            # ---- Load pinhole views (for segmented output) ----
            if sid not in pinhole_subs:
                continue
            spath = os.path.join(pinhole_dir, sid)
            pinhole_views = {}
            for view in ['front', 'back', 'left', 'right']:
                jpath = os.path.join(spath, f"{view}_pose.json")
                if not os.path.exists(jpath):
                    continue
                with open(jpath, 'r') as f:
                    m_v = json.load(f)
                rgb_v, _ = self.load_image_robust(os.path.join(spath, f"{view}_rgb"))
                depth_v = os.path.join(spath, m_v['depth_file'])
                if not os.path.exists(depth_v) or rgb_v is None:
                    continue

                d_img = tifffile.imread(depth_v)
                R_eds_v, t_eds_v = self.calculate_ground_pose_matrix_pinhole(m_v)
                focal = (d_img.shape[1] / 2.0) / np.tan(np.radians(m_v.get('fov', 90)) / 2.0)
                K_v = np.array([[focal, 0, d_img.shape[1] / 2.0],
                                [0, focal, d_img.shape[0] / 2.0],
                                [0, 0, 1]])
                pts_v, _ = self.unified_projection(d_img, K_v, R_eds_v, t_eds_v,
                                                   m_json=m_v, step=4, is_uav=False, use_formula=True)
                raw_arr = np.array([m_v['pitch'], m_v['roll'], np.degrees(m_v['heading']),
                                    m_v['latitude'], m_v['longitude'], m_v['elevation']])

                pinhole_views[view] = {
                    'rgb_src': rgb_v, 'depth_src': depth_v, 'depth': d_img,
                    'K': K_v, 'R_c2w': R_eds_v, 't_c2w': t_eds_v,
                    'voxels': self.get_voxels(pts_v), 'raw_data': raw_arr, 'pose_json': m_v
                }
            if pinhole_views:
                self.ground_pinhole_views[sid] = pinhole_views

        print(f"    Ground: {len(self.ground_data)} panoramas")

    # ------------------------------------------------------------------
    #  Pairing: panorama <-> panorama -> UAV
    # ------------------------------------------------------------------
    def find_quadruplets(self, output_dir, min_mutual_voxels=100,
                         min_pinhole_overlap_ratio=0.15):
        print(f"[*] Pairing (direct panorama pairing, EDS)...")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self._min_pinhole_ratio = min_pinhole_overlap_ratio

        pair_idx = 0
        for i in range(len(self.ground_data)):
            g1 = self.ground_data[i]
            for j in range(i + 1, len(self.ground_data)):
                g2 = self.ground_data[j]
                if np.linalg.norm(g1['center'] - g2['center']) > 30.0:
                    continue
                v_g12 = g1['voxels'].intersection(g2['voxels'])
                if len(v_g12) < min_mutual_voxels:
                    continue
                u_candidates = []
                for u in self.uav_data:
                    v_overlap = v_g12.intersection(u['voxels'])
                    if len(v_overlap) >= min_mutual_voxels:
                        u_candidates.append((len(v_overlap), u))
                if len(u_candidates) >= 2:
                    u_candidates.sort(key=lambda x: x[0], reverse=True)
                    u1, u2 = u_candidates[0][1], u_candidates[1][1]
                    final_v = v_g12.intersection(u1['voxels']).intersection(u2['voxels'])
                    if len(final_v) >= min_mutual_voxels:
                        self.save_quad(output_dir, pair_idx, g1, g2, u1, u2,
                                       len(final_v), v_g12)
                        pair_idx += 1
                        if pair_idx >= 100:
                            return
        print(f"    [debug] final pairs: {pair_idx}")

    # ------------------------------------------------------------------
    #  Save pair (panoramas primary + pinhole segmented output)
    # ------------------------------------------------------------------
    def save_quad(self, out_dir, idx, g1, g2, u1, u2, count, pano_overlap_voxels):
        folder = os.path.join(out_dir, f"pair_{idx}")
        os.makedirs(folder, exist_ok=True)

        meta = {
            "pair_id": idx, "overlap_voxels": count,
            "world_system": "EDS", "pairing_basis": "panorama", "images": []
        }
        f_pts_all, m_pts_all, f_rgb_all, f_seg_all = [], [], [], []

        # ---- Save 2 panoramas (ground primary) ----
        for g_data, prefix in [(g1, "ground_1"), (g2, "ground_2")]:
            station = g_data['station']

            pano_rgb_src = g_data['pano_rgb_path']
            rgb_ext = os.path.splitext(pano_rgb_src)[1]
            dst_pano_rgb = os.path.join(folder, f"{prefix}_pano_rgb{rgb_ext}")
            shutil.copy(pano_rgb_src, dst_pano_rgb)

            dst_pano_depth = os.path.join(folder, f"{prefix}_pano_depth.tiff")
            shutil.copy(g_data['pano_depth_path'], dst_pano_depth)

            orig_m = g_data['pano_params']
            raw_array = np.array([orig_m['pitch'], orig_m['roll'], np.degrees(orig_m['heading']),
                                  orig_m['latitude'], orig_m['longitude'], orig_m['elevation']])
            c2w_4x4 = np.eye(4)
            c2w_4x4[:3, :3] = g_data['R_c2w']
            c2w_4x4[:3, 3] = g_data['t_c2w']
            npy_payload = {'intrinsics': g_data['K'], 'c2w': c2w_4x4, 'raw_data': raw_array}
            dst_pano_npy = os.path.join(folder, f"{prefix}_pano_rgb.npy")
            np.save(dst_pano_npy, npy_payload)

            # Panorama point cloud
            pts_enu, cols_rgb, (u_idx, v_idx) = self.pano_to_pointcloud(
                orig_m, pano_rgb_src, g_data['pano_depth_path'], step=4)
            if len(pts_enu) > 0:
                p_f = pts_enu @ self.R_enu_to_eds.T
                p_m = p_f.copy()
                sc = [1.0, 0.0, 0.0]
                f_pts_all.append(p_f)
                m_pts_all.append(p_m)
                f_rgb_all.append(cols_rgb)
                f_seg_all.append(np.tile(sc, (len(p_f), 1)))

            meta["images"].append({
                "name": prefix, "type": "ground_pano", "station_id": station,
                "panorama_rgb": os.path.basename(dst_pano_rgb),
                "panorama_depth": os.path.basename(dst_pano_depth),
                "panorama_npy": os.path.basename(dst_pano_npy)
            })

        # ---- Save 2 UAV images (with corrected K_512) ----
        for u_data, prefix in [(u1, "uav_1"), (u2, "uav_2")]:
            rgb_dst = os.path.join(folder, f"{prefix}_rgb.jpg")
            depth_dst = os.path.join(folder, f"{prefix}_depth.tiff")
            npy_dst = os.path.join(folder, f"{prefix}_rgb.npy")

            left = (self.original_width - self.original_height) // 2
            img = cv2.imread(u_data['rgb_src'])
            cv2.imwrite(rgb_dst, cv2.resize(img[:, left:left+self.original_height], (512, 512)))
            tifffile.imwrite(depth_dst, u_data['depth_processed'].astype(np.float32))

            c2w_4x4 = np.eye(4)
            c2w_4x4[:3, :3] = u_data['R_c2w']
            c2w_4x4[:3, 3] = u_data['t_c2w']
            npy_payload = {'intrinsics': u_data['K'], 'c2w': c2w_4x4, 'raw_data': u_data['raw_data']}
            np.save(npy_dst, npy_payload)

            p_f, (u_idx, v_idx) = self.unified_projection(
                u_data['depth_processed'], u_data['K'], u_data['R_c2w'], u_data['t_c2w'],
                step=4, is_uav=True, use_formula=False)
            p_m = p_f.copy()

            img_c = cv2.cvtColor(cv2.imread(rgb_dst), cv2.COLOR_BGR2RGB)
            cols_rgb = img_c[v_idx, u_idx] / 255.0
            sc = [0.0, 0.0, 1.0]
            f_pts_all.append(p_f)
            m_pts_all.append(p_m)
            f_rgb_all.append(cols_rgb)
            f_seg_all.append(np.tile(sc, (len(p_f), 1)))

            meta["images"].append({
                "name": prefix, "type": "uav", "station_id": u_data.get('label')
            })

        # ---- Segment pinhole views: keep only those in the panorama overlap region ----
        for g_data, prefix in [(g1, "ground_1"), (g2, "ground_2")]:
            station = g_data['station']
            pinhole_views = self.ground_pinhole_views.get(station, {})
            if not pinhole_views:
                continue

            overlapping_views = []
            for view_name, v_data in pinhole_views.items():
                view_overlap = v_data['voxels'].intersection(pano_overlap_voxels)
                ratio = len(view_overlap) / max(len(v_data['voxels']), 1)
                if ratio >= self._min_pinhole_ratio:
                    overlapping_views.append((view_name, ratio, v_data))

            if overlapping_views:
                seg_info = {"station": station, "overlapping_pinhole_views": []}
                for view_name, ratio, v_data in overlapping_views:
                    ph_prefix = f"{prefix}_pinhole_{view_name}"

                    rgb_dst = os.path.join(folder, f"{ph_prefix}_rgb.jpg")
                    depth_dst = os.path.join(folder, f"{ph_prefix}_depth.tiff")
                    npy_dst = os.path.join(folder, f"{ph_prefix}_rgb.npy")

                    shutil.copy(v_data['rgb_src'], rgb_dst)
                    shutil.copy(v_data['depth_src'], depth_dst)

                    c2w_4x4 = np.eye(4)
                    c2w_4x4[:3, :3] = v_data['R_c2w']
                    c2w_4x4[:3, 3] = v_data['t_c2w']
                    npy_payload = {
                        'intrinsics': v_data['K'], 'c2w': c2w_4x4, 'raw_data': v_data['raw_data']
                    }
                    np.save(npy_dst, npy_payload)

                    # Pinhole point cloud (for visualization)
                    p_f, (u_idx, v_idx) = self.unified_projection(
                        v_data['depth'], v_data['K'], v_data['R_c2w'], v_data['t_c2w'],
                        m_json=v_data['pose_json'], step=4, is_uav=False, use_formula=True)
                    p_m, _ = self.unified_projection(
                        v_data['depth'], v_data['K'], v_data['R_c2w'], v_data['t_c2w'],
                        step=4, is_uav=False, use_formula=False)

                    img_c = cv2.cvtColor(cv2.imread(rgb_dst), cv2.COLOR_BGR2RGB)
                    cols_rgb = img_c[v_idx, u_idx] / 255.0
                    sc = [1.0, 0.0, 0.0]
                    f_pts_all.append(p_f)
                    m_pts_all.append(p_m)
                    f_rgb_all.append(cols_rgb)
                    f_seg_all.append(np.tile(sc, (len(p_f), 1)))

                    entry = {
                        "name": ph_prefix, "type": "ground_pinhole",
                        "station_id": station, "view": view_name,
                        "overlap_ratio": round(ratio, 3)
                    }
                    meta["images"].append(entry)
                    seg_info["overlapping_pinhole_views"].append(entry)

                meta.setdefault("pinhole_segmentation", []).append(seg_info)

        # ---- Save point clouds ----
        def save_pcd(name, pts_arr, cols_arr):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.vstack(pts_arr))
            pcd.colors = o3d.utility.Vector3dVector(np.vstack(cols_arr))
            o3d.io.write_point_cloud(os.path.join(folder, name), pcd)

        save_pcd("formula_rgb.ply", f_pts_all, f_rgb_all)
        save_pcd("formula_seg.ply", f_pts_all, f_seg_all)
        save_pcd("matrix_rgb.ply", m_pts_all, f_rgb_all)
        save_pcd("matrix_seg.ply", m_pts_all, f_seg_all)

        with open(os.path.join(folder, "quad_info.json"), 'w') as f:
            json.dump(meta, f, indent=4)
        print(f"  [OK] Pair {idx} saved (panorama pairing, EDS).")


def generate_triview_samples(scene_dir: str, output_dir: str = None,
                             voxel_size: float = 1.0,
                             min_mutual_voxels: int = 100,
                             min_pinhole_overlap_ratio: float = 0.15) -> dict:
    """Top-level wrapper for tri-view pairing (for pipeline.py).

    Expects scene_dir to contain:
        - metashape.xml (or {scene_id}.xml)
        - pinhole/   (pinhole camera data)
        - pano_rgb/  (panorama RGB)
        - pano_depth/ (panorama depth)
        - uav_rgb/   (UAV RGB)
        - uav_depth/ (UAV depth)

    Args:
        scene_dir: scene directory with subdirectories.
        output_dir: output directory for pairs (default: scene_dir/triview).
        voxel_size: voxel size for overlap matching.
        min_mutual_voxels: minimum mutual overlap voxels.
        min_pinhole_overlap_ratio: minimum pinhole overlap ratio.

    Returns:
        {"output_dir", "num_pairs"}
    """
    import glob

    if output_dir is None:
        output_dir = os.path.join(scene_dir, "triview")
    os.makedirs(output_dir, exist_ok=True)

    xml_files = glob.glob(os.path.join(scene_dir, "*.xml"))
    if not xml_files:
        print(f"No .xml found in {scene_dir}, skipping pairing.")
        return {"output_dir": output_dir, "num_pairs": 0}
    xml_path = xml_files[0]

    pinhole_dir = os.path.join(scene_dir, "pinhole")
    pano_rgb_dir = os.path.join(scene_dir, "pano_rgb")
    pano_depth_dir = os.path.join(scene_dir, "pano_depth")
    uav_rgb = os.path.join(scene_dir, "uav_rgb")
    uav_depth = os.path.join(scene_dir, "uav_depth")

    matcher = MutualOverlapMatcher(xml_path, voxel_size=voxel_size)
    matcher.collect_uav_data(uav_rgb, uav_depth)
    matcher.collect_ground_data(
        pinhole_dir=pinhole_dir,
        pano_rgb_dir=pano_rgb_dir,
        pano_depth_dir=pano_depth_dir,
    )
    matcher.find_quadruplets(output_dir,
                             min_mutual_voxels=min_mutual_voxels,
                             min_pinhole_overlap_ratio=min_pinhole_overlap_ratio)

    num_pairs = len(glob.glob(os.path.join(output_dir, "pair_*")))
    return {"output_dir": output_dir, "num_pairs": num_pairs}


if __name__ == "__main__":
    import argparse
    import sys as _sys

    _sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from config import VOXEL_SIZE_M

    parser = argparse.ArgumentParser(
        description="Tri-view pairing: find satellite / UAV / ground quadruplets")
    parser.add_argument("--scene_dir", type=str, required=True,
                        help="Scene directory (must contain .xml, pinhole/, pano_rgb/, "
                             "pano_depth/, uav_rgb/, uav_depth/)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for pairs (default: scene_dir/triview)")
    parser.add_argument("--voxel_size", type=float, default=VOXEL_SIZE_M,
                        help=f"Voxel size for overlap matching (default: {VOXEL_SIZE_M})")
    parser.add_argument("--min_mutual_voxels", type=int, default=100,
                        help="Minimum mutual overlap voxels (default: 100)")
    parser.add_argument("--min_pinhole_overlap_ratio", type=float, default=0.15,
                        help="Minimum pinhole overlap ratio (default: 0.15)")
    args = parser.parse_args()

    result = generate_triview_samples(
        scene_dir=args.scene_dir,
        output_dir=args.output_dir,
        voxel_size=args.voxel_size,
        min_mutual_voxels=args.min_mutual_voxels,
        min_pinhole_overlap_ratio=args.min_pinhole_overlap_ratio,
    )
    print(f"Done: {result}")
