"""
Satellite pose recovery for CrossGeo.
=====================================
For each ground image in a pair, generate the co-located satellite tile pose.
Traverses all area_multi / sub / pair_* folders and, for each ground_XX_rgb.npy,
writes the corresponding ground_XX_satellite_rgb.npy.

CrossGeo (paper Sec 2.1 -- Pose recovery): the satellite is modeled as a virtual
pinhole camera looking straight down (FOV 5 deg, altitude 5726 m).
"""
import os
import argparse
import numpy as np
from tqdm import tqdm

# ========== Satellite parameters (consistent with config.py) ==========
SAT_W, SAT_H = 1024, 1024
SAT_FOV = 5.0
SAT_ALTITUDE = 5726.0

# Satellite intrinsics
f_sat = (SAT_W / 2.0) / np.tan(np.radians(SAT_FOV / 2.0))
K_sat = np.array([
    [f_sat, 0, SAT_W / 2.0],
    [0, f_sat, SAT_H / 2.0],
    [0, 0, 1]
], dtype=np.float32)

# Satellite rotation R_c2w (Cam -> EDS)
# Camera Z points down (world +Y), camera X points right (world +Z),
# camera Y points down (world +X)
R_sat_c2w = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
], dtype=np.float32)


def add_satellite_pose_for_ground(pair_dir, ground_prefix):
    """Generate the satellite pose for a single ground image.

    Args:
        pair_dir: pair directory.
        ground_prefix: e.g. "ground_00".

    Returns:
        True if the satellite pose was written.
    """
    ref_npy = os.path.join(pair_dir, f"{ground_prefix}_rgb.npy")
    if not os.path.exists(ref_npy):
        return False

    ref_data = np.load(ref_npy, allow_pickle=True).item()
    ref_t_eds = ref_data['c2w'][:3, 3]
    ref_raw = ref_data['raw_data']  # [pitch, roll, heading, lat, lon, alt]
    ref_lat, ref_lon, ref_alt = ref_raw[3], ref_raw[4], ref_raw[5]

    # Satellite sits directly above the ground station
    alt_diff = SAT_ALTITUDE - ref_alt
    t_sat_eds = ref_t_eds.copy()
    t_sat_eds[1] -= alt_diff  # EDS Y is down; subtracting altitude goes up

    c2w_sat = np.eye(4, dtype=np.float32)
    c2w_sat[:3, :3] = R_sat_c2w
    c2w_sat[:3, 3] = t_sat_eds

    sat_raw_data = np.array([90.0, 0.0, 0.0, ref_lat, ref_lon, SAT_ALTITUDE])

    save_name = f"{ground_prefix}_satellite_rgb.npy"
    save_path = os.path.join(pair_dir, save_name)

    payload = {
        'intrinsics': K_sat,
        'c2w': c2w_sat,
        'raw_data': sat_raw_data,
        'type': 'satellite',
        'base_station': ground_prefix
    }

    np.save(save_path, payload)
    return True


def process_pair(pair_dir):
    """Process one pair folder: generate satellite poses for all ground_XX."""
    count = 0
    for fname in os.listdir(pair_dir):
        if fname.endswith("_rgb.npy") and fname.startswith("ground_"):
            prefix = fname.replace("_rgb.npy", "")
            if prefix.endswith("_satellite"):
                continue  # skip already-generated satellite files
            if add_satellite_pose_for_ground(pair_dir, prefix):
                count += 1
    return count


def main(base_dir):
    """Walk base_dir/area_multi/sub/pair_* and add satellite poses."""
    if not os.path.isdir(base_dir):
        print(f"[!] Directory does not exist: {base_dir}")
        return

    total_pairs = 0
    total_sats = 0

    area_dirs = sorted(os.listdir(base_dir))
    for area in area_dirs:
        area_path = os.path.join(base_dir, area)
        if not os.path.isdir(area_path):
            continue
        sub_dirs = sorted(os.listdir(area_path))
        for sub in sub_dirs:
            sub_path = os.path.join(area_path, sub)
            if not os.path.isdir(sub_path):
                continue
            pair_dirs = sorted([
                d for d in os.listdir(sub_path)
                if d.startswith("pair_") and os.path.isdir(os.path.join(sub_path, d))
            ])
            for pd in pair_dirs:
                pair_dir = os.path.join(sub_path, pd)
                n = process_pair(pair_dir)
                if n > 0:
                    total_pairs += 1
                    total_sats += n

    print(f"\n[OK] Done! Processed {total_pairs} pairs, "
          f"generated {total_sats} satellite pose files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add satellite poses for all ground images under a dataset root"
    )
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Root directory containing area_multi/sub/pair_* folders")
    args = parser.parse_args()
    main(args.base_dir)
