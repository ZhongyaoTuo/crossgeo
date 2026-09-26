#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Earth Studio .esp dense keyframe generator.
==================================================
Reads a GE Studio .esp camera file, replaces sparse keyframes with 600 dense
randomly-generated keyframes (within the original global range, endpoints
preserved), and writes the modified .esp.

Usage:
    python uav/modify_esp.py --input path/to/scene_old.esp --output path/to/scene.esp
"""
import json
import os
import argparse
import random

# Fixed random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def get_global_range(keyframes):
    """Get the min and max value across all keyframes."""
    values = [k['value'] for k in keyframes]
    return min(values), max(values)


def generate_global_random_keyframes(orig_keyframes, time_points,
                                     keep_endpoints=True):
    """Randomly generate keyframe values within the original global range.

    Args:
        orig_keyframes: original keyframe list (used only to determine range).
        time_points: sorted list of time points for new keyframes.
        keep_endpoints: if True, keep original first/last values at t=0 and t=1.
    """
    min_val, max_val = get_global_range(orig_keyframes)

    if keep_endpoints:
        first_val = orig_keyframes[0]['value']
        last_val = orig_keyframes[-1]['value']

    new_keyframes = []
    for t in time_points:
        if keep_endpoints and t == 0.0:
            value = first_val
        elif keep_endpoints and t == 1.0:
            value = last_val
        else:
            # Uniform random within the global range
            value = random.uniform(min_val, max_val)

        new_keyframes.append({
            "time": t,
            "value": value
        })

    return new_keyframes


def convert_camera_data(input_path, output_path):
    """Convert camera data to dense keyframe format.

    Main changes:
    1. Keep settings.name unchanged
    2. Replace cameraPositionGroup position keyframes (longitude, latitude,
       altitude) with 600 dense randomly-generated keyframes
    3. Add 600 keyframes for cameraRotationGroup (rotationX/Y/Z, with noise)
    4. Add 600 keyframes for cameraLensGroup fov
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    scene = data['scenes'][0]
    attributes = scene['attributes']

    # Find cameraGroup
    camera_group = None
    for attr in attributes:
        if attr.get('type') == 'cameraGroup':
            camera_group = attr
            break

    if camera_group is None:
        raise ValueError("cameraGroup not found")

    # Find cameraPositionGroup
    camera_pos_group = None
    for attr in camera_group.get('attributes', []):
        if attr.get('type') == 'cameraPositionGroup':
            camera_pos_group = attr
            break

    if camera_pos_group is None:
        raise ValueError("cameraPositionGroup not found")

    # Find position
    position = None
    for attr in camera_pos_group.get('attributes', []):
        if attr.get('type') == 'position':
            position = attr
            break

    if position is None:
        raise ValueError("position not found")

    # Get original keyframes for longitude, latitude, altitude
    orig_longitude = None
    orig_latitude = None
    orig_altitude = None

    for attr in position.get('attributes', []):
        if attr.get('type') == 'longitude':
            orig_longitude = attr
        elif attr.get('type') == 'latitude':
            orig_latitude = attr
        elif attr.get('type') == 'altitude':
            orig_altitude = attr

    lon_keyframes_orig = orig_longitude['keyframes']
    lat_keyframes_orig = orig_latitude['keyframes']
    alt_keyframes_orig = orig_altitude['keyframes']

    # Generate 600 keyframes (0 to 1, step 1/600)
    total_frames = 600
    new_keyframes_time = [i / total_frames for i in range(total_frames)]

    # Random keyframes within global range (preserve endpoints)
    longitude_keyframes = generate_global_random_keyframes(
        lon_keyframes_orig, new_keyframes_time, keep_endpoints=True)
    latitude_keyframes = generate_global_random_keyframes(
        lat_keyframes_orig, new_keyframes_time, keep_endpoints=True)
    altitude_keyframes = generate_global_random_keyframes(
        alt_keyframes_orig, new_keyframes_time, keep_endpoints=True)

    orig_longitude['keyframes'] = longitude_keyframes
    orig_latitude['keyframes'] = latitude_keyframes
    orig_altitude['keyframes'] = altitude_keyframes

    # 2. cameraRotationGroup - add 600 keyframes (with random noise)
    camera_rotation_group = None
    for attr in camera_group.get('attributes', []):
        if attr.get('type') == 'cameraRotationGroup':
            camera_rotation_group = attr
            break

    if camera_rotation_group:
        # rotationX keyframes
        rotation_x_frames = []
        base_rotation_x = 0.0

        for i in range(total_frames):
            time_val = i / total_frames
            if i == total_frames - 1:
                value = 0.3324475724154684
            else:
                random.seed(i)  # ensure reproducibility
                noise = random.uniform(-0.05, 0.05)
                base = base_rotation_x + (0.3324475724154684 - base_rotation_x) * time_val
                value = base + noise

            rotation_x_frames.append({"time": time_val, "value": value})

        rotation_x_frames[-1]['value'] = 0.3324475724154684

        # rotationY keyframes
        rotation_y_frames = []
        base_rotation_y = 0.0

        for i in range(total_frames):
            time_val = i / total_frames
            if i == total_frames - 1:
                value = 0.10077606300489514
            else:
                random.seed(i + 1000)
                noise = random.uniform(-0.02, 0.02)
                base = base_rotation_y + (0.10077606300489514 - base_rotation_y) * time_val
                value = base + noise

            rotation_y_frames.append({"time": time_val, "value": value})

        rotation_y_frames[-1]['value'] = 0.10077606300489514

        # rotationZ keyframes
        rotation_z_frames = []

        for i in range(total_frames):
            time_val = i / total_frames
            if i == total_frames - 1:
                value = 0.9988401968626126
            else:
                random.seed(i + 2000)
                if random.random() > 0.5:
                    value = random.uniform(0.0, 0.002)
                else:
                    value = random.uniform(0.998, 1.0)

            rotation_z_frames.append({"time": time_val, "value": value})

        rotation_z_frames[-1]['value'] = 0.9988401968626126

        for attr in camera_rotation_group.get('attributes', []):
            if attr.get('type') == 'rotationX':
                attr['value']['relative'] = 0.3324475724154684
                attr['keyframes'] = rotation_x_frames
            elif attr.get('type') == 'rotationY':
                attr['value']['relative'] = 0.10077606300489514
                attr['keyframes'] = rotation_y_frames
            elif attr.get('type') == 'rotationZ':
                attr['value']['relative'] = 0.9988401968626126
                attr['keyframes'] = rotation_z_frames

    # 3. cameraLensGroup - add 600 fov keyframes
    camera_lens_group = None
    for attr in camera_group.get('attributes', []):
        if attr.get('type') == 'cameraLensGroup':
            camera_lens_group = attr
            break

    if camera_lens_group:
        fov_frames = []
        fov_value = 0.33146067415730335

        for i in range(total_frames):
            time_val = i / total_frames
            fov_frames.append({"time": time_val, "value": fov_value})

        for attr in camera_lens_group.get('attributes', []):
            if attr.get('type') == 'fov':
                attr['value']['relative'] = fov_value
                attr['keyframes'] = fov_frames

    # 4. environmentGroup worldTime and clouddate ranges
    for attr in attributes:
        if attr.get('type') == 'environmentGroup':
            for sub_attr in attr.get('attributes', []):
                if sub_attr.get('type') == 'sunGroup':
                    for sun_attr in sub_attr.get('attributes', []):
                        if sun_attr.get('type') == 'worldTime':
                            sun_attr['value']['minValueRange'] = 1773641799987
                            sun_attr['value']['maxValueRange'] = 1773814599987
                elif sub_attr.get('type') == 'cloudGroup':
                    for cloud_attr in sub_attr.get('attributes', []):
                        if cloud_attr.get('type') == 'clouddate':
                            cloud_attr['value']['minValueRange'] = 1773640800000
                            cloud_attr['value']['maxValueRange'] = 1773720000000

    # Save output
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("Conversion complete!")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Keyframe count: {total_frames}")
    print(f"Longitude range: "
          f"{min(lon_keyframes_orig, key=lambda k: k['value'])['value']:.10f} -> "
          f"{max(lon_keyframes_orig, key=lambda k: k['value'])['value']:.10f}")
    print(f"Latitude range: "
          f"{min(lat_keyframes_orig, key=lambda k: k['value'])['value']:.10f} -> "
          f"{max(lat_keyframes_orig, key=lambda k: k['value'])['value']:.10f}")
    print(f"Altitude range: "
          f"{min(alt_keyframes_orig, key=lambda k: k['value'])['value']:.10e} -> "
          f"{max(alt_keyframes_orig, key=lambda k: k['value'])['value']:.10e}")
    print("Strategy: uniform random within global range (endpoints preserved)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate dense keyframes in a GE Studio .esp file"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="input .esp file path")
    parser.add_argument("--output", type=str, required=True,
                        help="output .esp file path")
    args = parser.parse_args()

    convert_camera_data(args.input, args.output)
