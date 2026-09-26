#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge ESP and JSON metadata into a CSV (one row per frame).
==========================================================
Combines Google Earth Studio .esp (rotation) and .json (GPS/altitude/FOV)
metadata into a single CSV file.

Usage:
    # Single video pair
    python uav/json2csv.py --json path/to/video.json --esp path/to/video.esp

    # Batch process a whole scene
    python uav/json2csv.py --data_root <root> --scene 0003
"""
import os
import json
import csv
import argparse
from pathlib import Path


def read_esp_rotations(esp_file_path):
    """Read an .esp file and extract rotationX/Y/Z per frame."""
    try:
        with open(esp_file_path, 'r', encoding='ascii') as f:
            data = json.load(f)

        rotations = []
        scene = data['scenes'][0]
        attributes = scene['attributes']

        for attr in attributes:
            if attr.get('type') == 'cameraRotationGroup':
                rotation_attributes = attr['attributes']
                rotationX_frames = []
                rotationY_frames = []
                rotationZ_frames = []

                for rot_attr in rotation_attributes:
                    rot_type = rot_attr.get('type')

                    if rot_type == 'rotationX':
                        for keyframe in rot_attr['keyframes']:
                            rotationX_frames.append(keyframe['value'])
                    elif rot_type == 'rotationY':
                        for keyframe in rot_attr['keyframes']:
                            rotationY_frames.append(keyframe['value'])
                    elif rot_type == 'rotationZ':
                        for keyframe in rot_attr['keyframes']:
                            rotationZ_frames.append(keyframe['value'])

                num_frames = len(rotationX_frames)
                if len(rotationY_frames) == num_frames and len(rotationZ_frames) == num_frames:
                    for i in range(num_frames):
                        rotations.append({
                            "frame": i,
                            "rotationX": rotationX_frames[i],
                            "rotationY": rotationY_frames[i],
                            "rotationZ": rotationZ_frames[i]
                        })
                break

        print(f"  Read {len(rotations)} rotation frames from .esp")
        return rotations

    except Exception as e:
        print(f"  Error reading .esp file: {e}")
        return []


def read_json_camera_data(json_file_path):
    """Read a .json file and extract lat/lon/alt/FOV per frame."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        camera_frames = data.get('cameraFrames', [])
        results = []

        for i, frame in enumerate(camera_frames):
            coordinate = frame.get('coordinate', {})
            fov_vertical = frame.get('fovVertical', 0)

            results.append({
                "frame": i,
                "latitude": coordinate.get('latitude', 0),
                "longitude": coordinate.get('longitude', 0),
                "altitude": coordinate.get('altitude', 0),
                "fov": fov_vertical
            })

        print(f"  Read {len(results)} camera frames from .json")
        return results

    except Exception as e:
        print(f"  Error reading .json file: {e}")
        return []


def convert_rotation_to_angles(rotation_params):
    """Convert .esp rotation parameters to actual angles."""
    rotationX = rotation_params.get('rotationX', 0)
    rotationY = rotation_params.get('rotationY', 0)
    rotationZ = rotation_params.get('rotationZ', 0)

    yaw = rotationX * 360
    pitch = rotationY * 180
    roll = rotationZ * 360

    yaw = yaw % 360
    pitch = pitch % 360
    roll = roll % 360

    if pitch > 180:
        pitch = pitch - 360

    return {"yaw": yaw, "pitch": pitch, "roll": roll}


def process_video_to_csv_rows(video_basename, esp_data, json_data,
                              image_ext='.jpeg', num_digits=3):
    """Merge ESP and JSON data into CSV rows.

    Args:
        video_basename: video base name (e.g. '0003_45_30').
        esp_data: ESP rotation data.
        json_data: JSON camera data (GPS, altitude, FOV).
        image_ext: image file extension.
        num_digits: frame index digit count (default 3: 000-999).

    Returns:
        List of CSV row dicts.
    """
    rows = []
    num_frames = min(len(esp_data), len(json_data))

    if num_frames == 0:
        print(f"  Warning: {video_basename} has no valid data")
        return rows

    if num_digits is None:
        num_digits = len(str(num_frames - 1))

    format_str = f"{{:0{num_digits}d}}"

    for i in range(num_frames):
        esp_frame = esp_data[i]
        json_frame = json_data[i]
        angles = convert_rotation_to_angles(esp_frame)

        frame_str = format_str.format(i)
        image_id = f"{video_basename}_{frame_str}{image_ext}"

        row = {
            'image_id': image_id,
            'longitude': json_frame['longitude'],
            'latitude': json_frame['latitude'],
            'altitude': json_frame['altitude'],
            'yaw': angles['yaw'],
            'pitch': angles['pitch'],
            'roll': angles['roll']
        }
        rows.append(row)

    return rows


def process_single_video_pair(json_file_path, esp_file_path,
                              video_basename=None, num_digits=3):
    """Process a single JSON+ESP pair and return CSV rows."""
    if not os.path.exists(json_file_path):
        print(f"[ERROR] JSON file does not exist: {json_file_path}")
        return []

    if not os.path.exists(esp_file_path):
        print(f"[ERROR] ESP file does not exist: {esp_file_path}")
        return []

    if video_basename is None:
        video_basename = os.path.splitext(os.path.basename(json_file_path))[0]

    print(f"\nProcessing: {video_basename}")
    print(f"  JSON: {json_file_path}")
    print(f"  ESP:  {esp_file_path}")

    esp_data = read_esp_rotations(esp_file_path)
    json_data = read_json_camera_data(json_file_path)

    rows = process_video_to_csv_rows(video_basename, esp_data, json_data,
                                     num_digits=num_digits)
    print(f"  Generated {len(rows)} rows")
    return rows


def process_scene_to_csv(data_root, scene_name, output_csv_path=None,
                         image_ext='.jpeg', num_digits=3):
    """Batch process all videos in a scene, output a single CSV file."""
    json_folder = os.path.join(data_root, 'geojsons', scene_name, f'{scene_name}_modify_v2')
    esp_folder = os.path.join(data_root, 'geojsons', scene_name, f'{scene_name}_modify')

    if output_csv_path is None:
        output_csv_path = os.path.join(data_root, 'raw_data_v2', scene_name, f'{scene_name}.csv')

    if not os.path.exists(json_folder):
        print(f'[ERROR] JSON folder not found: {json_folder}')
        return

    if not os.path.exists(esp_folder):
        print(f'[ERROR] ESP folder not found: {esp_folder}')
        return

    print(f'{"="*60}')
    print(f'Processing scene: {scene_name}')
    print(f'JSON dir: {json_folder}')
    print(f'ESP dir: {esp_folder}')
    print(f'Output CSV: {output_csv_path}')
    print(f'Frame index digits: {num_digits} (format: {"0"*num_digits}-{str(9)*num_digits})')
    print(f'{"="*60}')

    all_rows = []
    total_videos = 0
    skipped_videos = 0

    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]

    if not json_files:
        print(f'[ERROR] No JSON files found in {json_folder}')
        return

    print(f'Found {len(json_files)} JSON files\n')

    for json_file in json_files:
        video_basename = os.path.splitext(json_file)[0]
        json_path = os.path.join(json_folder, json_file)
        esp_path = os.path.join(esp_folder, f'{video_basename}.esp')

        total_videos += 1

        rows = process_single_video_pair(json_path, esp_path, video_basename, num_digits)

        if rows:
            all_rows.extend(rows)
        else:
            skipped_videos += 1

    print(f'\n{"="*60}')
    print(f'Processing complete!')
    print(f'Total videos: {total_videos}')
    print(f'Successful: {total_videos - skipped_videos}')
    print(f'Skipped/failed: {skipped_videos}')
    print(f'Total rows: {len(all_rows)}')
    print(f'{"="*60}')

    if all_rows:
        output_dir = os.path.dirname(output_csv_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        all_rows.sort(key=lambda x: x['image_id'])

        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['image_id', 'longitude', 'latitude', 'altitude', 'yaw', 'pitch', 'roll']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

        print(f'\nCSV saved: {output_csv_path}')

        print(f'\nFirst 5 rows preview:')
        for i, row in enumerate(all_rows[:5]):
            print(f"  {row['image_id']}: lat={row['latitude']}, "
                  f"lon={row['longitude']}, yaw={row['yaw']:.2f}")
    else:
        print('\n[Warning] No data generated')


def process_single_pair_to_csv(json_file_path, esp_file_path,
                               output_csv_path=None, image_ext='.jpeg',
                               num_digits=3):
    """Process a single JSON+ESP pair and output a CSV."""
    if output_csv_path is None:
        base_name = os.path.splitext(os.path.basename(json_file_path))[0]
        dir_name = os.path.dirname(json_file_path)
        output_csv_path = os.path.join(dir_name, f'{base_name}.csv')
    else:
        if os.path.isdir(output_csv_path):
            base_name = os.path.splitext(os.path.basename(json_file_path))[0]
            output_csv_path = os.path.join(output_csv_path, f'{base_name}.csv')
        elif not os.path.splitext(output_csv_path)[1]:
            output_csv_path = output_csv_path + '.csv'

    print(f'{"="*60}')
    print(f'Processing single file pair')
    print(f'Frame index format: {num_digits} digits')
    print(f'Output CSV: {output_csv_path}')
    print(f'{"="*60}')

    rows = process_single_video_pair(json_file_path, esp_file_path, num_digits=num_digits)

    if not rows:
        print('[ERROR] No data generated')
        return

    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image_id', 'longitude', 'latitude', 'altitude', 'yaw', 'pitch', 'roll']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'\nCSV saved: {output_csv_path}')
    print(f'Total rows: {len(rows)}')

    print(f'\nData preview:')
    print(f'First 3 rows:')
    for row in rows[:3]:
        print(f"  {row['image_id']}")
    if len(rows) > 6:
        print(f'...')
    print(f'Last 3 rows:')
    for row in rows[-3:]:
        print(f"  {row['image_id']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Merge ESP and JSON metadata into a CSV (one row per frame)'
    )
    parser.add_argument('--json', type=str, help='single JSON file path')
    parser.add_argument('--esp', type=str, help='single ESP file path')
    parser.add_argument('--output', type=str, help='output CSV file path (optional)')
    parser.add_argument('--data_root', type=str, default=None,
                        help='data root directory (for batch scene mode)')
    parser.add_argument('--scene', type=str,
                        help='scene name (e.g. 0003) for batch processing')
    parser.add_argument('--ext', type=str, default='.jpeg',
                        help='image file extension (default: .jpeg)')
    parser.add_argument('--digits', type=int, default=3,
                        help='frame index digit count (default: 3)')

    args = parser.parse_args()

    # Mode 1: single file pair
    if args.json and args.esp:
        process_single_pair_to_csv(args.json, args.esp, args.output,
                                   args.ext, args.digits)
    # Mode 2: batch scene
    elif args.scene:
        if not args.data_root:
            parser.error('--scene requires --data_root')
        process_scene_to_csv(args.data_root, args.scene, args.output,
                             args.ext, args.digits)
    else:
        print('[ERROR] Specify one of the following parameter combinations:')
        print('  1. --json and --esp  (process a single video)')
        print('  2. --scene and --data_root  (batch process a scene)')
        parser.print_help()
