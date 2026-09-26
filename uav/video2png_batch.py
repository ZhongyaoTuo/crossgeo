#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch extract frames from all videos in a directory.
====================================================
Scans a directory for all .mp4 files and extracts frames from each.
Used for batch UAV frame extraction from Google Earth Studio renders.

Usage:
    python uav/video2png_batch.py --data_dir path/to/videos/
"""
import os
import argparse
import cv2
import glob
from tqdm import tqdm


def extract_frames(video_path, output_folder, video_id,
                   image_format='jpeg', quality=95):
    """Extract all frames from a video and save as images."""
    os.makedirs(output_folder, exist_ok=True)

    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        print(f'  [ERROR] Cannot open video')
        return 0

    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'  Resolution: {width}x{height}, FPS: {fps:.2f}, '
          f'total frames: {total_frames}')

    ext = image_format.lower()
    save_params = ([cv2.IMWRITE_JPEG_QUALITY, quality]
                   if ext in ['jpg', 'jpeg'] else [])

    count = 0
    success, image = vidcap.read()
    while success:
        frame_filename = f'{video_id}_{count:03d}.{ext}'
        frame_path = os.path.join(output_folder, frame_filename)
        cv2.imwrite(frame_path, image, save_params)
        success, image = vidcap.read()
        count += 1

    vidcap.release()
    print(f'  [OK] Done: extracted {count} frames')
    return count


def main(data_dir):
    """Find all .mp4 files in data_dir and extract frames from each."""
    video_files = sorted(glob.glob(os.path.join(data_dir, "*.mp4")))
    if not video_files:
        print(f"[!] No .mp4 files found in {data_dir}")
        return

    print(f"[*] Found {len(video_files)} video files, starting extraction...\n")

    total_frames = 0
    for vp in video_files:
        basename = os.path.splitext(os.path.basename(vp))[0]
        print(f"  Processing: {basename}")

        # Output to data_dir/video_id/
        output_folder = os.path.join(data_dir, basename)

        n = extract_frames(vp, output_folder, basename)
        total_frames += n
        print()

    print(f"[OK] All done! Processed {len(video_files)} videos, "
          f"extracted {total_frames} frames")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Batch extract frames from all videos in a directory'
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help='directory containing .mp4 files')
    args = parser.parse_args()
    main(args.data_dir)
