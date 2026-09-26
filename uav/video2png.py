#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract frames from a video and save as images.
===============================================
Used to extract UAV frames from Google Earth Studio rendered .mp4 videos.

Usage:
    python uav/video2png.py --video path/to/video.mp4 \
        --output_dir output/ --video_id 0001_70_30
"""
import os
import argparse
import cv2


def extract_frames(video_path, output_base_folder, video_id,
                   image_format='jpeg', quality=95):
    """Extract all frames from a video and save as images.

    Args:
        video_path: path to the video file.
        output_base_folder: base output directory.
        video_id: video identifier (e.g. 0001_70_30).
        image_format: image format.
        quality: JPEG quality.

    Returns:
        Number of extracted frames.
    """
    output_folder = os.path.join(output_base_folder, video_id)
    os.makedirs(output_folder, exist_ok=True)

    print(f'Processing video: {video_id}')
    print(f'Video path: {video_path}')
    print(f'Output dir: {output_folder}')

    vidcap = cv2.VideoCapture(video_path)

    if not vidcap.isOpened():
        print(f'[ERROR] Cannot open video: {video_path}')
        return 0

    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f'  Resolution: {width}x{height}, FPS: {fps:.2f}, '
          f'total frames: {total_frames}')

    ext = image_format.lower()
    if ext in ['jpg', 'jpeg']:
        save_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    else:
        save_params = []

    count = 0
    success, image = vidcap.read()

    while success:
        # Filename: 0001_70_30_000.jpeg (3 digits, starting from 000)
        frame_filename = f'{video_id}_{count:03d}.{ext}'
        frame_path = os.path.join(output_folder, frame_filename)
        cv2.imwrite(frame_path, image, save_params)

        success, image = vidcap.read()
        count += 1

        if count % 100 == 0:
            print(f'  Extracted: {count}/{total_frames} frames')

    vidcap.release()

    print(f'[OK] Done: extracted {count} frames')
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract frames from a video as images'
    )
    parser.add_argument('--video', type=str, required=True,
                        help='path to the .mp4 video')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='output base directory')
    parser.add_argument('--video_id', type=str, required=True,
                        help='video identifier (e.g. 0001_70_30)')
    parser.add_argument('--format', type=str, default='jpeg',
                        help='image format (default: jpeg)')
    parser.add_argument('--quality', type=int, default=95,
                        help='JPEG quality (default: 95)')
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f'[ERROR] Video file does not exist: {args.video}')
        exit(1)

    extract_frames(args.video, args.output_dir, args.video_id,
                   args.format, args.quality)
