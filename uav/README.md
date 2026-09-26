# UAV Data Pipeline

This folder contains scripts for rendering UAV (drone) imagery in Google Earth Studio and recovering dense depth maps via COLMAP multi-view stereo.

## Tutorial

### Step 1: Render in Google Earth Studio

1. Open [Google Earth Studio](https://earth.google.com/studio/) and create a new project.
2. Set the camera altitude to 30–120m and pitch to 0°–90°.
3. Export as `.mp4` video + 3D Camera Tracking JSON (Global frame).
4. Export the flight trajectory as `.esp` file.

### Step 2: Extract Frames

Extract individual frames from the rendered video:

```bash
# Single video
python uav/video2png.py --video data/uav/0005/render.mp4 --output_dir data/uav/0005/frames

# Batch processing
python uav/video2png_batch.py --input_dir data/uav/ --output_dir data/uav/frames
```

### Step 3: Parse Camera Metadata

Convert the Google Earth Studio JSON + ESP trajectory into COLMAP-compatible format:

```bash
# Generate dense keyframes from ESP
python uav/modify_esp.py --esp data/uav/0005/trajectory.esp --output data/uav/0005/modified.esp

# Convert ESP + JSON to CSV metadata
python uav/json2csv.py --esp data/uav/0005/modified.esp --json data/uav/0005/camera.json --output data/uav/0005/metadata.csv

# Parse GE camera tracking JSON → COLMAP cameras
python uav/preprocess_ge.py --data_root data/uav --scene_id 0005
```

> The GE JSON → COLMAP conversion (ECEF→ENU→EDS) follows [AerialMegaDepth](https://github.com/kvuong2711/aerial-megadepth). The full source is included as a git submodule at [`aerial-megadepth/`](aerial-megadepth/).

### Step 4: Recover Dense Depth via COLMAP MVS

Run COLMAP multi-view stereo to recover dense depth maps:

```bash
python uav/colmap_mvs.py --scene_dir data/uav/0005 --colmap_empty data/uav/0005/sparse
```

**Output**: Per-frame depth maps (`*.tiff`) and COLMAP point cloud (`*.ply`).

## Files

| File | Description |
|------|-------------|
| `ges_utils.py` | GE JSON → COLMAP camera conversion (ECEF→ENU→EDS) |
| `preprocess_ge.py` | Extract frames + metadata from GE render output |
| `colmap_mvs.py` | COLMAP MVS dense depth recovery |
| `video2png.py` | Video → PNG frame extraction |
| `video2png_batch.py` | Batch video → PNG extraction |
| `json2csv.py` | ESP + JSON → CSV metadata |
| `modify_esp.py` | ESP dense keyframe generation |
