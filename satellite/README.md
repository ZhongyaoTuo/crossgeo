# Satellite Data Pipeline

This folder contains scripts for downloading satellite imagery from Google Maps and recovering camera poses and depth for CrossGeo.

## Tutorial

### Step 1: Download Satellite Tiles

Download a 500m × 500m Google Maps satellite tile centered at a given GPS coordinate, at zoom 19 (~0.5 m/px):

```bash
python satellite/download.py --lat 40.7128 --lon -74.0060 --scene_id 0005
```

**Output**: `satellite_rgb.jpg` (1024×1024 RGB tile) and metadata JSON.

> The download logic follows [andolg/satellite-imagery-downloader](https://github.com/andolg/satellite-imagery-downloader). The full source is included as a git submodule at [`satellite-imagery-downloader/`](satellite-imagery-downloader/).

### Step 2: Recover Satellite Pose

Each satellite tile is modeled as a **virtual pinhole camera** looking straight down with FOV 5° at altitude 5726m. Recover the pose (intrinsics + c2w in EDS frame):

```bash
python satellite/pose.py --base_dir data/scenes/0005_pair
```

**Output**: `ground_XX_satellite_rgb.npy` — satellite pose dict with `intrinsics` (3×3), `c2w` (4×4 in EDS frame).

### Step 3: Compute Satellite Depth

Project the UAV point cloud into the satellite virtual camera to obtain depth via Z-Buffer:

```bash
python satellite/depth.py --multi_base data/uav/0005 --geo_base data/scenes/0005_pair
```

**Output**: `ground_XX_satellite_depth.tiff` — float32 depth map.

## World Coordinate System (EDS)

All poses use the EDS frame: **X** → South, **Y** → Down, **Z** → East. See `utils/geo.py` for coordinate conversion utilities.
