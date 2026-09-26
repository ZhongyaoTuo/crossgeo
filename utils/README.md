# Shared Utilities & Tri-View Pairing

This folder contains shared utilities for coordinate conversion, COLMAP I/O, depth manipulation, visualization, and tri-view pairing.

## Tutorial

### Coordinate Conversion (`geo.py`)

Convert between GPS, ENU (East-North-Up), and EDS (East-Down-South) coordinate systems:

```python
from utils.geo import wgs84_to_eds, enu_to_eds, compute_tile_bounds

# WGS84 (lat, lon, alt) → EDS (x, y, z)
x, y, z = wgs84_to_eds(lat=40.7128, lon=-74.0060, alt=100.0)

# Compute satellite tile bounding box
bounds = compute_tile_bounds(lat=40.7128, lon=-74.0060, zoom=19, tile_size=1024)
```

**EDS frame**: X → South, Y → Down, Z → East.

### Depth Format Conversion (`depth2npy.py`)

Convert TIFF depth maps to NPY format for training:

```bash
python utils/depth2npy.py --dataset_dir data/scenes/0005_pair
```

### Depth Visualization (`tiff2png.py`)

Visualize float32 TIFF depth maps as colorized PNG:

```bash
python utils/tiff2png.py --target_pair_dir data/scenes/0005_pair/0005_60_30/pair_3
```

### Point Cloud Visualization (`ply_show.py`)

Visualize COLMAP point clouds or reconstructed 3D points:

```bash
python utils/ply_show.py --ply data/uav/0005/points3D.ply
```

### UAV Coverage Polygon (`scope.py`)

Compute the ground coverage polygon of a UAV flight trajectory:

```bash
python utils/scope.py --uav_dir data/uav/0005 --output data/uav/0005/scope.geojson
```

### Tri-View Pairing (`pairing/tri_view_pairing.py`)

Form tri-view pairs by voxel overlap scoring in the EDS world frame. Given satellite, UAV, and ground data with poses and depth, find the highest-scoring 6-image tuples (2 views per modality):

```bash
python utils/pairing/tri_view_pairing.py \
    --xml <metashape.xml> \
    --uav_dir <uav_depth> \
    --ground_dir <ground_data> \
    --output_dir data/pairs
```

### Dataset Split (`pairing/scene_split.py`)

Split scenes into train/val/test (75/5/5) ensuring no city appears in more than one split:

```bash
python utils/pairing/scene_split.py --scenes_dir data/scenes --output data/splits
```

## Files

| File | Description |
|------|-------------|
| `geo.py` | Geo coordinate conversion (GPS/ENU/EDS) |
| `colmap_io.py` | COLMAP model I/O |
| `depth2npy.py` | TIFF → NPY depth conversion |
| `ply_show.py` | Point cloud visualization |
| `scope.py` | UAV coverage polygon |
| `tiff2png.py` | Depth visualization (TIFF → PNG) |
| `pairing/tri_view_pairing.py` | Tri-view pairing (EDS, voxel overlap) |
| `pairing/scene_split.py` | Train/val/test split (75/5/5 scenes) |