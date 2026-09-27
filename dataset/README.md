# Dataset Metadata (85 Scenes)

This directory contains the **input parameters** for all **85 scenes** used in the CrossGeo paper — UAV flight trajectories, SfM reconstruction results, and ground-level pano IDs. Feed these into the pipeline scripts (`uav/`, `ground/`, `satellite/`) to collect raw imagery and process it into posed RGB + dense depth. **No Google-owned imagery or depth data is redistributed.**

## Statistics

| Item | Count |
|------|-------|
| Scenes | 85 |
| UAV ESP trajectories | 425 (85 × 5 routes) |
| Reconstruction XML files | 425 |
| Ground pano records | 22,142 |
| Unique ground pano IDs | 22,110 |

## Directory Structure

```
dataset/
├── metadata/
│   ├── scenes.csv                  # Per-scene summary (pano counts, route counts)
│   ├── files.csv                   # SHA-256 checksums for all files
│   └── duplicate_panoids.json      # 32 pano IDs shared across scenes
└── scenes/
    └── {scene_id}/                 # e.g., 0001, 0003, ..., 0496
        ├── uav/
        │   ├── esp/
        │   │   └── {scene_id}_{alt}_{pitch}.esp    # Google Earth Studio trajectory
        │   └── reconstruction_xml/
        │       └── {scene_id}_{alt}_{pitch}.xml    # Metashape SfM reconstruction
        └── ground/
            └── panoids.json                        # Street View pano ID list
```

### UAV Route Naming

Each scene has **5 flight routes** named `{altitude}_{pitch}`:

| Route | Altitude | Pitch | Description |
|-------|----------|-------|-------------|
| `45_30` | 45 m | 30° | Low altitude, oblique view |
| `45_60` | 45 m | 60° | Low altitude, steep oblique |
| `45_90` | 45 m | 90° | Low altitude, top-down (nadir) |
| `70_30` | 70 m | 30° | Mid altitude, oblique view |
| `100_30` | 100 m | 30° | High altitude, oblique view |

## How to Use

### 1. UAV: Render Images from ESP Trajectories

Each `.esp` file is a [Google Earth Studio](https://earth.google.com/studio/) project file. To render UAV imagery:

1. Open Google Earth Studio in your browser.
2. Import the `.esp` file (File → Import).
3. Render the animation as an MP4 video + 3D Camera Tracking JSON (Global frame).
4. Extract frames and recover depth using the UAV pipeline:

```bash
# Extract frames from rendered video
python uav/video2png.py \
    --video data/uav/{scene_id}/render.mp4 \
    --output_dir data/uav/{scene_id}/frames

# Parse camera metadata from ESP + tracking JSON
python uav/preprocess_ge.py --data_root data/uav --scene_id {scene_id}

# Recover dense depth via COLMAP MVS
python uav/colmap_mvs.py \
    --scene_dir data/uav/{scene_id} \
    --colmap_empty data/uav/{scene_id}/sparse
```

See [`uav/README.md`](../uav/README.md) for the full UAV pipeline tutorial.

### 2. UAV: Use Reconstruction XML (Pre-computed SfM)

Each `reconstruction_xml/*.xml` file is an [Agisoft Metashape](https://www.agisoft.com/) reconstruction document containing:

- **Camera intrinsics**: focal length, principal point, distortion coefficients (k1–k3, p1–p2)
- **Extrinsic poses**: per-frame camera positions and orientations
- **Covariance matrices**: uncertainty estimates for all parameters

You can use these to skip the SfM step and directly run MVS depth recovery:

```python
import xml.etree.ElementTree as ET

tree = ET.parse("dataset/scenes/0001/uav/reconstruction_xml/0001_45_60.xml")
root = tree.getroot()

# Parse camera calibration
sensor = root.find(".//sensor")
f = float(sensor.find(".//f").text)          # focal length (pixels)
cx = float(sensor.find(".//cx").text)        # principal point x
cy = float(sensor.find(".//cy").text)        # principal point y
# k1, k2, k3, p1, p2 for lens distortion
```

Or import directly into Metashape: File → Import → Import Cameras (XML).

### 3. Ground: Download Street View Panoramas

Each `ground/panoids.json` contains the list of Google Street View panorama IDs for that scene:

```json
{
  "scene_id": "0001",
  "panos": [
    {
      "pano_id": "WInCaZXs7cvbfedkH_30XA",
      "lat": 51.5075313962962,
      "lon": -0.1291737180601169,
      "elevation": 10.89904403686523,
      "heading": 4.803238992547217,
      "pitch": 0.004547747035473811,
      "roll": 6.254632491868411,
      "source_index": "0001_0"
    }
  ]
}
```

To download the actual RGB and depth data (requires a [Google Street View API key](https://developers.google.com/maps/documentation/streetview)):

```bash
python ground/download.py \
    --pano_list dataset/scenes/0001/ground/panoids.json \
    --output_dir data/ground/0001
```

See [`ground/README.md`](../ground/README.md) for the full ground pipeline tutorial.

### 4. Full Pipeline: Reproduce a Scene

Combine all three modalities to reproduce a complete scene:

```bash
# 1. Ground: download Street View RGB + depth
python ground/download.py \
    --pano_list dataset/scenes/0005/ground/panoids.json \
    --output_dir data/ground/0005

# 2. UAV: render in Google Earth Studio using the .esp file,
#    then recover depth via COLMAP
python uav/colmap_mvs.py --scene_dir data/uav/0005

# 3. Satellite: download co-located satellite tiles
python satellite/download.py --scene_config data/scenes/0005.json

# 4. Tri-view pairing
python utils/pairing/tri_view_pairing.py \
    --ground_dir data/ground/0005 \
    --uav_dir data/uav/0005 \
    --satellite_dir data/satellite/0005 \
    --output_dir data/pairs/0005
```

## File Integrity

Verify downloaded files against the released checksums:

```bash
# Check a specific file
sha256sum dataset/scenes/0001/uav/esp/0001_45_60.esp
# Expected: c9fad17d4c07c8e6989cf29afd8c48c235fe94cc64bd8e60175f33481d6d70ce

# Or use the metadata/files.csv for bulk verification
python -c "
import csv, hashlib, os
with open('dataset/metadata/files.csv') as f:
    for row in csv.DictReader(f):
        path = os.path.join('dataset', row['relative_path'].split('crossgeo-paper85/')[-1])
        if os.path.exists(path):
            sha = hashlib.sha256(open(path,'rb').read()).hexdigest()
            status = 'OK' if sha == row['sha256'] else 'MISMATCH'
            print(f'{status}: {path}')
"
```

## Notes

- **Google Earth Studio** requires a Google account and may have usage quotas.
- **Street View images** are NOT redistributed here — only pano IDs. You must download them yourself via the Google Street View API.
- **32 pano IDs** appear in more than one scene (see `metadata/duplicate_panoids.json`). This is expected for scenes that are geographically close.
- The `metadata/files.csv` references paths with a `data/crossgeo-paper85/` prefix — this is the original release path. In this repo, files are under `dataset/`.
