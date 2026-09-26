# CrossGeo Dataset

<p align="center">
  <a href="https://arxiv.org/abs/2605.07978">Paper</a> •
  <a href="#data-structure">Data Structure</a> •
  <a href="#open-source-status">Open-Source Status</a> •
  <a href="#citation">Citation</a> •
  <a href="#license">License</a>
</p>

A large-scale tri-view (satellite / UAV / ground) dataset for cross-view 3D reconstruction and camera localization, spanning 85 scenes across every continent except Antarctica.

> **Paper**: [Seeing Across Skies and Streets: Feedforward 3D Reconstruction from Satellite, Drone, and Ground Images](https://arxiv.org/abs/2605.07978) (arXiv:2605.07978)

## Overview

CrossGeo contains **277,812 images** (46,302 samples × 6 views) with full 6-DoF poses and dense metric depth across three modalities:

| Modality | Source | Collection |
|----------|--------|------------|
| Satellite | Google Maps | 500m × 500m tiles (1024×1024, FOV 5°, altitude 5726m) |
| UAV | Google Earth Studio | Rendered at altitude 30–120m, pitch 0°–90° (same as [AerialMegaDepth](https://github.com/kvuong2711/aerial-megadepth)) |
| Ground | Google Street View | **Pano IDs only** — images NOT redistributed (Google TOS) |

## Data Structure

Each scene is organized as `{scene_id}_pair/{scene_id}_{altitude}_{pitch}/pair_{N}/`. Every pair contains **5 trajectories** (2 ground + 2 UAV + 2 satellite, where each ground station has a co-located satellite tile):

```
pair_3/
├── quad_info.json                     # Pair metadata
├── ground_1_rgb.jpg                   # Ground RGB
├── ground_1_rgb.npy                   # Ground pose {intrinsics, c2w, raw_data}
├── ground_1_depth.tiff                # Ground depth (float32 TIFF)
├── ground_1_satellite.jpg             # Satellite tile (co-located)
├── ground_1_satellite_depth.tiff      # Satellite depth (Z-Buffer from UAV point cloud)
├── uav_1_rgb.jpg                      # UAV RGB
├── uav_1_depth.tiff                   # UAV depth (COLMAP MVS)
└── ...
```

### Pose Format (`*_rgb.npy`)

Python dict with: `intrinsics` (3×3), `c2w` (4×4 in EDS frame), `raw_data` `[pitch, roll, heading, lat, lon, alt]`.

### World Coordinate System (EDS)

**X** → South, **Y** → Down, **Z** → East

## Open-Source Status

| Component | Status | Notes |
|-----------|--------|-------|
| Satellite RGB download | ✅ Open source | `satellite/download.py` (ref: [andolg/satellite-imagery-downloader](https://github.com/andolg/satellite-imagery-downloader), WGS84 → EDS modified) |
| Satellite pose + depth | ✅ Open source | `satellite/pose.py`, `satellite/depth.py` |
| UAV data pipeline | ✅ Open source | `uav/` (ref: [AerialMegaDepth](https://github.com/kvuong2711/aerial-megadepth)) |
| UAV flight trajectories (`.esp`) | 🔜 TODO | Will be released |
| Ground pano IDs | 🔜 TODO | Will be released (images NOT redistributed) |
| Ground depth refinement | ✅ Open source | `ground/depth_refine.py` (CDM + DA3 + PDA) |
| Ground pano2pinhole | ✅ Open source | `ground/pano2pinhole.py` |
| Tri-view pairing | ✅ Open source | `utils/pairing/tri_view_pairing.py` |
| Pre-trained model (Cross3R) | 🔜 TODO | Will be released separately |

## Repository Structure

```
crossgeo/
├── satellite/                   # Satellite data
│   ├── download.py              #   RGB download (ref: andolg/satellite-imagery-downloader)
│   ├── pose.py                  #   Pose recovery (virtual camera, FOV 5°, alt 5726m) ★
│   └── depth.py                 #   Depth projection (Z-Buffer from UAV point cloud) ★
├── ground/                      # Ground data (Google Street View)
│   ├── download.py              #   RGB + depth download via official API (pano IDs only)
│   ├── pano2pinhole.py          #   Panorama → pinhole camera conversion ★
│   ├── cdm_decoder.py           #   Google CDM depth format decoder ★
│   └── depth_refine.py          #   Depth refinement (CDM + DepthAnything v3 + Prior Depth Anything)
├── uav/                         # UAV data (Google Earth Studio)
│   ├── ges_utils.py             #   GE JSON → COLMAP (ECEF→ENU→EDS) (ref: AerialMegaDepth)
│   ├── preprocess_ge.py         #   Extract frames + metadata (ref: AerialMegaDepth)
│   ├── colmap_mvs.py            #   COLMAP MVS depth recovery (ref: AerialMegaDepth)
│   ├── video2png.py             #   Video frame extraction ★
│   ├── video2png_batch.py       #   Batch frame extraction ★
│   ├── json2csv.py              #   ESP + JSON → CSV metadata ★
│   └── modify_esp.py            #   ESP dense keyframe generation ★
├── utils/                       # Shared utilities + pairing
│   ├── geo.py                   #   Geo coordinate conversion (GPS/ENU/EDS)
│   ├── colmap_io.py             #   COLMAP model I/O
│   ├── depth2npy.py             #   TIFF → NPY conversion ★
│   ├── ply_show.py              #   Point cloud visualization ★
│   ├── scope.py                 #   UAV coverage polygon ★
│   ├── tiff2png.py              #   Depth visualization ★
│   └── pairing/                 #   Tri-view pairing + dataset split
│       ├── tri_view_pairing.py  #     Tri-view pairing (EDS, voxel overlap) ★
│       └── scene_split.py       #     Train/val/test split (75/5/5 scenes)
├── config.py                    # Configuration
├── pipeline.py                  # Main pipeline entry
├── requirements.txt             # Dependencies
├── LICENSE                      # MIT (code) + Google data rights
└── CITATION.bib                 # Paper citation
```

> **★** = verified code from actual data processing.

## Usage

### 1. Satellite (Google Maps)

Download 500m × 500m tiles at zoom 19. RGB download follows [andolg/satellite-imagery-downloader](https://github.com/andolg/satellite-imagery-downloader); the WGS84/Web Mercator tile addressing is converted to EDS poses in `satellite/pose.py`.

```bash
python satellite/download.py --lat 40.7128 --lon -74.0060 --scene_id 0005
```

### 2. UAV (Google Earth Studio) — Same as AerialMegaDepth

UAV images are rendered in Google Earth Studio following the [AerialMegaDepth](https://github.com/kvuong2711/aerial-megadepth) pipeline:

1. Render in Google Earth Studio → export `.mp4` + 3D Camera Tracking JSON (Global frame)
2. Extract frames: `python uav/preprocess_ge.py --data_root data/uav --scene_id 0005`
3. Recover depth via COLMAP MVS: `python uav/colmap_mvs.py --scene_dir ... --colmap_empty ...`

### 3. Ground (Google Street View) — Pano IDs Only

Per Google's TOS, we **do not redistribute** Street View images or depth. We release only **pano IDs**. Users download RGB and depth themselves:

```bash
# Download RGB + depth from pano IDs (requires GOOGLE_STREETVIEW_API_KEY)
python ground/download.py --pano_list data/pano_ids/scene_0005.json --output_dir data/ground/0005

# Refine depth: CDM coarse depth + DepthAnything v3 → Prior Depth Anything
python ground/depth_refine.py --scene_dir data/ground/0005
```

### 4. Tri-View Pairing

Pairs are formed by voxel overlap scoring in the EDS world frame:

```bash
python utils/pairing/tri_view_pairing.py --xml <metashape.xml> --uav_dir <uav_depth> --ground_dir <ground_data>
```

### Full Pipeline

```bash
python pipeline.py --scene_config data/scenes/example.json
python pipeline.py --scene_config data/scenes/example.json --steps satellite ground
python pipeline.py --split --scenes_dir data/scenes
```

## Installation

```bash
pip install -r requirements.txt
pip install git+https://github.com/cvg/Hierarchical-Localization.git  # hloc
```

## Citation

```bibtex
@article{wang2026crossgeo,
  title   = {Seeing Across Skies and Streets: Feedforward 3D Reconstruction from Satellite, Drone, and Ground Images},
  author  = {Wang, Qiwei and Tuo, Zhongyao and Ze, Xianghui and Shi, Yujiao},
  journal = {arXiv preprint arXiv:2605.07978},
  year    = {2026}
}
```

## License

- **Code**: MIT License (see [LICENSE](LICENSE))
- **Google Earth/Maps/Street View data**: Owned by Google, non-commercial research use only.
- **Street View images**: NOT redistributed. Only pano IDs are released.

## Acknowledgements

- [satellite-imagery-downloader](https://github.com/andolg/satellite-imagery-downloader) — Satellite RGB download
- [AerialMegaDepth](https://github.com/kvuong2711/aerial-megadepth) — UAV data collection pipeline
- [Depth-Anything-3](https://github.com/Depth-Anything/Depth-Anything-3) — Relative depth estimation
- [Prior-Depth-Anything](https://github.com/sichengplus/Prior-Depth-Anything) — Depth fusion
- [hloc](https://github.com/cvg/Hierarchical-Localization) — Feature extraction and matching
