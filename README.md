<div align="center">

# CrossGeo: Dataset Construction Pipeline

**Seeing Across Skies and Streets: Feedforward 3D Reconstruction from Satellite, Drone, and Ground Images**

<a href="https://scholar.google.com/citations?user=V61LSk0AAAAJ&hl=en">Qiwei Wang</a><sup>1,†</sup>,
<a href="https://scholar.google.com/citations?user=hJQA_NQAAAAJ&hl=en">Zhongyao Tuo</a><sup>1,†</sup>,
<a href="https://scholar.google.com/citations?user=N0hjZrAAAAAJ&hl=en">Xianghui Ze</a><sup>2</sup>,
<a href="https://scholar.google.com/citations?user=rVsRpZEAAAAJ&hl=en">Yujiao Shi</a><sup>1,‡</sup>

<sup>1</sup>ShanghaiTech University &nbsp;&nbsp; <sup>2</sup>Nanjing University of Science and Technology

<sup>†</sup>Equal contribution &nbsp;&nbsp; <sup>‡</sup>Corresponding author

[[`arXiv`](https://arxiv.org/abs/2605.07978)]
[[`Bibtex`](#citation)]

<a href="https://arxiv.org/abs/2605.07978"><img src="https://img.shields.io/badge/arXiv-2605.07978-b31b1b.svg" alt="arXiv"></a>
<a href="https://github.com/ZhongyaoTuo/crossgeo/actions/workflows/ci.yml"><img src="https://github.com/ZhongyaoTuo/crossgeo/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
<img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python">
<img src="https://img.shields.io/badge/scenes-85-green.svg" alt="Scenes">
<img src="https://img.shields.io/badge/images-277K-green.svg" alt="Images">

</div>

> **This repository contains the data construction pipeline for the CrossGeo dataset introduced in the paper.** The pre-trained Cross3R model weights will be released separately.

## Overview

CrossGeo is a large-scale tri-view (satellite / UAV / ground) dataset for cross-view 3D reconstruction and camera localization, spanning **85 scenes** across every continent except Antarctica. It contains **277,812 images** (46,302 samples × 6 views) with full 6-DoF poses and dense metric depth across three modalities:

| Modality | Source | Collection |
|----------|--------|------------|
| Satellite | Google Maps | 500m × 500m tiles (1024×1024, FOV 5°, altitude 5726m) |
| UAV | Google Earth Studio | Rendered at altitude 30–120m, pitch 0°–90° (same as [AerialMegaDepth](https://github.com/kvuong2711/aerial-megadepth)) |
| Ground | Google Street View | **Pano IDs only** — images NOT redistributed (Google TOS) |

## Data Structure

Each scene is organized as `{scene_id}_pair/{scene_id}_{altitude}_{pitch}/pair_{N}/`. Every pair contains **6 views** (2 ground + 2 UAV + 2 satellite, where each ground station has a co-located satellite tile):

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
├── satellite/                   # Satellite data pipeline (see satellite/README.md)
├── ground/                      # Ground data pipeline (see ground/README.md)
├── uav/                         # UAV data pipeline (see uav/README.md)
├── utils/                       # Shared utilities + tri-view pairing (see utils/README.md)
├── config.py                    # Configuration
├── pipeline.py                  # Main pipeline entry
├── requirements.txt             # Dependencies
├── LICENSE                      # MIT (code) + Google data rights
└── CITATION.bib                 # Paper citation
```

### External Dependencies

The following folders contain code adapted from external open-source repositories, included as git submodules:

| Folder | Source Repository | Description |
|--------|-------------------|-------------|
| `satellite/satellite-imagery-downloader/` | [andolg/satellite-imagery-downloader](https://github.com/andolg/satellite-imagery-downloader) | Satellite RGB tile download (WGS84/Web Mercator addressing) |
| `uav/aerial-megadepth/` | [kvuong2711/aerial-megadepth](https://github.com/kvuong2711/aerial-megadepth) | UAV data collection pipeline (GE → COLMAP → MVS depth) |

## Installation

```bash
git clone --recursive https://github.com/ZhongyaoTuo/crossgeo.git
cd crossgeo
pip install -r requirements.txt
pip install git+https://github.com/cvg/Hierarchical-Localization.git  # hloc
```

## Quick Start

### Full Pipeline

```bash
python pipeline.py --scene_config data/scenes/example.json
python pipeline.py --scene_config data/scenes/example.json --steps satellite ground
python pipeline.py --split --scenes_dir data/scenes
```

### Step-by-Step

See the tutorial in each subfolder:
- **[satellite/README.md](satellite/README.md)** — Download satellite tiles & recover poses
- **[uav/README.md](uav/README.md)** — Render UAV imagery in Google Earth Studio & recover depth
- **[ground/README.md](ground/README.md)** — Download Street View panoramas & refine depth
- **[utils/README.md](utils/README.md)** — Tri-view pairing, coordinate conversion, visualization

## Acknowledgement

This codebase builds upon the following excellent open-source projects. We thank the respective authors for making their work publicly available:

- **[satellite-imagery-downloader](https://github.com/andolg/satellite-imagery-downloader)** — Satellite RGB tile download (`satellite/download.py`)
- **[AerialMegaDepth](https://github.com/kvuong2711/aerial-megadepth)** — UAV data collection pipeline (`uav/ges_utils.py`, `uav/preprocess_ge.py`, `uav/colmap_mvs.py`)
- **[Depth-Anything-3](https://github.com/Depth-Anything/Depth-Anything-3)** — Relative depth estimation
- **[Prior-Depth-Anything](https://github.com/sichengplus/Prior-Depth-Anything)** — Depth fusion
- **[hloc](https://github.com/cvg/Hierarchical-Localization)** — Feature extraction and matching

## Citation

If you find our work to be useful in your research, please consider citing our paper:

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
