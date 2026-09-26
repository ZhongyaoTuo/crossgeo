<div align="center">

# CrossGeo Dataset

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

This repository is part of the **CrossGeo** project introduced in the paper above. It provides the **data construction pipeline** used to build the CrossGeo dataset — a large-scale tri-view (satellite / UAV / ground) dataset for cross-view 3D reconstruction and camera localization, spanning **85 scenes** across every continent except Antarctica with **277,812 images** in total.

> The same pipeline is **scalable to unlimited scenes**: given any GPS coordinate, it automatically downloads satellite imagery, renders UAV views, retrieves ground panoramas, recovers poses, computes depth, and forms tri-view pairs. You can use it to build your own dataset of any size.

## Sample Data

A demo sample from scene `0005` is included in [`sample/`](sample/) for quick inspection:

```
sample/0005_45_60/pair_3/
├── quad_info.json                     # Pair metadata
├── ground_1_rgb.jpg + .npy            # Ground RGB + pose
├── ground_2_rgb.jpg + .npy
├── uav_1_rgb.jpg + .npy               # UAV RGB + pose
├── uav_2_rgb.jpg + .npy
├── ground_1_satellite.jpg + .npy      # Satellite tile + pose
├── ground_2_satellite.jpg + .npy
├── ground_1_depth_metric.png          # Refined depth visualization
├── ground_2_depth_metric.png
├── ground_1_satellite_depth_viz.png   # Satellite depth visualization
└── ground_2_satellite_depth_viz.png
```

> Large depth arrays (`.tiff`, `.npy`, `.pt`) are excluded from the repo. See the full data format below.

## Overview

CrossGeo contains **277,812 images** (46,302 samples × 6 views) with full 6-DoF poses and dense metric depth across three modalities:

| Modality | Source | Collection |
|----------|--------|------------|
| Satellite | Google Maps | 500m × 500m tiles (1024×1024, FOV 5°, altitude 5726m) |
| UAV | Google Earth Studio | Rendered at altitude 30–120m, pitch 0°–90° (same as [AerialMegaDepth](https://github.com/kvuong2711/aerial-megadepth)) |
| Ground | Google Street View | **Pano IDs only** — images NOT redistributed (Google TOS) |

### Pipeline at a Glance

<div align="center">
  <img src="dataset_pipeline.jpeg" alt="CrossGeo data sources" width="800">
</div>

*(a) Ground views and coarse depth from Google Street View. (b) UAV captures rendered in Google Earth together with co-located satellite tiles from Google Maps. (c) Tri-view pairing across the three modalities. (d) AnyVisLoc test-set augmentation.*

```
GPS coordinate
    ├──▶ satellite/     Google Maps       ──▶ RGB tile + virtual camera pose + Z-Buffer depth
    ├──▶ uav/           Google Earth      ──▶ rendered frames + COLMAP MVS depth
    └──▶ ground/        Google Street View──▶ pano RGB + CDM depth + refined depth
            └──▶ utils/pairing/           ──▶ voxel overlap scoring ──▶ tri-view pairs
```

## Data Structure

Each scene is organized as `{scene_id}_pair/{scene_id}_{altitude}_{pitch}/pair_{N}/`. Every pair contains **6 views** (2 ground + 2 UAV + 2 satellite):

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
| Satellite pipeline | ✅ Open source | `satellite/` (ref: [andolg/satellite-imagery-downloader](https://github.com/andolg/satellite-imagery-downloader)) |
| UAV pipeline | ✅ Open source | `uav/` (ref: [AerialMegaDepth](https://github.com/kvuong2711/aerial-megadepth)) |
| Ground pipeline | ✅ Open source | `ground/` (pano IDs only, no image redistribution) |
| Tri-view pairing | ✅ Open source | `utils/pairing/tri_view_pairing.py` |
| UAV flight trajectories (`.esp`) | 🔜 TODO | Will be released |
| Ground pano IDs | 🔜 TODO | Will be released |
| Pre-trained model (Cross3R) | 🔜 TODO | Will be released separately |

### Open-Source Scope & AnyVisLoc OOD Test Set

**This repository open-sources the CrossGeo training data construction pipeline only.** The real-world out-of-distribution (OOD) test set in the paper repurposes [**AnyVisLoc**](https://github.com/UAV-AVL/Benchmark) — a benchmark of UAV photographs captured by physical drones (CVPR 2026 Findings).

> **Why is AnyVisLoc not included here?** AnyVisLoc's coordinates are intentionally sanitized to protect geographic privacy: original UAV longitude/latitude are converted to UTM and then shifted to a local metric coordinate system, with heights shifted by the mean DSM elevation. **The original GPS coordinates are not publicly releasable.** Therefore, we only open-source the CrossGeo training set construction pipeline. Users who wish to evaluate on the OOD test set should obtain AnyVisLoc directly from its [official repository](https://github.com/UAV-AVL/Benchmark).
>
> **Important:** Per the AnyVisLoc ground-truth protocol, AnyVisLoc is designed for **horizontal visual localization only**. Do not use it for UAV height estimation or absolute-altitude estimation, as the released coordinates do not share a consistent elevation reference.

## Quick Start

### Installation

```bash
git clone --recursive https://github.com/ZhongyaoTuo/crossgeo.git
cd crossgeo
pip install -r requirements.txt
pip install git+https://github.com/cvg/Hierarchical-Localization.git  # hloc
```

### Collect a single scene

```bash
python pipeline.py --scene_config data/scenes/example.json
python pipeline.py --scene_config data/scenes/example.json --steps satellite ground
```

### Scale to more scenes

Add more scene config JSONs to `data/scenes/` and run:

```bash
python pipeline.py --split --scenes_dir data/scenes
```

### Step-by-step tutorials

- **[satellite/README.md](satellite/README.md)** — Download satellite tiles & recover poses
- **[uav/README.md](uav/README.md)** — Render UAV imagery in Google Earth Studio & recover depth
- **[ground/README.md](ground/README.md)** — Download Street View panoramas & refine depth
- **[utils/README.md](utils/README.md)** — Tri-view pairing, coordinate conversion, visualization

## External Dependencies

| Folder (submodule) | Source Repository | Description |
|---------------------|-------------------|-------------|
| `satellite/satellite-imagery-downloader/` | [andolg/satellite-imagery-downloader](https://github.com/andolg/satellite-imagery-downloader) | Satellite RGB tile download |
| `uav/aerial-megadepth/` | [kvuong2711/aerial-megadepth](https://github.com/kvuong2711/aerial-megadepth) | UAV data collection (GE → COLMAP → MVS) |

## Acknowledgement

This codebase builds upon the following excellent open-source projects. We thank the respective authors for making their work publicly available:

- **[satellite-imagery-downloader](https://github.com/andolg/satellite-imagery-downloader)** — Satellite RGB tile download
- **[AerialMegaDepth](https://github.com/kvuong2711/aerial-megadepth)** — UAV data collection pipeline
- **[AnyVisLoc](https://github.com/UAV-AVL/Benchmark)** — Real-world OOD test set (UAV visual localization benchmark)
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
