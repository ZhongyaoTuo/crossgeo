<div align="center">

# CrossGeo: Scalable Tri-View Data Acquisition Pipeline

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
<img src="https://img.shields.io/badge/scalable-∞-green.svg" alt="Scalable">
<img src="https://img.shields.io/badge/modalities-3-blue.svg" alt="Modalities">

</div>

## What is this?

This repository provides a **fully automated, scalable pipeline** for collecting tri-view (satellite / UAV / ground) data with full 6-DoF poses and dense metric depth from **any location on Earth**. Given a GPS coordinate, the pipeline automatically downloads satellite imagery, renders UAV views, retrieves ground-level panoramas, recovers poses, computes depth, and forms tri-view pairs.

**The pipeline is not limited to a fixed dataset — it can scale to collect an unlimited number of scenes.** CrossGeo (85 scenes, 277,812 images) is simply the dataset we produced with this pipeline for the paper. You can use it to build your own dataset of any size.

### Key Features

- **🌍 Globally scalable**: Pick any GPS coordinate and the pipeline handles the rest — satellite tiles from Google Maps, UAV renders from Google Earth Studio, ground panoramas from Google Street View.
- **🔄 Fully automated**: From raw download to tri-view pairing, the entire workflow runs end-to-end with a single command.
- **📐 Full 6-DoF poses + metric depth**: Every image comes with calibrated camera pose (intrinsics + c2w in EDS frame) and dense metric depth.
- **🛰️ Three modalities, one frame**: Satellite, UAV, and ground views are aligned in a unified EDS world coordinate system (X→South, Y→Down, Z→East).
- **🔗 Tri-view pairing**: Automatic voxel-overlap scoring forms cross-modal pairs without known relative poses.

## Pipeline Overview

```
GPS coordinate
    │
    ├──▶ satellite/          Google Maps ──▶ RGB tile + virtual camera pose + Z-Buffer depth
    ├──▶ uav/                Google Earth Studio ──▶ rendered frames + COLMAP MVS depth
    └──▶ ground/             Google Street View ──▶ pano RGB + CDM depth + refined depth
                │
                ▼
         utils/pairing/       Voxel overlap scoring ──▶ tri-view pairs (6 images per sample)
```

| Modality | Source | What you get |
|----------|--------|-------------|
| Satellite | Google Maps | 500m × 500m tiles (1024×1024, FOV 5°, altitude 5726m) + pose + depth |
| UAV | Google Earth Studio | Rendered frames at altitude 30–120m, pitch 0°–90° + COLMAP MVS depth |
| Ground | Google Street View | Pinhole views from panoramas + refined depth (CDM + DepthAnything v3 + PDA) |

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
# Run the full pipeline for one scene (satellite + UAV + ground + pairing)
python pipeline.py --scene_config data/scenes/example.json

# Or run specific steps only
python pipeline.py --scene_config data/scenes/example.json --steps satellite ground
```

### Scale to unlimited scenes

```bash
# Batch-collect multiple scenes and split into train/val/test
python pipeline.py --split --scenes_dir data/scenes
```

Just add more scene config JSONs to `data/scenes/` — the pipeline handles the rest.

### Step-by-step tutorials

Each subfolder contains a detailed tutorial:

- **[satellite/README.md](satellite/README.md)** — Download satellite tiles & recover poses
- **[uav/README.md](uav/README.md)** — Render UAV imagery in Google Earth Studio & recover depth
- **[ground/README.md](ground/README.md)** — Download Street View panoramas & refine depth
- **[utils/README.md](utils/README.md)** — Tri-view pairing, coordinate conversion, visualization

## CrossGeo Dataset

Using this pipeline, we produced **CrossGeo**, a 278K-image tri-view dataset spanning 85 scenes across every continent except Antarctica:

- **46,302 samples** × 6 views = **277,812 images**
- **75/5/5** scene-level train/val/test split (no city overlap across splits)
- Each sample: 2 satellite + 2 UAV + 2 ground views with poses + depth

### Data Structure

```
{scene_id}_pair/{scene_id}_{altitude}_{pitch}/pair_{N}/
├── quad_info.json                     # Pair metadata
├── ground_1_rgb.jpg + .npy + _depth.tiff
├── ground_1_satellite.jpg + _depth.tiff
├── uav_1_rgb.jpg + _depth.tiff
└── ...
```

### Pose Format (`*_rgb.npy`)

Python dict with: `intrinsics` (3×3), `c2w` (4×4 in EDS frame), `raw_data` `[pitch, roll, heading, lat, lon, alt]`.

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

## External Dependencies

| Folder (submodule) | Source Repository | Description |
|---------------------|-------------------|-------------|
| `satellite/satellite-imagery-downloader/` | [andolg/satellite-imagery-downloader](https://github.com/andolg/satellite-imagery-downloader) | Satellite RGB tile download |
| `uav/aerial-megadepth/` | [kvuong2711/aerial-megadepth](https://github.com/kvuong2711/aerial-megadepth) | UAV data collection (GE → COLMAP → MVS) |

## Acknowledgement

This codebase builds upon the following excellent open-source projects. We thank the respective authors for making their work publicly available:

- **[satellite-imagery-downloader](https://github.com/andolg/satellite-imagery-downloader)** — Satellite RGB tile download
- **[AerialMegaDepth](https://github.com/kvuong2711/aerial-megadepth)** — UAV data collection pipeline
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
