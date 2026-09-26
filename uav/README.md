# UAV Data Pipeline

This folder contains the UAV (drone) data collection pipeline for CrossGeo, using Google Earth Studio renders.

## Code Source

| File | Origin | Description |
|------|--------|-------------|
| `ges_utils.py` | [kvuong2711/aerial-megadepth](https://github.com/kvuong2711/aerial-megadepth) | GE JSON → COLMAP camera conversion (ECEF→ENU→EDS) |
| `preprocess_ge.py` | [kvuong2711/aerial-megadepth](https://github.com/kvuong2711/aerial-megadepth) | Frame extraction + metadata parsing from GE render output |
| `colmap_mvs.py` | [kvuong2711/aerial-megadepth](https://github.com/kvuong2711/aerial-megadepth) | COLMAP MVS dense depth recovery pipeline |
| `video2png.py` | Original (CrossGeo) | Video frame extraction |
| `video2png_batch.py` | Original (CrossGeo) | Batch frame extraction |
| `json2csv.py` | Original (CrossGeo) | ESP + JSON → CSV metadata |
| `modify_esp.py` | Original (CrossGeo) | ESP dense keyframe generation |

> For the original UAV data collection implementation, please refer to **[AerialMegaDepth](https://github.com/kvuong2711/aerial-megadepth)**.
>
> The full source repository is also included as a git submodule at [`aerial-megadepth/`](aerial-megadepth/). To clone with submodules:
> ```bash
> git clone --recursive https://github.com/ZhongyaoTuo/crossgeo.git
> ```