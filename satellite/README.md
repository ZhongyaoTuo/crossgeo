# Satellite Data Pipeline

This folder contains the satellite data collection pipeline for CrossGeo.

## Code Source

| File | Origin | Description |
|------|--------|-------------|
| `download.py` | [andolg/satellite-imagery-downloader](https://github.com/andolg/satellite-imagery-downloader) | Satellite RGB tile download (WGS84/Web Mercator addressing), modified for EDS coordinate conversion |
| `pose.py` | Original (CrossGeo) | Pose recovery via virtual camera (FOV 5°, altitude 5726m) |
| `depth.py` | Original (CrossGeo) | Depth projection via Z-Buffer from UAV point cloud |

> For the original satellite imagery download implementation, please refer to **[andolg/satellite-imagery-downloader](https://github.com/andolg/satellite-imagery-downloader)**.