# Ground Data Pipeline

This folder contains scripts for downloading Google Street View panoramas, decoding depth, converting to pinhole cameras, and refining depth for CrossGeo.

> **Important**: Per Google's Terms of Service, we **do NOT redistribute** Street View images or depth. We release only **pano IDs**. Users must download RGB and depth themselves via the official Google Street View API.

## Tutorial

### Step 1: Download Street View Panoramas

Given a list of pano IDs, download RGB panoramas and coarse depth via the Google Street View API:

```bash
python ground/download.py --pano_list data/pano_ids/scene_0005.json --output_dir data/ground/0005
```

**Prerequisites**: Set the `GOOGLE_STREETVIEW_API_KEY` environment variable.

**Output**: `pano_XXX.jpg` (equirectangular RGB) and `pano_XXX_depth.json` (coarse depth).

### Step 2: Decode CDM Depth

Google Street View depth is encoded in the CDM (Custom Depth Model) format. Decode it into a usable depth array:

```python
from ground.cdm_decoder import decode_cdm_depth

depth = decode_cdm_depth("data/ground/0005/pano_001_depth.json")
# depth: H×W float32 numpy array (equirectangular)
```

### Step 3: Convert Panorama to Pinhole Cameras

Split the equirectangular panorama into multiple pinhole camera views:

```bash
python ground/pano2pinhole.py --pano_dir data/ground/0005 --output_dir data/ground/0005/pinhole --fov 90 --n_views 4
```

**Output**: `ground_XX_rgb.jpg` + `ground_XX_rgb.npy` (pose dict with `intrinsics`, `c2w`, `raw_data`).

### Step 4: Refine Depth

Fuse the coarse CDM depth with a sharp relative depth from DepthAnything v3 via Prior Depth Anything:

```bash
python ground/depth_refine.py --scene_dir data/ground/0005
```

**Pipeline**: CDM coarse depth (metric scale) + DepthAnything v3 (sharp, relative) → Prior Depth Anything (sharp, metric).

**Output**: `ground_XX_depth.tiff` — refined float32 metric depth map.

> Samples whose rescaled depth has low Pearson correlation with the relative prediction are discarded.

## Files

| File | Description |
|------|-------------|
| `download.py` | Download RGB + depth from pano IDs via Google Street View API |
| `cdm_decoder.py` | Decode Google CDM depth format |
| `pano2pinhole.py` | Convert equirectangular panorama → pinhole camera views |
| `depth_refine.py` | Depth refinement (CDM + DepthAnything v3 + Prior Depth Anything) |