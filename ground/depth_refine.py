#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ground depth refinement: CDM coarse depth + DepthAnything v3 -> metric depth.
=============================================================================
CrossGeo (paper Sec 2.1 -- Metric depth):
    Ground depth = PriorDepthAnything(StreetView_CDM_depth, DepthAnythingV3(image))

Flow:
    1. Decode Google Street View CDM depth (ground/cdm_decoder.py)
    2. Predict relative depth with DepthAnything v3
    3. Fuse with Prior Depth Anything to obtain metric depth
    4. Safety check: Pearson correlation between rescaled and relative depth

Dependencies:
    - DepthAnything v3:     https://github.com/Depth-Anything/Depth-Anything-3
    - Prior Depth Anything:  https://github.com/sichengplus/Prior-Depth-Anything
"""
import os
import numpy as np

import sys
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    DEPTHANY_V3_CKPT, PRIOR_DEPTH_ANYTHING_CKPT,
    GROUND_DEPTH_PEARSON_THRESHOLD,
)


def decode_cdm_depth_from_file(depth_bin_path: str, pano_id: str = None,
                               dep_max: float = 100.0) -> np.ndarray:
    """Decode Google Street View CDM depth to a per-pixel depth map.

    Wrapper around ground/cdm_decoder.py:decode_cdm_depth.
    Reads the base64-encoded CDM data from file, decodes it, and
    returns only the depth map (the underlying function returns
    (width, height, depth)).

    Args:
        depth_bin_path: path to the raw CDM depth file (base64 content).
        pano_id: panorama ID (for logging).
        dep_max: maximum depth in meters (CDM decoding parameter).

    Returns:
        Depth map (H, W) in meters, or None if decoding fails.
    """
    sys.path.insert(0, str(os.path.dirname(os.path.abspath(__file__))))
    from cdm_decoder import decode_cdm_depth as _decode_cdm_depth_b64

    with open(depth_bin_path, "r") as f:
        depth_b64 = f.read().strip()

    result = _decode_cdm_depth_b64(depth_b64, dep_max=dep_max)
    if result is None or result[2] is None:
        if pano_id:
            print(f"  CDM decode failed for pano {pano_id}")
        return None
    _, _, depth = result
    return depth


def load_depth_anything_v3(ckpt_path: str = None):
    """Load DepthAnything v3 model.

    Ref: https://github.com/Depth-Anything/Depth-Anything-3
    """
    import torch
    ckpt_path = ckpt_path or DEPTHANY_V3_CKPT
    if not ckpt_path:
        raise EnvironmentError(
            "DEPTHANY_V3_CKPT must point to the DepthAnything v3 checkpoint. "
            "See: https://github.com/Depth-Anything/Depth-Anything-3"
        )
    try:
        from depth_anything_v3 import DepthAnythingV3
        model = DepthAnythingV3.from_pretrained(ckpt_path)
        model.eval()
        return model
    except ImportError:
        print("Warning: depth_anything_v3 not found. "
              "Install: https://github.com/Depth-Anything/Depth-Anything-3")
        return None


def predict_relative_depth(model, image: np.ndarray,
                           device: str = "cuda") -> np.ndarray:
    """Predict relative depth with DepthAnything v3.

    Args:
        model: DepthAnything v3 model.
        image: input image (H, W, 3), RGB.
        device: compute device.

    Returns:
        Relative depth map (H, W); larger = closer.
    """
    import torch
    from torchvision import transforms

    if model is None:
        raise RuntimeError("DepthAnything v3 model not loaded")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        depth = model(img_tensor)
    return depth.squeeze().cpu().numpy()


def fuse_depth(relative_depth: np.ndarray, cdm_depth: np.ndarray,
               pda_model=None) -> np.ndarray:
    """Fuse relative depth with CDM coarse metric depth.

    Uses Prior Depth Anything when available; falls back to least-squares
    linear alignment: metric = a * relative + b.

    Args:
        relative_depth: DepthAnything v3 output (sharp, relative).
        cdm_depth: decoded CDM depth (absolute scale, coarse).
        pda_model: Prior Depth Anything model (None = linear fallback).

    Returns:
        Fused metric depth map (H, W).
    """
    import torch

    if pda_model is not None:
        with torch.no_grad():
            fused = pda_model(
                torch.from_numpy(relative_depth).float(),
                torch.from_numpy(cdm_depth).float(),
            )
        return fused.cpu().numpy()

    valid = (cdm_depth > 0) & np.isfinite(cdm_depth)
    if valid.sum() < 10:
        return relative_depth

    rel = relative_depth[valid]
    coarse = cdm_depth[valid]
    A = np.vstack([rel, np.ones(len(rel))]).T
    a, b = np.linalg.lstsq(A, coarse, rcond=None)[0]
    fused = a * relative_depth + b
    fused[fused < 0] = 0
    return fused


def compute_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return 0.0
    xm, ym = x.mean(), y.mean()
    num = np.sum((x - xm) * (y - ym))
    den = np.sqrt(np.sum((x - xm) ** 2) * np.sum((y - ym) ** 2))
    return float(num / den) if den > 0 else 0.0


def refine_ground_depth(image: np.ndarray, cdm_depth: np.ndarray = None,
                        da3_model=None, pda_model=None,
                        device: str = "cuda") -> dict:
    """Refine ground depth for a single image.

    Flow:
        1. DepthAnything v3 -> relative depth
        2. Prior Depth Anything fuses CDM coarse depth
        3. Pearson check (discard if too low)

    Args:
        image: input image (H, W, 3), RGB.
        cdm_depth: decoded CDM depth (None = no prior).
        da3_model: DepthAnything v3 model.
        pda_model: Prior Depth Anything model.
        device: compute device.

    Returns:
        {"depth", "relative_depth", "valid", "pearson"}
    """
    rel = predict_relative_depth(da3_model, image, device)

    if cdm_depth is not None:
        metric = fuse_depth(rel, cdm_depth, pda_model)
        pearson = compute_pearson(rel, metric)
        valid = pearson >= GROUND_DEPTH_PEARSON_THRESHOLD
    else:
        metric = rel
        pearson = 1.0
        valid = True

    return {"depth": metric, "relative_depth": rel, "valid": valid, "pearson": pearson}


def process_scene(scene_dir: str, output_dir: str = None,
                  da3_model=None, pda_model=None,
                  device: str = "cuda") -> dict:
    """Process ground depth for an entire scene.

    Args:
        scene_dir: scene directory (contains *.jpg and *_depth.bin).
        output_dir: depth output directory.
        da3_model, pda_model: depth models.
        device: compute device.

    Returns:
        Processing statistics.
    """
    import cv2
    import glob

    if output_dir is None:
        output_dir = os.path.join(scene_dir, "depths")
    os.makedirs(output_dir, exist_ok=True)

    images = sorted(glob.glob(os.path.join(scene_dir, "*.jpg")))
    num_valid, num_total = 0, 0

    for img_path in images:
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        depth_bin = img_path.replace(".jpg", "_depth.bin")
        cdm_depth = None
        if os.path.exists(depth_bin):
            cdm_depth = decode_cdm_depth_from_file(depth_bin)

        result = refine_ground_depth(image, cdm_depth, da3_model, pda_model, device)
        num_total += 1
        if result["valid"]:
            num_valid += 1
            out = os.path.join(output_dir,
                               os.path.basename(img_path).replace(".jpg", ".npy"))
            np.save(out, result["depth"])

    print(f"Ground depth refined: {num_valid}/{num_total} valid "
          f"(Pearson >= {GROUND_DEPTH_PEARSON_THRESHOLD})")
    return {"num_valid": num_valid, "num_total": num_total, "output_dir": output_dir}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Refine ground depth (CDM + DA3 + PDA)")
    parser.add_argument("--scene_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    da3 = load_depth_anything_v3()
    result = process_scene(args.scene_dir, args.output_dir, da3_model=da3)
    print(f"Result: {result}")