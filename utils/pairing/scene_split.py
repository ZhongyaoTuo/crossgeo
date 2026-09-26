#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scene-level train / val / test split.
====================================
CrossGeo (paper Sec 2.1 -- Train/val/test split):
    - 85 scenes, scene-level split (no city in multiple splits)
    - Train: 75 scenes (38,962 samples)
    - Val:   5 scenes (3,614 samples), North American cities
    - Test:  5 scenes (3,726 samples), cities disjoint from train
"""
import os
import json
import random

import sys
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from config import NUM_TRAIN_SCENES, NUM_VAL_SCENES, NUM_TEST_SCENES, TOTAL_SCENES


def load_scene_list(scenes_dir: str) -> list:
    """Load all scenes from a directory."""
    scenes = []
    for name in sorted(os.listdir(scenes_dir)):
        scene_path = os.path.join(scenes_dir, name)
        if not os.path.isdir(scene_path):
            continue
        meta_path = os.path.join(scene_path, "meta.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        scenes.append({
            "scene_id": name,
            "city": meta.get("city", "unknown"),
            "continent": meta.get("continent", "unknown"),
            "lat": meta.get("lat_center", 0),
            "lon": meta.get("lon_center", 0),
        })
    return scenes


def split_by_city_and_continent(scenes: list,
                                num_train=NUM_TRAIN_SCENES, num_val=NUM_VAL_SCENES,
                                num_test=NUM_TEST_SCENES,
                                val_continent="North America", seed=42) -> dict:
    """Split scenes by city and continent (no city in multiple splits)."""
    random.seed(seed)
    by_city = {}
    for s in scenes:
        by_city.setdefault(s["city"], []).append(s)

    cities = list(by_city.keys())
    random.shuffle(cities)

    train, val, test = [], [], []

    val_cities = [c for c in cities if by_city[c][0]["continent"] == val_continent]
    random.shuffle(val_cities)
    for city in val_cities:
        if len(val) >= num_val:
            break
        val.extend(by_city[city])
        cities.remove(city)

    test_cities = list(cities)
    random.shuffle(test_cities)
    for city in test_cities:
        if len(test) >= num_test:
            break
        test.extend(by_city[city])
        cities.remove(city)

    for city in cities:
        train.extend(by_city[city])
        if len(train) >= num_train:
            break

    train, val, test = train[:num_train], val[:num_val], test[:num_test]

    tc = set(s["city"] for s in train)
    vc = set(s["city"] for s in val)
    sc = set(s["city"] for s in test)
    overlap = (tc & vc) | (tc & sc) | (vc & sc)
    if overlap:
        print(f"Warning: city overlap: {overlap}")

    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return {"train": train, "val": val, "test": test}


def create_scene_split(scenes_dir: str, output_dir: str = None, seed=42) -> dict:
    """Create scene-level split and save to JSON + txt files."""
    scenes = load_scene_list(scenes_dir)
    print(f"Loaded {len(scenes)} scenes")
    if len(scenes) < TOTAL_SCENES:
        print(f"Warning: {len(scenes)} < expected {TOTAL_SCENES}")

    split = split_by_city_and_continent(scenes, seed=seed)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "scene_split.json"), "w") as f:
            json.dump(split, f, indent=2)
        for name in ["train", "val", "test"]:
            ids = [s["scene_id"] for s in split[name]]
            with open(os.path.join(output_dir, f"{name}_scenes.txt"), "w") as f:
                f.write("\n".join(ids))

    return split


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CrossGeo scene split")
    parser.add_argument("--scenes_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/split")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_scene_split(args.scenes_dir, args.output, args.seed)