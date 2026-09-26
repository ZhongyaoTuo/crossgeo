#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geographic coordinate conversion utilities.
===========================================
GPS <-> ENU <-> pixel conversions, plus pose assembly in the CrossGeo EDS
world frame.

CrossGeo world frame (paper Sec 2.1 -- Pose recovery):
    X -> South
    Y -> Down
    Z -> East

Ref: aerial-megadepth/data_generation/ges_utils.py (ENU conversion logic).
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation

try:
    from pymap3d.enu import geodetic2enu
except ImportError:
    geodetic2enu = None


def rot_ecef2enu(lat: float, lon: float) -> np.ndarray:
    """ECEF -> ENU rotation matrix.

    Args:
        lat: latitude in degrees.
        lon: longitude in degrees.
    Returns:
        3x3 rotation matrix from ECEF to local ENU.
    """
    lamb = np.deg2rad(lon)
    phi = np.deg2rad(lat)
    sL, sP = np.sin(lamb), np.sin(phi)
    cL, cP = np.cos(lamb), np.cos(phi)
    return np.array([
        [     -sL,       cL,  0],
        [-sP * cL, -sP * sL, cP],
        [ cP * cL,  cP * sL, sP],
    ])


def gps_to_enu(lat: float, lon: float, alt: float,
               lat0: float, lon0: float, alt0: float) -> np.ndarray:
    """Convert GPS to ENU coordinates relative to (lat0, lon0, alt0).

    Returns: [east, north, up].
    """
    if geodetic2enu is None:
        raise ImportError("pymap3d is required: pip install pymap3d")
    e, n, u = geodetic2enu(lat, lon, alt, lat0, lon0, alt0)
    return np.array([e, n, u])


def enu_to_crossgeo_world(east: float, north: float, up: float) -> np.ndarray:
    """ENU -> CrossGeo world frame.

    CrossGeo: x=South, y=Down, z=East
    ENU:      east, north, up

    Mapping: x = -north (South), y = -up (Down), z = east (East)
    """
    return np.array([-north, -up, east])


def gps_to_crossgeo_world(lat: float, lon: float, alt: float,
                          lat0: float, lon0: float, alt0: float) -> np.ndarray:
    """GPS -> CrossGeo world coordinates (one step)."""
    e, n, u = gps_to_enu(lat, lon, alt, lat0, lon0, alt0)
    return enu_to_crossgeo_world(e, n, u)


def euler_to_rotation(rx: float, ry: float, rz: float,
                      sequence: str = "XYZ", degrees: bool = True) -> np.ndarray:
    """Euler angles to rotation matrix."""
    return Rotation.from_euler(sequence, [rx, ry, rz], degrees=degrees).as_matrix()


def build_camera_pose(lat: float, lon: float, alt: float,
                      rx: float, ry: float, rz: float,
                      lat0: float, lon0: float, alt0: float) -> np.ndarray:
    """Assemble a 4x4 camera-to-world (c2w) pose matrix.

    Used for Google Earth Studio / Street View pose recovery:
    1. GPS -> ENU local coordinates
    2. ECEF rotation -> ENU rotation
    3. Assemble c2w

    Args:
        lat, lon, alt: camera GPS.
        rx, ry, rz: camera Euler angles (ECEF frame, XYZ order).
        lat0, lon0, alt0: scene origin GPS.

    Returns:
        4x4 c2w matrix.
    """
    x, y, z = gps_to_enu(lat, lon, alt, lat0, lon0, alt0)
    R_ecef = euler_to_rotation(rx, ry, rz)
    rot = rot_ecef2enu(lat0, lon0)
    c2w = np.block([
        [rot @ R_ecef, np.array([x, y, z]).reshape(-1, 1)],
        [np.zeros((1, 3)), 1],
    ])
    return c2w


def haversine_distance(lat1: float, lon1: float,
                       lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in meters."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def latlon_to_meters_per_pixel(lat: float, zoom: int, tile_size: int = 256) -> float:
    """Meters per pixel at a given latitude and zoom level (Web Mercator)."""
    return 40075016.686 * math.cos(math.radians(lat)) / (2 ** zoom * tile_size)


def meters_to_latlon_offset(lat: float, dx_meters: float, dy_meters: float):
    """Convert meter offsets (east, north) to lat/lon offsets.

    Returns (dlat, dlon).
    """
    R = 6371000.0
    dlat = dy_meters / R * 180 / math.pi
    dlon = dx_meters / (R * math.cos(math.radians(lat))) * 180 / math.pi
    return dlat, dlon


def compute_tile_bounds(lat_center: float, lon_center: float,
                        size_meters: float):
    """Compute bounds of a size_meters x size_meters tile centered at GPS.

    Returns (lat_top, lon_left), (lat_bottom, lon_right).
    """
    half = size_meters / 2
    dlat, dlon = meters_to_latlon_offset(lat_center, half, half)
    return (lat_center + dlat, lon_center - dlon), (lat_center - dlat, lon_center + dlon)


def virtual_camera_altitude_fov(altitude_m: float, ground_size_m: float) -> float:
    """Compute FOV (degrees) from altitude and ground coverage.

    Used for the satellite virtual camera: altitude 5726 m, coverage 500 m
    -> FOV ~ 5 deg.
    """
    return 2 * math.degrees(math.atan(ground_size_m / 2 / altitude_m))
