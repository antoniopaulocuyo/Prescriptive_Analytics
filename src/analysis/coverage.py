"""
Builds spatial coverage relationships:
  - which population points the CURRENT network already covers
  - which candidate sites are too close to an existing station (no-overlap rule)
  - which candidate sites could cover which (currently-uncovered) population points

All distance search uses a BallTree with the haversine metric on (lat, lon) in
radians, which is exact on a sphere and far faster than looping pairwise
distances for national-scale point counts.
"""
import numpy as np
import geopandas as gpd
from sklearn.neighbors import BallTree

from . import config as cfg

EARTH_RADIUS_KM = 6371.0088  # mean Earth radius (IUGG)


def _to_radians(gdf: gpd.GeoDataFrame) -> np.ndarray:
    lat = gdf.geometry.y.to_numpy()
    lon = gdf.geometry.x.to_numpy()
    return np.radians(np.column_stack([lat, lon]))


def mark_existing_coverage(pop_points: gpd.GeoDataFrame, stations: gpd.GeoDataFrame,
                            radius_km: float = cfg.RADIUS_KM) -> gpd.GeoDataFrame:
    """Flags each population point as covered (True) or not by the CURRENT network."""
    tree = BallTree(_to_radians(stations), metric="haversine")
    idx = tree.query_radius(_to_radians(pop_points), r=radius_km / EARTH_RADIUS_KM)
    pop_points = pop_points.copy()
    pop_points["covered_existing"] = [len(i) > 0 for i in idx]
    return pop_points


def drop_candidates_near_existing(grid: gpd.GeoDataFrame, stations: gpd.GeoDataFrame,
                                   min_dist_km: float = cfg.NO_OVERLAP_KM) -> gpd.GeoDataFrame:
    """Removes candidate sites within `min_dist_km` of any existing station."""
    tree = BallTree(_to_radians(stations), metric="haversine")
    dist, _ = tree.query(_to_radians(grid), k=1)
    dist_km = dist.ravel() * EARTH_RADIUS_KM
    grid = grid.copy()
    grid["dist_to_existing_km"] = dist_km
    return grid[dist_km >= min_dist_km].reset_index(drop=True)


def build_coverage_sets(grid: gpd.GeoDataFrame, pop_points: gpd.GeoDataFrame,
                         radius_km: float = cfg.RADIUS_KM):
    """
    Returns:
        uncovered : the subset of pop_points NOT already covered by the
                    existing network (reset to a fresh 0..n-1 index -- this
                    index is what N_j's keys refer to)
        N_j       : dict {row position in `uncovered` -> [candidate_ids within
                    radius_km]}. Only currently-uncovered population points are
                    included, since the optimization objective is *newly*
                    covered population, matching the paper's formulation.
    """
    uncovered = pop_points[~pop_points["covered_existing"]].reset_index(drop=True)
    tree = BallTree(_to_radians(grid), metric="haversine")
    idx_lists = tree.query_radius(_to_radians(uncovered), r=radius_km / EARTH_RADIUS_KM)

    candidate_ids = grid["candidate_id"].to_numpy()
    N_j = {j: candidate_ids[idxs].tolist() for j, idxs in enumerate(idx_lists) if len(idxs) > 0}
    return uncovered, N_j
