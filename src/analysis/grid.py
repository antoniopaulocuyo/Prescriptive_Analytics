"""Builds the candidate monitoring-site grid and attaches province + built-up attributes."""
import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import Point

from . import config as cfg


def make_candidate_grid(country: gpd.GeoDataFrame, cell_km: float = cfg.CANDIDATE_CELL_KM) -> gpd.GeoDataFrame:
    """
    Generates a regular grid of candidate points at `cell_km` spacing, clipped
    to points falling within the country boundary.
    """
    country_proj = country.to_crs(cfg.CRS_PROJECTED)
    minx, miny, maxx, maxy = country_proj.total_bounds
    step_m = cell_km * 1000

    xs = np.arange(minx, maxx, step_m)
    ys = np.arange(miny, maxy, step_m)
    xx, yy = np.meshgrid(xs, ys)
    pts = gpd.GeoSeries([Point(x, y) for x, y in zip(xx.ravel(), yy.ravel())], crs=cfg.CRS_PROJECTED)

    grid = gpd.GeoDataFrame(geometry=pts)
    union = country_proj.geometry.union_all()
    grid = grid[grid.within(union)].reset_index(drop=True)
    grid["candidate_id"] = grid.index

    return grid.to_crs(cfg.CRS_WGS84)


def attach_province(points: gpd.GeoDataFrame, provinces: gpd.GeoDataFrame,
                     province_col: str = "province") -> gpd.GeoDataFrame:
    """
    Generic spatial join: tags any point GeoDataFrame (candidate grid OR
    population points) with the province it falls in. Points that don't land
    exactly inside a polygon (e.g. small islands, coastline snapping issues)
    are matched to their nearest province as a fallback.
    """
    joined = gpd.sjoin(points, provinces[[cfg.ADMIN2_NAME_FIELD, "geometry"]],
                        how="left", predicate="within")
    joined = joined.rename(columns={cfg.ADMIN2_NAME_FIELD: province_col}).drop(columns=["index_right"])

    missing = joined[province_col].isna()
    if missing.any():
        nearest = gpd.sjoin_nearest(
            points.loc[missing, ["geometry"]],
            provinces[[cfg.ADMIN2_NAME_FIELD, "geometry"]],
            how="left",
        )
        joined.loc[missing, province_col] = nearest[cfg.ADMIN2_NAME_FIELD].to_numpy()

    return joined


def attach_builtup(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Samples the GHSL built-up fraction at each candidate point location."""
    with rasterio.open(cfg.GHSL_MOSAIC_TIF) as src:
        coords = [(p.x, p.y) for p in grid.geometry]
        values = np.array([v[0] for v in src.sample(coords)], dtype=float)

    grid = grid.copy()
    grid["builtup_frac"] = values
    grid["is_eligible"] = grid["builtup_frac"] >= cfg.BUILTUP_THRESHOLD
    return grid
