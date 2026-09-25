"""Loaders for all raw inputs: boundaries, population, built-up raster, stations."""
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.merge import merge

from . import config as cfg


def load_country_boundary() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(cfg.ADMIN0_SHP)
    return gdf.to_crs(cfg.CRS_WGS84)


def load_provinces() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(cfg.ADMIN2_SHP).to_crs(cfg.CRS_WGS84)
    if cfg.ADMIN2_NAME_FIELD not in gdf.columns:
        raise KeyError(
            f"'{cfg.ADMIN2_NAME_FIELD}' not found in phl_admin2 attributes.\n"
            f"Available columns: {list(gdf.columns)}\n"
            f"Update ADMIN2_NAME_FIELD in config.py to the correct one."
        )
    return gdf


def load_population_points() -> gpd.GeoDataFrame:
    """
    Loads the 1km population-density ASCII XYZ file and converts density -> counts.

    Returns a GeoDataFrame of points, one per ~1km cell, with columns:
        lon, lat, density, cell_area_km2, population, geometry
    """
    df = pd.read_csv(cfg.POP_DENSITY_CSV)
    lon_col = cfg.POP_CSV_COLS["lon"]
    lat_col = cfg.POP_CSV_COLS["lat"]
    dens_col = cfg.POP_CSV_COLS["density"]
    missing = [c for c in (lon_col, lat_col, dens_col) if c not in df.columns]
    if missing:
        raise KeyError(
            f"Column(s) {missing} not found in {cfg.POP_DENSITY_CSV.name}.\n"
            f"Available columns: {list(df.columns)}\n"
            f"Update POP_CSV_COLS in config.py."
        )

    df = df.rename(columns={lon_col: "lon", lat_col: "lat", dens_col: "density"})
    df = df[df["density"] > 0].reset_index(drop=True)  # drop nodata / ocean cells

    # Per-row cell area in km^2. A "1km" lon/lat grid has cells that are ~1km in
    # the north-south direction everywhere, but shrink east-west as |lat| grows,
    # so area is NOT a flat 1 km^2 across the whole archipelago.
    lat_rad = np.radians(df["lat"].to_numpy())
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(lat_rad)
    cell_deg = 1.0 / km_per_deg_lat  # ~1km spacing expressed in degrees latitude
    df["cell_area_km2"] = (cell_deg * km_per_deg_lat) * (cell_deg * km_per_deg_lon)

    df["population"] = df["density"] * df["cell_area_km2"]

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=cfg.CRS_WGS84
    )
    return gdf


def build_ghsl_mosaic(force: bool = False) -> None:
    """Merges the GHSL built-up tiles once and caches the result to disk."""
    if cfg.GHSL_MOSAIC_TIF.exists() and not force:
        return
    srcs = [rasterio.open(p) for p in cfg.GHSL_TILES]
    mosaic, transform = merge(srcs)
    meta = srcs[0].meta.copy()
    meta.update(driver="GTiff", height=mosaic.shape[1], width=mosaic.shape[2], transform=transform)
    with rasterio.open(cfg.GHSL_MOSAIC_TIF, "w", **meta) as dst:
        dst.write(mosaic)
    for s in srcs:
        s.close()


def check_ghsl_covers_country(country: gpd.GeoDataFrame) -> bool:
    """Sanity check: does the merged GHSL mosaic bounding box contain the PH boundary?"""
    build_ghsl_mosaic()
    with rasterio.open(cfg.GHSL_MOSAIC_TIF) as src:
        left, bottom, right, top = src.bounds
    minx, miny, maxx, maxy = country.total_bounds
    covers = (left <= minx) and (bottom <= miny) and (right >= maxx) and (top >= maxy)
    if not covers:
        print(
            "[WARNING] GHSL mosaic does not fully cover the PH boundary.\n"
            f"  mosaic bounds : {(left, bottom, right, top)}\n"
            f"  country bounds: {(minx, miny, maxx, maxy)}\n"
            "  Candidate cells outside the mosaic will have no built-up value "
            "and will be marked ineligible by default -- check for a missing "
            "GHSL tile (e.g. Batanes, or Sulu/Tawi-Tawi)."
        )
    return covers


def load_stations() -> gpd.GeoDataFrame:
    df = pd.read_csv(cfg.STATION_LIST_CSV)
    missing = [c for c in (cfg.STATION_LON_COL, cfg.STATION_LAT_COL) if c not in df.columns]
    if missing:
        raise KeyError(
            f"Column(s) {missing} not found in {cfg.STATION_LIST_CSV.name}.\n"
            f"Available columns: {list(df.columns)}\n"
            f"Update STATION_LAT_COL / STATION_LON_COL in config.py."
        )
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[cfg.STATION_LON_COL], df[cfg.STATION_LAT_COL]),
        crs=cfg.CRS_WGS84,
    )
    return gdf
