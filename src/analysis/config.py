"""
Central configuration for the AQMN spatial-coverage optimization pipeline.

Edit the constants below to match your local project structure and to tune
the experiment parameters (station radius, number of new stations, grid size).
Anything marked TODO needs a one-time confirmation against your actual files
(open the .dbf / .csv once and check column names) before the pipeline is
trustworthy end-to-end.
"""
from pathlib import Path

# --- Project paths -----------------------------------------------------
# This file lives at <project_root>/src/analysis/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
GIS_DIR = DATA_DIR / "gis"

# Vector boundaries
ADMIN0_SHP = GIS_DIR / "phl_admin0.shp"
ADMIN2_SHP = GIS_DIR / "phl_admin2.shp"
ADMIN2_NAME_FIELD = "adm2_name"

# Population (1km density, ASCII XYZ -> loaded as CSV)
POP_DENSITY_CSV = GIS_DIR / "phl_pd_2020_1km_UNadj_ASCII_XYZ.csv"
POP_CSV_COLS = {"lon": "X", "lat": "Y", "density": "Z"}

# GHSL Built-up surface tiles (fraction built-up per cell, 3-arcsec ~ 100m, EPSG:4326)
GHSL_TILES = [
    GIS_DIR / "GHS_BUILT_S_E2020_GLOBE_R2023A_4326_3ss_V1_0_R7_C31.tif",
    GIS_DIR / "GHS_BUILT_S_E2020_GLOBE_R2023A_4326_3ss_V1_0_R8_C30.tif",
    GIS_DIR / "GHS_BUILT_S_E2020_GLOBE_R2023A_4326_3ss_V1_0_R8_C31.tif",
    GIS_DIR / "GHS_BUILT_S_E2020_GLOBE_R2023A_4326_3ss_V1_0_R9_C30.tif",
    GIS_DIR / "GHS_BUILT_S_E2020_GLOBE_R2023A_4326_3ss_V1_0_R9_C31.tif",
]

GHSL_MOSAIC_TIF = GIS_DIR / "ghsl_built_mosaic_phl.tif"  # cached merged output

# Stations
STATION_LIST_CSV = DATA_DIR / "station_list.csv"
STATION_LAT_COL = "lat"
STATION_LON_COL = "lon"

# Province-level population/area benchmark (for validation + min-station rule)
PROVINCE_STATS_CSV = DATA_DIR / "Population_LandArea_Density_Province.csv"

# Cached outputs
ABT_PARQUET = DATA_DIR / "gis_base_table.parquet"
RESULTS_DIR = DATA_DIR / "results"

# --- CRS -----------------------------------------------------------------
CRS_WGS84 = "EPSG:4326"
# PRS92 / Philippines Zone III (meters). Good enough for a national-scale grid;
# distances here are still computed via haversine (BallTree), this CRS is
# mainly used for area math and generating the regular candidate grid.
CRS_PROJECTED = "EPSG:3123"

# --- Model parameters (editable) -----------------------------------------
RADIUS_KM = 4.0                # station monitoring radius
CANDIDATE_CELL_KM = 2.0        # candidate grid resolution
N_NEW_STATIONS = 279           # total new stations to place
NO_OVERLAP_KM = 8.0            # min distance from an existing station (2x radius)
BUILTUP_THRESHOLD = 0.2        # min built-up fraction for a candidate to be "eligible"
MIN_STATIONS_PER_PROVINCE = 1  # floor used when splitting the budget across provinces
SOLVER_TIME_LIMIT_SEC = 300
