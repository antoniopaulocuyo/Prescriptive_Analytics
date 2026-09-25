"""
Run this BEFORE run_models.py to resolve the three TODOs in config.py.
It only reads headers/previews -- doesn't build anything, doesn't need the
GHSL mosaic, and takes a few seconds.

Run from the project root as:
    python -m src.analysis.check_inputs
"""
import geopandas as gpd
import pandas as pd

from . import config as cfg


def check_admin2():
    print("=" * 70)
    print(f"phl_admin2.shp  ->  {cfg.ADMIN2_SHP}")
    gdf = gpd.read_file(cfg.ADMIN2_SHP)
    print(f"  rows: {len(gdf)}")
    print(f"  columns: {list(gdf.columns)}")
    print(f"  CRS: {gdf.crs}")
    print("  first 3 rows (non-geometry columns):")
    print(gdf.drop(columns="geometry").head(3).to_string())
    print(f"\n  ==> current config.ADMIN2_NAME_FIELD = '{cfg.ADMIN2_NAME_FIELD}' "
          f"{'FOUND' if cfg.ADMIN2_NAME_FIELD in gdf.columns else '*** NOT FOUND ***'}")


def check_population_csv():
    print("=" * 70)
    print(f"population CSV  ->  {cfg.POP_DENSITY_CSV}")
    df = pd.read_csv(cfg.POP_DENSITY_CSV, nrows=5)
    print(f"  columns: {list(df.columns)}")
    print("  first 5 rows:")
    print(df.to_string())
    for key, col in cfg.POP_CSV_COLS.items():
        status = "FOUND" if col in df.columns else "*** NOT FOUND ***"
        print(f"  ==> current config.POP_CSV_COLS['{key}'] = '{col}'  {status}")


def check_stations():
    print("=" * 70)
    print(f"station list  ->  {cfg.STATION_LIST_CSV}")
    df = pd.read_csv(cfg.STATION_LIST_CSV, nrows=5)
    print(f"  columns: {list(df.columns)}")
    print("  first 5 rows:")
    print(df.to_string())
    for key, col in [("STATION_LAT_COL", cfg.STATION_LAT_COL), ("STATION_LON_COL", cfg.STATION_LON_COL)]:
        status = "FOUND" if col in df.columns else "*** NOT FOUND ***"
        print(f"  ==> current config.{key} = '{col}'  {status}")


def check_admin0():
    print("=" * 70)
    print(f"phl_admin0.shp  ->  {cfg.ADMIN0_SHP}")
    gdf = gpd.read_file(cfg.ADMIN0_SHP)
    print(f"  rows: {len(gdf)}, CRS: {gdf.crs}")
    print(f"  bounds: {gdf.total_bounds}")


def check_ghsl_tiles():
    print("=" * 70)
    print("GHSL built-up tiles:")
    import rasterio
    for p in cfg.GHSL_TILES:
        exists = p.exists()
        print(f"  {p.name}  exists={exists}")
        if exists:
            with rasterio.open(p) as src:
                print(f"    bounds={src.bounds}, crs={src.crs}, shape={src.shape}")


if __name__ == "__main__":
    check_admin0()
    check_admin2()
    check_population_csv()
    check_stations()
    check_ghsl_tiles()
    print("=" * 70)
    print("Done. Fix any '*** NOT FOUND ***' lines above by editing the "
          "matching constant in src/analysis/config.py, then re-run this "
          "script to confirm before moving on to run_models.py.")
