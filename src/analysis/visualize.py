"""
Static map visualizations of chosen new monitoring sites vs the existing
network, one per model plus a combined comparison grid. Pure matplotlib +
geopandas -- runs headless (no display needed), saves PNGs to disk.
"""
import matplotlib
matplotlib.use("Agg")  # headless-safe backend; must be set before pyplot import
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd

from . import config as cfg


def _site_points(chosen_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Builds a point GeoDataFrame from a chosen-sites table's lon/lat columns."""
    return gpd.GeoDataFrame(
        chosen_df,
        geometry=gpd.points_from_xy(chosen_df["lon"], chosen_df["lat"]),
        crs=cfg.CRS_WGS84,
    )


def plot_model_sites(model_name: str, chosen_df: pd.DataFrame,
                      country: gpd.GeoDataFrame, stations: gpd.GeoDataFrame,
                      out_path) -> None:
    """Saves one PNG: country outline + existing stations + this model's new sites."""
    sites = _site_points(chosen_df)

    fig, ax = plt.subplots(figsize=(8, 10))
    country.boundary.plot(ax=ax, color="black", linewidth=0.5)
    stations.plot(ax=ax, color="gray", markersize=6, label=f"Existing ({len(stations)})")
    sites.plot(ax=ax, color="crimson", markersize=8, label=f"New ({len(sites)})")
    ax.set_title(model_name, fontsize=11)
    ax.set_axis_off()
    ax.legend(loc="lower left", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_all_models(results: dict, grid_df: pd.DataFrame, country: gpd.GeoDataFrame,
                     stations: gpd.GeoDataFrame, out_path) -> None:
    """Saves one PNG with a 2x2 panel comparing every model's chosen sites side by side."""
    n = len(results)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 8 * nrows))
    axes = axes.ravel() if n > 1 else [axes]

    for ax, (name, r) in zip(axes, results.items()):
        chosen_df = grid_df[grid_df["candidate_id"].isin(r["chosen_ids"])]
        sites = _site_points(chosen_df)
        country.boundary.plot(ax=ax, color="black", linewidth=0.4)
        stations.plot(ax=ax, color="gray", markersize=3)
        sites.plot(ax=ax, color="crimson", markersize=4)
        ax.set_title(f"{name}\n({len(sites)} new stations)", fontsize=9)
        ax.set_axis_off()

    for ax in axes[n:]:  # hide any unused panels if results count is odd
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)