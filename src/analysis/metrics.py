"""Population-coverage metrics: PCR before/after each model, MARR, national summary."""
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import BallTree

from . import coverage as coverage_mod
from . import config as cfg


def pcr_before(pop_points: pd.DataFrame) -> float:
    total = pop_points["population"].sum()
    covered = pop_points.loc[pop_points["covered_existing"], "population"].sum()
    return covered / total


def pcr_after(pop_points: pd.DataFrame, newly_covered_population: float) -> float:
    total = pop_points["population"].sum()
    covered_before = pop_points.loc[pop_points["covered_existing"], "population"].sum()
    return (covered_before + newly_covered_population) / total


def compute_marr(pop_points: gpd.GeoDataFrame, all_stations: gpd.GeoDataFrame,
                  radius_km: float = cfg.RADIUS_KM) -> float:
    """
    Monitoring Area Repetition Rate (MARR): population-weighted average number
    of REDUNDANT stations covering an already-covered population point, using
    the full network (existing + newly placed stations for this model).

        MARR = sum_p[ pop_p * (n_stations_covering_p - 1) ] / sum_p[ pop_p ]
               over points p covered by >= 1 station

    0.0 -> every covered point is served by exactly one station (no overlap)
    1.0 -> covered points are served by two stations on average
    etc.

    WORKING DEFINITION -- MARR is not otherwise specified in this codebase.
    This mirrors pcr_after's convention of scoring the combined existing+new
    network, population-weighted like the rest of metrics.py. Confirm against
    your source formula (e.g. the paper optimize.py's MCLP docstring
    references) and adjust this function if the definition differs --
    everything that calls it only depends on the float it returns.

    Parameters
    ----------
    pop_points   : full population GeoDataFrame (not just uncovered), with a
                   "population" column and point geometry in CRS_WGS84.
    all_stations : GeoDataFrame of every station in the network being scored
                   (existing stations + this model's newly chosen sites),
                   point geometry in CRS_WGS84.
    """
    if len(all_stations) == 0 or len(pop_points) == 0:
        return 0.0

    tree = BallTree(coverage_mod._to_radians(all_stations), metric="haversine")
    idx_lists = tree.query_radius(
        coverage_mod._to_radians(pop_points), r=radius_km / coverage_mod.EARTH_RADIUS_KM
    )
    counts = np.array([len(i) for i in idx_lists])

    covered_mask = counts >= 1
    pop = pop_points["population"].to_numpy()
    covered_pop = pop[covered_mask]
    covered_counts = counts[covered_mask]

    total_covered_pop = covered_pop.sum()
    if total_covered_pop == 0:
        return 0.0

    return float(np.sum(covered_pop * (covered_counts - 1)) / total_covered_pop)


def summarize_models(results: dict, pop_points: pd.DataFrame,
                      marr_values: Optional[dict] = None) -> pd.DataFrame:
    """
    Parameters
    ----------
    results     : {model_name: {"chosen_ids": [...], "newly_covered_pop": float}}
    marr_values : optional {model_name: float}, from compute_marr() per model.
                  Omit to leave "marr" out of the summary table.

    Returns a tidy comparison table across all four models.
    """
    base_pcr = pcr_before(pop_points)
    rows = []
    for name, r in results.items():
        row = {
            "model": name,
            "n_stations_placed": len(r["chosen_ids"]),
            "newly_covered_population": r["newly_covered_pop"],
            "pcr_before": base_pcr,
            "pcr_after": pcr_after(pop_points, r["newly_covered_pop"]),
            "pcr_gain_pp": (pcr_after(pop_points, r["newly_covered_pop"]) - base_pcr) * 100,
        }
        if marr_values is not None:
            row["marr"] = marr_values.get(name)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("pcr_after", ascending=False).reset_index(drop=True)