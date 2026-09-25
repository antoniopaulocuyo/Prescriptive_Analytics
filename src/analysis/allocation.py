"""Splits a national station budget across provinces, proportional to each
province's share of currently-uncovered population -- this is what makes
Models 2 and 3 'equitable/equal coverage' rather than a single national pot."""
import pandas as pd

from . import config as cfg


def compute_province_budgets(uncovered_pop_with_province: pd.DataFrame,
                              total_new_stations: int = cfg.N_NEW_STATIONS,
                              min_per_province: int = cfg.MIN_STATIONS_PER_PROVINCE) -> dict:
    """
    Parameters
    ----------
    uncovered_pop_with_province : DataFrame with columns ["province", "population"],
        one row per currently-uncovered population point.

    Returns
    -------
    dict {province_name: n_new_stations}, summing exactly to total_new_stations.
    """
    prov_pop = uncovered_pop_with_province.groupby("province")["population"].sum()
    shares = prov_pop / prov_pop.sum()

    budgets = (shares * total_new_stations).round().astype(int)
    budgets = budgets.clip(lower=min_per_province)

    # Rounding can drift the total off target; nudge the largest-share
    # provinces up/down by 1 until it matches exactly.
    diff = total_new_stations - budgets.sum()
    if diff != 0:
        order = shares.sort_values(ascending=False).index.tolist()
        i = 0
        while diff != 0:
            prov = order[i % len(order)]
            step = 1 if diff > 0 else -1
            if step < 0 and budgets[prov] <= min_per_province:
                i += 1
                continue
            budgets[prov] += step
            diff -= step
            i += 1

    return budgets.to_dict()
