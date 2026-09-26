"""Splits a national station budget across provinces, proportional to each
province's share of currently-uncovered population -- this is what makes
Models 2 and 3 'equitable/equal coverage' rather than a single national pot."""
from typing import Optional

import pandas as pd

from . import config as cfg


def compute_province_capacity(grid: pd.DataFrame, use_eligibility: bool,
                               province_col: str = "province") -> dict:
    """
    Counts available candidate sites per province, matching the same scoping
    a province-level optimize.solve_mclp() call would see: eligible-only sites
    when use_eligibility=True (Model 2), all sites when False (Model 3).

    Pass the result as max_per_province to compute_province_budgets so a
    province is never assigned more new stations than it has room for.
    """
    df = grid[grid["is_eligible"]] if use_eligibility else grid
    return df.groupby(province_col)["candidate_id"].nunique().to_dict()


def compute_province_budgets(uncovered_pop_with_province: pd.DataFrame,
                              total_new_stations: int = cfg.N_NEW_STATIONS,
                              min_per_province: int = cfg.MIN_STATIONS_PER_PROVINCE,
                              max_per_province: Optional[dict] = None) -> dict:
    """
    Parameters
    ----------
    uncovered_pop_with_province : DataFrame with columns ["province", "population"],
        one row per currently-uncovered population point.
    max_per_province : optional dict {province_name: max_n_new_stations}, e.g. the
        count of available (eligible) candidate sites in that province. Pass this
        whenever the budget will be handed to optimize.solve_mclp for a scoped
        subset, so a province is never assigned more stations than it can place.
        Provinces absent from this dict are treated as uncapped.

    Returns
    -------
    dict {province_name: n_new_stations}, summing exactly to total_new_stations.

    Raises
    ------
    ValueError
        If total_new_stations can't be reconciled with min_per_province (too few
        stations for the number of provinces) or with max_per_province (too many
        stations for the available capacity). Both are checked up front so the
        function fails fast with a clear message instead of spinning forever in
        the rounding-adjustment loop below.
    """
    prov_pop = uncovered_pop_with_province.groupby("province")["population"].sum()
    provinces = prov_pop.index.tolist()
    n_provinces = len(provinces)

    floor_total = min_per_province * n_provinces
    if total_new_stations < floor_total:
        raise ValueError(
            f"total_new_stations ({total_new_stations}) is less than "
            f"min_per_province ({min_per_province}) x n_provinces ({n_provinces}) "
            f"= {floor_total}. Raise N_NEW_STATIONS, lower MIN_STATIONS_PER_PROVINCE, "
            f"or reduce the number of provinces in scope."
        )

    if max_per_province is not None:
        total_capacity = sum(max_per_province.get(p, float("inf")) for p in provinces)
        if total_new_stations > total_capacity:
            raise ValueError(
                f"total_new_stations ({total_new_stations}) exceeds total candidate "
                f"capacity across provinces ({total_capacity:,.0f}). Lower N_NEW_STATIONS "
                f"or widen the candidate grid (e.g. smaller CANDIDATE_CELL_KM)."
            )
        capacity_floor = sum(min(max_per_province.get(p, float("inf")), min_per_province)
                              for p in provinces)
        if capacity_floor < floor_total:
            # A province's own capacity is below min_per_province -- min_per_province
            # itself is infeasible for it, independent of total_new_stations.
            short = [p for p in provinces
                     if max_per_province.get(p, float("inf")) < min_per_province]
            raise ValueError(
                f"min_per_province ({min_per_province}) exceeds available candidate "
                f"capacity in: {short}. Lower MIN_STATIONS_PER_PROVINCE or expand the "
                f"candidate grid for those provinces."
            )

    shares = prov_pop / prov_pop.sum()

    budgets = (shares * total_new_stations).round().astype(int)
    budgets = budgets.clip(lower=min_per_province)

    if max_per_province is not None:
        for p in provinces:
            cap = max_per_province.get(p)
            if cap is not None:
                budgets[p] = min(budgets[p], cap)

    # Rounding (and capping) can drift the total off target; nudge the
    # largest-share provinces up/down by 1 until it matches exactly. Bounded
    # by min_per_province on the way down and max_per_province on the way up,
    # both already validated as feasible above, so this always terminates.
    diff = total_new_stations - budgets.sum()
    if diff != 0:
        order = shares.sort_values(ascending=False).index.tolist()
        i = 0
        stall_guard = 0
        max_stalls = len(order) + 1  # one full pass with zero progress means stuck
        while diff != 0:
            prov = order[i % len(order)]
            step = 1 if diff > 0 else -1

            blocked = False
            if step < 0 and budgets[prov] <= min_per_province:
                blocked = True
            if step > 0 and max_per_province is not None:
                cap = max_per_province.get(prov)
                if cap is not None and budgets[prov] >= cap:
                    blocked = True

            if blocked:
                i += 1
                stall_guard += 1
                if stall_guard > max_stalls:
                    # Should be unreachable given the feasibility checks above;
                    # fail loudly rather than loop forever if it ever is.
                    raise RuntimeError(
                        "compute_province_budgets: could not converge on a feasible "
                        "allocation despite passing feasibility checks -- this is a bug, "
                        "please report the inputs that triggered it."
                    )
                continue

            budgets[prov] += step
            diff -= step
            i += 1
            stall_guard = 0

    return budgets.to_dict()