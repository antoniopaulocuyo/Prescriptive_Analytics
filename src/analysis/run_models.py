"""
End-to-end runner: builds the ABT (candidate grid + province + eligibility),
builds the coverage relationships, runs Models 1-4, and writes a comparison
summary + per-model chosen-site tables + map visualizations to
data/results/n<N_NEW_STATIONS>/.

Run from the project root as:
    python -m src.analysis.run_models
"""
import time

import pandas as pd
import geopandas as gpd

from . import config as cfg
from . import io_utils
from . import grid as grid_mod
from . import coverage
from . import optimize
from . import allocation
from . import metrics
from . import visualize

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional -- falls back to a plain loop, no progress bar
    def tqdm(iterable, **kwargs):
        return iterable


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}min"


class _Stage:
    """Small context manager that prints how long a pipeline stage took."""
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        print(f"{self.label}...")
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            print(f"  done in {_fmt_elapsed(time.time() - self._t0)}")


def build_abt(force_rebuild: bool = False) -> gpd.GeoDataFrame:
    """
    Builds (or loads a cached) analytical base table: the candidate grid with
    province, built-up eligibility, and distance-to-existing-station attached.

    Caching: saved as GeoParquet (geometry + CRS preserved) at cfg.ABT_PARQUET
    after the first build. Later calls load straight from that file unless
    force_rebuild=True or the file doesn't exist.

    IMPORTANT: the cache bakes in CANDIDATE_CELL_KM and NO_OVERLAP_KM, since
    both affect which candidate points even exist in the saved grid. If you
    change either in config.py, delete data/analytical_base_table.parquet (or
    pass force_rebuild=True) so it rebuilds. BUILTUP_THRESHOLD is safe to
    change freely between runs -- eligibility is recomputed from the cached
    raw builtup_frac every time, cache or not.
    """
    provinces = io_utils.load_provinces()
    stations = io_utils.load_stations()

    if cfg.ABT_PARQUET.exists() and not force_rebuild:
        print(f"Loading cached ABT from {cfg.ABT_PARQUET} (use --rebuild-abt to force a rebuild)")
        grid = gpd.read_parquet(cfg.ABT_PARQUET)
        grid["is_eligible"] = grid["builtup_frac"] >= cfg.BUILTUP_THRESHOLD
        return grid, stations, provinces

    print("No cached ABT found (or rebuild forced) -- building candidate grid from scratch...")
    country = io_utils.load_country_boundary()
    io_utils.check_ghsl_covers_country(country)

    grid = grid_mod.make_candidate_grid(country)
    grid = grid_mod.attach_province(grid, provinces)
    grid = grid_mod.attach_builtup(grid)
    grid = coverage.drop_candidates_near_existing(grid, stations)

    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_parquet(cfg.ABT_PARQUET)  # GeoParquet -- keeps geometry + CRS, unlike a plain DataFrame
    return grid, stations, provinces


def prepare_population(stations: gpd.GeoDataFrame, provinces: gpd.GeoDataFrame):
    pop_points = io_utils.load_population_points()
    pop_points = coverage.mark_existing_coverage(pop_points, stations)
    pop_points = grid_mod.attach_province(pop_points, provinces)
    return pop_points


def run_model_1_national_modified(grid, uncovered_pop, N_j):
    chosen, new_pop, status = optimize.solve_mclp(
        grid, uncovered_pop, N_j, n_new=cfg.N_NEW_STATIONS, use_eligibility=True
    )
    print(f"[Model 1] status={status}, stations={len(chosen)}, newly_covered_pop={new_pop:,.0f}")
    return {"chosen_ids": chosen, "newly_covered_pop": new_pop}


def run_model_4_national_simple(grid, uncovered_pop, N_j):
    chosen, new_pop, status = optimize.solve_mclp(
        grid, uncovered_pop, N_j, n_new=cfg.N_NEW_STATIONS, use_eligibility=False
    )
    print(f"[Model 4] status={status}, stations={len(chosen)}, newly_covered_pop={new_pop:,.0f}")
    return {"chosen_ids": chosen, "newly_covered_pop": new_pop}


def run_model_by_province(grid, uncovered_pop, N_j, province_budgets: dict,
                           use_eligibility: bool, label: str):
    all_chosen, total_new_pop = [], 0.0
    for province, n_new in tqdm(province_budgets.items(), total=len(province_budgets),
                                 desc=f"[{label}] provinces", unit="province"):
        subset_ids = grid.loc[grid["province"] == province, "candidate_id"]
        if n_new <= 0 or subset_ids.empty:
            continue
        chosen, new_pop, status = optimize.solve_mclp(
            grid, uncovered_pop, N_j, n_new=n_new,
            use_eligibility=use_eligibility, candidate_subset=subset_ids,
        )
        if status != "Optimal":
            print(f"  [{label}] {province}: status={status} (n_new={n_new})")
        all_chosen.extend(chosen)
        total_new_pop += new_pop
    print(f"[{label}] stations={len(all_chosen)}, newly_covered_pop={total_new_pop:,.0f}")
    return {"chosen_ids": all_chosen, "newly_covered_pop": total_new_pop}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild-abt", action="store_true",
        help="Force rebuilding the candidate grid/ABT even if a cached parquet exists "
             "(needed after changing CANDIDATE_CELL_KM or NO_OVERLAP_KM in config.py).",
    )
    args = parser.parse_args()

    run_start = time.time()

    with _Stage("Building ABT (candidate grid, province tags, built-up eligibility)"):
        grid, stations, provinces = build_abt(force_rebuild=args.rebuild_abt)
        # Keep lon/lat as plain columns before dropping geometry -- otherwise
        # every downstream CSV (including the chosen-site tables) loses the
        # coordinates entirely, since geometry is what actually carries them.
        grid_df = pd.DataFrame(
            grid.assign(lon=grid.geometry.x, lat=grid.geometry.y).drop(columns="geometry")
        )

    with _Stage("Loading population points and marking existing coverage"):
        pop_points = prepare_population(stations, provinces)

    with _Stage("Building candidate<->population coverage sets"):
        uncovered_pop, N_j = coverage.build_coverage_sets(grid, pop_points)
        uncovered_pop_df = pd.DataFrame(uncovered_pop.drop(columns="geometry"))

    with _Stage("Computing province budgets"):
        # Model 2 (eligibility-filtered) and Model 3 (unfiltered) see different
        # candidate pools per province, so each needs its own capacity-aware
        # budget -- sharing one budget dict risks handing Model 2 more stations
        # than a province has eligible sites for.
        capacity_modified = allocation.compute_province_capacity(grid_df, use_eligibility=True)
        capacity_simple = allocation.compute_province_capacity(grid_df, use_eligibility=False)
        province_budgets_modified = allocation.compute_province_budgets(
            uncovered_pop_df, max_per_province=capacity_modified
        )
        province_budgets_simple = allocation.compute_province_budgets(
            uncovered_pop_df, max_per_province=capacity_simple
        )

    results = {}
    with _Stage("[Model 1] solving (national, modified framework)"):
        results["Model 1: National + Modified Framework"] = run_model_1_national_modified(
            grid_df, uncovered_pop_df, N_j
        )
    with _Stage("[Model 2] solving (per-province, modified framework)"):
        results["Model 2: Province + Modified Framework"] = run_model_by_province(
            grid_df, uncovered_pop_df, N_j, province_budgets_modified,
            use_eligibility=True, label="Model 2",
        )
    with _Stage("[Model 3] solving (per-province, simple distance)"):
        results["Model 3: Province + Simple Distance"] = run_model_by_province(
            grid_df, uncovered_pop_df, N_j, province_budgets_simple,
            use_eligibility=False, label="Model 3",
        )
    with _Stage("[Model 4] solving (national, simple distance)"):
        results["Model 4: National + Simple Distance"] = run_model_4_national_simple(
            grid_df, uncovered_pop_df, N_j
        )

    with _Stage("Computing MARR (Monitoring Area Repetition Rate) per model"):
        # MARR needs the FULL network (existing + this model's new sites), so
        # rebuild point geometry for the chosen candidates from the lon/lat
        # columns we preserved above and combine with the existing stations.
        marr_values = {}
        for name, r in results.items():
            chosen_df = grid_df[grid_df["candidate_id"].isin(r["chosen_ids"])]
            new_sites = gpd.GeoDataFrame(
                chosen_df,
                geometry=gpd.points_from_xy(chosen_df["lon"], chosen_df["lat"]),
                crs=cfg.CRS_WGS84,
            )
            all_stations_for_model = gpd.GeoDataFrame(
                pd.concat([stations[["geometry"]], new_sites[["geometry"]]], ignore_index=True),
                geometry="geometry", crs=cfg.CRS_WGS84,
            )
            marr_values[name] = metrics.compute_marr(pop_points, all_stations_for_model)

    summary = metrics.summarize_models(results, pop_points, marr_values=marr_values)

    # Results are namespaced by station count so re-running with a different
    # N_NEW_STATIONS never overwrites a previous run's output.
    run_tag = f"n{cfg.N_NEW_STATIONS}"
    run_results_dir = cfg.RESULTS_DIR / run_tag
    run_results_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(run_results_dir / f"model_comparison_{run_tag}.csv", index=False)

    with _Stage("Writing chosen-site tables and map visualizations"):
        country = io_utils.load_country_boundary()
        for name, r in results.items():
            chosen_df = grid_df[grid_df["candidate_id"].isin(r["chosen_ids"])]
            safe_name = name.split(":")[0].replace(" ", "_").lower()
            chosen_df.to_csv(run_results_dir / f"{safe_name}_{run_tag}_sites.csv", index=False)
            visualize.plot_model_sites(
                name, chosen_df, country, stations,
                out_path=run_results_dir / f"{safe_name}_{run_tag}_map.png",
            )
        visualize.plot_all_models(
            results, grid_df, country, stations,
            out_path=run_results_dir / f"all_models_{run_tag}_comparison.png",
        )

    print("\n=== Model comparison ===")
    print(summary.to_string(index=False))
    print(f"\nWritten to {run_results_dir}")
    print(f"Total runtime: {_fmt_elapsed(time.time() - run_start)}")


if __name__ == "__main__":
    main()