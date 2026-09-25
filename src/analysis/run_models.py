"""
End-to-end runner: builds the ABT (candidate grid + province + eligibility),
builds the coverage relationships, runs Models 1-4, and writes a comparison
summary + per-model chosen-site tables to data/results/.

Run from the project root as:
    python -m src.analysis.run_models
"""
import pandas as pd
import geopandas as gpd

from . import config as cfg
from . import io_utils
from . import grid as grid_mod
from . import coverage
from . import optimize
from . import allocation
from . import metrics


def build_abt() -> gpd.GeoDataFrame:
    """Builds (or loads a cached) analytical base table: the candidate grid
    with province, built-up eligibility, and distance-to-existing-station
    already attached. Returns a GeoDataFrame (geometry kept in memory; the
    parquet cache on disk drops geometry to stay lightweight -- see below)."""
    country = io_utils.load_country_boundary()
    provinces = io_utils.load_provinces()
    io_utils.check_ghsl_covers_country(country)
    stations = io_utils.load_stations()

    grid = grid_mod.make_candidate_grid(country)
    grid = grid_mod.attach_province(grid, provinces)
    grid = grid_mod.attach_builtup(grid)
    grid = coverage.drop_candidates_near_existing(grid, stations)

    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    grid.drop(columns="geometry").to_parquet(cfg.ABT_PARQUET)
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
    for province, n_new in province_budgets.items():
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
    print("Building ABT (candidate grid, province tags, built-up eligibility)...")
    grid, stations, provinces = build_abt()
    grid_df = pd.DataFrame(grid.drop(columns="geometry"))

    print("Loading population points and marking existing coverage...")
    pop_points = prepare_population(stations, provinces)

    print("Building candidate<->population coverage sets...")
    uncovered_pop, N_j = coverage.build_coverage_sets(grid, pop_points)
    uncovered_pop_df = pd.DataFrame(uncovered_pop.drop(columns="geometry"))

    province_budgets = allocation.compute_province_budgets(uncovered_pop_df)

    results = {}
    results["Model 1: National + Modified Framework"] = run_model_1_national_modified(
        grid_df, uncovered_pop_df, N_j
    )
    results["Model 2: Province + Modified Framework"] = run_model_by_province(
        grid_df, uncovered_pop_df, N_j, province_budgets,
        use_eligibility=True, label="Model 2",
    )
    results["Model 3: Province + Simple Distance"] = run_model_by_province(
        grid_df, uncovered_pop_df, N_j, province_budgets,
        use_eligibility=False, label="Model 3",
    )
    results["Model 4: National + Simple Distance"] = run_model_4_national_simple(
        grid_df, uncovered_pop_df, N_j
    )

    summary = metrics.summarize_models(results, pop_points)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(cfg.RESULTS_DIR / "model_comparison.csv", index=False)

    for name, r in results.items():
        chosen_df = grid_df[grid_df["candidate_id"].isin(r["chosen_ids"])]
        safe_name = name.split(":")[0].replace(" ", "_").lower()
        chosen_df.to_csv(cfg.RESULTS_DIR / f"{safe_name}_sites.csv", index=False)

    print("\n=== Model comparison ===")
    print(summary.to_string(index=False))
    print(f"\nWritten to {cfg.RESULTS_DIR}")


if __name__ == "__main__":
    main()
