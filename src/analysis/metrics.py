"""Population-coverage metrics: PCR before/after each model, national summary."""
import pandas as pd


def pcr_before(pop_points: pd.DataFrame) -> float:
    total = pop_points["population"].sum()
    covered = pop_points.loc[pop_points["covered_existing"], "population"].sum()
    return covered / total


def pcr_after(pop_points: pd.DataFrame, newly_covered_population: float) -> float:
    total = pop_points["population"].sum()
    covered_before = pop_points.loc[pop_points["covered_existing"], "population"].sum()
    return (covered_before + newly_covered_population) / total


def summarize_models(results: dict, pop_points: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    results : {model_name: {"chosen_ids": [...], "newly_covered_pop": float}}

    Returns a tidy comparison table across all four models.
    """
    base_pcr = pcr_before(pop_points)
    rows = []
    for name, r in results.items():
        rows.append({
            "model": name,
            "n_stations_placed": len(r["chosen_ids"]),
            "newly_covered_population": r["newly_covered_pop"],
            "pcr_before": base_pcr,
            "pcr_after": pcr_after(pop_points, r["newly_covered_pop"]),
            "pcr_gain_pp": (pcr_after(pop_points, r["newly_covered_pop"]) - base_pcr) * 100,
        })
    return pd.DataFrame(rows).sort_values("pcr_after", ascending=False).reset_index(drop=True)
