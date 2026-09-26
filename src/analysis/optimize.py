"""
Generic maximal-covering-location MILP (mirrors the paper's eq. 4-6), used to
implement all four models by toggling two switches:

  use_eligibility  -> True  = "Modified Framework" (impervious/built-up constraint)
                       False = "Simple Distance"    (pure distance-based coverage)

  candidate_subset -> None            = national scope
                       province's ids = province-partitioned scope

Model 1 = solve_mclp(..., use_eligibility=True,  candidate_subset=None)      [national, modified]
Model 2 = one call per province with use_eligibility=True                   [province, modified]
Model 3 = one call per province with use_eligibility=False                  [province, simple]
Model 4 = solve_mclp(..., use_eligibility=False, candidate_subset=None)     [national, simple]
"""
from typing import Optional, Iterable
import pulp
import pandas as pd

from . import config as cfg


def solve_mclp(grid: pd.DataFrame, uncovered_pop: pd.DataFrame, N_j: dict,
                n_new: int, use_eligibility: bool = True,
                candidate_subset: Optional[Iterable[int]] = None,
                time_limit_sec: int = cfg.SOLVER_TIME_LIMIT_SEC):
    """
    Solves:
        maximize   sum_j w_j * z_j
        subject to sum_i x_i == n_new
                   z_j <= sum_{i in N_j} x_i      for every population point j
                   x_i == 0 for ineligible i      (only if use_eligibility)
                   x_i, z_j in {0, 1}

    Returns
    -------
    chosen_ids        : list of selected candidate_id values
    newly_covered_pop : float, total newly covered population
    status            : PuLP solver status string ("Optimal", "Infeasible", ...)
    """
    cand_df = grid if candidate_subset is None else grid[grid["candidate_id"].isin(candidate_subset)]
    candidate_ids = cand_df["candidate_id"].tolist()

    if n_new > len(candidate_ids):
        print(
            f"[WARNING] solve_mclp: requested n_new={n_new} but only "
            f"{len(candidate_ids)} candidate sites are available in this subset "
            f"-- capping to {len(candidate_ids)}. If this is unexpected, check "
            f"that province budgets were computed with the correct "
            f"max_per_province capacity for this eligibility setting."
        )
        n_new = len(candidate_ids)

    prob = pulp.LpProblem("mclp", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in candidate_ids}

    # Only build z_j / constraints for population points reachable from this
    # candidate subset -- keeps province-level sub-problems small.
    relevant_j = {j: [i for i in ids if i in x] for j, ids in N_j.items()}
    relevant_j = {j: ids for j, ids in relevant_j.items() if ids}
    z = {j: pulp.LpVariable(f"z_{j}", cat="Binary") for j in relevant_j}

    w = uncovered_pop["population"].to_dict()
    prob += pulp.lpSum(w[j] * z[j] for j in z)               # objective (eq. 4)
    prob += pulp.lpSum(x.values()) == n_new                  # station budget (eq. 5)

    for j, ids in relevant_j.items():
        prob += z[j] <= pulp.lpSum(x[i] for i in ids)        # coverage link

    if use_eligibility:
        elig = cand_df.set_index("candidate_id")["is_eligible"]
        for i in candidate_ids:
            if not bool(elig.get(i, False)):
                prob += x[i] == 0                            # impervious constraint

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_sec)
    prob.solve(solver)

    chosen_ids = [i for i, var in x.items() if pulp.value(var) is not None and pulp.value(var) > 0.5]
    newly_covered = sum(w[j] for j in z if pulp.value(z[j]) is not None and pulp.value(z[j]) > 0.5)
    return chosen_ids, newly_covered, pulp.LpStatus[prob.status]