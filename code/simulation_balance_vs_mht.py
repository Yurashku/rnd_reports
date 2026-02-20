"""Pure-Python simulation for balance checks under multiple testing and rerandomization rules."""

from __future__ import annotations

import argparse
import csv
import math
import random
from statistics import mean, variance


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def welch_pvalue(a: list[float], b: list[float]) -> float:
    """Approximate two-sided p-value using normal approximation."""
    ma, mb = mean(a), mean(b)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 1.0
    va, vb = variance(a), variance(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return 1.0
    z = abs((ma - mb) / se)
    return 2.0 * (1.0 - normal_cdf(z))


def compute_smd(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    va, vb = variance(a), variance(b)
    pooled_sd = math.sqrt((va + vb) / 2.0)
    if pooled_sd == 0:
        return 0.0
    return (mean(a) - mean(b)) / pooled_sd


def simulate_once(n: int, n_features: int, alpha: float, smd_thr: float) -> tuple[bool, bool, bool, bool]:
    x = [[random.gauss(0, 1) for _ in range(n_features)] for _ in range(n)]
    grp = [random.randint(0, 1) for _ in range(n)]

    pvals = []
    smds = []
    for j in range(n_features):
        a = [x[i][j] for i in range(n) if grp[i] == 0]
        b = [x[i][j] for i in range(n) if grp[i] == 1]
        pvals.append(welch_pvalue(a, b))
        smds.append(abs(compute_smd(a, b)))

    accept_raw = all(p >= alpha for p in pvals)
    accept_bonf = all(p >= alpha / n_features for p in pvals)
    accept_smd = max(smds) <= smd_thr
    any_sig = any(p < alpha for p in pvals)
    return accept_raw, accept_bonf, accept_smd, any_sig


def run_simulation(n_iter: int, n: int, feature_grid: list[int], alpha: float, smd_thr: float, seed: int) -> list[dict[str, float]]:
    random.seed(seed)
    rows = []
    for n_features in feature_grid:
        acc_raw = acc_bonf = acc_smd = any_sig = 0
        for _ in range(n_iter):
            r_raw, r_bonf, r_smd, r_any = simulate_once(n, n_features, alpha, smd_thr)
            acc_raw += int(r_raw)
            acc_bonf += int(r_bonf)
            acc_smd += int(r_smd)
            any_sig += int(r_any)

        rows.append(
            {
                "n_features": n_features,
                "accept_raw_p": acc_raw / n_iter,
                "accept_bonferroni": acc_bonf / n_iter,
                "accept_smd": acc_smd / n_iter,
                "p_any_sig": any_sig / n_iter,
            }
        )
    return rows


def save_csv(rows: list[dict[str, float]], output: str) -> None:
    fieldnames = ["n_features", "accept_raw_p", "accept_bonferroni", "accept_smd", "p_any_sig"]
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulate acceptance rates for rerandomization rules.")
    p.add_argument("--n-iter", type=int, default=500)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--feature-grid", type=str, default="5,10,20,50")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--smd-thr", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="report/simulation_results.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    grid = [int(x.strip()) for x in args.feature_grid.split(",") if x.strip()]
    rows = run_simulation(args.n_iter, args.n, grid, args.alpha, args.smd_thr, args.seed)
    save_csv(rows, args.output)

    for row in rows:
        print(row)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
