"""Figure 1: cost gap of the Sinkhorn plan against epsilon (DECISIONS.md D4).

For each run we record two gaps, both in units of median(C):

- true gap   = <P, C> - LP optimum, using the certified exact LP (exact.py);
- bound gap  = <P, C> - lp_dual_bound(f, g), the dual lower bound from the
  solver's own potentials (metrics.py). The bound is never above the LP
  optimum, so the bound gap is a guaranteed upper bound on the overpay that
  needs no exact solve.

Only converged runs enter the gap curves: a capped plan is off its marginals,
so its cost can sit below the LP (implementation.md, plotting traps). The
strip under each panel shows the fraction of seeds that converged (author-
approved option B). Every run is in the CSV.

Usage, from the repo root: python experiments/fig1_cost_gap.py [run|plot]
"""

import sys

from _common import FIG_DIR, INK_2, SERIES, ResumableCSV, single_thread, style

single_thread()

import numpy as np

from sinkhorn_lab.exact import solve_exact
from sinkhorn_lab.metrics import lp_dual_bound
from sinkhorn_lab.problems import gaussian_clouds, random_cost
from sinkhorn_lab.solvers import log_sinkhorn, vanilla_sinkhorn

NAME = "fig1_cost_gap"
CSV = FIG_DIR / f"{NAME}.csv"
PNG = FIG_DIR / f"{NAME}.png"

N = 100
SEEDS = range(5)
EPS = np.logspace(0, -2, 9)          # 1 ... 0.01, in units of median(C)
NORMALIZE = "median"
SEPARATION = 0.0                     # gaussian_clouds only
TOL = 1e-9
MAX_ITER = 100_000
FAMILIES = {
    "gaussian_clouds": lambda seed: gaussian_clouds(N, N, separation=SEPARATION, normalize=NORMALIZE, seed=seed),
    "random_cost": lambda seed: random_cost(N, N, normalize=NORMALIZE, seed=seed),
}
SOLVERS = {"vanilla": vanilla_sinkhorn, "log": log_sinkhorn}

COLUMNS = ["family", "n", "seed", "eps", "normalize", "c_scale", "solver", "stabilize", "tol", "max_iter",
           "status", "n_iter", "err", "cost", "lp_cost", "dual_bound", "true_gap", "bound_gap"]
KEY = ["family", "seed", "eps", "solver"]


def run(csv_path=CSV):
    out = ResumableCSV(csv_path, COLUMNS, KEY)
    for family, make in FAMILIES.items():
        for seed in SEEDS:
            todo = [(e, s) for e in EPS for s in SOLVERS
                    if not out.has(family=family, seed=seed, eps=repr(float(e)), solver=s)]
            if not todo:
                continue
            p = make(seed)
            lp = solve_exact(p).cost
            for eps, sname in todo:
                r = SOLVERS[sname](p.a, p.b, p.C, eps, tol=TOL, max_iter=MAX_ITER)
                finite = bool(np.all(np.isfinite(r.f)) and np.all(np.isfinite(r.g)))
                bound = lp_dual_bound(r.f, r.g, p.a, p.b, p.C) if finite else np.nan
                out.write({
                    "family": family, "n": N, "seed": seed, "eps": repr(float(eps)), "normalize": NORMALIZE,
                    "c_scale": p.meta["c_scale"], "solver": sname, "stabilize": "rowcol", "tol": TOL,
                    "max_iter": MAX_ITER, "status": r.status, "n_iter": r.n_iter, "err": r.err,
                    "cost": r.cost, "lp_cost": lp, "dual_bound": bound,
                    "true_gap": r.cost - lp, "bound_gap": r.cost - bound,
                })
                print(f"{family} seed={seed} eps={eps:.3g} {sname}: {r.status} n_iter={r.n_iter}", flush=True)


def _load(csv_path):
    import csv
    with open(csv_path, newline="") as fh:
        return list(csv.DictReader(fh))


def plot(csv_path=CSV, png_path=PNG):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)

    rows = _load(csv_path)
    families = list(dict.fromkeys(r["family"] for r in rows))
    fig, grid = plt.subplots(2, len(families), figsize=(9, 5.0), layout="constrained", sharex="col",
                             gridspec_kw={"height_ratios": [3, 1]})
    axes = grid[0]
    notes = []
    for ax, axf, family in zip(grid[0], grid[1], families):
        # The log solver converges wherever vanilla does (same plan, tested), and also where vanilla fails.
        rs = [r for r in rows if r["family"] == family and r["solver"] == "log"]
        eps_vals = sorted({float(r["eps"]) for r in rs}, reverse=True)
        excluded = 0
        nonpos = 0
        for col, label, ls, color in [("true_gap", "true gap  ⟨P,C⟩ − LP", "-", SERIES[0]),
                                      ("bound_gap", "bound gap  ⟨P,C⟩ − dual bound", "--", SERIES[1])]:
            xs, med, lo, hi = [], [], [], []
            for e in eps_vals:
                conv = [float(r[col]) for r in rs if float(r["eps"]) == e and r["status"] == "converged"]
                if col == "true_gap":
                    excluded += sum(1 for r in rs if float(r["eps"]) == e and r["status"] != "converged")
                pos = [v for v in conv if v > 0]
                if col == "true_gap":
                    nonpos += len(conv) - len(pos)
                if not pos:
                    continue
                xs.append(e)
                med.append(np.median(pos))
                lo.append(np.percentile(pos, 25))
                hi.append(np.percentile(pos, 75))
            ax.plot(xs, med, ls, color=color, marker="o", label=label)
            ax.fill_between(xs, lo, hi, color=color, alpha=0.15, linewidth=0)
        # Option B: fraction of seeds converged at each eps (capped runs have no valid gap, so A does not apply).
        frac = [sum(1 for r in rs if float(r["eps"]) == e and r["status"] == "converged")
                / max(1, sum(1 for r in rs if float(r["eps"]) == e)) for e in eps_vals]
        axf.plot(eps_vals, frac, "-o", color=INK_2, markersize=3)
        axf.set(ylim=(-0.05, 1.05), yticks=[0, 0.5, 1], xlabel="ε  (units of median C)")
        ax.set(xscale="log", yscale="log", title=family.replace("_", " "))
        ax.invert_xaxis()
        notes.append(f"{family}: {excluded} non-converged runs excluded, {nonpos} non-positive gaps dropped")
    axes[0].set_ylabel("cost gap  (units of median C)")
    grid[1, 0].set_ylabel("converged\n(fraction)", fontsize=8)
    axes[0].legend(loc="lower left")
    n_seeds = len({r["seed"] for r in rows})
    fig.suptitle(f"Cost gap of the log-domain Sinkhorn plan, n = {rows[0]['n']}, median over {n_seeds} seeds "
                 "(band: interquartile range)", fontsize=10)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    for line in notes:
        print(line)


if __name__ == "__main__":
    what = sys.argv[1:] or ["run", "plot"]
    if "run" in what:
        run()
    if "plot" in what:
        plot()
