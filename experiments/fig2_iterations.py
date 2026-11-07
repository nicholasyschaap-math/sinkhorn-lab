"""Figure 2: iterations to converge against epsilon, for several n (DECISIONS.md D4).

Primary run uses median(C) normalization; the same sweep is repeated with
max(C) normalization as a check on whether the separation between the
n-curves holds. The check is printed by plot().

Two tolerances are run (author-approved option C): tol = 1e-9, the solver
default, and tol = 1e-6. Sinkhorn at small eps reaches ~1e-6 quickly and then
enters a slow final phase, so the pair separates the fast phase from the tail.

Solver rule: vanilla runs everywhere. Where vanilla converges, the log solver
takes exactly the same number of iterations (tests/test_log_solver.py), so it
is run only where vanilla reports numerical_failure.

Capped runs (author-approved options A and B):
- A: each curve is the median over seeds with capped runs counted as
  "at least the cap" (right-censored). The median is exact while fewer than
  half the seeds are capped; otherwise the point is drawn as an open
  triangle on the cap line, meaning "median >= cap".
- B: the panel under each plot shows the fraction of seeds that converged
  at tol = 1e-9.

Usage, from the repo root: python experiments/fig2_iterations.py [run|plot]
"""

import sys

from _common import FIG_DIR, INK_2, ORDINAL_BLUE, ResumableCSV, single_thread, style

single_thread()

import numpy as np

from sinkhorn_lab.problems import gaussian_clouds, random_cost
from sinkhorn_lab.solvers import log_sinkhorn, vanilla_sinkhorn

NAME = "fig2_iterations"
CSV = FIG_DIR / f"{NAME}.csv"
PNG = FIG_DIR / f"{NAME}.png"

NS = [25, 50, 100, 200]
SEEDS = range(5)
EPS = np.logspace(0, -2.5, 11)        # 1 ... ~0.003, in units of the normalizing scale
NORMALIZATIONS = ["median", "max"]    # median is primary, max is the check
SEPARATION = 0.0
TOLS = [1e-9, 1e-6]                   # solver default, and the loose tolerance (option C)
MAX_ITER = 100_000
FAMILIES = {
    "gaussian_clouds": lambda n, norm, seed: gaussian_clouds(n, n, separation=SEPARATION, normalize=norm, seed=seed),
    "random_cost": lambda n, norm, seed: random_cost(n, n, normalize=norm, seed=seed),
}

COLUMNS = ["family", "normalize", "n", "seed", "eps", "c_scale", "solver", "stabilize", "tol", "max_iter",
           "status", "n_iter", "err"]
KEY = ["family", "normalize", "n", "seed", "eps", "tol", "solver"]


def run(csv_path=CSV):
    out = ResumableCSV(csv_path, COLUMNS, KEY)
    for tol in TOLS:
        for norm in NORMALIZATIONS:
            for family, make in FAMILIES.items():
                for n in NS:
                    for seed in SEEDS:
                        p = None
                        for eps in EPS:
                            key = {"family": family, "normalize": norm, "n": n, "seed": seed,
                                   "eps": repr(float(eps)), "tol": tol}
                            if not out.has(**key, solver="vanilla"):
                                p = p or make(n, norm, seed)
                                _run_one(out, p, key, "vanilla", vanilla_sinkhorn, eps)
                            failed = out.get(**key, solver="vanilla")["status"] == "numerical_failure"
                            if failed and not out.has(**key, solver="log"):
                                p = p or make(n, norm, seed)
                                _run_one(out, p, key, "log", log_sinkhorn, eps)


def _run_one(out, p, key, sname, solver, eps):
    r = solver(p.a, p.b, p.C, eps, tol=key["tol"], max_iter=MAX_ITER)
    out.write({**key, "c_scale": p.meta["c_scale"], "solver": sname, "stabilize": "rowcol",
               "max_iter": MAX_ITER, "status": r.status, "n_iter": r.n_iter, "err": r.err})
    print(f"tol={key['tol']:g} {key['normalize']} {key['family']} n={key['n']} seed={key['seed']} eps={eps:.3g} "
          f"{sname}: {r.status} n_iter={r.n_iter}", flush=True)


def _load(csv_path):
    import csv
    with open(csv_path, newline="") as fh:
        return list(csv.DictReader(fh))


def _iterations(rows):
    """One result per (family, normalize, n, seed, eps, tol): vanilla, or log where vanilla failed."""
    by = {}
    for r in rows:
        k = (r["family"], r["normalize"], int(r["n"]), int(r["seed"]), float(r["eps"]), float(r["tol"]))
        by.setdefault(k, {})[r["solver"]] = r
    result = {}
    for k, d in by.items():
        if "vanilla" not in d:
            continue
        r = d["log"] if d["vanilla"]["status"] == "numerical_failure" and "log" in d else d["vanilla"]
        result[k] = (r["status"], int(r["n_iter"]), r["solver"], int(r["max_iter"]))
    return result


def _censored_median(runs):
    """Median iterations with non-converged runs counted as +inf (at least the cap)."""
    vals = np.array([c if s == "converged" else np.inf for s, c, _, _ in runs], dtype=float)
    return float(np.median(vals))


def plot(csv_path=CSV, png_path=PNG):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    style(plt)

    rows = _load(csv_path)
    it = _iterations(rows)
    families = list(dict.fromkeys(r["family"] for r in rows))
    norms = list(dict.fromkeys(r["normalize"] for r in rows))
    ns = sorted({int(r["n"]) for r in rows})
    tols = sorted({float(r["tol"]) for r in rows})
    tol_main, tol_loose = min(tols), max(tols)
    cap = max(int(r["max_iter"]) for r in rows)
    styles = {tol_loose: "--", tol_main: "-"}      # main last, so a single tolerance draws solid

    fig, axes = plt.subplots(2 * len(norms), len(families), figsize=(9.5, 9.0), layout="constrained",
                             sharex="col", squeeze=False, gridspec_kw={"height_ratios": [3, 1] * len(norms)})
    medians = {}
    for i, norm in enumerate(norms):
        for j, family in enumerate(families):
            ax, axf = axes[2 * i, j], axes[2 * i + 1, j]
            not_conv = from_log = 0
            for n, color in zip(ns, ORDINAL_BLUE):
                for tol in tols:
                    sel = {k: v for k, v in it.items() if k[:3] == (family, norm, n) and k[5] == tol}
                    eps_vals = sorted({k[4] for k in sel}, reverse=True)
                    xs, ys, xc, fx, fy = [], [], [], [], []
                    for e in eps_vals:
                        runs = [v for k, v in sel.items() if k[4] == e]
                        if tol == tol_main:
                            not_conv += sum(1 for r in runs if r[0] != "converged")
                            fx.append(e)
                            fy.append(sum(1 for r in runs if r[0] == "converged") / len(runs))
                        from_log += sum(1 for r in runs if r[2] == "log" and r[0] == "converged")
                        m = _censored_median(runs)
                        if np.isfinite(m):
                            xs.append(e)
                            ys.append(m)
                            medians[(tol, norm, family, n, e)] = m
                        else:
                            xc.append(e)
                    ax.plot(xs, ys, styles[tol], color=color, marker="o" if tol == tol_main else None,
                            markersize=4)
                    if xc:
                        ax.plot(xc, [cap] * len(xc), linestyle="none", marker="^", markersize=6,
                                markerfacecolor="none", markeredgecolor=color, markeredgewidth=1.2)
                    if tol == tol_main:
                        axf.plot(fx, fy, "-", color=color, marker="o", markersize=3)
            ax.axhline(cap, color=INK_2, linewidth=0.8, linestyle=":")
            ax.set(xscale="log", yscale="log", title=f"{family.replace('_', ' ')}, C / {norm}(C)")
            ax.text(0.98, 0.04, f"tol {tol_main:g}: {not_conv} runs hit the cap\n"
                                f"{from_log} counts from log solver (vanilla failed)",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=INK_2)
            axf.set(xscale="log", ylim=(-0.05, 1.05), yticks=[0, 0.5, 1])
            if j == 0:
                ax.set_ylabel("iterations (median over seeds)")
                axf.set_ylabel(f"converged\nat {tol_main:g}", fontsize=8)
            if i == len(norms) - 1:
                axf.set_xlabel("ε  (units of the normalizing scale)")
    for j in range(len(families)):
        axes[0, j].invert_xaxis()       # columns share x, so this flips the whole column
    handles = [Line2D([], [], color=c, marker="o", markersize=4, label=f"n = {n}") for n, c in zip(ns, ORDINAL_BLUE)]
    handles += [Line2D([], [], color=INK_2, linestyle=styles[t], label=f"tol = {t:g}") for t in tols]
    handles += [Line2D([], [], color=INK_2, linestyle="none", marker="^", markerfacecolor="none",
                       label=f"median ≥ cap ({cap:,})")]
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles))
    n_seeds = len({r["seed"] for r in rows})
    fig.suptitle(f"Iterations to converge, median over {n_seeds} seeds with capped runs counted as ≥ cap",
                 fontsize=10)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    for tol in tols:
        _separation_check(medians, tol, families, norms, ns)


def _separation_check(medians, tol, families, norms, ns):
    """Does the ordering and spacing of the n-curves under median hold under max?"""
    if len(ns) < 2:
        print("Separation check skipped: fewer than two n values in the CSV")
        return
    print(f"Separation check (n-curves), tol = {tol:g}: at each eps where every n has a finite median")
    for family in families:
        for norm in norms:
            eps_all = sorted({e for (t, nm, f, _, e) in medians if t == tol and nm == norm and f == family},
                             reverse=True)
            orders, spreads = [], []
            for e in eps_all:
                vals = [medians.get((tol, norm, family, n, e)) for n in ns]
                if any(v is None for v in vals):
                    continue
                orders.append(tuple(np.argsort(vals, kind="stable")))
                spreads.append(max(vals) / min(vals))
            same = sum(1 for o in orders if o == tuple(range(len(ns))))
            rev = sum(1 for o in orders if o == tuple(reversed(range(len(ns)))))
            spread = f"{np.median(spreads):.2f}" if spreads else "n/a"
            print(f"  {family:16s} {norm:6s}: {len(orders)} eps values; iterations increase with n at {same}, "
                  f"decrease at {rev}, mixed at {len(orders) - same - rev}; median max/min ratio across n {spread}")


if __name__ == "__main__":
    what = sys.argv[1:] or ["run", "plot"]
    if "run" in what:
        run()
    if "plot" in what:
        plot()
