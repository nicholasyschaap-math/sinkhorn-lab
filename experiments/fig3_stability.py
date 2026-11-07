"""Figure 3: run status over epsilon and cloud separation, one panel per method (DECISIONS.md D4).

Raw C (no normalization), so separation stays in the cost scale. Fixed n.
Panels: vanilla without the D2 shift, vanilla with it, and log-domain with it.

Note (DECISIONS.md D4, open point): for squared-Euclidean cost, separation
adds a term in i plus a term in j to C, so every plan is unchanged and the
D2 shift removes most of the numerical effect. Differences between the
first two panels show what the shift absorbs. The CSV also records the
candidate failure predictors (c_max, c_min, c_shift_max) for later analysis.

Usage, from the repo root: python experiments/fig3_stability.py [run|plot]
"""

import sys

from _common import (
    FIG_DIR,
    STATUS,
    STATUS_HATCH,
    STATUS_LABEL,
    ResumableCSV,
    single_thread,
    style,
)

single_thread()

import numpy as np

from sinkhorn_lab.problems import gaussian_clouds
from sinkhorn_lab.solvers import log_sinkhorn, rowcol_shift, vanilla_sinkhorn

NAME = "fig3_stability"
CSV = FIG_DIR / f"{NAME}.csv"
PNG = FIG_DIR / f"{NAME}.png"

N = 50
SEED = 0
SEPARATIONS = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0]
EPS = np.logspace(-2.5, 1.5, 17)          # raw cost units
NORMALIZE = "none"
TOL = 1e-9
MAX_ITER = 100_000
METHODS = {
    "vanilla, no shift": (vanilla_sinkhorn, "none"),
    "vanilla, row/col shift": (vanilla_sinkhorn, "rowcol"),
    "log-domain, row/col shift": (log_sinkhorn, "rowcol"),
}

COLUMNS = ["method", "solver", "stabilize", "n", "seed", "separation", "eps", "normalize", "tol", "max_iter",
           "c_max", "c_min", "c_shift_max", "status", "n_iter", "err"]
KEY = ["method", "separation", "eps"]


def run(csv_path=CSV):
    out = ResumableCSV(csv_path, COLUMNS, KEY)
    for sep in SEPARATIONS:
        p = gaussian_clouds(N, N, separation=sep, normalize=NORMALIZE, seed=SEED)
        c_shift_max = rowcol_shift(p.C)[0].max()
        for method, (solver, stab) in METHODS.items():
            for eps in EPS:
                if out.has(method=method, separation=repr(sep), eps=repr(float(eps))):
                    continue
                r = solver(p.a, p.b, p.C, eps, stabilize=stab, tol=TOL, max_iter=MAX_ITER)
                out.write({"method": method, "solver": solver.__name__, "stabilize": stab, "n": N, "seed": SEED,
                           "separation": repr(sep), "eps": repr(float(eps)), "normalize": NORMALIZE, "tol": TOL,
                           "max_iter": MAX_ITER, "c_max": p.C.max(), "c_min": p.C.min(),
                           "c_shift_max": c_shift_max, "status": r.status, "n_iter": r.n_iter, "err": r.err})
                print(f"sep={sep} {method} eps={eps:.3g}: {r.status} n_iter={r.n_iter}", flush=True)


def _load(csv_path):
    import csv
    with open(csv_path, newline="") as fh:
        return list(csv.DictReader(fh))


def _log_edges(x):
    lx = np.log10(np.asarray(sorted(x)))
    mid = (lx[1:] + lx[:-1]) / 2
    return 10 ** np.concatenate([[lx[0] - (mid[0] - lx[0])], mid, [lx[-1] + (lx[-1] - mid[-1])]])


def plot(csv_path=CSV, png_path=PNG):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    style(plt)
    plt.rcParams["hatch.color"] = "#0b0b0b"
    plt.rcParams["hatch.linewidth"] = 0.5

    rows = _load(csv_path)
    eps_vals = sorted({float(r["eps"]) for r in rows})
    seps = sorted({float(r["separation"]) for r in rows})
    xe = _log_edges(eps_vals)
    methods = list(dict.fromkeys(r["method"] for r in rows))
    fig, axes = plt.subplots(1, len(methods), figsize=(12, 4.2), layout="constrained", sharey=True)
    for ax, method in zip(axes, methods):
        counts = {s: 0 for s in STATUS}
        for r in rows:
            if r["method"] != method:
                continue
            i, j = eps_vals.index(float(r["eps"])), seps.index(float(r["separation"]))
            st = r["status"]
            counts[st] += 1
            ax.add_patch(Rectangle((xe[i], j), xe[i + 1] - xe[i], 1, facecolor=STATUS[st],
                                   hatch=STATUS_HATCH[st], edgecolor="#fcfcfb", linewidth=1.0))
        ax.set(xscale="log", xlim=(xe[0], xe[-1]), ylim=(0, len(seps)), xlabel="ε  (raw cost units)", title=method)
        ax.set_yticks(np.arange(len(seps)) + 0.5, [f"{s:g}" for s in seps])
        ax.grid(False)
        ax.set_title(method + "\n" + "converged {}, cap {}, failure {}".format(*counts.values()), fontsize=9)
    axes[0].set_ylabel("cloud separation")
    handles = [Patch(facecolor=STATUS[s], hatch=STATUS_HATCH[s], edgecolor="#fcfcfb", label=STATUS_LABEL[s])
               for s in STATUS]
    fig.legend(handles=handles, loc="outside lower center", ncols=3)
    r0 = rows[0]
    fig.suptitle(f"Run status by ε and separation: gaussian clouds, n = {r0['n']}, seed {r0['seed']}, raw C, "
                 f"tol = {float(r0['tol']):g}, cap = {int(r0['max_iter']):,} iterations", fontsize=10)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    what = sys.argv[1:] or ["run", "plot"]
    if "run" in what:
        run()
    if "plot" in what:
        plot()
