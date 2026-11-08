# Implementation guide

Reference for agents working in this repo. `AGENTS.md` holds the hard rules;
this file holds the layout, interfaces and conventions behind them.
`docs/derivation.md` is the only formula source.

## Ownership

The author writes these by hand. Agents may read them but must not edit them:

- `src/sinkhorn_lab/solvers.py`
- `src/sinkhorn_lab/metrics.py`
- `docs/derivation.md`
- Captions and claims in `README.md`

Agents may write everything else when asked: generators, the exact baseline,
tests, experiment scripts, CI, and the clean-clone check. If a task seems to
need a change to an author-owned file, stop and describe the change instead.

## Layout by stage

| Stage | Code | Tests | Output |
| --- | --- | --- | --- |
| Setup | `pyproject.toml`, `src/sinkhorn_lab/__init__.py`, `.github/workflows/ci.yml` | `tests/test_smoke.py` | |
| 1. Baseline | `problems.py`, `exact.py` | `test_problems.py`, `test_exact.py` | |
| 2. Vanilla | `solvers.py`, `metrics.py` (author) | `test_solvers.py`, `test_metrics.py` | |
| 3. Failure hunt | `experiments/failure_sweep.py` | Failing instance added to `test_solvers.py` | `figures/failure_sweep.csv` |
| 4. Log domain | `solvers.py` (author), `experiments/compare_solvers.py` | `test_log_solver.py`, `test_independent.py` | |
| 5. Experiments | `experiments/fig1_cost_gap.py`, `fig2_iterations.py`, `fig3_stability.py`, `fig4_timing.py` | None | `figures/*.csv`, `figures/*.png` |
| 6. Release | None | Clean-clone run | `README.md`, `v0.1.0` tag |

All `src` files are in `src/sinkhorn_lab/`. Install with `pip install -e ".[dev]"`.

## Interfaces

### `problems.py`

```python
@dataclass(frozen=True)
class Problem:
    a: np.ndarray   # source weights, shape (n,)
    b: np.ndarray   # target weights, shape (m,)
    C: np.ndarray   # cost, shape (n, m)
    meta: dict      # name, seed, parameters
```

- Every generator takes an explicit `seed` and uses `np.random.default_rng(seed)`.
  Never use global `np.random.seed`.
- Every generator calls `_validate(a, b, C)`: shapes match, weights are
  nonnegative, and total masses are equal.
- Generators: `gaussian_clouds(n, m, d=2, separation=0.0, normalize="none", seed=0)`,
  `random_cost(n, m, normalize="none", seed=0)`, `uniform_1d(n, seed=0)`.
  `uniform_1d` stores its points in `meta["x"]` and `meta["y"]` for the
  sorted-formula oracle.
- `normalize` is `"none"`, `"median"` or `"max"`: C is divided by 1,
  median(C) or max(C), and the divisor is stored in `meta["c_scale"]`.
  Normalizing rescales the effective ε. Which one each figure uses is set in
  DECISIONS.md, D4.
- Weights may contain zeros; total mass must be positive and equal
  (DECISIONS.md, D1).

### `exact.py`

`solve_exact(problem, max_iter=1_000_000, tol=1e-8) -> ExactResult(cost, plan, f, g)`

- A thin wrapper over `ot.emd(..., log=True)` that adds no algorithm.
- Raises if `log["warning"]` is set or either marginal of the plan is off by
  more than `tol`. It never returns an unverified answer.
- This is the only file that imports `ot`.

### Solvers (author fills this in after stage 2)

Tests and experiments depend on this contract. Read it from here, not from
the solver body.

```text
vanilla_sinkhorn(a, b, C, eps, tol=?, max_iter=?) -> result
log_sinkhorn(a, b, C, eps, tol=?, max_iter=?) -> result
result fields: P, converged, n_iter, status, err   # confirm or edit
status values: "converged" | "max_iter" | "numerical_failure"   # confirm or edit
stopping rule: ?
```

Decided (DECISIONS.md):

- D1: zero-weight points are dropped internally; their duals are NaN and
  their rows/columns of P are zero.
- D2: the kernel is built from the row-then-column shifted cost on the
  support. Cost and duals are reported on the original C
  (cost ⟨P, C⟩, duals f + α, g + β), matching `solve_exact`.

## POT details that cause mistakes

- `ot.emd` rescales `b` to the total of `a` after a six-decimal check. Our
  generators validate mass first, so this never hides an error.
- `ot.sinkhorn` defaults to `numItermax=1000`. Set it explicitly in any comparison.
- POT's entropy term has no −1, so its objective differs from the book's by a
  constant. Compare transport plans, never objective values.

## Testing rules

- Never weaken, skip or delete a test to make it pass. Report the failure.
- Use `pytest.approx` for floats. Test tolerances must be looser than the
  solver's stopping tolerance.
- Solver tests cover several independent properties: both marginals, P → abᵀ
  for large ε, cost approaching the LP value as ε shrinks, and cost not
  beating the LP beyond the slack set by the marginal error. Marginals alone
  miss a flipped sign in K.
- After adding a test, break the code on purpose and confirm the test fails.
  Report which mutation you tried.
- Tests must be fast and deterministic. Timing never goes in `tests/`.
- When a sweep finds a failing instance, add its seed and parameters as a test.
- Commands: `pytest -q`, `pytest -x`, `pytest --lf`, `pytest -k <word>`, `ruff check .`

## Experiment conventions

Figures (DECISIONS.md, D4):

| Figure | Script | Content | Normalization |
| --- | --- | --- | --- |
| 1 | `fig1_cost_gap.py` | Cost gap | `"median"` |
| 2 | `fig2_iterations.py` | Iterations | `"median"`; rerun with `"max"` as a check, report whether the separation holds |
| 3 | `fig3_stability.py` | Stability, ε against separation, one panel per method, fixed small n ≈ 50 | `"none"` (raw C) |
| 4 | `fig4_timing.py` | Timing: cost per iteration (left), time to convergence (right) | `"median"` |

Use `gaussian_clouds` and `random_cost` with the same `normalize` when both
families share an ε axis.

- One script per figure, run from the repo root as `python experiments/<name>.py`.
- Each script has `run()`, which writes `figures/<name>.csv`, and `plot()`,
  which reads only that CSV. Restyling never reruns the sweep.
- Grids, seeds, sizes and tolerances are constants at the top of the script.
- Write every run to the CSV, including failed and capped runs, with a
  `status` column. Never drop rows silently.
- Commit both the CSV and the PNG.
- Matplotlib: use `fig, ax = plt.subplots(layout="constrained")` and `ax.*`
  methods, save with `fig.savefig(path, dpi=200)`, then `plt.close(fig)`.
  Build paths from `ROOT = Path(__file__).resolve().parents[1]`.

### Plotting traps

- Log axes silently drop zero, negative, `nan` and `inf` values. Count and
  report those rows.
- Cost gaps can be negative when P is not feasible. Filter on status before
  plotting.
- A capped run's iteration count is the cap, not a measurement. Mark or
  exclude capped runs.
- For grid plots on log axes, compute cell edges in log space and use
  `shading="flat"`. `shading="nearest"` clips the outer cells.

## Timing experiment (`experiments/fig4_timing.py`, figure 4)

Goal: find where vanilla stops being reliable, and what the log solver costs there.
The figure plots cost per iteration on the left and time to convergence on
the right. Costs are median-normalized (D4).

- Where vanilla converges, it should be faster than log, so there is no
  runtime crossover. The cutoff is a stability threshold, priced by the log
  solver's slowdown.
- Test candidate predictors of failure against the data. Do not assume one.
  The expected strong predictor is the reduced cost range `C_shift.max() / eps`,
  where `C_shift` is the row-then-column shifted cost of D2: it controls
  underflow in the shifted kernel. Also test `C.max() / eps`,
  `(C.max() - C.min()) / eps` and `eps` alone.
- Time with `time.perf_counter()`. Discard a warm-up run, repeat each
  configuration 5–7 times, and report the median.
- Set `OMP_NUM_THREADS=1` before importing NumPy, and record the CPU in the CSV.
- Vary `n`, because the slowdown may grow with problem size.
- Compare times only between runs that reached the same tolerance.
- CSV columns: `n, seed, eps, normalize, c_scale, c_max, c_min, c_shift_max, solver, status, n_iter, setup_s, iter_s, final_err, cpu`.
  Cost columns are for the normalized C passed to the solver.
