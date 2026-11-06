# Agents Log

Log of agent work on substantive aspects of the project's design.

## 2026-11-06 — CI workflow, dependencies, smoke test

Task: organize the CI workflow (Setup stage in implementation.md) and add
dependencies to pyproject.toml.

Changes:
- `pyproject.toml`: `requires-python = ">=3.12"`; runtime deps `numpy`,
  `scipy`, `pot`; `dev` extra `pytest`, `ruff`, `matplotlib`. Unpinned.
- `.github/workflows/ci.yml`: on push to main, PRs and manual dispatch;
  Python 3.12 and 3.13 on ubuntu-latest; `pip install -e ".[dev]"`,
  `ruff check .`, `pytest -q`. Read-only token, superseded runs cancelled.
- `tests/test_smoke.py`: imports `sinkhorn_lab`, `.solvers`, `.metrics`.
  Needed so CI has a test to run (pytest exits 5 with none).

Verification (fresh clone + fresh venv in scratch, Python 3.13.7):
- `pip install -e ".[dev]"` exit 0; `ruff check .` passed; `pytest -q` 3 passed.
- Workflow YAML parsed with PyYAML: one job, five steps.
- Mutations in the scratch copy only (author files untouched):
  syntax error in solvers.py → 1 failed; `raise ImportError` in
  `__init__.py` → 3 failed. Restored → 3 passed.

Not verified: the workflow on GitHub runners (pending push).

## 2026-11-06 — Rules file rename, metrics.py ownership, CLAUDE.md imports

Task: give the author ownership of metrics.py in the rules, rename the
rules file for compatibility, and load implementation.md in every session.

Changes:
- `Agents.md` → `AGENTS.md` (matches the reference in implementation.md;
  Linux file systems are case-sensitive). Content otherwise unchanged
  except the next item.
- `AGENTS.md` Rules: the hand-written-file rule now covers
  `src/sinkhorn_lab/metrics.py` as well as `solvers.py` (wording made plural).
- `CLAUDE.md`: imports `@AGENTS.md` and `@implementation.md`.

Held, per author: implementation.md still says exact.py is the only file
that imports `ot`, until the author decides whether solvers/metrics need it.

Verification: `ruff check .` passed; `pytest -q` 3 passed.

Follow-up (2026-11-06): pushed; first GitHub Actions run 42274051703
passed on Python 3.12 and 3.13.

## 2026-11-06 — DECISIONS D1–D3; baseline built, held for author review

Task: record the author's zero-weight and cost-shift directives in
DECISIONS.md and build the stage 1 baseline (problems.py, exact.py, tests).

Committed now:
- `DECISIONS.md`: D1 (zero weights: solve on support, NaN duals), D2 (shift
  C by its minimum before the Gibbs kernel), D3 (exact baseline returns only
  certified answers). Reasoning formalized from the author's directives; the
  author may reword. Open points listed under D2.

Built, tested, not yet committed (awaiting author answers):
- `src/sinkhorn_lab/problems.py`: `Problem`, `_validate`, `gaussian_clouds`,
  `random_cost`, `uniform_1d`; arrays made read-only.
- `src/sinkhorn_lab/exact.py`: `solve_exact` → `ExactResult(cost, plan, f, g)`
  with input guards, support solve, min-shift, optimality certificate.
- `tests/test_problems.py`, `tests/test_exact.py`: 96 new tests (99 total).

Finding — POT `ot.emd` false infeasibility on negative costs:
- Raw `ot.emd` returns "Problem infeasible" when all costs are negative and
  near-constant below −N/(N−1), N = n + m. Bisection on C = 1e-6·U + c:
  n=2: −1.333334; n=5: −1.111112; n=30: −1.016950 (= N/(N−1) for N = 4, 10, 60).
  20/20 seeds fail at c = −3 and −100 for n = 5, 8, 30; 0/20 after
  subtracting min C. Wide-spread negative costs (max −1.5, min −99) passed.
- Probable cause (unverified, compiled code): artificial-arc cost in the
  network simplex is sized from max(0, max C).
- Fix: `exact.py` passes C_s − min C_s to `ot.emd` and adds min C_s to g.
  Regression test `test_near_constant_negative_cost`.

Verification:
- `ruff check .` passed; `pytest -q` 99 passed (also with
  `-W error::UserWarning`; the iteration-cap test filters POT's expected
  numItermax warning only).
- Mutations, scratch copy (author files untouched), all caught:
  skip input validation → 2 failed; remove min-shift entirely → 4 failed;
  forget c_min on g → 52; zero-weight duals 0 not NaN → 1; skip certificate
  → 3; ignore POT warning → 1; flip reduced-cost sign → 54; generator uses
  global RNG → 1; drop equal-mass check → 3; uniform_1d |x−y| not (x−y)² → 21.
  Restored → 99 passed.

## 2026-11-06 — Stage 1 baseline committed; normalize argument; D4

Author answers: NaN duals off support in exact.py; keep the certificate and
the min-shift; solvers report on original C; replace `scale_c` with
per-figure normalization; figure set is now four (D4).

Changes:
- `src/sinkhorn_lab/problems.py`: as logged above, with
  `gaussian_clouds(..., normalize="none" | "median" | "max")` replacing
  `scale_c`; divisor stored in `meta["c_scale"]`; default "none" (agent
  choice: experiment scripts set it explicitly per D4).
- `src/sinkhorn_lab/exact.py`: as logged above.
- `tests/test_problems.py`, `tests/test_exact.py`: normalize tests replace the
  scale_c test; exact tests run on median- and max-normalized clouds.
- `DECISIONS.md`: D1 exact convention confirmed; D2 reported quantities
  resolved (original C); D3 confirmed; D4 added (figures, normalization).

Not changed: `implementation.md` still lists `scale_c`, three figures plus
the timing script, and `C.max() / eps` as a predictor (flagged to author).

Verification:
- `ruff check .` passed; `pytest -q -W error::UserWarning` 117 passed.
- Normalization mutations (scratch copy), all caught: median→mean (3
  failed), median/max swapped (4), normalization not applied (4), c_scale
  not recorded (4), unknown value silently treated as none (4).
  Restored → 117 passed.

## 2026-11-06 — Row/column shift (D2 revised), random_cost normalize, implementation.md

Author directives: keep `normalize="none"` default; `random_cost` needs
`normalize` so both families share one ε axis; D2 changes from a global
min-shift to a row-then-column shift; reflect the new failure predictor;
`implementation.md` may be updated; the `ot` import line stays (still only
`exact.py`).

Changes:
- `problems.py`: shared `_normalize(C, normalize) -> (C / s, s)`;
  `random_cost(n, m, normalize="none", seed=0)` records `normalize`, `c_scale`.
- `tests/test_problems.py`: random_cost normalization tests; unknown-value
  rejection for both generators.
- `tests/test_exact.py`: `test_row_column_shift_has_same_plan` (D2 claim on
  the baseline), `test_separation_does_not_change_exact_plan`.
- `DECISIONS.md`: D2 rewritten for the row/column shift (proof that every
  row and column gets an exact 0; duals reported as f + α, g + β; shift on
  the support; predictor `C_shift.max()/eps`; exact.py's scalar shift kept
  as an independent workaround). D4: random_cost normalize; open point on
  separation and figure 3.
- `implementation.md`: generator signatures, D1/D2 notes on the solver
  contract, figure table (fig3_stability.py, fig4_timing.py replace
  fig3_success_map.py and timing_crossover.py), predictor list, CSV
  columns (+ normalize, c_scale, c_min, c_shift_max, cpu).
  `exact.py`-only `ot` import line left as is.

Finding — separation does not change the problem for squared Euclidean cost:
- Moving the target by s·e₁ adds a term in i plus a term in j, so plans are
  invariant. Exact plans at n = 50, seed 0, s ∈ {0,1,2,4,8,16}: bit-identical.
- Raw range max C − min C: 26.0 → 265.7 (s = 0 → 16). Row/column-shifted
  max: 24.7 → 36.7. The D2 shift absorbs most of the separation's numerical
  effect. Flagged to the author for figure 3.
- Row/column shift checked at n = 50 on gaussian (s = 0, 4, 4 median) and
  random_cost: every row and column has an exact 0, all entries ≥ 0.

Verification:
- `ruff check .` passed; `pytest -q -W error::UserWarning` 139 passed.
- Mutations (scratch copy), all caught: random_cost ignores normalize (8
  failed); random_cost c_scale fixed at 1 (4); helper median→mean (6);
  helper accepts unknown values (8); separation added on all axes (1).
  Restored → 139 passed.
