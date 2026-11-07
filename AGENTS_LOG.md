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

## 2026-11-06 — Commit author's solvers.py (vanilla + log domain); .gitignore

Committed on the author's explicit approval. `solvers.py` body is
author-written; my only edits to it were the import lines
(`dataclass`, `numpy`, `scipy.special.logsumexp`), added when asked.
`metrics.py` (one prompted `import numpy` line) is left uncommitted at the
author's instruction; it would fail ruff F401 until used.

- `.gitignore`: added `.DS_Store` (author request).
- `src/sinkhorn_lab/solvers.py`: `SinkhornResult`, `rowcol_shift`,
  `softmin`, `slack`, `vanilla_sinkhorn`, `log_sinkhorn`.

Verification before commit:
- `ruff check src/sinkhorn_lab/solvers.py` passed; `pytest -q` 139 passed.
- Real file, warnings as errors: `max_iter=0` and `3` report err equal to
  the true row error of the returned P in both solvers (96.04, 1.4307);
  vanilla and log agree at ε = 0.5 (89 iterations each, plans within
  2.4e-17); vanilla at ε = 0.01 returns `numerical_failure`.

## 2026-11-06 — solvers.py: converged on the last allowed iteration (explicit author approval)

**Explicit approval:** the author approved this edit to the author-owned
`solvers.py` in chat and asked for it to go in this push. No other edit was
made to the file apart from the one in the next entry.

- `vanilla_sinkhorn` and `log_sinkhorn`: after the err recompute that runs
  after the loop when `status == "max_iter"`, if `err < tol` then
  `status = "converged"` and `converge = True`. Three lines in each
  function.
- Side effect: the Edit tool also removed trailing spaces at the end of the
  `err = ...` line it touched in each function. The code is unchanged.
- Reason: with the check-before-update loop, a run converging on update n
  reported `"max_iter"` with `err < tol` when `max_iter = n`.
- Verified on the clouds_sep2 test problem at ε = 1 (converges in 41
  iterations):
  - `max_iter` = 40 gives `max_iter` with err 1.51e-9;
  - `max_iter` = 41 gives `converged` with `n_iter` 41 and err 9.10e-10;
  - `max_iter` = 42 gives the same result as 41.
  Both solvers behave identically.

## 2026-11-06 — solvers.py: default max_iter 100,000 (explicit author approval)

**Explicit approval:** the author approved this edit to the author-owned
`solvers.py` in chat.

- Both signatures: `max_iter=10_000` became `max_iter=100_000`.
- Reason (author): a wider range of starting parameters gives richer data
  for the figures. Noted in DECISIONS.md D10: a run that never converges
  now takes about 40 minutes for the log solver at n = 1000, so experiment
  scripts should set `max_iter` explicitly.

## 2026-11-06 — Solver tests: tests/test_solvers.py, tests/test_log_solver.py

The author reviewed and approved the tests for accuracy and scope, on
condition that a very large n test was added.

- `tests/test_solvers.py` runs both solvers on every test, unless marked
  "vanilla". The LP references don't use POT (`linear_sum_assignment`,
  HiGHS `linprog`), and every test treats a warning as a failure. It covers:
  - `rowcol_shift`;
  - converged-run properties: status, `err` equal to the true row error,
    row error < 1e-8, column error < 1e-12, P finite and ≥ 0,
    cost = ⟨P, C⟩;
  - the Gibbs form of the reported duals on the original C;
  - cost ≥ LP minus the slack the marginal error allows;
  - the gap to the LP shrinking at every step of ε = 1 → 0.03;
  - P → abᵀ at ε = 1e5, and the single-source case;
  - invariance to row and column offsets, and to scaling C and ε together;
  - `none` and `rowcol` agreeing;
  - D1 matching the reduced problem;
  - capped runs (0, 1, 3, 50) reporting the true error;
  - converging on the last allowed iteration;
  - tighter `tol` taking more iterations;
  - determinism and inputs never modified;
  - 9 kinds of bad input;
  - vanilla failure detection, unshifted and shifted;
  - large n (2000 × 2000, median-normalized, ε = 1): converged-run
    properties against the `linear_sum_assignment` LP, and agreement
    between the two solvers.
- `tests/test_log_solver.py` covers:
  - `softmin`: the direct formula, where the direct formula underflows, and
    the hard-min limit;
  - `slack`, entry by entry;
  - agreement with vanilla (n_iter, P, f + g, cost) on 4 problems with
    both stabilize options, and on capped runs;
  - converging where vanilla underflows without the shift;
  - staying finite and improving where vanilla fails with the shift;
  - no `numerical_failure` at ε = 1e-4 and 1e-8.
- A problem with non-uniform weights (Dirichlet) was added after the first
  round of deliberate breaks: the generators only make uniform weights,
  and a g update using mean(b) was caught by only 1 test.

Mutations, applied to scratch copies of `src/` (the real file was never
mutated), with the number of tests each one failed:

| Mutation | Tests failed |
|---|---|
| sign of K flipped | 29 |
| α dropped from f | 13 |
| α added twice | 13 |
| stop check on the column marginal | 30 |
| no post-loop err recompute (vanilla) | 7 |
| `n_iter` counting the check | 22 |
| `softmin` sign flipped | 42 |
| sign flipped inside the stop check's `exp` (log) | 29 |
| g missing from the log plan | 37 |
| β dropped (log) | 13 |
| no post-loop err recompute (log) | 9 |
| β not subtracted in `rowcol_shift` | 23 |
| off-support duals set to 0 | 4 |
| cost computed on a shifted C | 20 |
| `stabilize` check removed | 4 |
| negative-`a` check removed | 2 |
| g update using mean(b) (log), after adding the non-uniform problem | 4 |
| f update using mean(a) (log), after adding the non-uniform problem | 4 |
| v update using mean(b) (vanilla), after adding the non-uniform problem | 4 |

For the new tests specifically:
- removing the at-cap fix fails the at-cap test for both solvers;
- the sign flip of K fails 2 of the large-n tests;
- dropping β fails 1 of the large-n tests;
- f update using mean(a) passes the large-n tests, because their weights
  are uniform. The non-uniform tests catch it.

The vanilla zero check (`u_s == 0` or `v_s == 0`) has no test. Removing it
only delays the failure flag by one iteration (480-run sweep, 61 runs
affected, same status in all). See DECISIONS.md D11.

Result: `pytest -q` 275 passed in 7.2 s; `ruff check` passes on everything
except the author's uncommitted `metrics.py`.

## 2026-11-06 — DECISIONS.md: D5–D12, D4 figure note, D1 status

The author asked me to paraphrase their descriptions and format them as
markdown. The author will review the text by hand after upload. Where the
author's descriptions were brief, I added context from earlier sessions:
measurements, reasons for rejected alternatives, and test names.

- D1: "will implement" became "implements", with the test name.
- D4: the four-figure set reduces redundancy across axes; the planned ratio
  figure is dropped in favour of spread and iteration-count data.
- New sections:
  - D5: separate vanilla and log functions; a hybrid solver is deferred
    until there is data;
  - D6: the `SinkhornResult` fields and the scope of what is reported;
  - D7: vanilla as derived; check-before-update loop order, with the timing
    numbers and why checking at intervals was rejected;
  - D8: the `softmin` and `slack` helpers;
  - D9: input guards, with an open point on unequal mass and non-finite C;
  - D10: default `max_iter` of 100,000; `err` reports the returned plan;
    converging on the last allowed iteration counts as converged;
  - D11: failure detection, for reference;
  - D12: the test suite's approval.
