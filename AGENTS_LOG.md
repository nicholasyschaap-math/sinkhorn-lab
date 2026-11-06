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
