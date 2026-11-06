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
