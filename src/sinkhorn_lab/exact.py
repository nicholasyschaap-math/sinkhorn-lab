"""Exact (unregularized) OT baseline: a verified wrapper over POT's ot.emd.

The wrapper adds no algorithm. It guards the inputs so POT cannot silently
alter them, solves on the support of the weights, and certifies the answer
before returning it:

- Inputs are validated with ``problems._validate`` (equal mass to 1e-12
  relative). POT itself only checks mass to 6 decimals and then rescales b to
  sum(a), which would hide a mass error; it also substitutes uniform weights
  for empty inputs and casts the plan to integer for integer weights. All three
  are prevented here.
- Zero weights are dropped before calling POT and their duals are reported as
  NaN (DECISIONS.md, D1). POT would otherwise fill them with estimated values.
- POT receives the support cost shifted by its minimum (C - min C >= 0, as in
  DECISIONS.md, D2), and min C is added back to g. ot.emd falsely reports
  "Problem infeasible" when every cost is negative and roughly constant below
  -N/(N-1), N = n + m (AGENTS_LOG.md, 2026-11-06). The shift leaves the
  optimal plan unchanged; the cost is recomputed on the original C.
- The returned plan and duals are an optimality certificate, checked against
  the caller's original a, b, C: primal feasibility (P >= 0, both marginals
  within ``tol``), dual feasibility (f_i + g_j <= C_ij), complementary
  slackness (f_i + g_j = C_ij where P_ij > 0) and zero duality gap. If any
  check fails, or POT reports a warning, ExactSolverError is raised. An
  unverified answer is never returned.

Duals are determined only up to f + c, g - c; POT's centering is kept.
This is the only module that imports ``ot``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ot

from .problems import Problem, _validate


class ExactSolverError(RuntimeError):
    """The exact LP solve failed or its answer could not be certified."""


@dataclass(frozen=True)
class ExactResult:
    cost: float  # <P, C> on the original cost
    plan: np.ndarray  # shape (n, m); rows/cols with zero weight are zero
    f: np.ndarray  # shape (n,); NaN where a_i == 0
    g: np.ndarray  # shape (m,); NaN where b_j == 0


def solve_exact(problem: Problem, max_iter=1_000_000, tol=1e-8) -> ExactResult:
    a = np.array(problem.a, dtype=np.float64)
    b = np.array(problem.b, dtype=np.float64)
    C = np.array(problem.C, dtype=np.float64)
    try:
        _validate(a, b, C)
    except ValueError as exc:
        raise ExactSolverError(f"invalid problem: {exc}") from exc

    rows = np.flatnonzero(a > 0)
    cols = np.flatnonzero(b > 0)
    a_s, b_s = a[rows], b[cols]
    C_s = np.ascontiguousarray(C[np.ix_(rows, cols)])

    c_min = float(C_s.min())
    P_s, log = ot.emd(a_s, b_s, C_s - c_min, numItermax=int(max_iter), log=True)
    if log["warning"] is not None:
        raise ExactSolverError(f"ot.emd: {log['warning']}")
    P_s = np.asarray(P_s, dtype=np.float64)
    u = np.asarray(log["u"], dtype=np.float64)
    v = np.asarray(log["v"], dtype=np.float64) + c_min  # duals for the original C

    P = np.zeros_like(C)
    P[np.ix_(rows, cols)] = P_s
    f = np.full(a.shape, np.nan)
    g = np.full(b.shape, np.nan)
    f[rows] = u
    g[cols] = v
    cost = float(np.sum(P * C))

    _certify(a, b, C, P, f, g, rows, cols, cost, tol)
    return ExactResult(cost=cost, plan=P, f=f, g=g)


def _certify(a, b, C, P, f, g, rows, cols, cost, tol) -> None:
    """Raise ExactSolverError unless (P, f, g) is a certified optimal pair."""
    problems = []

    if not np.all(np.isfinite(P)) or P.min() < 0:
        problems.append("plan has negative or non-finite entries")
    err_a = float(np.max(np.abs(P.sum(axis=1) - a)))
    err_b = float(np.max(np.abs(P.sum(axis=0) - b)))
    if err_a > tol or err_b > tol:
        problems.append(f"marginal error {max(err_a, err_b):.3e} > tol {tol:.1e}")

    fs, gs = f[rows], g[cols]
    if not (np.all(np.isfinite(fs)) and np.all(np.isfinite(gs))):
        problems.append("non-finite duals on the support")
    else:
        C_s = C[np.ix_(rows, cols)]
        reduced = C_s - fs[:, None] - gs[None, :]  # >= 0 for dual feasibility
        scale = max(1.0, float(np.max(np.abs(C_s))))
        if reduced.min() < -tol * scale:
            problems.append(f"dual infeasible: min(C - f - g) = {reduced.min():.3e}")
        P_s = P[np.ix_(rows, cols)]
        active = P_s > 0
        if active.any():
            slack = float(np.max(np.abs(reduced[active])))
            if slack > tol * scale:
                problems.append(f"complementary slackness violated by {slack:.3e}")
        dual_obj = float(a[rows] @ fs + b[cols] @ gs)
        gap = abs(dual_obj - cost)
        if gap > tol * max(1.0, abs(cost)):
            problems.append(f"duality gap {gap:.3e}")

    if problems:
        raise ExactSolverError("uncertified LP solution: " + "; ".join(problems))
