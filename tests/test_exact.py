"""Tests for the exact LP baseline.

Every optimal value is checked against a reference independent of POT:
the sorted-matching formula in 1-D, scipy's linear_sum_assignment for
uniform n = n problems, or scipy's HiGHS LP for general weights.
"""

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment, linprog

import sinkhorn_lab.exact as exact_mod
from sinkhorn_lab.exact import ExactSolverError, solve_exact
from sinkhorn_lab.problems import (
    Problem,
    gaussian_clouds,
    random_cost,
    uniform_1d,
)

TOL = 1e-8


def make(a, b, C):
    """Hand-built Problem (bypasses generators, as a user would)."""
    return Problem(a=np.asarray(a), b=np.asarray(b), C=np.asarray(C), meta={})


def lp_value(a, b, C):
    """Reference optimal value from scipy's HiGHS LP solver."""
    n, m = C.shape
    A_eq = np.vstack([np.kron(np.eye(n), np.ones((1, m))), np.kron(np.ones((1, n)), np.eye(m))])
    res = linprog(C.ravel(), A_eq=A_eq, b_eq=np.concatenate([a, b]), bounds=(0, None), method="highs")
    assert res.status == 0
    return res.fun


def assert_certified(res, p):
    a, b, C = p.a, p.b, p.C
    P = res.plan
    assert P.shape == C.shape and P.dtype == np.float64
    assert P.min() >= 0
    assert np.abs(P.sum(axis=1) - a).max() <= TOL
    assert np.abs(P.sum(axis=0) - b).max() <= TOL
    assert res.cost == pytest.approx(float(np.sum(P * C)), rel=1e-12, abs=1e-15)
    rows, cols = a > 0, b > 0
    f, g = res.f[rows], res.g[cols]
    assert np.all(np.isfinite(f)) and np.all(np.isfinite(g))
    reduced = C[np.ix_(rows, cols)] - f[:, None] - g[None, :]
    assert reduced.min() >= -1e-9
    active = P[np.ix_(rows, cols)] > 0
    assert np.abs(reduced[active]).max() <= 1e-9
    assert a[rows] @ f + b[cols] @ g == pytest.approx(res.cost, rel=1e-9, abs=1e-12)


# --- agreement with independent references ---------------------------------

@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("n", [1, 2, 7, 40])
def test_uniform_1d_matches_sorted_formula(n, seed):
    p = uniform_1d(n, seed=seed)
    res = solve_exact(p)
    x, y = np.sort(p.meta["x"]), np.sort(p.meta["y"])
    assert res.cost == pytest.approx(np.mean((x - y) ** 2), rel=1e-9, abs=1e-14)
    assert_certified(res, p)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize(
    "gen",
    [
        lambda s: random_cost(9, 9, seed=s),
        lambda s: gaussian_clouds(9, 9, seed=s),
        lambda s: gaussian_clouds(9, 9, d=3, separation=3.0, normalize="median", seed=s),
        lambda s: gaussian_clouds(9, 9, d=3, separation=3.0, normalize="max", seed=s),
    ],
    ids=["random_cost", "gaussian_clouds", "gaussian_clouds_median", "gaussian_clouds_max"],
)
def test_square_uniform_matches_assignment(gen, seed):
    p = gen(seed)
    res = solve_exact(p)
    r, c = linear_sum_assignment(p.C)
    assert res.cost == pytest.approx(p.C[r, c].mean(), rel=1e-9, abs=1e-14)
    assert_certified(res, p)


@pytest.mark.parametrize("seed", range(5))
def test_general_weights_match_highs(seed):
    rng = np.random.default_rng(100 + seed)
    n, m = 6, 8
    a = rng.random(n)
    a /= a.sum()
    b = rng.random(m)
    b /= b.sum()
    b *= a.sum() / b.sum()
    C = rng.random((n, m)) * 10
    p = make(a, b, C)
    res = solve_exact(p)
    assert res.cost == pytest.approx(lp_value(a, b, C), rel=1e-7, abs=1e-10)
    assert_certified(res, p)


def test_rectangular_generator():
    p = gaussian_clouds(5, 11, seed=0)
    res = solve_exact(p)
    assert res.cost == pytest.approx(lp_value(p.a, p.b, p.C), rel=1e-7)
    assert_certified(res, p)


def test_unnormalized_mass():
    a = np.array([2.0, 3.0, 5.0])
    b = np.array([4.0, 6.0])
    C = np.array([[1.0, 4.0], [2.0, 0.5], [3.0, 3.0]])
    p = make(a, b, C)
    res = solve_exact(p)
    assert res.cost == pytest.approx(lp_value(a, b, C), rel=1e-9)
    assert_certified(res, p)


# --- zero weights (DECISIONS.md, D1) ---------------------------------------

def zero_weight_problem():
    rng = np.random.default_rng(7)
    a = np.array([0.2, 0.0, 0.3, 0.0, 0.5])
    b = np.array([0.0, 0.4, 0.35, 0.25, 0.0, 0.0])
    C = rng.random((5, 6))
    return make(a, b, C)


def test_zero_weights_solved_on_support():
    p = zero_weight_problem()
    res = solve_exact(p)
    za, zb = p.a == 0, p.b == 0
    assert np.all(res.plan[za, :] == 0) and np.all(res.plan[:, zb] == 0)
    assert np.all(np.isnan(res.f[za])) and np.all(np.isnan(res.g[zb]))
    assert np.all(np.isfinite(res.f[~za])) and np.all(np.isfinite(res.g[~zb]))
    assert res.cost == pytest.approx(lp_value(p.a, p.b, p.C), rel=1e-7)
    assert_certified(res, p)


def test_zero_weights_equal_reduced_problem():
    p = zero_weight_problem()
    rows, cols = p.a > 0, p.b > 0
    reduced = make(p.a[rows], p.b[cols], p.C[np.ix_(rows, cols)])
    full, red = solve_exact(p), solve_exact(reduced)
    assert full.cost == pytest.approx(red.cost, rel=1e-12)
    np.testing.assert_array_equal(full.plan[np.ix_(rows, cols)], red.plan)
    np.testing.assert_array_equal(full.f[rows], red.f)
    np.testing.assert_array_equal(full.g[cols], red.g)


def test_single_point_support():
    a = np.array([0.0, 1.0, 0.0])
    b = np.array([0.25, 0.0, 0.75])
    C = np.array([[9.0, 9.0, 9.0], [1.0, 9.0, 2.0], [9.0, 9.0, 9.0]])
    res = solve_exact(make(a, b, C))
    assert res.cost == pytest.approx(0.25 * 1.0 + 0.75 * 2.0)
    np.testing.assert_array_equal(res.plan[1], [0.25, 0.0, 0.75])


# --- cost shift (DECISIONS.md, D2) -----------------------------------------

@pytest.mark.parametrize("shift", [-3.0, 0.7, 50.0])
def test_cost_shift_changes_cost_by_shift_times_mass_only(shift):
    p = random_cost(8, 8, seed=11)
    base = solve_exact(p)
    shifted = solve_exact(make(p.a, p.b, p.C + shift))
    assert shifted.cost == pytest.approx(base.cost + shift * p.a.sum(), rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(shifted.plan, base.plan, atol=1e-15)


def test_min_subtracted_cost_has_same_plan():
    p = gaussian_clouds(10, 10, separation=2.0, seed=4)
    base = solve_exact(p)
    shifted = solve_exact(make(p.a, p.b, p.C - p.C.min()))
    assert shifted.cost == pytest.approx(base.cost - p.C.min() * p.a.sum(), rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(shifted.plan, base.plan, atol=1e-15)


@pytest.mark.parametrize("n", [2, 5, 30])
def test_near_constant_negative_cost(n):
    # Regression: raw ot.emd reports "Problem infeasible" for all-negative,
    # near-constant costs below -N/(N-1), N = 2n. The wrapper must solve it.
    p = random_cost(n, n, seed=0)
    C = p.C * 1e-6 - 3.0
    q = make(p.a, p.b, C)
    res = solve_exact(q)
    r, c = linear_sum_assignment(C)
    assert res.cost == pytest.approx(C[r, c].mean(), rel=1e-12)
    assert_certified(res, q)


@pytest.mark.parametrize(
    "gen",
    [lambda: gaussian_clouds(30, 30, separation=2.0, seed=3), lambda: random_cost(12, 9, seed=5)],
    ids=["gaussian_clouds", "random_cost"],
)
def test_row_column_shift_has_same_plan(gen):
    # D2: C - alpha 1^T - 1 beta^T changes every admissible plan's cost by the
    # constant <a, alpha> + <b, beta>, so the optimal plan is unchanged.
    p = gen()
    alpha = p.C.min(axis=1)
    S = p.C - alpha[:, None]
    beta = S.min(axis=0)
    S = S - beta[None, :]
    assert np.all(S.min(axis=1) == 0) and np.all(S.min(axis=0) == 0)
    base, shifted = solve_exact(p), solve_exact(make(p.a, p.b, S))
    assert shifted.cost == pytest.approx(base.cost - p.a @ alpha - p.b @ beta, abs=1e-12)
    np.testing.assert_allclose(shifted.plan, base.plan, atol=1e-15)


@pytest.mark.parametrize("separation", [1.0, 4.0, 16.0])
def test_separation_does_not_change_exact_plan(separation):
    # For squared Euclidean cost, translating the target adds a term in i plus
    # a term in j, so the optimal plan is the same for every separation
    # (DECISIONS.md, D4 open point). Same seed -> same base points.
    base = solve_exact(gaussian_clouds(30, 30, seed=0)).plan
    moved = solve_exact(gaussian_clouds(30, 30, separation=separation, seed=0)).plan
    np.testing.assert_allclose(moved, base, atol=1e-15)


# --- the wrapper never returns a corrupted answer --------------------------

def test_rejects_mass_mismatch_that_pot_would_rescale():
    # sum(b) = 1 + 1e-7: inside ot.emd's 6-decimal check, so POT would
    # silently rescale b. The wrapper must refuse.
    p = make(np.ones(4) / 4, np.ones(4) / 4 * (1 + 1e-7), np.ones((4, 4)))
    with pytest.raises(ExactSolverError, match="masses differ"):
        solve_exact(p)


def test_rejects_zero_total_mass_instead_of_uniform_fallback():
    # POT replaces empty weight vectors with uniform ones; all-zero weights
    # would become empty after dropping zeros.
    p = make(np.zeros(3), np.zeros(2), np.ones((3, 2)))
    with pytest.raises(ExactSolverError, match="positive"):
        solve_exact(p)


def test_integer_weights_are_not_truncated():
    # POT casts the plan to the weights' dtype; integer weights would truncate.
    p = make(np.array([1, 2]), np.array([2, 1]), np.array([[0.0, 1.0], [1.0, 0.0]]))
    res = solve_exact(p)
    assert res.plan.dtype == np.float64
    np.testing.assert_allclose(res.plan, [[1.0, 0.0], [1.0, 1.0]], atol=1e-15)
    assert res.cost == pytest.approx(1.0)


@pytest.mark.filterwarnings("ignore:numItermax reached:UserWarning")
def test_rejects_iteration_cap():
    with pytest.raises(ExactSolverError, match="ot.emd"):
        solve_exact(random_cost(30, 30, seed=0), max_iter=5)


def test_does_not_modify_inputs():
    a = np.array([0.5, 0.0, 0.5])
    b = np.array([0.5, 0.5])
    C = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 0.0]])
    a0, b0, C0 = a.copy(), b.copy(), C.copy()
    solve_exact(make(a, b, C))
    np.testing.assert_array_equal(a, a0)
    np.testing.assert_array_equal(b, b0)
    np.testing.assert_array_equal(C, C0)


def _patch_emd(monkeypatch, corrupt):
    real = exact_mod.ot.emd

    def fake(a, b, M, **kw):
        P, log = real(a, b, M, **kw)
        return corrupt(P.copy(), log)

    monkeypatch.setattr(exact_mod.ot, "emd", fake)


def test_certificate_rejects_suboptimal_plan(monkeypatch):
    # Feasible but not optimal: swap two rows' matches in an assignment.
    def corrupt(P, log):
        return P[[1, 0] + list(range(2, P.shape[0]))], log

    _patch_emd(monkeypatch, corrupt)
    with pytest.raises(ExactSolverError, match="uncertified"):
        solve_exact(random_cost(6, 6, seed=1))


def test_certificate_rejects_marginal_error(monkeypatch):
    def corrupt(P, log):
        P[0, :] *= 1 + 1e-6
        return P, log

    _patch_emd(monkeypatch, corrupt)
    with pytest.raises(ExactSolverError, match="marginal"):
        solve_exact(random_cost(6, 6, seed=1))


def test_certificate_rejects_infeasible_duals(monkeypatch):
    def corrupt(P, log):
        log = dict(log)
        log["u"] = log["u"] + 1e-3
        return P, log

    _patch_emd(monkeypatch, corrupt)
    with pytest.raises(ExactSolverError, match="uncertified"):
        solve_exact(random_cost(6, 6, seed=1))
