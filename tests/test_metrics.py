"""Tests for metrics.lp_dual_bound: agreement of the dual lower bound with the LP optimum.

The LP optimum comes from the certified exact baseline (exact.py). The bound
is computed from Sinkhorn duals across a sweep of eps, in units of each
problem's median cost on the support. Capped runs are kept: any finite duals
give a valid bound after the c-transform.
"""

import itertools

import numpy as np
import pytest

from sinkhorn_lab.exact import solve_exact
from sinkhorn_lab.metrics import lp_dual_bound
from sinkhorn_lab.problems import Problem, gaussian_clouds, random_cost, uniform_1d
from sinkhorn_lab.solvers import log_sinkhorn, vanilla_sinkhorn

pytestmark = pytest.mark.filterwarnings("error")

EPS = [1.0, 0.3, 0.1, 0.03, 0.01]      # in units of median(C) on the support
MAX_ITER = 20_000


def _nonuniform():
    rng = np.random.default_rng(10)
    p = gaussian_clouds(25, 20, separation=1.0, seed=10)
    return Problem(a=rng.dirichlet(np.ones(25)), b=rng.dirichlet(np.ones(20)), C=p.C, meta={})


def _zero_weights():
    p = gaussian_clouds(24, 18, seed=8)
    a, b = p.a.copy(), p.b.copy()
    a[[0, 5, 9]] = 0.0
    b[[3, 11]] = 0.0
    return Problem(a=a / a.sum(), b=b / b.sum(), C=p.C, meta={})


PROBLEMS = {
    "clouds_30x30": lambda: gaussian_clouds(30, 30, normalize="median", seed=1),
    "clouds_25x40_sep2": lambda: gaussian_clouds(25, 40, separation=2.0, normalize="median", seed=2),
    "random_30x20": lambda: random_cost(30, 20, normalize="median", seed=3),
    "uniform_1d_30": lambda: uniform_1d(30, seed=4),
    "nonuniform_25x20": _nonuniform,
    "zero_weights_24x18": _zero_weights,
}


def _scale(p):
    return np.median(p.C[np.ix_(p.a > 0, p.b > 0)])


@pytest.fixture(scope="module", params=list(PROBLEMS), ids=list(PROBLEMS))
def sweep(request):
    """LP optimum and dual bounds along the eps sweep, for one problem."""
    p = PROBLEMS[request.param]()
    exact = solve_exact(p)
    s = _scale(p)
    runs = []
    for e in EPS:
        r = log_sinkhorn(p.a, p.b, p.C, e * s, max_iter=MAX_ITER)
        runs.append((e, r, lp_dual_bound(r.f, r.g, p.a, p.b, p.C)))
    return p, exact, runs


def test_bound_never_exceeds_lp(sweep):
    _p, exact, runs = sweep
    for _, _, bound in runs:
        assert bound <= exact.cost + 1e-12 * max(1.0, abs(exact.cost))


def test_bound_gap_contains_true_gap(sweep):
    # cost - bound >= cost - LP: the bound never understates the overpay.
    _p, exact, runs = sweep
    for _, r, bound in runs:
        assert r.cost - bound >= r.cost - exact.cost - 1e-12 * max(1.0, abs(exact.cost))


def test_bound_tightens_as_eps_shrinks(sweep):
    _p, exact, runs = sweep
    gaps = [(exact.cost - bound) / abs(exact.cost) for _, _, bound in runs]
    for g0, g1 in itertools.pairwise(gaps):
        assert g1 <= g0 + 1e-12
    assert gaps[-1] < 1e-2                  # within 1% of the LP optimum at eps = 0.01 median(C)
    assert gaps[-1] <= 0.1 * gaps[0]        # at least 10x tighter than at eps = median(C)


def test_exact_duals_recover_lp_optimum(sweep):
    p, exact, _ = sweep
    assert lp_dual_bound(exact.f, exact.g, p.a, p.b, p.C) == pytest.approx(exact.cost, rel=1e-10, abs=1e-12)


def test_bound_matches_for_both_solvers():
    p = PROBLEMS["clouds_25x40_sep2"]()
    eps = 0.1 * _scale(p)
    v = vanilla_sinkhorn(p.a, p.b, p.C, eps)
    lg = log_sinkhorn(p.a, p.b, p.C, eps)
    assert v.converged and lg.converged
    assert lp_dual_bound(v.f, v.g, p.a, p.b, p.C) == pytest.approx(
        lp_dual_bound(lg.f, lg.g, p.a, p.b, p.C), rel=1e-10)


@pytest.mark.parametrize("seed", range(5))
def test_any_finite_duals_give_a_valid_bound(seed):
    # Weak duality after the c-transform holds for arbitrary f, g, not only Sinkhorn output.
    rng = np.random.default_rng(seed)
    p = random_cost(15, 12, seed=seed)
    lp = solve_exact(p).cost
    f, g = rng.normal(0, 5, 15), rng.normal(0, 5, 12)
    assert lp_dual_bound(f, g, p.a, p.b, p.C) <= lp + 1e-12


def test_invariant_to_constant_dual_shift():
    p = _nonuniform()
    r = log_sinkhorn(p.a, p.b, p.C, 0.3 * _scale(p))
    base = lp_dual_bound(r.f, r.g, p.a, p.b, p.C)
    assert lp_dual_bound(r.f + 3.7, r.g - 3.7, p.a, p.b, p.C) == pytest.approx(base, rel=1e-12)


def test_zero_weights_match_reduced_problem():
    p = _zero_weights()
    rows, cols = p.a > 0, p.b > 0
    eps = 0.3 * _scale(p)
    r = log_sinkhorn(p.a, p.b, p.C, eps)
    assert np.isnan(r.f[~rows]).all() and np.isnan(r.g[~cols]).all()
    a, b, C = p.a[rows], p.b[cols], p.C[np.ix_(rows, cols)]
    red = log_sinkhorn(a, b, C, eps)
    assert lp_dual_bound(r.f, r.g, p.a, p.b, p.C) == pytest.approx(lp_dual_bound(red.f, red.g, a, b, C), rel=1e-12)


def _bad_inputs():
    a, b, C = np.ones(3) / 3, np.ones(2) / 2, np.ones((3, 2))
    f, g = np.zeros(3), np.zeros(2)
    return [
        ("shape_C", (f, g, a, b, np.ones((2, 3)))),
        ("shape_f", (np.zeros(2), g, a, b, C)),
        ("nan_on_support", (np.array([np.nan, 0.0, 0.0]), g, a, b, C)),
        ("a_negative", (f, g, np.array([1.5, -0.5, 0.0]), b, C)),
        ("b_negative", (f, g, a, np.array([1.5, -0.5]), C)),
        ("a_all_zero", (f, g, np.zeros(3), b, C)),
    ]


@pytest.mark.parametrize("args", [x for _, x in _bad_inputs()], ids=[n for n, _ in _bad_inputs()])
def test_rejects_bad_input(args):
    with pytest.raises(ValueError):
        lp_dual_bound(*args)
