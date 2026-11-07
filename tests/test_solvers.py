"""Property tests shared by both Sinkhorn solvers, plus vanilla-only failure tests.

LP references are independent of POT: scipy's linear_sum_assignment for
square uniform problems and scipy's HiGHS LP otherwise. Every test runs with
warnings as errors, so a solver that starts warning fails here.
"""

import itertools

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment, linprog

from sinkhorn_lab.problems import Problem, gaussian_clouds, random_cost, uniform_1d
from sinkhorn_lab.solvers import log_sinkhorn, rowcol_shift, vanilla_sinkhorn

pytestmark = pytest.mark.filterwarnings("error")

TOL = 1e-9          # solver default stopping tolerance
MARG_TOL = 1e-8     # test tolerance on the checked (row) marginal, looser than TOL
COL_TOL = 1e-12     # column marginal is exact up to rounding after the g/v update

SOLVERS = [vanilla_sinkhorn, log_sinkhorn]
SOLVER_IDS = ["vanilla", "log"]

def nonuniform_weights():
    """Generators all use uniform weights; this one does not, so a solver that
    confuses a with b, or uses a weight's mean, is caught."""
    rng = np.random.default_rng(10)
    p = gaussian_clouds(25, 20, separation=1.0, seed=10)
    a, b = rng.dirichlet(np.ones(25)), rng.dirichlet(np.ones(20))
    return Problem(a=a, b=b, C=p.C, meta={"name": "nonuniform"})


PROBLEMS = {
    "clouds_sep2": lambda: gaussian_clouds(40, 40, separation=2.0, seed=1),
    "clouds_rect": lambda: gaussian_clouds(30, 55, seed=2),
    "random_rect": lambda: random_cost(35, 25, seed=3),
    "uniform_1d": lambda: uniform_1d(40, seed=4),
    "nonuniform": nonuniform_weights,
}


def row_err(P, a):
    return np.abs(P.sum(axis=1) - a).sum()


def col_err(P, b):
    return np.abs(P.sum(axis=0) - b).sum()


def lp_ref(a, b, C):
    """Optimal transport cost, independent of POT."""
    n, m = C.shape
    if n == m and np.allclose(a, 1 / n) and np.allclose(b, 1 / m):
        r, c = linear_sum_assignment(C)
        return C[r, c].sum() / n
    A_eq = np.vstack([np.kron(np.eye(n), np.ones((1, m))), np.kron(np.ones((1, n)), np.eye(m))])
    res = linprog(C.ravel(), A_eq=A_eq, b_eq=np.concatenate([a, b]), bounds=(0, None), method="highs")
    assert res.status == 0
    return res.fun


def gibbs(f, g, C, eps):
    return np.exp((f[:, None] + g[None, :] - C) / eps)


# --- rowcol_shift -----------------------------------------------------------

@pytest.mark.parametrize("make", PROBLEMS.values(), ids=PROBLEMS.keys())
def test_rowcol_shift_properties(make):
    C = make().C
    Cs, alpha, beta = rowcol_shift(C)
    assert np.all(Cs >= 0)
    assert np.all(Cs.min(axis=1) == 0) and np.all(Cs.min(axis=0) == 0)
    np.testing.assert_allclose(Cs + alpha[:, None] + beta[None, :], C, rtol=0, atol=1e-12 * np.abs(C).max())


# --- converged-run properties -------------------------------------------------

@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
@pytest.mark.parametrize("make", PROBLEMS.values(), ids=PROBLEMS.keys())
@pytest.mark.parametrize("eps", [1.0, 0.3])
def test_converged_run_properties(solver, make, eps):
    p = make()
    r = solver(p.a, p.b, p.C, eps)
    assert r.status == "converged" and r.converged is True
    assert 0 < r.n_iter < 10_000
    assert r.err < TOL
    assert r.err == pytest.approx(row_err(r.P, p.a), rel=1e-6, abs=1e-15)
    assert row_err(r.P, p.a) < MARG_TOL
    assert col_err(r.P, p.b) < COL_TOL
    assert r.P.shape == p.C.shape and np.all(np.isfinite(r.P)) and np.all(r.P >= 0)
    assert r.cost == pytest.approx(np.sum(r.P * p.C), rel=1e-14)


@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
@pytest.mark.parametrize("make", PROBLEMS.values(), ids=PROBLEMS.keys())
def test_plan_is_gibbs_form_of_reported_duals(solver, make):
    # Duals are reported on the original C (D2). This catches a flipped sign
    # in K or a missing/doubled alpha, beta, which marginals alone miss.
    p = make()
    eps = 0.3
    r = solver(p.a, p.b, p.C, eps)
    np.testing.assert_allclose(gibbs(r.f, r.g, p.C, eps), r.P, rtol=1e-10, atol=1e-16)


@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
@pytest.mark.parametrize("make", PROBLEMS.values(), ids=PROBLEMS.keys())
def test_cost_never_beats_lp(solver, make):
    p = make()
    lp = lp_ref(p.a, p.b, p.C)
    for eps in [1.0, 0.3, 0.1]:
        r = solver(p.a, p.b, p.C, eps, max_iter=2000)
        # A plan off the marginals by err can undercut the LP by at most err * max|C|.
        slack = (row_err(r.P, p.a) + col_err(r.P, p.b)) * np.abs(p.C).max()
        assert r.cost >= lp - slack - 1e-12


@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
@pytest.mark.parametrize("name", ["clouds_rect", "random_rect", "uniform_1d", "nonuniform"])
def test_cost_approaches_lp_as_eps_shrinks(solver, name):
    p = PROBLEMS[name]()
    lp = lp_ref(p.a, p.b, p.C)
    gaps = []
    for eps in [1.0, 0.3, 0.1, 0.03]:
        r = solver(p.a, p.b, p.C, eps)
        assert r.converged
        gaps.append(r.cost - lp)
    assert all(g >= 0 for g in gaps)
    assert all(g1 < g0 for g0, g1 in itertools.pairwise(gaps))
    assert gaps[-1] < 0.1 * gaps[0]


@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
def test_large_eps_gives_independent_coupling(solver):
    p = gaussian_clouds(30, 20, separation=1.0, seed=5)
    r = solver(p.a, p.b, p.C, 1e5)
    assert r.converged
    np.testing.assert_allclose(r.P, np.outer(p.a, p.b), rtol=0, atol=1e-6)


@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
def test_single_source_point_is_exact(solver):
    b = np.array([0.2, 0.5, 0.3])
    r = solver(np.array([1.0]), b, np.array([[3.0, 1.0, 2.0]]), 0.1)
    assert r.converged
    np.testing.assert_allclose(r.P, b[None, :], rtol=1e-12)


# --- invariances (D2 bookkeeping) -------------------------------------------

@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
@pytest.mark.parametrize("stabilize", ["none", "rowcol"])
def test_row_column_offsets_leave_plan_unchanged(solver, stabilize):
    # C + r_i + s_j has the same plan; the duals absorb the offsets.
    p = random_cost(20, 15, seed=6)
    rng = np.random.default_rng(0)
    r_off, s_off = rng.uniform(-2, 2, 20), rng.uniform(-2, 2, 15)
    C2 = p.C + r_off[:, None] + s_off[None, :]
    eps = 0.1
    r1 = solver(p.a, p.b, p.C, eps, stabilize=stabilize)
    r2 = solver(p.a, p.b, C2, eps, stabilize=stabilize)
    np.testing.assert_allclose(r2.P, r1.P, rtol=1e-8, atol=1e-14)
    fg1 = r1.f[:, None] + r1.g[None, :]
    fg2 = r2.f[:, None] + r2.g[None, :]
    np.testing.assert_allclose(fg2, fg1 + r_off[:, None] + s_off[None, :], rtol=0, atol=1e-8)
    assert r2.cost == pytest.approx(np.sum(r2.P * C2), rel=1e-14)


@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
def test_plan_depends_only_on_c_over_eps(solver):
    p = gaussian_clouds(25, 25, seed=7)
    eps, k = 0.5, 37.0
    r1 = solver(p.a, p.b, p.C, eps)
    r2 = solver(p.a, p.b, k * p.C, k * eps)
    np.testing.assert_allclose(r2.P, r1.P, rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(r2.f[:, None] + r2.g, k * (r1.f[:, None] + r1.g), rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
@pytest.mark.parametrize("make", PROBLEMS.values(), ids=PROBLEMS.keys())
def test_stabilize_options_agree_where_both_converge(solver, make):
    p = make()
    r0 = solver(p.a, p.b, p.C, 0.3, stabilize="none")
    r1 = solver(p.a, p.b, p.C, 0.3, stabilize="rowcol")
    assert r0.converged and r1.converged
    np.testing.assert_allclose(r1.P, r0.P, rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(r1.f[:, None] + r1.g, r0.f[:, None] + r0.g, rtol=0, atol=1e-8)


# --- D1: zero weights -------------------------------------------------------

@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
@pytest.mark.parametrize("stabilize", ["none", "rowcol"])
def test_zero_weights_match_reduced_problem(solver, stabilize):
    p = gaussian_clouds(12, 10, seed=8)
    a, b = p.a.copy(), p.b.copy()
    a[[0, 5]] = 0.0
    b[[3, 9]] = 0.0
    a /= a.sum()
    b /= b.sum()
    rows, cols = a > 0, b > 0
    eps = 0.5
    r = solver(a, b, p.C, eps, stabilize=stabilize)
    red = solver(a[rows], b[cols], p.C[np.ix_(rows, cols)], eps, stabilize=stabilize)
    assert r.converged and red.converged and r.n_iter == red.n_iter
    assert np.all(np.isnan(r.f[~rows])) and np.all(np.isnan(r.g[~cols]))
    assert np.all(np.isfinite(r.f[rows])) and np.all(np.isfinite(r.g[cols]))
    assert np.all(r.P[~rows, :] == 0) and np.all(r.P[:, ~cols] == 0)
    np.testing.assert_array_equal(r.P[np.ix_(rows, cols)], red.P)
    np.testing.assert_array_equal(r.f[rows], red.f)
    np.testing.assert_array_equal(r.g[cols], red.g)
    assert r.cost == pytest.approx(red.cost, rel=1e-14)


# --- status, n_iter, err ----------------------------------------------------

@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
@pytest.mark.parametrize("cap", [0, 1, 3, 50])
def test_capped_run_reports_error_of_returned_plan(solver, cap):
    p = gaussian_clouds(30, 30, separation=3.0, seed=0)
    r = solver(p.a, p.b, p.C, 0.05, max_iter=cap)
    assert r.status == "max_iter" and r.converged is False
    assert r.n_iter == cap
    assert np.isfinite(r.err) and r.err > TOL
    assert r.err == pytest.approx(row_err(r.P, p.a), rel=1e-9)
    assert np.all(np.isfinite(r.P))


@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
def test_tighter_tolerance_takes_more_iterations(solver):
    p = PROBLEMS["clouds_rect"]()
    loose = solver(p.a, p.b, p.C, 0.3, tol=1e-4)
    tight = solver(p.a, p.b, p.C, 0.3, tol=1e-10)
    assert loose.converged and tight.converged
    assert loose.n_iter < tight.n_iter
    assert loose.err < 1e-4 and tight.err < 1e-10


@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
def test_deterministic_and_does_not_mutate_inputs(solver):
    p = random_cost(15, 12, seed=9)
    a, b, C = p.a.copy(), p.b.copy(), p.C.copy()
    r1 = solver(a, b, C, 0.2)
    r2 = solver(a, b, C, 0.2)
    np.testing.assert_array_equal(a, p.a)
    np.testing.assert_array_equal(b, p.b)
    np.testing.assert_array_equal(C, p.C)
    np.testing.assert_array_equal(r1.P, r2.P)
    assert r1.n_iter == r2.n_iter and r1.err == r2.err


# --- input errors -----------------------------------------------------------

def _bad_inputs():
    a, b, C = np.ones(3) / 3, np.ones(2) / 2, np.ones((3, 2))
    return [
        ("eps_zero", {"a": a, "b": b, "C": C, "eps": 0.0}),
        ("eps_negative", {"a": a, "b": b, "C": C, "eps": -1.0}),
        ("eps_nan", {"a": a, "b": b, "C": C, "eps": np.nan}),
        ("stabilize_typo", {"a": a, "b": b, "C": C, "eps": 1.0, "stabilize": "rowcols"}),
        ("stabilize_none_obj", {"a": a, "b": b, "C": C, "eps": 1.0, "stabilize": None}),
        ("a_negative", {"a": np.array([1.5, -0.5, 0.0]), "b": b, "C": C, "eps": 1.0}),
        ("b_negative", {"a": a, "b": np.array([1.5, -0.5]), "C": C, "eps": 1.0}),
        ("a_all_zero", {"a": np.zeros(3), "b": b, "C": C, "eps": 1.0}),
        ("b_all_zero", {"a": a, "b": np.zeros(2), "C": C, "eps": 1.0}),
    ]


@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
@pytest.mark.parametrize("kwargs", [k for _, k in _bad_inputs()], ids=[n for n, _ in _bad_inputs()])
def test_rejects_bad_input(solver, kwargs):
    with pytest.raises(ValueError):
        solver(**kwargs)


# --- vanilla failure detection ----------------------------------------------

def test_vanilla_flags_underflow_without_shift():
    # Every kernel entry underflows: C/eps > 1000 everywhere.
    p = random_cost(20, 20, seed=0)
    r = vanilla_sinkhorn(p.a, p.b, p.C + 1000.0, 1.0, stabilize="none")
    assert r.status == "numerical_failure" and r.converged is False
    assert r.n_iter == 1 and np.isnan(r.err)


def test_vanilla_shift_rescues_constant_offset():
    p = random_cost(20, 20, seed=0)
    r = vanilla_sinkhorn(p.a, p.b, p.C + 1000.0, 1.0, stabilize="rowcol")
    ref = vanilla_sinkhorn(p.a, p.b, p.C, 1.0, stabilize="rowcol")
    assert r.converged
    np.testing.assert_allclose(r.P, ref.P, rtol=1e-10, atol=1e-16)


def test_vanilla_flags_failure_with_shift():
    # Separated clouds at small eps: the shifted kernel still underflows and
    # the scalings overflow mid-run (n_iter 333 when written).
    p = gaussian_clouds(20, 20, separation=3.0, seed=0)
    r = vanilla_sinkhorn(p.a, p.b, p.C, 0.003, stabilize="rowcol", max_iter=1000)
    assert r.status == "numerical_failure" and r.converged is False
    assert 1 < r.n_iter < 1000 and np.isnan(r.err)


@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
def test_converging_on_the_last_allowed_iteration_reports_converged(solver):
    p = PROBLEMS["clouds_sep2"]()
    n = solver(p.a, p.b, p.C, 1.0).n_iter
    at_cap = solver(p.a, p.b, p.C, 1.0, max_iter=n)
    assert at_cap.status == "converged" and at_cap.converged is True
    assert at_cap.n_iter == n and at_cap.err < TOL
    short = solver(p.a, p.b, p.C, 1.0, max_iter=n - 1)
    assert short.status == "max_iter" and short.converged is False and short.err >= TOL


# --- large n ----------------------------------------------------------------

LARGE_N = 2000


@pytest.fixture(scope="module")
def large_problem():
    p = gaussian_clouds(LARGE_N, LARGE_N, normalize="median", seed=0)
    r, c = linear_sum_assignment(p.C)
    return p, p.C[r, c].sum() / LARGE_N


@pytest.fixture(scope="module")
def large_runs(large_problem):
    p, _ = large_problem
    return {s.__name__: s(p.a, p.b, p.C, 1.0) for s in SOLVERS}


@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
def test_large_n_converged_run_properties(solver, large_problem, large_runs):
    p, lp = large_problem
    r = large_runs[solver.__name__]
    assert r.status == "converged" and r.converged is True
    assert r.err < TOL and r.err == pytest.approx(row_err(r.P, p.a), rel=1e-6, abs=1e-15)
    assert row_err(r.P, p.a) < MARG_TOL and col_err(r.P, p.b) < COL_TOL
    assert np.all(np.isfinite(r.P)) and np.all(r.P >= 0)
    assert r.cost == pytest.approx(np.sum(r.P * p.C), rel=1e-12)
    assert r.cost > lp
    np.testing.assert_allclose(gibbs(r.f, r.g, p.C, 1.0), r.P, rtol=1e-10, atol=1e-18)


def test_large_n_solvers_agree(large_runs):
    v, lg = large_runs["vanilla_sinkhorn"], large_runs["log_sinkhorn"]
    assert v.n_iter == lg.n_iter
    np.testing.assert_allclose(lg.P, v.P, rtol=1e-8, atol=1e-16)
    assert lg.cost == pytest.approx(v.cost, rel=1e-10)
