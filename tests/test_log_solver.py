"""Log-domain solver: agreement with vanilla, and stability where vanilla fails.

Shared properties (marginals, LP gap, D1, D2, status, errors) run against
both solvers in test_solvers.py.
"""

import numpy as np
import pytest

from sinkhorn_lab.problems import gaussian_clouds, random_cost, uniform_1d
from sinkhorn_lab.solvers import log_sinkhorn, slack, softmin, vanilla_sinkhorn

pytestmark = pytest.mark.filterwarnings("error")


def row_err(P, a):
    return np.abs(P.sum(axis=1) - a).sum()


# --- helpers ----------------------------------------------------------------

@pytest.mark.parametrize("axis", [0, 1])
def test_softmin_matches_direct_formula(axis):
    A = np.random.default_rng(0).uniform(0, 3, (6, 4))
    eps = 0.7
    direct = -eps * np.log(np.exp(-A / eps).sum(axis=axis))
    np.testing.assert_allclose(softmin(A, eps, axis), direct, rtol=1e-13)


def test_softmin_is_stable_where_direct_formula_underflows():
    # exp(-A/eps) underflows to 0 for every entry; softmin stays exact.
    A = 1000.0 + np.array([[0.0, 1e-3, 2e-3]])
    eps = 1e-3
    expected = 1000.0 - eps * np.log(1 + np.exp(-1.0) + np.exp(-2.0))
    assert softmin(A, eps, 1)[0] == pytest.approx(expected, rel=1e-15)


def test_softmin_approaches_hard_min_as_eps_shrinks():
    A = np.random.default_rng(1).uniform(0, 5, (5, 7))
    np.testing.assert_allclose(softmin(A, 1e-6, 1), A.min(axis=1), atol=1e-5)
    np.testing.assert_allclose(softmin(A, 1e-6, 0), A.min(axis=0), atol=1e-5)


def test_slack_definition():
    rng = np.random.default_rng(2)
    C, f, g = rng.random((4, 3)), rng.random(4), rng.random(3)
    S = slack(C, f, g)
    for i in range(4):
        for j in range(3):
            assert S[i, j] == pytest.approx(C[i, j] - f[i] - g[j], abs=1e-15)


# --- agreement with vanilla -------------------------------------------------

AGREE = [
    ("clouds_sep2_0.1", lambda: gaussian_clouds(40, 40, separation=2.0, seed=1), 0.1),
    ("clouds_rect_0.03", lambda: gaussian_clouds(30, 55, seed=2), 0.03),
    ("random_0.03", lambda: random_cost(35, 25, seed=3), 0.03),
    ("uniform_1d_0.03", lambda: uniform_1d(40, seed=4), 0.03),
]


@pytest.mark.parametrize("stabilize", ["none", "rowcol"])
@pytest.mark.parametrize("make, eps", [(m, e) for _, m, e in AGREE], ids=[n for n, _, _ in AGREE])
def test_matches_vanilla_where_vanilla_converges(make, eps, stabilize):
    p = make()
    v = vanilla_sinkhorn(p.a, p.b, p.C, eps, stabilize=stabilize)
    lg = log_sinkhorn(p.a, p.b, p.C, eps, stabilize=stabilize)
    assert v.converged and lg.converged
    assert lg.n_iter == v.n_iter
    np.testing.assert_allclose(lg.P, v.P, rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(lg.f[:, None] + lg.g, v.f[:, None] + v.g, rtol=0, atol=1e-8)
    assert lg.cost == pytest.approx(v.cost, rel=1e-10)


@pytest.mark.parametrize("cap", [0, 1, 7])
def test_matches_vanilla_on_capped_runs(cap):
    p = gaussian_clouds(25, 25, separation=1.0, seed=3)
    v = vanilla_sinkhorn(p.a, p.b, p.C, 0.2, max_iter=cap)
    lg = log_sinkhorn(p.a, p.b, p.C, 0.2, max_iter=cap)
    assert lg.status == v.status == "max_iter"
    np.testing.assert_allclose(lg.P, v.P, rtol=1e-10, atol=1e-16)
    assert lg.err == pytest.approx(v.err, rel=1e-9)


# --- stability where vanilla fails ------------------------------------------

def test_converges_where_vanilla_underflows_without_shift():
    p = random_cost(20, 20, seed=0)
    C = p.C + 1000.0
    assert vanilla_sinkhorn(p.a, p.b, C, 1.0, stabilize="none").status == "numerical_failure"
    lg = log_sinkhorn(p.a, p.b, C, 1.0, stabilize="none")
    ref = log_sinkhorn(p.a, p.b, p.C, 1.0, stabilize="rowcol")
    assert lg.converged
    np.testing.assert_allclose(lg.P, ref.P, rtol=1e-8, atol=1e-14)
    assert lg.err == pytest.approx(row_err(lg.P, p.a), rel=1e-6, abs=1e-15)


@pytest.mark.parametrize("stabilize", ["none", "rowcol"])
def test_stays_finite_where_vanilla_fails_with_shift(stabilize):
    p = gaussian_clouds(20, 20, separation=3.0, seed=0)
    eps = 0.003
    assert vanilla_sinkhorn(p.a, p.b, p.C, eps, stabilize="rowcol", max_iter=1000).status == "numerical_failure"
    early = log_sinkhorn(p.a, p.b, p.C, eps, stabilize=stabilize, max_iter=50)
    late = log_sinkhorn(p.a, p.b, p.C, eps, stabilize=stabilize, max_iter=1000)
    for r in (early, late):
        assert r.status == "max_iter"
        assert np.all(np.isfinite(r.P)) and np.all(np.isfinite(r.f)) and np.all(np.isfinite(r.g))
        assert np.isfinite(r.err) and r.err == pytest.approx(row_err(r.P, p.a), rel=1e-9)
        assert np.abs(r.P.sum(axis=0) - p.b).sum() < 1e-12
    assert late.err < early.err


@pytest.mark.parametrize("eps", [1e-4, 1e-8])
def test_extreme_eps_never_reports_numerical_failure(eps):
    p = gaussian_clouds(15, 15, separation=5.0, seed=4)
    r = log_sinkhorn(p.a, p.b, p.C, eps, max_iter=30)
    assert r.status == "max_iter"
    assert np.all(np.isfinite(r.P)) and np.isfinite(r.err) and np.isfinite(r.cost)
