import numpy as np
import pytest

from sinkhorn_lab.problems import (
    MASS_RTOL,
    Problem,
    _validate,
    gaussian_clouds,
    random_cost,
    uniform_1d,
)

GENERATORS = [
    ("gaussian_clouds", lambda seed: gaussian_clouds(7, 5, seed=seed)),
    ("gaussian_clouds_d3_sep", lambda seed: gaussian_clouds(6, 9, d=3, separation=4.0, seed=seed)),
    ("gaussian_clouds_median", lambda seed: gaussian_clouds(5, 5, normalize="median", seed=seed)),
    ("gaussian_clouds_max", lambda seed: gaussian_clouds(5, 5, normalize="max", seed=seed)),
    ("random_cost", lambda seed: random_cost(4, 6, seed=seed)),
    ("random_cost_median", lambda seed: random_cost(4, 6, normalize="median", seed=seed)),
    ("random_cost_max", lambda seed: random_cost(4, 6, normalize="max", seed=seed)),
    ("uniform_1d", lambda seed: uniform_1d(8, seed=seed)),
]
IDS = [name for name, _ in GENERATORS]
GENS = [gen for _, gen in GENERATORS]


@pytest.mark.parametrize("gen", GENS, ids=IDS)
def test_generator_output_is_valid(gen):
    p = gen(0)
    assert isinstance(p, Problem)
    n, m = p.C.shape
    assert p.a.shape == (n,) and p.b.shape == (m,)
    for x in (p.a, p.b, p.C):
        assert x.dtype == np.float64
        assert np.all(np.isfinite(x))
    assert np.all(p.a >= 0) and np.all(p.b >= 0)
    assert p.a.sum() == pytest.approx(1.0) and p.b.sum() == pytest.approx(1.0)
    assert np.all(p.C >= 0)
    _validate(p.a, p.b, p.C)


@pytest.mark.parametrize("gen", GENS, ids=IDS)
def test_generator_is_deterministic_in_seed(gen):
    p1, p2, p3 = gen(3), gen(3), gen(4)
    np.testing.assert_array_equal(p1.C, p2.C)
    np.testing.assert_array_equal(p1.a, p2.a)
    np.testing.assert_array_equal(p1.b, p2.b)
    assert not np.array_equal(p1.C, p3.C)
    assert p1.meta["seed"] == 3 and p3.meta["seed"] == 4


@pytest.mark.parametrize("gen", GENS, ids=IDS)
def test_generator_does_not_touch_global_rng(gen):
    np.random.seed(12345)
    before = np.random.random()
    np.random.seed(12345)
    gen(0)
    after = np.random.random()
    assert before == after


@pytest.mark.parametrize("gen", GENS, ids=IDS)
def test_problem_arrays_are_read_only(gen):
    p = gen(0)
    for x in (p.a, p.b, p.C):
        with pytest.raises(ValueError):
            x[0] = 1.0


def test_gaussian_clouds_cost_matches_points():
    p = gaussian_clouds(4, 3, d=2, separation=2.5, seed=1)
    x, y = p.meta["x"], p.meta["y"]
    expected = ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=-1)
    np.testing.assert_allclose(p.C, expected, rtol=1e-12)
    assert p.meta["separation"] == 2.5


def test_gaussian_clouds_separation_shifts_target_mean():
    p = gaussian_clouds(2000, 2000, d=2, separation=5.0, seed=0)
    shift = p.meta["y"].mean(axis=0) - p.meta["x"].mean(axis=0)
    assert shift[0] == pytest.approx(5.0, abs=0.2)
    assert shift[1] == pytest.approx(0.0, abs=0.2)


def test_gaussian_clouds_default_is_raw_cost():
    p = gaussian_clouds(6, 6, seed=2)
    assert p.meta["normalize"] == "none" and p.meta["c_scale"] == 1.0
    x, y = p.meta["x"], p.meta["y"]
    np.testing.assert_array_equal(p.C, ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=-1))


@pytest.mark.parametrize("n, m", [(6, 6), (7, 4), (50, 50)])
def test_gaussian_clouds_median_normalization(n, m):
    raw = gaussian_clouds(n, m, separation=3.0, seed=2)
    med = gaussian_clouds(n, m, separation=3.0, normalize="median", seed=2)
    assert np.median(med.C) == pytest.approx(1.0, rel=1e-12)
    assert med.meta["c_scale"] == pytest.approx(np.median(raw.C), rel=1e-12)
    np.testing.assert_allclose(med.C, raw.C / np.median(raw.C), rtol=1e-12)
    np.testing.assert_allclose(med.C * med.meta["c_scale"], raw.C, rtol=1e-12)


def test_gaussian_clouds_max_normalization():
    raw = gaussian_clouds(6, 9, separation=3.0, seed=2)
    mx = gaussian_clouds(6, 9, separation=3.0, normalize="max", seed=2)
    assert mx.C.max() == pytest.approx(1.0, rel=1e-12)
    assert mx.meta["c_scale"] == pytest.approx(raw.C.max(), rel=1e-12)
    np.testing.assert_allclose(mx.C, raw.C / raw.C.max(), rtol=1e-12)


def test_gaussian_clouds_normalizations_share_points():
    ps = [gaussian_clouds(5, 5, normalize=k, seed=8) for k in ("none", "median", "max")]
    for p in ps[1:]:
        np.testing.assert_array_equal(p.meta["x"], ps[0].meta["x"])
        np.testing.assert_array_equal(p.meta["y"], ps[0].meta["y"])


@pytest.mark.parametrize("gen", [gaussian_clouds, random_cost], ids=["gaussian_clouds", "random_cost"])
@pytest.mark.parametrize("bad", ["mean", "Median", None, True])
def test_rejects_unknown_normalization(gen, bad):
    with pytest.raises(ValueError, match="normalize"):
        gen(4, 4, normalize=bad, seed=0)


def test_random_cost_default_is_raw_cost():
    p = random_cost(6, 5, seed=2)
    assert p.meta["normalize"] == "none" and p.meta["c_scale"] == 1.0
    np.testing.assert_array_equal(p.C, np.random.default_rng(2).random((6, 5)))


@pytest.mark.parametrize("n, m", [(6, 6), (7, 4), (50, 50)])
def test_random_cost_median_normalization(n, m):
    raw = random_cost(n, m, seed=2)
    med = random_cost(n, m, normalize="median", seed=2)
    assert np.median(med.C) == pytest.approx(1.0, rel=1e-12)
    assert med.meta["c_scale"] == pytest.approx(np.median(raw.C), rel=1e-12)
    np.testing.assert_allclose(med.C, raw.C / np.median(raw.C), rtol=1e-12)
    np.testing.assert_allclose(med.C * med.meta["c_scale"], raw.C, rtol=1e-12)


def test_random_cost_max_normalization():
    raw = random_cost(6, 9, seed=2)
    mx = random_cost(6, 9, normalize="max", seed=2)
    assert mx.C.max() == pytest.approx(1.0, rel=1e-12)
    assert mx.meta["c_scale"] == pytest.approx(raw.C.max(), rel=1e-12)
    np.testing.assert_allclose(mx.C, raw.C / raw.C.max(), rtol=1e-12)


def test_uniform_1d_stores_points_and_cost():
    p = uniform_1d(6, seed=5)
    x, y = p.meta["x"], p.meta["y"]
    assert x.shape == (6,) and y.shape == (6,)
    np.testing.assert_allclose(p.C, (x[:, None] - y[None, :]) ** 2, rtol=1e-12)


# --- _validate -------------------------------------------------------------

def test_validate_accepts_zero_weights():
    a = np.array([0.5, 0.0, 0.5])
    b = np.array([0.0, 1.0])
    _validate(a, b, np.ones((3, 2)))


def test_validate_accepts_mass_other_than_one():
    _validate(np.array([2.0, 3.0]), np.array([5.0]), np.ones((2, 1)))


@pytest.mark.parametrize(
    "a, b, C, match",
    [
        (np.ones(3) / 3, np.ones(2) / 2, np.ones((2, 3)), "shape"),
        (np.ones((3, 1)) / 3, np.ones(2) / 2, np.ones((3, 2)), "1-D"),
        (np.array([]), np.array([]), np.ones((0, 0)), "non-empty"),
        (np.array([1.5, -0.5]), np.array([1.0]), np.ones((2, 1)), "nonnegative"),
        (np.array([0.0, 0.0]), np.array([0.0]), np.ones((2, 1)), "positive"),
        (np.array([0.5, 0.5]), np.array([1.0 + 1e-9]), np.ones((2, 1)), "masses differ"),
        (np.array([np.nan, 1.0]), np.array([1.0]), np.ones((2, 1)), "non-finite"),
        (np.array([1.0]), np.array([1.0]), np.array([[np.inf]]), "non-finite"),
        (np.array([True]), np.array([True]), np.ones((1, 1)), "real-valued"),
    ],
)
def test_validate_rejects(a, b, C, match):
    with pytest.raises(ValueError, match=match):
        _validate(a, b, C)


def test_validate_mass_tolerance_is_relative():
    big = 1e6
    _validate(np.array([big]), np.array([big * (1 + 0.5 * MASS_RTOL)]), np.ones((1, 1)))
    with pytest.raises(ValueError, match="masses differ"):
        _validate(np.array([big]), np.array([big * (1 + 10 * MASS_RTOL)]), np.ones((1, 1)))
