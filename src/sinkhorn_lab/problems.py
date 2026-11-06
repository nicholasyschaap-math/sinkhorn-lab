"""Problem container and seeded generators for discrete optimal transport.

Weights may contain zeros (see DECISIONS.md, D1); total mass must be positive
and equal on both sides. Every generator takes an explicit ``seed`` and draws
only from ``np.random.default_rng(seed)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Relative tolerance for the equal-mass check. Generators produce masses that
# agree to a few ulps; anything looser than this is a construction error, not
# rounding, and must not be silently rescaled away downstream.
MASS_RTOL = 1e-12


@dataclass(frozen=True)
class Problem:
    a: np.ndarray  # source weights, shape (n,)
    b: np.ndarray  # target weights, shape (m,)
    C: np.ndarray  # cost, shape (n, m)
    meta: dict = field(default_factory=dict)  # name, seed, parameters


def _validate(a, b, C) -> None:
    """Raise ValueError unless (a, b, C) is a well-posed balanced OT problem.

    Checks: a and b are 1-D, C is 2-D with shape (len(a), len(b)); all entries
    finite; weights nonnegative (zeros allowed); total mass positive and equal
    to relative tolerance MASS_RTOL.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    C = np.asarray(C)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(f"a and b must be 1-D, got shapes {a.shape} and {b.shape}")
    if a.size == 0 or b.size == 0:
        raise ValueError("a and b must be non-empty")
    if C.shape != (a.size, b.size):
        raise ValueError(f"C must have shape {(a.size, b.size)}, got {C.shape}")
    for name, x in (("a", a), ("b", b), ("C", C)):
        if not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x):
            raise ValueError(f"{name} must be real-valued, got dtype {x.dtype}")
        if not np.all(np.isfinite(x)):
            raise ValueError(f"{name} contains non-finite values")
    if np.any(a < 0) or np.any(b < 0):
        raise ValueError("weights must be nonnegative")
    mass_a, mass_b = float(a.sum()), float(b.sum())
    if mass_a <= 0 or mass_b <= 0:
        raise ValueError("total mass must be positive")
    if abs(mass_a - mass_b) > MASS_RTOL * max(mass_a, mass_b):
        raise ValueError(
            f"total masses differ: sum(a)={mass_a!r}, sum(b)={mass_b!r} "
            f"(rtol {MASS_RTOL})"
        )


def _make(a, b, C, meta) -> Problem:
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    C = np.array(C, dtype=np.float64)
    _validate(a, b, C)
    for x in (a, b, C):
        x.setflags(write=False)  # solvers must not modify problem data in place
    return Problem(a=a, b=b, C=C, meta=meta)


def _sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between rows of x (n, d) and y (m, d)."""
    diff = x[:, None, :] - y[None, :, :]
    return np.einsum("ijk,ijk->ij", diff, diff)


NORMALIZATIONS = ("none", "median", "max")


def gaussian_clouds(n, m, d=2, separation=0.0, normalize="none", seed=0) -> Problem:
    """Two standard Gaussian clouds in R^d with uniform weights.

    Source points x_i ~ N(0, I_d); target points y_j ~ N(separation * e_1, I_d).
    Raw cost C_ij = ||x_i - y_j||^2, then divided by a scale chosen by
    ``normalize`` (DECISIONS.md, D4):

    - "none":   raw C (scale 1). Keeps the separation visible in the cost scale.
    - "median": C / median(C), so the median cost is 1.
    - "max":    C / max(C), so the largest cost is 1.

    The divisor is stored in meta["c_scale"]; raw C = C * meta["c_scale"].
    Normalizing is a multiplication, so it changes the effective epsilon:
    (C / s, eps) is the same entropic problem as (C, s * eps).
    """
    if normalize not in NORMALIZATIONS:
        raise ValueError(f"normalize must be one of {NORMALIZATIONS}, got {normalize!r}")
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    y = rng.standard_normal((m, d))
    y[:, 0] += separation
    C = _sq_dists(x, y)
    if normalize == "median":
        c_scale = float(np.median(C))
    elif normalize == "max":
        c_scale = float(C.max())
    else:
        c_scale = 1.0
    if not c_scale > 0:
        raise ValueError(f"cannot normalize by {normalize}: scale is {c_scale!r}")
    C = C / c_scale
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
    meta = {
        "name": "gaussian_clouds",
        "seed": seed,
        "n": n,
        "m": m,
        "d": d,
        "separation": separation,
        "normalize": normalize,
        "c_scale": c_scale,
        "x": x,
        "y": y,
    }
    return _make(a, b, C, meta)


def random_cost(n, m, seed=0) -> Problem:
    """Unstructured cost C_ij ~ U[0, 1) i.i.d., uniform weights."""
    rng = np.random.default_rng(seed)
    C = rng.random((n, m))
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
    meta = {"name": "random_cost", "seed": seed, "n": n, "m": m}
    return _make(a, b, C, meta)


def uniform_1d(n, seed=0) -> Problem:
    """n points vs n points, each drawn U[0, 1), uniform weights, C_ij = (x_i - y_j)^2.

    Points are stored unsorted in meta["x"] and meta["y"]. For this convex cost
    and equal uniform weights the optimal plan matches sorted x to sorted y, so
    the exact cost is mean((sort(x) - sort(y))^2): an oracle independent of any
    LP solver.
    """
    rng = np.random.default_rng(seed)
    x = rng.random(n)
    y = rng.random(n)
    C = (x[:, None] - y[None, :]) ** 2
    a = np.full(n, 1.0 / n)
    b = np.full(n, 1.0 / n)
    meta = {"name": "uniform_1d", "seed": seed, "n": n, "p": 2, "x": x, "y": y}
    return _make(a, b, C, meta)
