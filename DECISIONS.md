# Design Decisions

Record of design decisions for sinkhorn-lab and the reasoning behind them.

Notation: a ∈ ℝⁿ₊, b ∈ ℝᵐ₊ weights with ⟨a, 1⟩ = ⟨b, 1⟩; C ∈ ℝⁿˣᵐ cost;
ε > 0 regularization; K = exp(−C/ε) the Gibbs kernel; f, g dual potentials.
The admissible plans are U(a, b) = {P ≥ 0 : P1 = a, Pᵀ1 = b}. Formulas here
are explanatory; `docs/derivation.md` is the formula source and wins on any
conflict.

---

## D1. Zero weights are accepted and handled on the support

*Status: decided by the author, 2026-11-06.*

**Decision.** Weights a and b may contain zeros. Validation rejects only
ill-posed inputs: negative or non-finite entries, zero total mass, or unequal
total mass. The solvers drop zero-weight points internally, solve on the
support, embed the plan back with zero rows and columns, and report the
duals of zero-weight points as NaN.

**Reasoning.**

- *Practice.* Zero weights occur naturally in data (empty histogram bins,
  pruned or masked points). Rejecting them at the door would make the
  solvers unusable on ordinary inputs.
- *Purpose of the project.* The project is partly optimization and partly a
  search for failure cases. Restricting the input domain in validation would
  remove observations before they can be made. Validation should exclude only
  problems that have no answer, not problems that are hard to answer.
- *Nothing is lost by dropping zeros.* Let I = {i : aᵢ > 0} and
  J = {j : bⱼ > 0}. For P ∈ U(a, b) and i ∉ I, Σⱼ Pᵢⱼ = 0 with Pᵢⱼ ≥ 0
  forces row i of P to be zero; the same holds for columns j ∉ J. So zero
  padding is a bijection between U(a_I, b_J) and U(a, b). It preserves
  ⟨P, C⟩, and with 0 log 0 = 0 it also preserves the entropy. Both the
  exact and the entropic problems therefore have the same solutions on
  (a_I, b_J, C_IJ) as on (a, b, C).
- *Why the duals are NaN.* For i ∉ I, fᵢ multiplies aᵢ = 0 in the dual
  objective, so the problem does not determine it.
  - In the LP dual, any fᵢ with fᵢ + gⱼ ≤ Cᵢⱼ is optimal.
  - In the entropic dual, the optimality condition
    fᵢ = ε log aᵢ − ε log Σⱼ exp((gⱼ − Cᵢⱼ)/ε) gives fᵢ = −∞. Equivalently,
    the scaling uᵢ = aᵢ / (Kv)ᵢ is 0.

  Reporting a finite value would invent a number. NaN says "undetermined"
  and propagates visibly if anything uses it by mistake.
- *Numerics.* Keeping zero weights inside the iterations produces
  log 0 = −∞ terms. In the log domain these can meet +∞ and become NaN
  (−∞ + ∞), which would be indistinguishable from a real numerical failure.
  Dropping them first keeps genuine failures identifiable.

**Where it applies.**

- `problems._validate` accepts zeros. It requires positive, equal total mass
  to relative tolerance 1e-12.
- `exact.solve_exact` follows the same convention: it solves on the support
  and returns NaN duals off the support. POT's own `ot.emd` would instead
  fill those duals with estimated finite values. *(Agent proposal for the
  baseline, awaiting author confirmation.)*
- `solvers.py` (author) will implement it for both Sinkhorn variants.

---

## D2. Cost is shifted by its minimum before forming the Gibbs kernel

*Status: decided by the author, 2026-11-06.*

**Decision.** The solvers form the kernel from C̃ = C − c_min·11ᵀ, where
c_min = minᵢⱼ Cᵢⱼ, so that K̃ = exp(−C̃/ε) has entries in (0, 1]. The
largest entry equals exp(0) = 1, at the position of the minimum cost. This is
a translation of C, not a multiplication, so the effective ε is unchanged.
The shift does not replace the log-sum-exp (log-domain) iteration. It gives
the vanilla solver a better-conditioned starting kernel, so ε can be pushed
closer to 0 before the log-domain method, and its cost, becomes necessary.

**Reasoning.**

- *The shift changes nothing about the solution.* For any P ∈ U(a, b),
  ⟨P, C̃⟩ = ⟨P, C⟩ − c_min·Σᵢⱼ Pᵢⱼ = ⟨P, C⟩ − c_min·⟨a, 1⟩, a constant on
  U(a, b).
  - The exact problem and the entropic problem ⟨P, C⟩ − εH(P) therefore have
    the same minimizers for C̃ as for C.
  - (f, g) is optimal for C̃ if and only if (f, g + c_min) is optimal for C,
    up to the usual (f + α, g − α) freedom.
  - The objective value shifts by exactly c_min·⟨a, 1⟩.
- *Why it helps.* K̃ = e^{c_min/ε}·K is a uniform rescaling of K, and Sinkhorn
  absorbs a uniform factor into the scalings: diag(u)·K̃·diag(v) =
  diag(e^{c_min/ε}u)·K·diag(v). Mathematically nothing changes. In floating
  point, though, K itself may not be representable. If c_min/ε is large, for
  example c_min = 10 and ε = 0.01, then every entry of K is at most
  e^{−1000}, which is 0 in float64. The whole kernel underflows even though
  the problem is perfectly well conditioned relative to its own costs. After
  the shift the kernel always contains a 1, and an entry underflows only if
  its own excess cost (Cᵢⱼ − c_min)/ε is large.
- *Underflow depends on the ratio of cost to ε, not on either alone.* In
  float64, exp(−x) leaves the normal range for x > 708.4 (log of the smallest
  normal number) and becomes exactly 0 for x ≳ 745.1. After the shift, every
  entry of K̃ is a normal number exactly when (max C − min C)/ε < 708.4. So
  the threshold depends on the cost range divided by ε. Scaling C and ε
  together by the same factor leaves K̃ unchanged, and adding a constant to C
  leaves it unchanged too. Neither C nor ε alone determines underflow. The
  author's derivations (`docs/derivation.md`, to be uploaded) develop this.
- *What the shift does not fix.*
  - Only one entry is guaranteed to equal 1. A row i whose smallest excess
    cost satisfies minⱼ C̃ᵢⱼ/ε ≳ 745 still has an all-zero row in K̃, giving
    (K̃v)ᵢ = 0 and a division by zero.
  - The scalings u and v can still overflow or underflow over the iterations
    as ε → 0.

  These are what the in-loop log-sum-exp stabilization handles, at its extra
  computational cost.

**Consequences and open points.**

- The exact baseline (`exact.py`) also shifts by the support's minimum cost
  before calling `ot.emd`. Raw `ot.emd` falsely reports "Problem infeasible"
  when all costs are negative and roughly constant below −N/(N−1),
  N = n + m (see `AGENTS_LOG.md`). The plan is unchanged by the argument
  above, min C is added back to g, and the cost and certificate are computed
  on the original C.
- *Open (author):* should solvers report cost and duals on the original C
  (⟨P, C⟩, g + c_min) or on C̃? This affects how stage 2 tests compare solver
  output with `solve_exact`, which reports on the original C.
- *Open (author):* `implementation.md` lists `C.max() / eps` as a candidate
  failure predictor for the timing experiment. Under D2, the quantity the
  argument above points to is the shifted range `(C.max() − C.min()) / eps`.
  The two coincide when min C ≈ 0, which is nearly true for the current
  generators. `implementation.md` has not been changed.

---

## D3. The exact baseline returns only certified answers

*Status: requirement from the author, 2026-11-06. Mechanism proposed by the
agent.*

**Decision.** `solve_exact` must always return the exact LP optimum, and
the wrapper itself must be able to show that float handling of the weights
did not corrupt it. Otherwise it raises `ExactSolverError`.

**Mechanism.**

- *Guard the inputs against POT's silent changes.* `ot.emd` checks equal
  mass only to 6 decimals and then rescales b to sum(a). It replaces empty
  weight vectors with uniform ones. It casts the plan to integers when the
  weights are integers. The wrapper validates mass to 1e-12 relative,
  rejects zero total mass, and converts everything to float64 before
  calling POT.
- *Certify the output against the caller's original (a, b, C).* LP duality
  says that (P, f, g) is optimal exactly when all of the following hold:
  - P ≥ 0, and both marginals match a and b within `tol`;
  - fᵢ + gⱼ ≤ Cᵢⱼ on the support (dual feasibility);
  - fᵢ + gⱼ = Cᵢⱼ wherever Pᵢⱼ > 0 (complementary slackness);
  - ⟨a, f⟩ + ⟨b, g⟩ = ⟨P, C⟩ (zero duality gap).

  Each condition is checked to `tol`, scaled by max(1, max|C|) for the dual
  conditions. Feasibility alone, which is what `implementation.md` asks for,
  would not detect a feasible but suboptimal plan.
