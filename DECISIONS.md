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
  fill those duals with estimated finite values. *(Confirmed by the author,
  2026-11-06.)*
- `solvers.py` (author) will implement it for both Sinkhorn variants.

---

## D2. Cost is shifted by row and column minima before forming the Gibbs kernel

*Status: decided by the author, 2026-11-06. The first version, a single
global shift C − min C, was revised the same day to the row-then-column
shift below.*

**Decision.** On the support (after D1 drops zero weights), the solvers form
the kernel from the doubly shifted cost

  αᵢ = minⱼ Cᵢⱼ,  βⱼ = minᵢ (Cᵢⱼ − αᵢ),  C̃ᵢⱼ = Cᵢⱼ − αᵢ − βⱼ,

which in code is `C_shift = C - C.min(axis=1, keepdims=True)` followed by
`C_shift = C_shift - C_shift.min(axis=0, keepdims=True)`.

- *Every row and every column of C̃ contains an exact 0, and C̃ ≥ 0.* After
  the row step, each row has a 0 at its minimum and all entries are ≥ 0.
  Each column of the row-shifted matrix therefore has a minimum βⱼ ≥ 0.
  Subtracting it puts a 0 in every column. A row's existing zero sits in a
  column whose minimum is already 0, so it is untouched.
- *So every row and every column of K̃ = exp(−C̃/ε) has an entry exactly
  equal to exp(0) = 1.*
- These are translations of C, not multiplications, so the effective ε is
  unchanged.

The shift does not replace the log-sum-exp (log-domain) iteration. It gives
the vanilla solver a better-conditioned starting kernel, so ε can be pushed
closer to 0 before the log-domain method, and its cost, becomes necessary.

**Reasoning.**

- *The shift changes nothing about the solution.* For any P ∈ U(a, b),
  ⟨P, C̃⟩ = ⟨P, C⟩ − Σᵢ αᵢ Σⱼ Pᵢⱼ − Σⱼ βⱼ Σᵢ Pᵢⱼ = ⟨P, C⟩ − ⟨a, α⟩ − ⟨b, β⟩,
  a constant on U(a, b). The exact problem and the entropic problem
  ⟨P, C⟩ − εH(P) therefore have the same minimizers for C̃ as for C.
  - (f, g) is optimal for C̃ if and only if (f + α, g + β) is optimal for C,
    up to the usual (f + c, g − c) freedom.
  - The objective value shifts by exactly ⟨a, α⟩ + ⟨b, β⟩.
  - The argument holds for any α, β; the minima are one particular choice.
- *Why it helps.* K̃ = diag(e^{α/ε})·K·diag(e^{β/ε}), and Sinkhorn absorbs
  diagonal factors into its scalings: diag(u)·K̃·diag(v) =
  diag(e^{α/ε}u)·K·diag(e^{β/ε}v). Mathematically nothing changes. In
  floating point, though, K itself may not be representable.
  - If a row's smallest cost satisfies αᵢ/ε ≳ 745, the whole row of K
    underflows to 0. Then (Kv)ᵢ = 0 and the update uᵢ = aᵢ/(Kv)ᵢ divides by
    zero, even though nothing about the problem is degenerate.
  - The same holds for columns and (Kᵀu)ⱼ.
  - After the shift, every row and column of K̃ contains a 1. So
    (K̃v)ᵢ ≥ v_{j*(i)} > 0, where j*(i) is the column of row i's zero, and
    likewise for (K̃ᵀu)ⱼ. Neither sum can underflow to 0 on the first
    iterations.
- *A single global shift was weaker.* Subtracting only min C guarantees one
  entry equal to 1 in the whole kernel. Any row whose own minimum lies far
  above the global minimum can still underflow. The row and column shift
  guarantees a 1 per row and per column.
- *Underflow depends on the ratio of cost to ε, not on either alone.* In
  float64, exp(−x) leaves the normal range for x > 708.4 (log of the smallest
  normal number) and becomes exactly 0 for x ≳ 745.1.
  - After the shift, an entry of K̃ underflows only if its reduced cost C̃ᵢⱼ/ε
    is large.
  - Every entry of K̃ is a normal number exactly when max C̃ / ε < 708.4.
  - Scaling C and ε together by the same factor leaves K̃ unchanged, and
    adding any α 1ᵀ + 1 βᵀ to C leaves it unchanged too. Neither C nor ε alone
    determines underflow.
  - The author's derivations (`docs/derivation.md`, to be uploaded) develop
    this.
- *What the shift does not fix.* The scalings u and v can still overflow or
  underflow over the iterations as ε → 0. Entries with large C̃ᵢⱼ/ε still
  underflow individually, which can starve the sums of all but one term.
  These are what the in-loop log-sum-exp stabilization handles, at its extra
  computational cost.

**Consequences.**

- *Reported quantities (author, 2026-11-06).* Solvers report cost and duals
  on the original C: cost ⟨P, C⟩ and duals (f + α, g + β). The shift is
  internal to the solver. This matches `solve_exact`, so stage 2 tests
  compare directly.
- *Shift on the support.* α and β are computed after zero weights are
  dropped (D1), on C restricted to the support. Computing them on the full C
  would still be a valid shift, but rows or columns with zero weight could
  set βⱼ or αᵢ and remove the exact zero on the support.
- *Likely failure predictor.* By the argument above, the quantity that
  controls underflow is the reduced cost range relative to ε,
  `C_shift.max() / eps`. Under the earlier global shift it was
  `(C.max() − C.min()) / eps`. The two can differ a lot (see D4, open
  point). The timing experiment in `implementation.md` tests
  `C_shift.max() / eps` as the expected strong predictor, alongside
  `C.max() / eps` and `eps` alone, and does not assume any of them.
- *The exact baseline uses its own, independent shift.* `exact.py` subtracts
  only the scalar min C on the support before calling `ot.emd`. Raw `ot.emd`
  falsely reports "Problem infeasible" when all costs are negative and
  roughly constant below −N/(N−1), N = n + m (see `AGENTS_LOG.md`), and the
  scalar shift is enough to avoid that. min C is added back to g, and the
  cost and certificate are computed on the original C. This is a workaround
  in the baseline, not part of the solvers' design. The tests also check
  that the row and column shift leaves the exact plan unchanged
  (`test_row_column_shift_has_same_plan`).

---

## D3. The exact baseline returns only certified answers

*Status: requirement from the author, 2026-11-06. The mechanism was proposed
by the agent and confirmed by the author, including the scalar min-shift
workaround (D2, Consequences).*

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

---

## D4. Figure set and per-figure cost normalization

*Status: decided by the author, 2026-11-06.*

**Figures.** There are four figures (this supersedes the three-figure plan in
`implementation.md`):

1. **Cost gap.**
2. **Iterations.**
3. **Stability: ε against separation.** One panel per method, at a fixed
   small n of about 50.
4. **Timing.** Cost per iteration on the left, time to convergence on the
   right. This takes over the timing experiment described in
   `implementation.md`.

**Normalization.** Generators can divide the cost by a scale s:
C ← C / s. This is a multiplication, unlike the D2 shift, so it changes the
effective regularization: (C / s, ε) is the same entropic problem as
(C, s·ε). Normalizing therefore fixes the units of ε, and choosing s chooses
what an ε sweep means. Per figure:

| Figure | Normalization | Reason |
|---|---|---|
| 1. Cost gap | median(C) | Puts ε in units of a typical cost. The median is fairly constant in n. |
| 2. Iterations | median(C); max(C) as a separate check | Median is primary. The run is repeated with max normalization to check whether the separation seen under median remains the same; the result is reported either way. |
| 3. Stability vs separation | none (raw C) | Normalizing would divide out the cost growth that separation causes: the median and max of the squared distances grow with separation. Raw C keeps the separation in the cost scale, at a fixed small n of about 50. |
| 4. Timing | median(C) | Same as figure 1. The median's stability in n keeps ε comparable as n varies. |

**Implementation.** `gaussian_clouds(..., normalize="none" | "median" | "max")`
replaces the earlier `scale_c` flag. The divisor is stored in
`meta["c_scale"]`, so raw C = C · `meta["c_scale"]`. The default is `"none"`,
so normalization only happens when an experiment script asks for it.
Normalization is applied before validation. The solver's D2 shift then acts
on the normalized C.

`random_cost(..., normalize=...)` takes the same argument, so both families
can be placed on one ε axis.

**Interaction with D2.** Normalization divides C̃ by the same s, so the D2
predictor C̃_max/ε becomes C̃_max/(s·ε) in raw units. Under max normalization
C̃_max ≤ 1, so the predictor is at most 1/ε.

**Open (author): separation and figure 3.**
- *Separation leaves the problem itself unchanged.* For squared Euclidean
  cost, moving the target by s·e₁ changes Cᵢⱼ by −2s(xᵢ₁ − zⱼ₁) + s²: a term
  in i plus a term in j. By the D2 argument, the exact plan and every
  entropic plan are therefore identical for every separation. Checked for
  the exact plan at n = 50, s ∈ {0, 1, 2, 4, 8, 16}: the plans are
  bit-identical. Regression test: `test_separation_does_not_change_exact_plan`.
- *Separation affects the numerics only, and the D2 shift absorbs most of
  it.* At n = 50 (seed 0), the raw range max C − min C grows from 26.0 at
  s = 0 to 265.7 at s = 16. The row-and-column-shifted range C̃_max only
  grows from 24.7 to 36.7.
- *Consequence.* A raw-C stability figure will show separation dependence
  mostly through the size of the costs. A vanilla solver that applies the
  D2 shift internally may show much weaker dependence on separation than
  the raw range suggests. That may be the intended finding, or it may call
  for plotting against C̃_max/ε or a different axis. The author decides.
