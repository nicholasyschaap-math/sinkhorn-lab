# Derivations

This file derives the vanilla, log-domain and shifted Sinkhorn updates used in
`src/sinkhorn_lab/solvers.py`, starting from the Lagrangian of the entropic
problem. It is the formula source for the code: where it differs from the
printed text of the references, this file is the one to follow.

Some of the claims below were also checked numerically on small random instances.

## Notation and conventions

| Symbol | Meaning |
|---|---|
| $a \in \mathbb{R}^n_{>0}$, $b \in \mathbb{R}^m_{>0}$ | Source and target weights, each summing to 1 |
| $C \in \mathbb{R}^{n \times m}$ | Cost matrix |
| $\varepsilon > 0$ | Regularization strength |
| $\langle P, C \rangle$ | $\sum_{i,j} P_{ij} C_{ij}$ |
| $`(f \oplus g)_{ij}`$ | $f_i + g_j$ |
| $\odot$, $/$ | Entrywise product and division |
| $`\mathbf{1}_n`$ | The all-ones vector of length $n$ |
| $K$ | The Gibbs kernel, $K_{ij} = e^{-C_{ij}/\varepsilon}$ |
| $S(f, g)$ | The slack matrix, $S(f,g) = C - f \oplus g$ |
| $\ell$ | Iteration index |

**Entropy.** This file uses the convention of Peyré and Cuturi, eq. (4.1):

```math
H(P) = -\sum_{i,j} P_{ij}\,(\log P_{ij} - 1), \qquad 0 \log 0 = 0.
```

Cuturi (2013) and POT drop the $-1$. Over couplings of unit mass the two
entropies differ by a constant, so they give the same optimal plan but
objective values that differ by $\varepsilon$.

**Slack.** The name comes from the unregularized dual, whose potentials must
satisfy $f_i + g_j \le C_{ij}$ (Peyré and Cuturi, (2.21)). $S_{ij}$ measures
how far that constraint is from binding. Entropic potentials need not satisfy
it exactly, but $S$ still appears naturally in the log-domain updates of
Section 4.

## Helpful Background

The derivations start from the Lagrangian of the regularized problem. The
material leading up to it is in these places. Section, proposition and
equation numbers for Peyré and Cuturi refer to the current arXiv version.

| Topic | Peyré and Cuturi | Cuturi (2013) |
|---|---|---|
| Couplings $U(a,b)$ and the Kantorovich problem $L_C(a,b)$ | §2.3, eqs. (2.10)–(2.11) | §2.1–2.2: $U(r,c)$ and $d_M$ |
| Dual problem and the constraint $f \oplus g \le C$ | §2.5, Prop. 2.4, eqs. (2.20)–(2.21) | |
| Entropy and the regularized problem $L^\varepsilon_C(a,b)$ | §4.1, eqs. (4.1)–(4.2) | §3.1 and eq. (2), with $\lambda = 1/\varepsilon$ |
| Limits as $\varepsilon \to 0$ and $\varepsilon \to \infty$ | §4.1, Prop. 4.1, eqs. (4.3)–(4.4) | |
| The Lagrangian and the scaling form | §4.2, Prop. 4.3 and its proof, eq. (4.12) | §4.1, eq. (3) |
| Sinkhorn's iteration | §4.2, eqs. (4.14)–(4.15) | §4.1, Algorithm 1 |
| Convergence and its rate | §4.2, Thms. 4.1–4.2, Remarks 4.14–4.15 | Sinkhorn and Knopp (1967) |
| Dual problem of the regularized problem | §4.4, Prop. 4.4, eqs. (4.30)–(4.31) | |
| Sinkhorn as block coordinate ascent | §4.4, Remark 4.21, eqs. (4.35)–(4.36) | |
| Soft-min and log-domain iterations | §4.4, Remarks 4.22–4.23, eqs. (4.37)–(4.44) | |
| Primal and dual bounds | §4.5, Def. 4.1, Props. 4.5 and 4.7, Remark 4.26 | |
| Absorbing scalings into potentials | Remark 4.23 | |

Cuturi (2013) writes the marginals as $r, c$, the cost as $M$ and the
regularization as $\lambda = 1/\varepsilon$, and states the scaling form as a
consequence of Sinkhorn's theorem rather than deriving it from a Lagrangian.
Schmitzer (2019) develops the absorption of scalings into potentials
in detail; Section 5 is a one-time version of that idea.

## 1. From the Lagrangian to the scaling form

The regularized problem (Peyré and Cuturi, (4.2)) is

```math
L^\varepsilon_C(a,b) = \min_{P \in U(a,b)} \; \langle P, C \rangle - \varepsilon H(P),
\qquad U(a,b) = \{P \ge 0 : P\mathbf{1}_m = a,\; P^\top \mathbf{1}_n = b\}.
```

Introduce multipliers $f \in \mathbb{R}^n$ and $g \in \mathbb{R}^m$ for the two
marginal constraints:

```math
\mathcal{E}(P, f, g) = \langle P, C \rangle - \varepsilon H(P)
- \langle f, P\mathbf{1}_m - a \rangle - \langle g, P^\top \mathbf{1}_n - b \rangle.
```

Written in coordinates,

```math
\mathcal{E}(P, f, g) = \sum_{i,j} P_{ij} C_{ij}
+ \varepsilon \sum_{i,j} P_{ij}(\log P_{ij} - 1)
- \sum_i f_i \Big(\sum_j P_{ij} - a_i\Big)
- \sum_j g_j \Big(\sum_i P_{ij} - b_j\Big).
```

Differentiate with respect to one entry $P_{ij}$. The terms $f_i a_i$ and
$g_j b_j$ do not depend on $P$, and
$\frac{\partial}{\partial P_{ij}} \big[P_{ij}(\log P_{ij} - 1)\big] = \log P_{ij}$, so

```math
\frac{\partial \mathcal{E}}{\partial P_{ij}} = C_{ij} + \varepsilon \log P_{ij} - f_i - g_j = 0.
```

Solving for $P_{ij}$:

```math
\varepsilon \log P_{ij} = f_i - C_{ij} + g_j
\quad\Longrightarrow\quad
P_{ij} = e^{f_i/\varepsilon}\, e^{-C_{ij}/\varepsilon}\, e^{g_j/\varepsilon}.
```

With $u_i = e^{f_i/\varepsilon}$, $K_{ij} = e^{-C_{ij}/\varepsilon}$ and
$v_j = e^{g_j/\varepsilon}$, this is the scaling form

```math
P_{ij} = u_i K_{ij} v_j, \qquad P = \mathrm{diag}(u)\, K \,\mathrm{diag}(v).
```

**Why no multiplier for $P \ge 0$ is needed.** The derivative of
$\varepsilon P_{ij}(\log P_{ij} - 1)$ tends to $-\infty$ as $P_{ij} \to 0^+$,
so the minimizer has every entry strictly positive and the constraint
$P \ge 0$ is never active. The objective is strictly convex on $U(a,b)$, so
the solution is unique. This is the argument behind Proposition 4.3.

## 2. Sinkhorn's updates from the scaling form

Substitute the scaling form into the two marginal constraints.

Row sums: $`\sum_j u_i K_{ij} v_j = u_i (Kv)_i = a_i`$.

Column sums: $`\sum_i u_i K_{ij} v_j = v_j (K^\top u)_j = b_j`$.

In vector form (Peyré and Cuturi, (4.14)):

```math
u \odot (Kv) = a, \qquad v \odot (K^\top u) = b.
```

Each equation is easy to solve for one vector when the other is fixed, so we
alternate. Starting from $`v^{(0)} = \mathbf{1}_m`$ (eq. (4.15)):

```math
u^{(\ell+1)} = \frac{a}{K v^{(\ell)}}, \qquad
v^{(\ell+1)} = \frac{b}{K^\top u^{(\ell+1)}}.
```

The $v$ update uses the new $u^{(\ell+1)}$. Combining the two steps gives a
single update in $v$:

```math
v^{(\ell+1)} = \frac{b}{K^\top \big(a / (K v^{(\ell)})\big)}.
```

Right after a $v$ update the column marginal holds exactly by construction, so
convergence should be measured on the row marginal,
$\lVert u^{(\ell)} \odot (K v^{(\ell)}) - a \rVert_1$. Theorem 4.2 gives linear
convergence.

## 3. The dual function and block coordinate ascent

For fixed $(f, g)$, $\mathcal{E}(\cdot, f, g)$ is strictly convex in $P$, so
the stationary point of Section 1,
$P_{ij}(f,g) = e^{(f_i + g_j - C_{ij})/\varepsilon}$, is its minimizer.
Substituting it back gives the Lagrange dual function
$Q(f,g) = \min_P \mathcal{E}(P, f, g)$. Term by term:

```math
-\varepsilon H(P) = \varepsilon \sum_{i,j} P_{ij}\big(\log P_{ij} - 1\big)
= \sum_{i,j} P_{ij}(f_i + g_j - C_{ij}) - \varepsilon \sum_{i,j} P_{ij},
```

```math
-\langle f, P\mathbf{1}_m - a \rangle - \langle g, P^\top \mathbf{1}_n - b \rangle
= -\sum_{i,j} P_{ij}(f_i + g_j) + \langle f, a \rangle + \langle g, b \rangle,
```

and $\langle P, C \rangle = \sum_{i,j} P_{ij} C_{ij}$. In the sum, the
$P_{ij}C_{ij}$ and $P_{ij}(f_i + g_j)$ terms cancel, leaving

```math
Q(f,g) = \langle f, a \rangle + \langle g, b \rangle
- \varepsilon \sum_{i,j} e^{(f_i + g_j - C_{ij})/\varepsilon}
= \langle f, a \rangle + \langle g, b \rangle
- \varepsilon \big\langle e^{f/\varepsilon}, K e^{g/\varepsilon} \big\rangle.
```

This is the dual problem of Proposition 4.4, eq. (4.30). The problem is convex
and $P = ab^\top$ is strictly feasible, so strong duality holds:
$\max_{f,g} Q(f,g) = L^\varepsilon_C(a,b)$.

**The gradient is the marginal error.**

```math
\frac{\partial Q}{\partial f_i} = a_i - e^{f_i/\varepsilon} \sum_j K_{ij} e^{g_j/\varepsilon}
= a_i - \big(P(f,g)\mathbf{1}_m\big)_i,
\qquad
\frac{\partial Q}{\partial g_j} = b_j - \big(P(f,g)^\top \mathbf{1}_n\big)_j.
```

**Exact block maximization.** Setting the $f$ block of the gradient to zero
with $g$ fixed:

```math
e^{f_i/\varepsilon} = \frac{a_i}{\big(K e^{g/\varepsilon}\big)_i}
\quad\Longrightarrow\quad
f_i = \varepsilon \log a_i - \varepsilon \log \big(K e^{g/\varepsilon}\big)_i,
```

and similarly

```math
g_j = \varepsilon \log b_j - \varepsilon \log \big(K^\top e^{f/\varepsilon}\big)_j.
```

With $u = e^{f/\varepsilon}$ and $v = e^{g/\varepsilon}$ these are
$u = a / (Kv)$ and $v = b / (K^\top u)$, the updates of Section 2.

$Q$ is concave: it is a linear function minus a sum of exponentials of affine
functions, each of which is convex. With one block fixed, $Q$ is concave in the
other, so each stationary point above is that block's exact maximizer.
Sinkhorn is therefore block coordinate ascent on $Q$ (Remark 4.21).

Two cautions:

- $`Q(f + c\mathbf{1}_n,\, g - c\mathbf{1}_m) = Q(f, g)`$ for every constant
  $c$, so optimal potentials are unique only up to this shift. Compare
  potentials from different runs up to a constant.
- Concavity alone does not guarantee that block coordinate ascent converges.
  Convergence and its rate come from Theorem 4.2.

## 4. Log-domain updates

For small $\varepsilon$, entries $e^{-C_{ij}/\varepsilon}$ of the kernel
underflow to 0. If an entire row of $K$ underflows, $`(Kv)_i = 0`$ and the
update $`u_i = a_i/(Kv)_i`$ divides by zero. Working with $f, g$ directly avoids
forming $K$ and its products.

**Soft-min.** For a vector $z$ (eq. (4.37)),

```math
\mathrm{min}_\varepsilon(z) = -\varepsilon \log \sum_k e^{-z_k/\varepsilon}.
```

It satisfies $`\mathrm{min}_\varepsilon(z + c) = \mathrm{min}_\varepsilon(z) + c`$
for any constant $c$. With $c = -\underline{z}$, where
$\underline{z} = \min_k z_k$, this gives the evaluation used in practice
(eq. (4.42)):

```math
\mathrm{min}_\varepsilon(z) = \underline{z}
- \varepsilon \log \sum_k e^{-(z_k - \underline{z})/\varepsilon}.
```

The largest term in the sum is $e^0 = 1$, so the sum is at least 1 and
neither underflows to 0 nor overflows. This is what makes the log domain
stable for any $\varepsilon > 0$.

**Updates.** The $f$ update from Section 3 is a soft-min over each row:

```math
f_i^{(\ell+1)} = \mathrm{min}_\varepsilon\big(C_{ij} - g_j^{(\ell)}\big)_j + \varepsilon \log a_i.
```

Since $C_{ij} - g_j = S_{ij} + f_i$, the translation property gives the form
written with the slack:

```math
f_i^{(\ell+1)} = f_i^{(\ell)}
+ \mathrm{min}_\varepsilon\Big(S\big(f^{(\ell)}, g^{(\ell)}\big)_{ij}\Big)_j
+ \varepsilon \log a_i.
```

Likewise, with the slack evaluated at the **updated** $f^{(\ell+1)}$:

```math
g_j^{(\ell+1)} = g_j^{(\ell)}
+ \mathrm{min}_\varepsilon\Big(S\big(f^{(\ell+1)}, g^{(\ell)}\big)_{ij}\Big)_i
+ \varepsilon \log b_j.
```

These are eqs. (4.43)–(4.44). Using $f^{(\ell)}$ in the $g$ update instead
turns the method into a simultaneous update, which is not block coordinate
ascent. In numerical tests that version failed to converge, while the
alternating version converged to machine precision.

**Interpretation.** The plan defined by a pair of potentials is
$`P(f,g)_{ij} = e^{-S(f,g)_{ij}/\varepsilon}`$. A soft-min over a row of $S$ is
therefore
$`\mathrm{min}_\varepsilon\big(S_{i\cdot}\big) = -\varepsilon \log p_i`$,
where $`p_i = \big(P(f^{(\ell)}, g^{(\ell)})\mathbf{1}_m\big)_i`$ is the current
row sum. Writing $q_j$ for the column sums of $P(f^{(\ell+1)}, g^{(\ell)})$,
the updates become

```math
f_i^{(\ell+1)} = f_i^{(\ell)} + \varepsilon \log \frac{a_i}{p_i},
\qquad
g_j^{(\ell+1)} = g_j^{(\ell)} + \varepsilon \log \frac{b_j}{q_j}.
```

Each correction is zero exactly when the corresponding marginal is matched and
shrinks as the marginals approach $a$ and $b$. The size of the correction,
$\varepsilon \lvert \log(a_i/p_i) \rvert$, measures a relative error, which
gives small-mass bins more weight than the absolute $L^1$ error of Section 2.
To compare solvers, stop both on the same marginal error.

**Cost.** Each iteration forms the slack matrix twice, at
$(f^{(\ell)}, g^{(\ell)})$ for the $f$ update and at $(f^{(\ell+1)}, g^{(\ell)})$
for the $g$ update, and each soft-min evaluates an exponential for every entry
of its slack matrix. One iteration therefore costs $2nm$ exponentials,
compared with two matrix–vector products for vanilla Sinkhorn (Remark 4.23).

## 5. Vanilla Sinkhorn with a row/column shift

**The shift.** Let

```math
\alpha_i = \min_j C_{ij}, \qquad
\beta_j = \min_i \big(C_{ij} - \alpha_i\big), \qquad
C' = C - \alpha \oplus \beta.
```

Then $C' \ge 0$, and every row and every column of $C'$ contains a zero. In
the notation of Section 4, $C' = S(\alpha, \beta)$: the shift is the slack
evaluated at the potentials $(\alpha, \beta)$.

**The plan does not change.** For every $P \in U(a,b)$,

```math
\langle P, \alpha \oplus \beta \rangle = \langle \alpha, a \rangle + \langle \beta, b \rangle,
```

which does not depend on $P$, and $H(P)$ does not involve $C$. Replacing $C$
with $C'$ therefore shifts the objective by the same constant for every
feasible $P$, so the minimizer is unchanged.

**Scalings.** The shifted kernel is
$`K'_{ij} = e^{-(C_{ij} - \alpha_i - \beta_j)/\varepsilon}`$, so

```math
K' = \mathrm{diag}\big(e^{\alpha/\varepsilon}\big)\, K \,\mathrm{diag}\big(e^{\beta/\varepsilon}\big).
```

With shifted potentials $f' = f - \alpha$ and $g' = g - \beta$,

```math
u'_i = e^{f_i/\varepsilon} e^{-\alpha_i/\varepsilon}, \qquad
v'_j = e^{g_j/\varepsilon} e^{-\beta_j/\varepsilon},
```

and the plan is the same in both coordinates:

```math
P = \mathrm{diag}(u')\, K' \,\mathrm{diag}(v') = \mathrm{diag}(u)\, K \,\mathrm{diag}(v).
```

**Only floating-point behavior changes.** Running the iteration of Section 2 on
$K'$ from $`v'^{(0)} = \mathbf{1}_m`$ produces exactly the iterates of the
original iteration on $K$ started from $v^{(0)} = e^{\beta/\varepsilon}$, with
$u^{(\ell)} = u'^{(\ell)} \odot e^{\alpha/\varepsilon}$ and
$v^{(\ell)} = v'^{(\ell)} \odot e^{\beta/\varepsilon}$ at every step. The
difference is numerical. Every row and column of $K'$ contains an entry equal
to 1, so the first iteration cannot divide by zero because a whole row of the
kernel underflowed. The shift does not stop $u'$ and $v'$ themselves from
overflowing or underflowing in later iterations; below that point the log
domain is required.

**Recovering the outputs.**

- The plan needs no rescaling: $`P = \mathrm{diag}(u')\, K' \,\mathrm{diag}(v')`$.
- The potentials are recovered in log form:
  $f = \varepsilon \log u' + \alpha$ and $g = \varepsilon \log v' + \beta$.
- Compute the transport cost as $\langle P, C \rangle$ with the original $C$.
  The identity
  $`\langle P, C \rangle = \langle P, C' \rangle + \langle \alpha, P\mathbf{1}_m \rangle + \langle \beta, P^\top\mathbf{1}_n \rangle`$
  reduces to adding $\langle \alpha, a \rangle + \langle \beta, b \rangle$ only
  when $P$ matches both marginals exactly, which Sinkhorn iterates do not.
- The dual value can be computed in shifted coordinates:
  $Q(f, g; C) = Q(f', g'; C') + \langle \alpha, a \rangle + \langle \beta, b \rangle$.
  The exponential term is identical in both, so this holds for any $f, g$.

**Procedure.**

1. Compute $\alpha$, $\beta$ and $C'$, and build $K' = e^{-C'/\varepsilon}$.
2. Initialize $`v' = \mathbf{1}_m`$ and iterate as in Section 2.
3. Form $`P = \mathrm{diag}(u')\, K' \,\mathrm{diag}(v')`$.
4. Recover $f$ and $g$ with the formulas above.

## Differences from the printed text

Checked against the current arXiv version of Peyré and Cuturi.

- **Eqs. (4.39) and (4.41)** use $f^{(\ell)}$ in the $g$ update. Follow
  eqs. (4.36) and (4.44), which use $f^{(\ell+1)}$.

## References

- Peyré, G. and Cuturi, M. (2019). [Computational Optimal Transport](https://arxiv.org/abs/1803.00567). *Foundations and Trends in Machine Learning*, 11(5–6).
- Cuturi, M. (2013). [Sinkhorn Distances: Lightspeed Computation of Optimal Transport](https://papers.nips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html). *Advances in Neural Information Processing Systems 26*.
- Sinkhorn, R. and Knopp, P. (1967). Concerning nonnegative matrices and doubly stochastic matrices. *Pacific Journal of Mathematics*, 21(2), 343–348.
- Schmitzer, B. (2019). [Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems](https://arxiv.org/abs/1610.06519). *SIAM Journal on Scientific Computing*.
