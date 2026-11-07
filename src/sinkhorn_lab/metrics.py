import numpy as np


def lp_dual_bound(f, g, a, b, C):
    #Creates a valid lower bound on the optimal plan to obtain a sense of how much sinkhorn overpays
    #This is derived from the inequality fi + gj <= Cij, transform guarantees this and we select largest of candidates
    f, g, a, b, C = map(np.asarray, (f, g, a, b, C))
    if C.shape != (a.size, b.size) or f.shape != a.shape or g.shape != b.shape:
        raise ValueError("shapes of f, g, a, b and C do not match")

    #Reduce to the support as in solvers.py: zero-weight duals are NaN by design (D1)
    if (a < 0).any():
        raise ValueError("Values of a must remain positive")
    if (b < 0).any():
        raise ValueError("Values of b must remain positive")
    rows = a > 0
    cols = b > 0
    if not rows.any() or not cols.any():
        raise ValueError("a and b must each have at least one positive entry")
    f, a = f[rows], a[rows]
    g, b = g[cols], b[cols]
    C = C[np.ix_(rows, cols)]

    if np.isnan(f).any() or np.isnan(g).any():
        raise ValueError("f and g must not contain nan on the support")

    g1 = np.min(C - f[:, None], axis=0)      # g1_j = min_i (C_ij - f_i)
    f1 = np.min(C - g1[None, :], axis=1)     # f1_i = min_j (C_ij - g1_j)

    f2 = np.min(C - g[None, :], axis=1)      #Same steps in reverse order
    g2 = np.min(C - f2[:, None], axis=0)

    def value(ft, gt):
        # Skip zero-mass bins to avoid nan
        return ft[a > 0] @ a[a > 0] + gt[b > 0] @ b[b > 0]
        #Max of candidates is best bounding
    return max(value(f1, g1), value(f2, g2))
