from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


#Bundles necessary metrics after running sinkhorn for easy return
@dataclass(frozen=True)
class SinkhornResult:
    P: np.ndarray
    f: np.ndarray
    g: np.ndarray
    cost: float        # <P, C> on the original C
    converged: bool
    n_iter: int
    status: str        # "converged" | "max_iter" | "numerical_failure"
    err: float
    
#Performs cost matrix shifts to create more stable kernel matrix
def rowcol_shift(C): 
    alpha = C.min(axis=1)
    C1 = C - alpha[:, None]
    beta = C1.min(axis=0)
    return C1 - beta[None, :], alpha, beta



def softmin(A, eps, axis):
    #Soft-min over one axis: -eps * log(sum(exp(-A / eps))). axis=1 is minrow and axis=0 is mincol
    return -eps * logsumexp(-A / eps, axis=axis)

def slack(C, f, g):
    S = C - f[:,None] - g
    return S

def vanilla_sinkhorn(a, b, C, eps, *, stabilize="rowcol", tol=1e-9, max_iter=100_000):
    if not eps > 0:
        raise ValueError("Invalid Regularization")
    if stabilize not in ("none", "rowcol"):
        raise ValueError(f"unknown stabilize option: {stabilize!r}")
    
    #Handle zero-weight (or less than zero) entries and resize
    if (a < 0).any():
        raise ValueError("Values of a must remain positive")
    if (b < 0).any():
        raise ValueError("Values of b must remain positive")
    rows = a > 0
    cols = b > 0
    if not rows.any() or not cols.any():
        raise ValueError("a and b must each have at least one positive entry")

    a_sub, b_sub = a[rows], b[cols]
    C_sub = C[np.ix_(rows, cols)]
    #Raise for unkown stabilize so typos dont run vanilla when intending rowcol
    Cs, alpha, beta = rowcol_shift(C_sub) if stabilize == "rowcol" else (C_sub, 0.0, 0.0)
    
    #initialize Gibbs Kernel
    K = np.exp(-Cs / eps)

    #initialize scaling vectors for projections
    u_s = np.ones_like(a_sub)           #the _s stands for a shift, could be 0 if unstabilized
    v_s = np.ones_like(b_sub) 
    #main loop for algorithm
    converge = False
    status = "max_iter"
    err = np.inf       #Tracks final state of loop in each case
    n_iter = 0    #Tracks loop iteration count
    for i in range(max_iter):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            multi = K @ v_s
            #Tracks stopping criteria across loops
            err = np.abs(u_s * multi - a_sub).sum()
        #Stopping criteria
        if err < tol:
            status = "converged"
            converge = True
            break
        n_iter += 1
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"): #Silence errors thrown for nonconvergent case (better in logs)
            u_s = a_sub / multi
            v_s = b_sub / (K.T @ u_s)
        #Check for numerical failures (0 checks are indicators kernel has blown up)
        if not (np.all(np.isfinite(u_s)) and np.all(np.isfinite(v_s))) or np.any(u_s == 0) or np.any(v_s == 0):
            status = "numerical_failure"
            err = np.nan
            break
        

    if status == "max_iter":
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            err = np.abs(u_s * (K @ v_s) - a_sub).sum()
        if err < tol:    #last update converged but the loop ended before its check
            status = "converged"
            converge = True
    with np.errstate(divide="ignore", invalid="ignore"):    #Same motivation as earlier instance of this context
        P_sub = u_s[:, None] * K * v_s[None,:]
        f_sub = eps * np.log(u_s) + alpha          # duals in log form 
        g_sub = eps * np.log(v_s) + beta

    P = np.zeros_like(C)
    P[np.ix_(rows, cols)] = P_sub          
    #dropped rows and cols remain 0
    f = np.full(a.shape, np.nan)
    f[rows] = f_sub
    g = np.full(b.shape, np.nan)
    g[cols] = g_sub
    transport_cost = np.sum(P * C)
    results = SinkhornResult(P,f,g,transport_cost,converge,n_iter,status,err)
    return results


def log_sinkhorn(a, b, C, eps, *, stabilize="rowcol", tol=1e-9, max_iter=100_000):
    if not eps > 0:
        raise ValueError("Invalid Regularization")
    if stabilize not in ("none", "rowcol"):
        raise ValueError(f"unknown stabilize option: {stabilize!r}")
    
    #Handle zero-weight (or less than zero) entries and resize
    if (a < 0).any():
        raise ValueError("Values of a must remain positive")
    if (b < 0).any():
        raise ValueError("Values of b must remain positive")
    rows = a > 0
    cols = b > 0
    if not rows.any() or not cols.any():
        raise ValueError("a and b must each have at least one positive entry")

    a_sub, b_sub = a[rows], b[cols]
    C_sub = C[np.ix_(rows, cols)]
    #Raise for unkown stabilize so typos dont run vanilla when intending rowcol
    Cs, alpha, beta = rowcol_shift(C_sub) if stabilize == "rowcol" else (C_sub, 0.0, 0.0)

    #initialize scaling vectors for projections
    f_s = np.zeros_like(a_sub)           #the _s stands for a shift, could be 0 if unstabilized
    g_s = np.zeros_like(b_sub)
    
    #main loop for algorithm
    converge = False
    status = "max_iter"
    err = np.inf       #Tracks final state of loop in each case
    n_iter = 0    #Tracks loop iteration count
    for i in range(max_iter):
        S = slack(Cs,f_s,g_s)
        sm = softmin(S,eps,1)
        #Tracks stopping criteria across loops
        err = np.abs(np.exp(-sm/eps) - a_sub).sum() #recycle sm calculation for err and f_s with early check
        #Stopping criteria
        if err < tol:
            status = "converged"
            converge = True
            break
        n_iter += 1 #increase when we actually begin new update loop
        f_s = sm + f_s + (eps * np.log(a_sub))
        S = slack(Cs,f_s,g_s)                                
        g_s = softmin(S,eps,0) + g_s + (eps * np.log(b_sub)) 
        #Check for numerical failures (0 checks are indicators kernel has blown up)
        if not (np.all(np.isfinite(f_s)) and np.all(np.isfinite(g_s))):
            status = "numerical_failure"
            err = np.nan
            break
        

    if status == "max_iter":
        err = np.abs(np.exp(-softmin(slack(Cs, f_s, g_s), eps, 1) / eps) - a_sub).sum()
        if err < tol:    #last update converged but the loop ended before its check
            status = "converged"
            converge = True
    with np.errstate(divide="ignore", invalid="ignore"):    #Same motivation as earlier instance of this context
        P_sub = np.exp((f_s[:, None] + g_s[None,:] - Cs) / eps)
        f_sub = f_s + alpha          # duals in log form 
        g_sub = g_s + beta

    P = np.zeros_like(C)
    P[np.ix_(rows, cols)] = P_sub          
    #dropped rows and cols remain 0
    f = np.full(a.shape, np.nan)
    f[rows] = f_sub
    g = np.full(b.shape, np.nan)
    g[cols] = g_sub
    transport_cost = np.sum(P * C)
    results = SinkhornResult(P,f,g,transport_cost,converge,n_iter,status,err)
    return results
    
    

    
    
    
    
    
   
