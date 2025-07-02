import torch

def rel_err(X: torch.Tensor, Y: torch.Tensor):
    """‖X − Y‖ / (‖Y‖ + ε)"""
    return (X - Y).norm() / (Y.norm() + 1e-16)

def ns_pinv(A: torch.Tensor, steps: int = 20, diagnostics: bool = False):
    """
    Moore–Penrose pseudo-inverse via Newton–Schulz iteration (2-D only).

    Parameters
    ----------
    A : (m, n) tensor
        Input matrix (real or complex).
    steps : int
        Iteration count.
    diagnostics : bool
        If True, also return a list of per-step relative errors.

    Returns
    -------
    pinv : (n, m) tensor
        The pseudo-inverse of `A`.
    errs : list[float]           (only if diagnostics=True)
        Relative error after each iteration.
    """
    assert A.ndim == 2, "This simplified version accepts a single 2-D matrix"

    transposed = A.shape[0] > A.shape[1]   # make the working copy fat
    M = A.T if transposed else A           # shape: (m≤n, n)

    scale = M.norm() + 1e-16                # stabilising scale factor
    M = M / scale
    Y = M.T                                # initial guess (n, m)

    if diagnostics:
        M_pinv = torch.linalg.pinv(M)      # reference (scaled) solution
        errs = [rel_err(Y, M_pinv)]

    for _ in range(steps):                 # Newton–Schulz refinement
        Y = 2 * Y - Y @ M @ Y
        if diagnostics:
            errs.append(rel_err(Y, M_pinv))

    pinv = Y / scale                             # undo scaling
    if transposed:
        pinv = pinv.T                            # restore original orientation

    return (pinv, errs) if diagnostics else pinv