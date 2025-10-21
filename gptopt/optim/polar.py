from itertools import repeat
import torch


@torch.compile  ## I had to comment this out, it was throwing an error
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T

    return X


polar_express_coeffs_list = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]

@torch.compile
def PolarExpress(G: torch.Tensor, steps, frob_eps=1e-2, deflation_eps=1e-2):
    assert G.ndim >= 2, "Input tensor must have at least two dimensions."
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):  # opposite convention from our other code
        X = X.mT
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + frob_eps))

    hs = polar_express_coeffs_list[:steps] + list(repeat(polar_express_coeffs_list[-1], steps - len(polar_express_coeffs_list)))
    for a, b, c in hs:
        a = a / (1 + deflation_eps)
        b = b / (1 + deflation_eps)
        c = c / (1 + deflation_eps)
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X


def SVDPolarFactor(G: torch.Tensor):
    assert G.ndim == 2, "Input tensor must have exactly two dimensions."

    # Staying consistent with PolarExpress().
    X = G.clone() # torch doesn't support SVD for bfloat16
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Compute polar factor from full SVD.
    # Note: We could try using different solvers through the `driver` argument.
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    X = U @ Vh

    # Staying consistent with PolarExpress().
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X.bfloat16() # cast to bfloat16 at the end to stay consistent

    return X
