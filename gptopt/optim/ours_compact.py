from itertools import chain, islice, repeat
import torch

# # How to generate these lists:
# from itertools import islice
# from matsign.methods import OursFixedL, Ours
# hs = list(OursFixedL(l=1e-3, cushion=1e-1, center_squred_svs=False, max_iters=10)(1e-3))  # centered
# hs = list(islice(Ours(cushion=1e-1, center_squred_svs=False).uncentered_sequence(1e-3), 10))  # uncentered
# [tuple(float(x) for x in h.coef) for h in hs]


# Coefficients for the optimal polynomial beginning with l = 1e-3, u = 1
our_coeffs_list = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]

our_coeffs_list_uncentered = [
    (4.160846870077099, -11.847032555063748, 8.68618568498665),
    (4.160846870077099, -11.847032555063748, 8.68618568498665),
    (4.160846870077099, -11.847032555063748, 8.68618568498665),
    (3.967691083242876, -10.357066768902493, 7.389375685659618),
    (3.19518365315184, -5.643620864919531, 3.4484372117676916),
    (2.122473839842304, -1.7963511350724446, 0.6738772952301406),
    (1.8772232236073114, -1.254450403151285, 0.3772271795439737),
    (1.875, -1.25, 0.375),
]



coeffs_list = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),  # subsequent coeffs equal this numerically
]
# safety factor for numerical stability (but exclude last polynomial)
coeffs_list = [(a / 1.01, b / 1.01**3, c / 1.01**5)
                for (a, b, c) in coeffs_list[:-1]] + [coeffs_list[-1]]

@torch.compile
def PolarExpress(G: torch.Tensor, steps: int) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()  # for speed
    if G.size(-2) > G.size(-1): X = X.mT  # this reduces FLOPs
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 +1e-7)
    hs = coeffs_list[:steps] + list( 
        repeat(coeffs_list[-1], steps - len(coeffs_list)))
    for a, b, c in hs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X  # X <- aX + bX^3 + cX^5
    if G.size(-2) > G.size(-1): X = X.mT
    return X



def deflate(abc, deflation_eps):
    a, b, c = abc
    return a / (1 + deflation_eps), b / (1 + deflation_eps)**3, c / (1 + deflation_eps)**5



@torch.compile
def ours_compact(G: torch.Tensor, steps: int, fast_apply_restart: int = 1, pinpoint_top: bool = False, deflation_eps: float = 1e-2, max_sigma1: float = 0.97, centered: bool = True, cast_to_bfloat16: bool = True):
    assert G.ndim >= 2, "Input tensor must have at least two dimensions."
    assert steps > 0, "Number of steps must be positive."
    if cast_to_bfloat16:
        X = G.bfloat16()
    else:
        X = G
    if G.size(-2) > G.size(-1):  # opposite convention from our other code
        X = X.mT
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + deflation_eps) + 1e-7)

    if pinpoint_top:
        XXT = X @ X.mT
        v = torch.ones((*XXT.shape[:-2], XXT.shape[-1], 1), device=XXT.device, dtype=XXT.dtype) / XXT.shape[-1]  # cheaper than a Gaussian
        v = XXT @ v
        v = XXT @ v
        v = XXT @ v  # these don't seem to be more expensive. could add more
        # v = XXT @ v
        # v = XXT @ v
        # If there is a batch of inputs, take the min over the batch
        sigma1_lower = (torch.linalg.vector_norm(XXT @ v, dim=-2, keepdim=False) / torch.linalg.vector_norm(v, dim=-2, keepdim=False)).min().sqrt()
        # For numerical stability, we don't want sigma1_lower too close to 1. So take min with max_sigma1
        # If this matrix does not have a large outlying singular value or power method failed to find it, i.e. sigma1_lower <= sqrt(0.5),
        # then we fall back to a cubic polynomial that maps [0, 1] to [0, 1]. After setting sigma1_lower = sqrt(0.5), the following coefficients actually form such a polynomial anyway
        # In the past we introduced a conditional here and if sigma1_lower <= sqrt(0.5), we just skipped applying the polynomial.
        # But this is (1) wasteful since we've already computed XXT. (2) causes some rare but very cursed problems with torch.compile (3) introduces significant wall clock time overhead
        sigma1_lower = torch.clamp(sigma1_lower, min=0.70711, max=max_sigma1)
        sigma_2_upper = torch.sqrt(1 - sigma1_lower**2)
        denom = sigma1_lower * sigma_2_upper * (sigma1_lower + sigma_2_upper)
        a_init = (sigma1_lower ** 2 + (sigma1_lower * sigma_2_upper) + sigma_2_upper ** 2) / denom
        b_init = -1 / denom
        X = a_init * X + b_init * XXT @ X
        steps -= 1  # Since we're basically applying one polynomial step


    # NOTE: it's very important to make `hs` a plain list, not an iterator.
    # Don't do any CPU operations inside the loop, just GPU ops.
    # Otherwise it could seriously slow down the code.
    all_coeffs = our_coeffs_list if centered else our_coeffs_list_uncentered
    hs = [deflate(coeff, deflation_eps) for coeff in chain(
        islice(all_coeffs, steps),
        repeat(all_coeffs[-1], steps - len(all_coeffs)),
    )]

    if steps == 1 or (X.size(-1) / X.size(-2) < 1.5 * steps / (steps - 1)):
        # In this case, it's not worth it to do fast apply
        fast_apply_restart = 1

    if fast_apply_restart == 1:
        # This is just copying Keller
        for a, b, c in hs:
            A = X @ X.mT
            B = b * A + c * A @ A
            X = a * X + B @ X
    else:
        if fast_apply_restart is None:
            # one big block
            fast_apply_restart = len(hs)
        # TODO! change this so fast_apply_restart is the maximum. e.g., with 4 iterations and max run 3, we should do 2 + 2 not 3 + 1.
        # also if there are shorter blocks, we should do those first
        all_chunked_hs = [hs[i:i + fast_apply_restart] for i in range(0, len(hs), fast_apply_restart)]
        I = torch.eye(X.size(-2), device=X.device, dtype=X.dtype)
        for chunk in all_chunked_hs:
            XXT = X @ X.mT
            a, b, c = chunk[0]
            Qi = a*I + b*XXT + c * XXT @ XXT
            for a, b, c in chunk[1:]:
                Ri = Qi @ XXT @ Qi.mT
                TEMP = b*Ri + c * Ri @ Ri  # Could use identity again, but optiming for storing TEMP to match Keller
                Qi = a*Qi + TEMP @ Qi
            X = Qi @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X