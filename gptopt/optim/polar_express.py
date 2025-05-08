from functools import cache
from itertools import islice
from math import acos, factorial, inf, sqrt, tan
from typing import final

import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import root_scalar
import torch


# TODO! should we allow for batched tensors at input or just matrices?


def add_scaled_identity(c, X):
    return c * torch.eye(*X.shape[-2:], dtype=X.dtype, device=X.device) + X
    # X.diagonal_scatter(X.diagonal() + c)


def horner(X, poly: Polynomial):
    result = poly.coef[-1] * torch.eye(*X.shape[-2:], dtype=X.dtype, device=X.device)
    for coeff in poly.coef[-2::-1]:
        result = add_scaled_identity(coeff, X @ result)
    return result


class PolarFactorizer:
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        pass


class ExactPolarFactorizer(PolarFactorizer):
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        U, _, V = torch.svd(X)
        return U @ V.T


class NakatsukasaPolarFactorizer(PolarFactorizer):
    pass


class Normalizer:
    def __call__(self, X):
        pass


class PerfectNormalizer(Normalizer):
    def __call__(self, X):
        s = torch.linalg.svdvals(X)
        # TODO: what happens when X is a batch of matrices? do we want to use the same lower bound for all of them?
        return X / s.max(axis=-1).values, (s.min() / s.max()).item()

    def __str__(self):
        return f"[s_min, s_max]"


class FrobeniusNormalizer(Normalizer):
    def __init__(self, absolute_lower_bound: float = 0):
        self.absolute_lower_bound = absolute_lower_bound

    def __call__(self, X):
        return X / torch.linalg.matrix_norm(X, ord='fro', keepdim=True), self.absolute_lower_bound

    def __str__(self):
        return f"[{np.format_float_scientific(self.absolute_lower_bound, exp_digits=1)}, ||X||_F]"


class SmartNormalizer(Normalizer):
    def __init__(self, absolute_lower_bound: float = 0):
        self.absolute_lower_bound = absolute_lower_bound

    @staticmethod
    def power_method(X):
        v = torch.randn(X.shape[-1], device=X.device, dtype=X.dtype)
        for _ in range(3):
            v /= torch.linalg.vector_norm(v)
            v = X @ v
        return torch.linalg.vector_norm(v)

    @staticmethod
    def normalizing_polynomial(z):
        """Assumes that Frobenius norm is 1 and the largest singular value is ≤ z
        Derived from:
        p[x_] = a*x + b*x^3
        Solve[p[z] == 1 && p[Sqrt[1 - z^2]] == 1, {a, b}]
        """
        k = sqrt(1 - z**2)  # = upper bound on sigma_2
        denom = z * k * (2 * z**2 - 1)
        b = (k - z) 
        a = ((z**2) * (z + k) - k)
        return Polynomial([a, b]) / denom

    def __call__(self, X):
        if X.shape[-1] > X.shape[-2]:
            return self.__call__(X.transpose(-1, -2)).transpose(-1, -2)

        frob_norm = torch.linalg.matrix_norm(X, ord='fro').item()
        X /= frob_norm
        XTX = X.mT @ X
        z = sqrt(self.power_method(XTX))  # lower bound on sigma_max = sigma_1
        if z**2 > 0.5:
            X = X @ horner(XTX, self.normalizing_polynomial(z))
        return X, self.absolute_lower_bound

    def __str__(self):
        return f"[{np.format_float_scientific(self.absolute_lower_bound, exp_digits=1)}, smart]"


class MaxColRowNormalizer(Normalizer):
    def __init__(self, absolute_lower_bound: float = 0):
        self.absolute_lower_bound = absolute_lower_bound

    def __call__(self, X):
        max_col = torch.linalg.vector_norm(X, dim=-2, ord=1, keepdim=True).max(dim=-1, keepdim=True).values
        max_row = torch.linalg.vector_norm(X, dim=-1, ord=1, keepdim=True).max(dim=-2, keepdim=True).values
        norm_upper = torch.sqrt(max_col * max_row)
        return X / norm_upper, self.absolute_lower_bound


class LanczosNormalizer(Normalizer):
    pass


class PolynomialSignIteration:
    def __call__(self, l: float, iter: int) -> Polynomial:
        pass


# TODO: this @final is convenient but a little uncomfortable. This *could* be subclassed, no?
@final
class PolynomialPolarFactorizer:
    def __init__(self, normalizer: Normalizer, polynomial_sign_iteration: PolynomialSignIteration, use_fast_apply: bool = True, deflation_eps: float = 0, cast: str = None):
        self.normalizer = normalizer
        self.polynomial_sign_iteration = polynomial_sign_iteration
        self.use_fast_apply = use_fast_apply
        self.deflation_eps = deflation_eps
        self.cast_dtype = dict(bfloat16=torch.bfloat16, float16=torch.float16, float32=torch.float32, float64=torch.float64).get(cast, None)

    def __str__(self):
        return f"{self.normalizer} -> {self.polynomial_sign_iteration}"

    def __call__(self, X, iters):
        if self.cast_dtype:
            X = X.to(self.cast_dtype)

        if X.shape[-1] > X.shape[-2]:
            return self.__call__(X.transpose(-1, -2), iters).transpose(-1, -2)

        if self.use_fast_apply and X.shape[-2] / X.shape[-1] > 1.5:
            X, l = self.normalizer(X)
            hs = list(islice(self.polynomial_sign_iteration(l), iters))
            hs = [self.deflate(h) for h in hs]
            return self.fast_apply(X, hs)
        else:
            return next(islice(self.sequence(X), iters-1, None))

    def sequence(self, X):
        if X.shape[-1] > X.shape[-2]:
            yield from (out.transpose(-1, -2) for out in self.sequence(X.transpose(-1, -2)))
        else:
            X, l = self.normalizer(X)
            for h in self.polynomial_sign_iteration(l):
                X = X @ horner(X.mT @ X, self.deflate(h))
                # self.deflation_eps = 0
                yield X

    @staticmethod
    def fast_apply(X, hs):
        XTX = X.mT @ X
        Qi = horner(XTX, hs[0])
        for h in hs[1:]:
            Qi = Qi @ horner((Qi.mT @ XTX @ Qi), h)
        return X @ Qi

    def deflate(self, h):
        c = 1 + self.deflation_eps
        return h(Polynomial.identity() / c**2) / c


class FixedPolynomialSignIteration(PolynomialSignIteration):
    def __init__(self, poly: Polynomial):
        self.poly = poly

    def __call__(self, l=None):
        while True:
            yield self.poly


class TaylorPolynomialSignIteration(FixedPolynomialSignIteration):
    def __init__(self, degree: int):
        # polynomial Pade
        assert degree % 2 == 1
        h_deg = degree // 2
        tilde_tilde_h = Polynomial.identity() * Polynomial(np.cumprod([(1+2*i)/(2*factorial(i+1)) for i in range(0, h_deg)]))
        poly = tilde_tilde_h(1 - Polynomial.identity()) + 1
        super().__init__(poly)

    def __str__(self):
        return f"Taylor({2 * self.poly.degree() + 1})"


class NewtonSchultz(TaylorPolynomialSignIteration):
    def __init__(self):
        super().__init__(3)

    def __str__(self):
        return "Newton-Schultz"


class Taylor5(TaylorPolynomialSignIteration):
    def __init__(self):
        super().__init__(5)


class Keller(FixedPolynomialSignIteration):
    def __init__(self):
        super().__init__(Polynomial([3.4445, -4.7750, 2.0315])) 

    def __str__(self):
        return "Keller"


class ChenChow(PolynomialSignIteration):
    def __init__(self, cushion: float = 0):
        """See Chen and Chow page 13.
        cushion (henceforth `t`) is a lower bound on the value of p(1).
        Solve p(1) = 3/2 a - 1/2 a^3 = t
        => a^3 - 3 a + 2t = 0
        Using Cardano's formula, u1 and u2 are complex numbers with unit norm with real part -t
        so the solution is a = u1^(1/3)+u2^(1/3) = 2*cos(acos(-t)/3)
        Solving for l in terms of a, we have l = (sqrt(12/a^2 - 3) - 1) / 2
        Simplifying, l = (tan(arccos(-t)/3) * sqrt(3) - 1) / 2
        """
        self.min_l = (tan(acos(-cushion)/3)*sqrt(3)-1)/2

    def __call__(self, l):
        while True:
            h = self.adapted_ns(max(l, self.min_l))
            yield h
            # For these polynomials, the upper bound is 1 and the lower bound occurs at the endpoint l
            l = l * h(l**2)

    @staticmethod
    def adapted_ns(l):
        assert l > 0
        a = sqrt(3/(1 + l + l**2))
        # This is ns(a*x)
        return Polynomial([1.5 * a, -0.5 * a**3])

    def __str__(self):
        return "Chen & Chow"


class CenteredPolynomialSignIteration(PolynomialSignIteration):
    def __init__(self, center_squred_svs: bool = False):
        """It's a bit weird that the residual starts at 3 for the centered polynomials.
        This is because the top singular value is about 2, so squaring it gives 4, and finally subtracting I we get 3.
        Instead, we can recenter the squared singular values, meaning that top singular value is never more than sqrt(2).
        """
        self.center_squred_svs = center_squred_svs

    def __call__(self, l):
        u = 1
        while True:
            h = self.adapted_l_u(l, u)
            yield h
            # We assume that the lower bound occurs at l
            # By construction, the polynomial is centered around 1
            l = l * h(l**2)
            u = 2 - l

    def adapted_l_u(self, l, u):
        if u - l <= 1e-8 * u:
            h_base = self.adapted_l_1(1)
            return h_base(Polynomial.identity() / max(l, u)**2)

        h_base = self.adapted_l_1(l / u)
        base_new_l = (l / u) * h_base((l / u)**2)
        assert base_new_l <= 1 + 1e-14
        if self.center_squred_svs:
            scalar_upper = sqrt(2 / (1 + base_new_l**2))
        else:
            scalar_upper = 2 / (1 + base_new_l)
        return (scalar_upper/u) * h_base(Polynomial.identity() / u**2)

    def adapted_l_1(self, l):
        pass


class CenteredChenChow(CenteredPolynomialSignIteration):
    def __init__(self, cushion: float = 0, center_squred_svs: bool = False):
        super().__init__(center_squred_svs=center_squred_svs)
        self.chen_chow = ChenChow(cushion=cushion)

    def adapted_l_1(self, l):
        return self.chen_chow.adapted_ns(max(l, self.chen_chow.min_l))
    
    def __str__(self):
        return "Centered Chen & Chow"


class Jiacheng(PolynomialSignIteration):
    abc_list = [
        (3955/1024, -8306/1024, 5008/1024),
        (3735/1024, -6681/1024, 3463/1024),
        (3799/1024, -6499/1024, 3211/1024),
        (4019/1024, -6385/1024, 2906/1024),
        (2677/1024, -3029/1024, 1162/1024),
        (2172/1024, -1833/1024,  682/1024)
    ]

    def __call__(self, l: float):
        for abc in Jiacheng.abc_list:
            yield Polynomial(abc)
        #
        #
        # TODO: what do we do after 6 iterations? repeat the last? or switch to deg 5 optimal?
        #
        #
        while True:
            yield Polynomial(Jiacheng.abc_list[-1])

    def __str__(self):
        return "Jiacheng"


class Pole(CenteredPolynomialSignIteration):
    def __init__(self, cushion: float = 0, center_squred_svs: bool = False):
        super().__init__(center_squred_svs=center_squred_svs)
        self.min_l = self.backout_one_sided_error(1 - cushion)

    def __str__(self):
        if self.min_l == 0:
            return "Pole"
        else:
            return f"Pole > {np.format_float_scientific(self.min_l, precision=3, exp_digits=1)}"

    @staticmethod
    @cache
    def optimal_quintic(l):
        if not 0 <= l <= 1:
            raise ValueError(f"l must be in [0, 1], got {l}")
        if 1 - 5e-6 <= l:
            # Above this threshold, the equioscillating polynomials is numerically equal to this
            return TaylorPolynomialSignIteration(5).poly
        # This initialization becomes exact as l -> 1
        q = (3*l + 1) / 4
        r = (l + 3) / 4
        E, old_E = inf, None
        while not old_E or abs(old_E - E) > 1e-15:
            old_E = E
            LHS = np.hstack((np.vander(np.array([l, q, r, 1]), 6, increasing=True)[:,1::2], (-1)**np.arange(4).reshape(-1,1)))
            a, b, c, E = np.linalg.solve(LHS, np.ones(4))
            q, r = np.sqrt((-3*b + np.array([-1, 1]) * sqrt(9*b**2 - 20*a*c)) / (10*c))
        return Polynomial([a, b, c]) / (1+E)

    def adapted_l_1(self, l):
        assert l <= 1
        l = max(l, self.min_l)
        return self.optimal_quintic(l)

    def backout_one_sided_error(self, error):
        assert 0 <= error <= 1
        def f(l):
            h = self.optimal_quintic(l)
            return (1 - l*h(l**2)) - error
        res = root_scalar(f, method='brentq', bracket=(0, 1), x0=1-error)
        assert res.converged
        return res.root

    def backout_poly(self, error):
        # TODO! this finds a polynomial with the right one sided error, but it isn't centered
        return self(self.backout_one_sided_error(error))


class PoleFixed(FixedPolynomialSignIteration):
    def __init__(self, error):
        super().__init__(Pole().backout_poly(error))


class PolePackaged(PolynomialPolarFactorizer):
    def __init__(self):
        super().__init__(
            normalizer=SmartNormalizer(l=1e-6),
            polynomial_sign_iteration=Pole(cushion=1e-2),
            use_fast_apply=False,
            deflation_eps=1e-2
        )

Pole_packaged = PolynomialPolarFactorizer(
    normalizer=SmartNormalizer(1e-3),
    polynomial_sign_iteration=Pole(cushion=1e-2),
    use_fast_apply=False,
    deflation_eps=1e-3,
    cast="bfloat16",
)

# Example usage:
# `u = Pole_packaged(G, num_steps)`