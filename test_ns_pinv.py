import pytest
import torch
from gptopt.linalg_utils import rel_err, ns_pinv


@pytest.fixture(scope="module")
def setup():
    """Setup fixture for reproducibility and tolerance."""
    torch.manual_seed(0)  # reproducibility
    return 1e-5  # accuracy threshold


def test_identity_matrix(setup):
    """Test ns_pinv with identity matrix (trivial fixed-point)."""
    tol = setup
    A = torch.eye(5)
    assert rel_err(ns_pinv(A), A) < tol


def test_square_full_rank_matrix(setup):
    """Test ns_pinv with square, full-rank random matrix."""
    tol = setup
    A = torch.randn(6, 6)
    assert rel_err(ns_pinv(A), torch.linalg.pinv(A)) < tol


def test_tall_full_rank_matrix(setup):
    """Test ns_pinv with tall (m > n) full-rank matrix."""
    tol = setup
    A = torch.randn(8, 3)
    assert rel_err(ns_pinv(A), torch.linalg.pinv(A)) < tol


def test_wide_full_rank_matrix(setup):
    """Test ns_pinv with wide (m < n) full-rank matrix."""
    tol = setup
    A = torch.randn(3, 8)
    assert rel_err(ns_pinv(A), torch.linalg.pinv(A)) < tol


def test_rank_deficient_matrix(setup):
    """Test ns_pinv with rank-deficient matrix (duplicate columns)."""
    tol = setup
    B = torch.randn(6, 4)
    A = torch.cat([B, B[:, :1]], dim=1)  # rank ≤ 4 < 5
    assert rel_err(ns_pinv(A), torch.linalg.pinv(A)) < tol


def test_all_zeros_matrix(setup):
    """Test ns_pinv with all-zeros matrix (pseudo-inverse should be zeros)."""
    tol = setup
    A = torch.zeros(4, 7)
    assert torch.allclose(ns_pinv(A), torch.zeros_like(A.T), atol=tol)


def test_ill_conditioned_matrix(setup):
    """Test ns_pinv with ill-conditioned square matrix (singular values differ by many orders)."""
    tol = setup
    U, _ = torch.linalg.qr(torch.randn(6, 6))
    V, _ = torch.linalg.qr(torch.randn(6, 6))
    s = torch.tensor([1.0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10])
    A = U @ torch.diag(s) @ V.T
    assert rel_err(ns_pinv(A), torch.linalg.pinv(A)) < tol