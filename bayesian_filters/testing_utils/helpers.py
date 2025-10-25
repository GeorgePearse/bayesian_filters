"""Helper functions for testing Bayesian filters."""

import numpy as np


def assert_matrix_psd(M, rtol=1e-5, atol=1e-8, name="Matrix"):
    """Assert that matrix is positive semi-definite.

    Parameters
    ----------
    M : np.ndarray
        Matrix to test
    rtol : float
        Relative tolerance for eigenvalue checks
    atol : float
        Absolute tolerance for eigenvalue checks
    name : str
        Name of matrix for error messages

    Raises
    ------
    AssertionError
        If matrix is not positive semi-definite
    """
    M = np.asarray(M)

    # Check symmetry
    if not np.allclose(M, M.T, rtol=rtol, atol=atol):
        raise AssertionError(f"{name} is not symmetric")

    # Check eigenvalues
    eigvals = np.linalg.eigvalsh(M)
    min_eigval = np.min(eigvals)

    if min_eigval < -atol:
        raise AssertionError(f"{name} is not positive semi-definite. Minimum eigenvalue: {min_eigval}")


def assert_matrix_symmetric(M, rtol=1e-5, atol=1e-8, name="Matrix"):
    """Assert that matrix is symmetric.

    Parameters
    ----------
    M : np.ndarray
        Matrix to test
    rtol : float
        Relative tolerance
    atol : float
        Absolute tolerance
    name : str
        Name of matrix for error messages

    Raises
    ------
    AssertionError
        If matrix is not symmetric
    """
    M = np.asarray(M)

    if not np.allclose(M, M.T, rtol=rtol, atol=atol):
        diff = np.max(np.abs(M - M.T))
        raise AssertionError(f"{name} is not symmetric. Max difference: {diff}")


def assert_filter_stable(P, max_variance=1e10):
    """Assert that filter covariance matrix is stable.

    Checks that diagonal elements don't explode.

    Parameters
    ----------
    P : np.ndarray
        Filter covariance matrix
    max_variance : float
        Maximum allowed variance on diagonal

    Raises
    ------
    AssertionError
        If filter appears unstable
    """
    P = np.asarray(P)
    diag = np.diag(P)

    if np.any(diag > max_variance):
        raise AssertionError(f"Filter covariance is unstable. Max diagonal: {np.max(diag)}")

    if np.any(np.isnan(diag)):
        raise AssertionError("Filter covariance contains NaN values")

    if np.any(np.isinf(diag)):
        raise AssertionError("Filter covariance contains infinite values")


def compute_condition_number(M):
    """Compute condition number of matrix.

    Parameters
    ----------
    M : np.ndarray
        Square matrix

    Returns
    -------
    float
        Condition number (ratio of largest to smallest singular value)
    """
    M = np.asarray(M)
    s = np.linalg.svd(M, compute_uv=False)
    return np.max(s) / np.max(np.min(s), 1e-15)


def assert_well_conditioned(M, max_condition=1e6, name="Matrix"):
    """Assert that matrix is well-conditioned.

    Parameters
    ----------
    M : np.ndarray
        Matrix to test
    max_condition : float
        Maximum allowed condition number
    name : str
        Name of matrix for error messages

    Raises
    ------
    AssertionError
        If matrix is ill-conditioned
    """
    cond = compute_condition_number(M)

    if cond > max_condition:
        raise AssertionError(f"{name} is ill-conditioned. Condition number: {cond:.2e}")


def assert_convergence(errors, tolerance=0.1, rtol=0.05):
    """Assert that error sequence converges.

    Parameters
    ----------
    errors : array_like
        Sequence of errors
    tolerance : float
        Final error should be below this
    rtol : float
        Relative tolerance for "no divergence"

    Raises
    ------
    AssertionError
        If sequence doesn't converge
    """
    errors = np.asarray(errors)

    # Check final error
    if errors[-1] > tolerance:
        raise AssertionError(f"Errors did not converge. Final error: {errors[-1]}")

    # Check for divergence in later part
    final_half = errors[len(errors) // 2 :]
    if np.max(final_half) > np.min(final_half) * (1 + rtol):
        raise AssertionError(f"Errors appear to diverge. Min: {np.min(final_half)}, Max: {np.max(final_half)}")


def mahalanobis_distance(x, mean, cov):
    """Compute Mahalanobis distance.

    Parameters
    ----------
    x : array_like
        Vector
    mean : array_like
        Mean vector
    cov : array_like
        Covariance matrix

    Returns
    -------
    float
        Mahalanobis distance
    """
    x = np.asarray(x).flatten()
    mean = np.asarray(mean).flatten()
    cov = np.asarray(cov)

    diff = x - mean
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    return np.sqrt(np.dot(diff, np.dot(cov_inv, diff)))


def normalized_error(estimate, truth):
    """Compute normalized estimation error.

    Parameters
    ----------
    estimate : array_like
        Estimated values
    truth : array_like
        True values

    Returns
    -------
    np.ndarray
        Normalized errors
    """
    estimate = np.asarray(estimate).flatten()
    truth = np.asarray(truth).flatten()

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        error = (estimate - truth) / np.abs(truth)
        error = np.nan_to_num(error, nan=0.0, posinf=0.0, neginf=0.0)

    return error
