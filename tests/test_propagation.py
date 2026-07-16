"""Tests for CR3BP trajectory propagation."""

import numpy as np
import pytest

from tno_dynamics.dynamics import jacobi_constant
from tno_dynamics.propagation import propagate_cr3bp


def test_propagation_succeeds_with_six_state_rows() -> None:
    """A nominal CR3BP trajectory integrates from the supplied initial state."""
    state0 = np.array([0.8, 0.0, 0.0, 0.0, 0.25, 0.0])

    solution = propagate_cr3bp(state0, (0.0, 0.1), 0.0121505856)

    assert solution.success
    assert solution.y.shape[0] == 6
    np.testing.assert_array_equal(solution.y[:, 0], state0)


def test_propagation_respects_t_eval() -> None:
    """Requested output times are returned without alteration."""
    state0 = np.array([0.8, 0.0, 0.0, 0.0, 0.25, 0.0])
    t_eval = np.linspace(0.0, 1.0, 21)

    solution = propagate_cr3bp(state0, (0.0, 1.0), 0.0121505856, t_eval=t_eval)

    assert solution.success
    np.testing.assert_array_equal(solution.t, t_eval)


def test_jacobi_constant_is_conserved() -> None:
    """A nonsingular Earth-Moon trajectory conserves the Jacobi constant."""
    mu = 0.0121505856
    state0 = np.array([0.8, 0.0, 0.0, 0.0, 0.25, 0.0])
    t_eval = np.linspace(0.0, 5.0, 201)

    solution = propagate_cr3bp(state0, (0.0, 5.0), mu, t_eval=t_eval)
    C = np.array([jacobi_constant(state, mu) for state in solution.y.T])

    assert solution.success
    assert np.max(np.abs(C - C[0])) < 1e-9


def test_invalid_initial_state_shape_is_rejected() -> None:
    """The propagation boundary requires one six-dimensional initial state."""
    with pytest.raises(ValueError, match=r"shape \(6,\)"):
        propagate_cr3bp(np.zeros(5), (0.0, 1.0), 0.0121505856)






def test_jacobi_constant_is_conserved() -> None:
    """Verify approximate Jacobi-constant conservation during propagation."""

    mu = 0.0121505856

    state0 = np.array(
        [
            0.8,
            0.0,
            0.0,
            0.0,
            0.25,
            0.0,
        ]
    )

    t_eval = np.linspace(0.0, 5.0, 201)

    solution = propagate_cr3bp(
        state0,
        (0.0, 5.0),
        mu,
        t_eval=t_eval,
    )

    jacobi_values = np.array(
        [
            jacobi_constant(state, mu)
            for state in solution.y.T
        ]
    )

    max_jacobi_drift = np.max(
        np.abs(jacobi_values - jacobi_values[0])
    )

    print(f"Maximum Jacobi drift: {max_jacobi_drift:.3e}")
    assert solution.success
    assert max_jacobi_drift < 1e-9