"""Tests for symmetric halo-orbit differential correction."""

import numpy as np
import pytest

from tno_dynamics.propagation import propagate_cr3bp
from tno_dynamics.targeting import correct_halo_orbit


MU = 0.01215058560962404
REFERENCE_STATE = np.array(
    [-0.41456184803140111, 0.0, 0.90753120433295065, 0.0, 1.4076145460136695, 0.0]
)
REFERENCE_PERIOD = 3.1233143922761588


def test_correct_halo_orbit() -> None:
    """The corrector recovers symmetry and full-period closure near the reference orbit."""
    state_guess = REFERENCE_STATE.copy()
    state_guess[0] += 1.0e-6
    state_guess[4] -= 1.0e-6
    original_guess = state_guess.copy()

    corrected_state, half_period, residual_history = correct_halo_orbit(state_guess, MU)

    assert corrected_state.shape == (6,)
    assert corrected_state[2] == state_guess[2]
    np.testing.assert_array_equal(corrected_state[[1, 3, 5]], state_guess[[1, 3, 5]])
    np.testing.assert_array_equal(state_guess, original_guess)
    assert residual_history.ndim == 1
    assert np.all(np.diff(residual_history) < 0.0)
    assert residual_history[-1] < 1.0e-10
    assert half_period == pytest.approx(REFERENCE_PERIOD / 2.0, rel=0.0, abs=1.0e-9)

    solution = propagate_cr3bp(
        corrected_state,
        (0.0, 2.0 * half_period),
        MU,
        rtol=1.0e-13,
        atol=1.0e-13,
    )

    assert solution.success
    np.testing.assert_allclose(solution.y[:, -1], corrected_state, rtol=0.0, atol=1.0e-9)


def test_correct_halo_orbit_rejects_incorrect_shape() -> None:
    """The targeting boundary requires a six-dimensional initial state."""
    with pytest.raises(ValueError, match=r"shape \(6,\)"):
        correct_halo_orbit(np.zeros(5), MU)
