"""Tests for one symmetric halo-family pseudo-arclength step."""

import numpy as np
import pytest

from tno_dynamics.continuation import continue_halo_orbit
from tno_dynamics.events import y_plane_crossing_downward
from tno_dynamics.propagation import propagate_cr3bp
from tno_dynamics.targeting import correct_halo_orbit


MU = 0.01215058560962404
REFERENCE_STATE = np.array(
    [-0.41456184803140111, 0.0, 0.90753120433295065, 0.0, 1.4076145460136695, 0.0]
)
CONTROL_INDICES = np.array([0, 2, 4])


def test_continue_halo_orbit() -> None:
    """One pseudo-arclength step produces another symmetric periodic orbit."""
    second_guess = REFERENCE_STATE.copy()
    second_guess[2] += 1.0e-5
    current_state, _, _ = correct_halo_orbit(second_guess, MU)
    previous_state = REFERENCE_STATE.copy()
    previous_copy = previous_state.copy()
    current_copy = current_state.copy()

    u_previous = previous_state[CONTROL_INDICES]
    u_current = current_state[CONTROL_INDICES]
    tangent = u_current - u_previous
    tangent = tangent / np.linalg.norm(tangent)
    u_predictor = u_current + 1.0e-5 * tangent

    next_state, half_period, residual_history = continue_halo_orbit(
        previous_state,
        current_state,
        MU,
        step_size=1.0e-5,
    )

    assert next_state.shape == (6,)
    np.testing.assert_array_equal(previous_state, previous_copy)
    np.testing.assert_array_equal(current_state, current_copy)
    np.testing.assert_array_equal(next_state[[1, 3, 5]], current_state[[1, 3, 5]])
    assert residual_history[-1] < 1.0e-10
    assert residual_history[-1] < residual_history[0]
    assert not np.array_equal(next_state, current_state)

    u_next = next_state[CONTROL_INDICES]
    arclength_residual = tangent @ (u_next - u_predictor)
    assert abs(arclength_residual) < 1.0e-12

    crossing = propagate_cr3bp(
        next_state,
        (0.0, 5.0),
        MU,
        rtol=1.0e-12,
        atol=1.0e-12,
        events=y_plane_crossing_downward,
    )
    state_f = crossing.y_events[0][0]
    assert abs(state_f[3]) < 1.0e-10
    assert abs(state_f[5]) < 1.0e-10

    full_period = propagate_cr3bp(
        next_state,
        (0.0, 2.0 * half_period),
        MU,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    closure_error = np.linalg.norm(full_period.y[:, -1] - next_state)
    assert closure_error < 1.0e-8


def test_identical_continuation_states_are_rejected() -> None:
    """A secant tangent cannot be formed from identical continuation vectors."""
    with pytest.raises(ValueError, match="must not be identical"):
        continue_halo_orbit(REFERENCE_STATE, REFERENCE_STATE, MU, step_size=1.0e-5)
