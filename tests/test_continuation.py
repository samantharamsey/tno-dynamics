"""Tests for one symmetric halo-family pseudo-arclength step."""

import numpy as np
import pytest

from tno_dynamics.continuation import continue_halo_orbit, grow_halo_family
from tno_dynamics.events import y_plane_crossing_downward
from tno_dynamics.propagation import propagate_cr3bp
from tno_dynamics.targeting import correct_halo_orbit


MU = 0.01215058560962404
REFERENCE_STATE = np.array(
    [-0.41456184803140111, 0.0, 0.90753120433295065, 0.0, 1.4076145460136695, 0.0]
)
CONTROL_INDICES = np.array([0, 2, 4])
NEAR_BIFURCATION_GUESS = np.array(
    [0.823390678, 0.0, 0.001659057, 0.0, 0.126372300, 0.0]
)


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


def test_grow_halo_family() -> None:
    """Four family members retain symmetry and periodic closure."""
    first_state, first_half_period, _ = correct_halo_orbit(NEAR_BIFURCATION_GUESS, MU)
    second_guess = first_state.copy()
    second_guess[2] += 2.0e-4
    second_state, second_half_period, _ = correct_halo_orbit(second_guess, MU)
    first_copy = first_state.copy()
    second_copy = second_state.copy()

    family_states, half_periods, residual_histories = grow_halo_family(
        first_state,
        second_state,
        first_half_period,
        second_half_period,
        MU,
        number_of_orbits=4,
        step_size=2.0e-4,
    )

    assert family_states.shape == (4, 6)
    assert half_periods.shape == (4,)
    assert len(residual_histories) == 2
    np.testing.assert_array_equal(family_states[:2], np.array([first_state, second_state]))
    np.testing.assert_array_equal(
        half_periods[:2], np.array([first_half_period, second_half_period])
    )
    np.testing.assert_array_equal(first_state, first_copy)
    np.testing.assert_array_equal(second_state, second_copy)
    np.testing.assert_array_equal(family_states[:, [1, 3, 5]], np.zeros((4, 3)))
    assert all(history[-1] < 1.0e-10 for history in residual_histories)
    assert np.all(half_periods > 0.0)
    assert not np.array_equal(family_states[-1], second_state)

    solution = propagate_cr3bp(
        family_states[-1],
        (0.0, 2.0 * half_periods[-1]),
        MU,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    closure_error = np.linalg.norm(solution.y[:, -1] - family_states[-1])
    assert closure_error < 1.0e-8


def test_grow_halo_family_rejects_invalid_inputs() -> None:
    """The family boundary enforces two six-state seeds and at least two members."""
    with pytest.raises(ValueError, match="at least 2"):
        grow_halo_family(np.zeros(6), np.ones(6), 1.0, 1.0, MU, number_of_orbits=1)
    with pytest.raises(ValueError, match="first_state"):
        grow_halo_family(np.zeros(5), np.ones(6), 1.0, 1.0, MU)
    with pytest.raises(ValueError, match="second_state"):
        grow_halo_family(np.zeros(6), np.ones(5), 1.0, 1.0, MU)
