"""Tests for CR3BP plane-crossing events."""

import numpy as np

from tno_dynamics.events import y_plane_crossing_downward
from tno_dynamics.propagation import propagate_cr3bp, propagate_cr3bp_with_stm


MU = 0.0121505856
STATE0 = np.array([0.8, 0.0, 0.0, 0.0, 0.25, 0.0])
T_SPAN = (0.0, 5.0)


def test_downward_y_plane_event_configuration() -> None:
    """The event returns y and is terminal only in the downward direction."""
    state = np.array([0.8, 0.2, 0.0, 0.0, -0.1, 0.0])

    assert y_plane_crossing_downward(1.0, state, MU) == state[1]
    assert y_plane_crossing_downward.terminal is True
    assert y_plane_crossing_downward.direction == -1


def test_state_propagation_detects_later_downward_crossing() -> None:
    """State propagation ignores the initial upward departure and stops later."""
    solution = propagate_cr3bp(STATE0, T_SPAN, MU, events=y_plane_crossing_downward)
    event_times = solution.t_events[0]
    event_states = solution.y_events[0]

    assert solution.success
    assert event_times.shape == (1,)
    assert event_times[0] > 0.0
    assert abs(event_states[0, 1]) < 1.0e-12
    assert event_states[0, 4] < 0.0


def test_stm_propagation_detects_same_downward_crossing() -> None:
    """The event reads physical y from an augmented state and stops at the same time."""
    state_solution = propagate_cr3bp(STATE0, T_SPAN, MU, events=y_plane_crossing_downward)
    stm_solution = propagate_cr3bp_with_stm(
        STATE0,
        T_SPAN,
        MU,
        events=y_plane_crossing_downward,
    )

    assert stm_solution.success
    assert stm_solution.t_events[0].shape == (1,)
    assert stm_solution.y_events[0].shape == (1, 42)
    np.testing.assert_allclose(
        stm_solution.t_events[0],
        state_solution.t_events[0],
        rtol=0.0,
        atol=1.0e-11,
    )
