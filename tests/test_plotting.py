"""Tests for CR3BP family plotting."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tno_dynamics.plotting import plot_halo_family_3d

plt.switch_backend("Agg")


@pytest.fixture
def halo_family() -> tuple[np.ndarray, np.ndarray, list[SimpleNamespace]]:
    """Return a small family using the project's array and OdeResult-like structures."""
    phase = np.linspace(0.0, 2.0 * np.pi, 25)
    states = np.array(
        [
            [0.82, 0.0, 0.01, 0.0, 0.12, 0.0],
            [0.81, 0.0, 0.02, 0.0, 0.13, 0.0],
        ]
    )
    periods = np.array([2.7, 2.8])
    trajectories = []
    for state in states:
        history = np.zeros((6, phase.size))
        history[0] = state[0] + 0.01 * np.cos(phase)
        history[1] = 0.03 * np.sin(phase)
        history[2] = state[2] * np.cos(phase)
        trajectories.append(SimpleNamespace(y=history))
    return states, periods, trajectories


@pytest.mark.parametrize("color_by", ["jacobi", "period", "max_z", "initial_x"])
def test_plot_halo_family_3d_color_parameters(
    halo_family: tuple[np.ndarray, np.ndarray, list[SimpleNamespace]],
    color_by: str,
) -> None:
    """Every supported color parameter produces one line per orbit and a colorbar."""
    states, periods, trajectories = halo_family

    fig, ax = plot_halo_family_3d(
        states,
        periods,
        trajectories,
        0.01215058560962404,
        color_by=color_by,
        l1_x=0.836915125772357,
    )

    assert len(ax.lines) == len(states)
    assert len(fig.axes) == 2
    assert ax.get_xlabel() == "x (nd)"
    assert ax.get_ylabel() == "y (nd)"
    assert ax.get_zlabel() == "z (nd)"
    assert ax.get_title() == "Earth-Moon L1 Halo Family"
    plt.close(fig)


def test_plot_halo_family_3d_rejects_mismatched_trajectories(
    halo_family: tuple[np.ndarray, np.ndarray, list[SimpleNamespace]],
) -> None:
    """A trajectory is required for each accepted family member."""
    states, periods, trajectories = halo_family

    with pytest.raises(ValueError, match="one full orbit"):
        plot_halo_family_3d(states, periods, trajectories[:-1], 0.01215058560962404)
