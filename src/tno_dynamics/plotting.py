"""Plotting utilities for CR3BP orbit families."""

from collections.abc import Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from numpy.typing import ArrayLike

from tno_dynamics.dynamics.cr3bp import jacobi_constant

ColorParameter = Literal["jacobi", "period", "max_z", "initial_x"]

_COLOR_LABELS = {
    "jacobi": "Jacobi constant, C",
    "period": "Period (nd)",
    "max_z": "Maximum |z| (nd)",
    "initial_x": "Initial x (nd)",
}


def plot_halo_family_3d(
    family_states: ArrayLike,
    periods: ArrayLike,
    trajectories: Sequence[Any],
    mu: float,
    *,
    color_by: ColorParameter = "jacobi",
    cmap: str = "cool",
    line_width: float = 1.2,
    elevation: float = 25.0,
    azimuth: float = -60.0,
    l1_x: float | None = None,
) -> tuple[Any, Any]:
    """Plot full nondimensional CR3BP halo trajectories in the rotating frame."""
    family_states = np.asarray(family_states, dtype=float)
    periods = np.asarray(periods, dtype=float)
    if family_states.ndim != 2 or family_states.shape[1] != 6:
        raise ValueError("family_states must have shape (N, 6)")
    if periods.shape != (len(family_states),):
        raise ValueError("periods must have shape (N,)")
    if len(trajectories) != len(family_states):
        raise ValueError("trajectories must contain one full orbit per family state")
    if color_by not in _COLOR_LABELS:
        choices = ", ".join(_COLOR_LABELS)
        raise ValueError(f"color_by must be one of: {choices}")

    orbit_states = [np.asarray(solution.y, dtype=float) for solution in trajectories]
    if any(states.ndim != 2 or states.shape[0] != 6 for states in orbit_states):
        raise ValueError("each trajectory.y must have shape (6, M)")

    color_values = {
        "jacobi": np.array([jacobi_constant(state, mu) for state in family_states]),
        "period": periods,
        "max_z": np.array([np.max(np.abs(states[2])) for states in orbit_states]),
        "initial_x": family_states[:, 0],
    }[color_by]

    norm = Normalize(vmin=float(np.min(color_values)), vmax=float(np.max(color_values)))
    colormap = plt.get_cmap(cmap)
    mappable = ScalarMappable(norm=norm, cmap=colormap)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for states, value in zip(orbit_states, color_values, strict=True):
        ax.plot(
            states[0],
            states[1],
            states[2],
            color=colormap(norm(value)),
            linewidth=line_width,
        )

    ax.scatter(-mu, 0.0, 0.0, color="royalblue", s=45, label="Earth")
    ax.scatter(1.0 - mu, 0.0, 0.0, color="silver", edgecolor="black", s=30, label="Moon")
    if l1_x is not None:
        ax.scatter(l1_x, 0.0, 0.0, color="black", marker="x", s=50, label="L1")

    positions = np.hstack([states[:3] for states in orbit_states])
    spans = np.ptp(positions, axis=1)
    padding = np.maximum(0.08 * spans, 1.0e-5)
    lower = np.min(positions, axis=1) - padding
    upper = np.max(positions, axis=1) + padding
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_zlim(lower[2], upper[2])
    ax.set_box_aspect(np.maximum(upper - lower, 1.0e-5))

    ax.set_xlabel("x (nd)")
    ax.set_ylabel("y (nd)")
    ax.set_zlabel("z (nd)")
    ax.set_title("Earth-Moon L1 Halo Family")
    ax.view_init(elev=elevation, azim=azimuth)
    ax.legend(loc="upper left")

    colorbar = fig.colorbar(mappable, ax=ax, pad=0.1, shrink=0.75)
    colorbar.set_label(_COLOR_LABELS[color_by])
    fig.tight_layout()
    return fig, ax
