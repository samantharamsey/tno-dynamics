"""Equations of motion for the circular restricted three-body problem."""

import numpy as np
from numpy.typing import NDArray


def cr3bp_equations(t: float, state: NDArray[np.float64], mu: float) -> NDArray[np.float64]:
    """Evaluate the nondimensional CR3BP equations in the rotating frame."""
    del t

    x, y, z, xdot, ydot, zdot = state
    r1 = np.sqrt((x + mu) ** 2 + y**2 + z**2)
    r2 = np.sqrt((x - 1.0 + mu) ** 2 + y**2 + z**2)

    xddot = (
        2.0 * ydot
        + x
        - (1.0 - mu) * (x + mu) / r1**3
        - mu * (x - 1.0 + mu) / r2**3
    )
    yddot = (
        -2.0 * xdot
        + y
        - (1.0 - mu) * y / r1**3
        - mu * y / r2**3
    )
    zddot = -(1.0 - mu) * z / r1**3 - mu * z / r2**3

    return np.array([xdot, ydot, zdot, xddot, yddot, zddot], dtype=np.float64)


def jacobi_constant(state: NDArray[np.float64], mu: float) -> float:
    """Compute the nondimensional CR3BP Jacobi constant."""

    x, y, z, xdot, ydot, zdot = state
    r1 = np.sqrt((x + mu) ** 2 + y**2 + z**2)
    r2 = np.sqrt((x - 1.0 + mu) ** 2 + y**2 + z**2)

    Omega = 0.5 * (x**2 + y**2) + (1.0 - mu) / r1 + mu / r2
    C = 2.0 * Omega - (xdot**2 + ydot**2 + zdot**2)

    return float(C)
