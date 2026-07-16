"""Variational quantities for the circular restricted three-body problem."""

import numpy as np
from numpy.typing import NDArray

from tno_dynamics.dynamics.cr3bp import cr3bp_equations


def cr3bp_jacobian(state: NDArray[np.float64], mu: float) -> NDArray[np.float64]:
    """Evaluate the analytical state Jacobian of the CR3BP equations."""
    x, y, z, _, _, _ = state
    x1 = x + mu
    x2 = x - 1.0 + mu
    r1 = np.sqrt(x1**2 + y**2 + z**2)
    r2 = np.sqrt(x2**2 + y**2 + z**2)

    Omega_xx = (
        1.0
        - (1.0 - mu) / r1**3
        - mu / r2**3
        + 3.0 * (1.0 - mu) * x1**2 / r1**5
        + 3.0 * mu * x2**2 / r2**5
    )
    Omega_yy = (
        1.0
        - (1.0 - mu) / r1**3
        - mu / r2**3
        + 3.0 * (1.0 - mu) * y**2 / r1**5
        + 3.0 * mu * y**2 / r2**5
    )
    Omega_zz = (
        -(1.0 - mu) / r1**3
        - mu / r2**3
        + 3.0 * (1.0 - mu) * z**2 / r1**5
        + 3.0 * mu * z**2 / r2**5
    )
    Omega_xy = 3.0 * (1.0 - mu) * x1 * y / r1**5 + 3.0 * mu * x2 * y / r2**5
    Omega_xz = 3.0 * (1.0 - mu) * x1 * z / r1**5 + 3.0 * mu * x2 * z / r2**5
    Omega_yz = 3.0 * (1.0 - mu) * y * z / r1**5 + 3.0 * mu * y * z / r2**5

    A = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [Omega_xx, Omega_xy, Omega_xz, 0.0, 2.0, 0.0],
            [Omega_xy, Omega_yy, Omega_yz, -2.0, 0.0, 0.0],
            [Omega_xz, Omega_yz, Omega_zz, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    return A


def cr3bp_variational_equations(
    t: float,
    augmented_state: NDArray[np.float64],
    mu: float,
) -> NDArray[np.float64]:
    """Evaluate the CR3BP state and state-transition-matrix equations."""
    state = augmented_state[:6]
    Phi = augmented_state[6:].reshape((6, 6), order="C")

    state_dot = cr3bp_equations(t, state, mu)
    A = cr3bp_jacobian(state, mu)
    Phi_dot = A @ Phi

    return np.concatenate((state_dot, Phi_dot.reshape(36, order="C")))
