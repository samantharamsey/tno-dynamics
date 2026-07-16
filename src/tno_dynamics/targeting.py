"""Differential correction for symmetric CR3BP halo orbits."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tno_dynamics.dynamics import cr3bp_equations
from tno_dynamics.events import y_plane_crossing_downward
from tno_dynamics.propagation import propagate_cr3bp_with_stm


def correct_halo_orbit(
    state_guess: ArrayLike,
    mu: float,
    tol: float = 1e-10,
    max_iterations: int = 20,
    t_max: float = 5.0,
) -> tuple[NDArray[np.float64], float, NDArray[np.float64]]:
    """Correct x0 and ydot0 for a symmetric northern halo orbit."""
    state0 = np.asarray(state_guess, dtype=float).copy()
    if state0.shape != (6,):
        raise ValueError(f"state_guess must have shape (6,), got {state0.shape}")

    residual_history: list[float] = []

    for _ in range(max_iterations):
        solution = propagate_cr3bp_with_stm(
            state0,
            (0.0, t_max),
            mu,
            events=y_plane_crossing_downward,
        )
        if solution.t_events[0].size == 0:
            raise RuntimeError("no downward y = 0 crossing found")

        half_period = float(solution.t_events[0][0])
        augmented_state_f = solution.y_events[0][0]
        state_f = augmented_state_f[:6]
        Phi = augmented_state_f[6:].reshape((6, 6), order="C")

        F = np.array([state_f[3], state_f[5]])
        residual_history.append(float(np.linalg.norm(F)))

        if residual_history[-1] < tol:
            return state0, half_period, np.asarray(residual_history)

        state_dot_f = cr3bp_equations(half_period, state_f, mu)
        xddot_f = state_dot_f[3]
        zddot_f = state_dot_f[5]
        ydot_f = state_f[4]

        if abs(ydot_f) < 1.0e-14:
            raise RuntimeError("downward y = 0 crossing is tangent")

        DF = np.array(
            [
                [
                    Phi[3, 0] - xddot_f * Phi[1, 0] / ydot_f,
                    Phi[3, 4] - xddot_f * Phi[1, 4] / ydot_f,
                ],
                [
                    Phi[5, 0] - zddot_f * Phi[1, 0] / ydot_f,
                    Phi[5, 4] - zddot_f * Phi[1, 4] / ydot_f,
                ],
            ]
        )
        correction = np.linalg.solve(DF, -F)

        state0[0] += correction[0]
        state0[4] += correction[1]

    raise RuntimeError(f"halo corrector did not converge in {max_iterations} iterations")
