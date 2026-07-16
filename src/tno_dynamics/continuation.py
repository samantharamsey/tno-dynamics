"""Pseudo-arclength continuation for symmetric CR3BP halo orbits."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tno_dynamics.dynamics import cr3bp_equations
from tno_dynamics.events import y_plane_crossing_downward
from tno_dynamics.propagation import propagate_cr3bp_with_stm


def continue_halo_orbit(
    previous_state: ArrayLike,
    current_state: ArrayLike,
    mu: float,
    step_size: float,
    tol: float = 1e-10,
    max_iterations: int = 20,
    t_max: float = 5.0,
) -> tuple[NDArray[np.float64], float, NDArray[np.float64]]:
    """Take one pseudo-arclength step along a symmetric halo family."""
    previous_state = np.asarray(previous_state, dtype=float).copy()
    current_state = np.asarray(current_state, dtype=float).copy()
    if previous_state.shape != (6,):
        raise ValueError(f"previous_state must have shape (6,), got {previous_state.shape}")
    if current_state.shape != (6,):
        raise ValueError(f"current_state must have shape (6,), got {current_state.shape}")

    control_indices = np.array([0, 2, 4])
    u_previous = previous_state[control_indices]
    u_current = current_state[control_indices]
    tangent = u_current - u_previous
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm == 0.0:
        raise ValueError("continuation vectors must not be identical")
    tangent = tangent / tangent_norm

    u_predictor = u_current + step_size * tangent
    state0 = current_state.copy()
    state0[control_indices] = u_predictor
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

        u = state0[control_indices]
        F = np.array([state_f[3], state_f[5]])
        arclength_residual = tangent @ (u - u_predictor)
        G = np.array([F[0], F[1], arclength_residual])
        residual_history.append(float(np.linalg.norm(G)))

        if residual_history[-1] < tol:
            return state0, half_period, np.asarray(residual_history)

        state_dot_f = cr3bp_equations(half_period, state_f, mu)
        xddot_f = state_dot_f[3]
        ydot_f = state_f[4]
        zddot_f = state_dot_f[5]

        if abs(ydot_f) < 1.0e-14:
            raise RuntimeError("downward y = 0 crossing is tangent")

        dxdot_du = (
            Phi[3, control_indices] - xddot_f * Phi[1, control_indices] / ydot_f
        )
        dzdot_du = (
            Phi[5, control_indices] - zddot_f * Phi[1, control_indices] / ydot_f
        )
        DG = np.array([dxdot_du, dzdot_du, tangent])
        correction = np.linalg.solve(DG, -G)

        state0[control_indices] += correction

    raise RuntimeError(f"halo continuation did not converge in {max_iterations} iterations")


def grow_halo_family(
    first_state: ArrayLike,
    second_state: ArrayLike,
    first_half_period: float,
    second_half_period: float,
    mu: float,
    number_of_orbits: int = 25,
    step_size: float = 2e-4,
    tol: float = 1e-10,
    max_iterations: int = 20,
    t_max: float = 5.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[NDArray[np.float64]]]:
    """Grow a symmetric halo family from two corrected seed orbits."""
    first_state = np.asarray(first_state, dtype=float).copy()
    second_state = np.asarray(second_state, dtype=float).copy()
    if first_state.shape != (6,):
        raise ValueError(f"first_state must have shape (6,), got {first_state.shape}")
    if second_state.shape != (6,):
        raise ValueError(f"second_state must have shape (6,), got {second_state.shape}")
    if number_of_orbits < 2:
        raise ValueError("number_of_orbits must be at least 2")

    family_states = [first_state.copy(), second_state.copy()]
    half_periods = [first_half_period, second_half_period]
    residual_histories = []

    while len(family_states) < number_of_orbits:
        next_state, next_half_period, residual_history = continue_halo_orbit(
            family_states[-2],
            family_states[-1],
            mu,
            step_size,
            tol=tol,
            max_iterations=max_iterations,
            t_max=t_max,
        )
        family_states.append(next_state)
        half_periods.append(next_half_period)
        residual_histories.append(residual_history)

    return np.asarray(family_states), np.asarray(half_periods), residual_histories
