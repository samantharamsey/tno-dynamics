"""Trajectory propagation for the circular restricted three-body problem."""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from tno_dynamics.dynamics import cr3bp_equations
from tno_dynamics.dynamics.variational import cr3bp_variational_equations


def propagate_cr3bp(
    state0: ArrayLike,
    t_span: tuple[float, float],
    mu: float,
    t_eval: NDArray[np.float64] | None = None,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    events: Any = None,
) -> Any:
    """Propagate the nondimensional CR3BP equations in the rotating frame."""
    state0 = np.asarray(state0, dtype=float)
    if state0.shape != (6,):
        raise ValueError(f"state0 must have shape (6,), got {state0.shape}")

    return solve_ivp(
        cr3bp_equations,
        t_span,
        state0,
        method="DOP853",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        args=(mu,),
        events=events,
    )


def propagate_cr3bp_with_stm(
    state0: ArrayLike,
    t_span: tuple[float, float],
    mu: float,
    t_eval: NDArray[np.float64] | None = None,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    events: Any = None,
) -> Any:
    """Propagate the nondimensional CR3BP state and STM in the rotating frame."""
    state0 = np.asarray(state0, dtype=float)
    if state0.shape != (6,):
        raise ValueError(f"state0 must have shape (6,), got {state0.shape}")

    Phi0 = np.eye(6)
    augmented_state0 = np.concatenate((state0, Phi0.reshape(36, order="C")))

    return solve_ivp(
        cr3bp_variational_equations,
        t_span,
        augmented_state0,
        method="DOP853",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        args=(mu,),
        events=events,
    )
