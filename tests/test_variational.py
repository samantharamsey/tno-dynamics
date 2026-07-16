"""Tests for the analytical CR3BP state Jacobian."""

import numpy as np

from tno_dynamics.dynamics import cr3bp_equations
from tno_dynamics.dynamics.variational import cr3bp_jacobian, cr3bp_variational_equations
from tno_dynamics.propagation import propagate_cr3bp, propagate_cr3bp_with_stm


MU = 0.0121505856
STATE = np.array([0.8, 0.1, 0.05, 0.02, 0.25, -0.01])


def test_cr3bp_jacobian_structure() -> None:
    """The kinematic and Coriolis blocks have the required structure."""
    A = cr3bp_jacobian(STATE, MU)

    assert A.shape == (6, 6)
    np.testing.assert_array_equal(A[:3, :3], np.zeros((3, 3)))
    np.testing.assert_array_equal(A[:3, 3:], np.eye(3))
    assert A[3, 4] == 2.0
    assert A[4, 3] == -2.0


def test_position_block_is_symmetric() -> None:
    """The effective-potential Hessian is symmetric."""
    position_block = cr3bp_jacobian(STATE, MU)[3:, :3]

    np.testing.assert_array_equal(position_block, position_block.T)


def test_analytical_jacobian_matches_central_difference() -> None:
    """The analytical matrix agrees with a central finite-difference Jacobian."""
    step = 1.0e-7
    numerical_A = np.empty((6, 6))

    for column in range(6):
        displacement = np.zeros(6)
        displacement[column] = step
        forward = cr3bp_equations(0.0, STATE + displacement, MU)
        backward = cr3bp_equations(0.0, STATE - displacement, MU)
        numerical_A[:, column] = (forward - backward) / (2.0 * step)

    analytical_A = cr3bp_jacobian(STATE, MU)

    np.testing.assert_allclose(analytical_A, numerical_A, rtol=1.0e-7, atol=1.0e-8)


def test_variational_equations_with_identity_stm() -> None:
    """The augmented derivative contains A when the STM is identity."""
    augmented_state = np.concatenate((STATE, np.eye(6).reshape(36, order="C")))

    augmented_state_dot = cr3bp_variational_equations(0.0, augmented_state, MU)
    Phi_dot = augmented_state_dot[6:].reshape((6, 6), order="C")

    assert augmented_state_dot.shape == (42,)
    np.testing.assert_array_equal(Phi_dot, cr3bp_jacobian(STATE, MU))


def test_stm_propagation_succeeds_with_identity_initial_stm() -> None:
    """The augmented propagation starts from state0 and the identity STM."""
    solution = propagate_cr3bp_with_stm(STATE, (0.0, 0.5), MU)
    Phi0 = solution.y[6:, 0].reshape((6, 6), order="C")

    assert solution.success
    assert solution.y.shape[0] == 42
    np.testing.assert_array_equal(Phi0, np.eye(6))


def test_propagated_stm_matches_initial_condition_sensitivity() -> None:
    """The propagated STM agrees with central initial-condition differences."""
    t_span = (0.0, 0.5)
    step = 1.0e-7
    rtol = 1.0e-12
    atol = 1.0e-12
    numerical_Phi = np.empty((6, 6))

    for column in range(6):
        displacement = np.zeros(6)
        displacement[column] = step
        plus = propagate_cr3bp(STATE + displacement, t_span, MU, rtol=rtol, atol=atol)
        minus = propagate_cr3bp(STATE - displacement, t_span, MU, rtol=rtol, atol=atol)
        numerical_Phi[:, column] = (plus.y[:, -1] - minus.y[:, -1]) / (2.0 * step)

    solution = propagate_cr3bp_with_stm(STATE, t_span, MU, rtol=rtol, atol=atol)
    analytical_Phi = solution.y[6:, -1].reshape((6, 6), order="C")

    np.testing.assert_allclose(analytical_Phi, numerical_Phi, rtol=1.0e-7, atol=2.0e-8)
