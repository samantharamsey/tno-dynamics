"""Tests for the nondimensional CR3BP equations of motion."""

import numpy as np
import pytest

from tno_dynamics.dynamics import cr3bp_equations, jacobi_constant


def test_derivative_shape_and_velocity_components() -> None:
    """The derivative has six components and begins with the state velocity."""
    state = np.array([0.8, 0.1, -0.2, 0.3, -0.4, 0.5], dtype=np.float64)

    derivative = cr3bp_equations(1.25, state, 0.01)

    assert derivative.shape == (6,)
    assert derivative.dtype == np.float64
    np.testing.assert_array_equal(derivative[:3], state[3:])


def test_l4_is_an_equilibrium() -> None:
    """The triangular L4 point has zero derivative at zero rotating velocity."""
    mu = 0.0121505856
    l4_state = np.array(
        [0.5 - mu, np.sqrt(3.0) / 2.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )

    derivative = cr3bp_equations(0.0, l4_state, mu)

    np.testing.assert_allclose(derivative, np.zeros(6), rtol=0.0, atol=5.0e-15)


def test_jacobi_constant_is_finite_for_valid_state() -> None:
    """A finite nonsingular state has a finite Jacobi constant."""
    state = np.array([0.8, 0.1, -0.2, 0.3, -0.4, 0.5], dtype=np.float64)

    C = jacobi_constant(state, 0.01)

    assert np.isfinite(C)


def test_velocity_increase_reduces_jacobi_constant_by_squared_speed() -> None:
    """Changing only velocity changes C by the negative squared-speed change."""
    stationary_state = np.array([0.8, 0.1, -0.2, 0.0, 0.0, 0.0], dtype=np.float64)
    moving_state = stationary_state.copy()
    moving_state[3:] = np.array([0.3, -0.4, 0.5])
    squared_speed = np.dot(moving_state[3:], moving_state[3:])

    C_stationary = jacobi_constant(stationary_state, 0.01)
    C_moving = jacobi_constant(moving_state, 0.01)

    assert C_moving == pytest.approx(C_stationary - squared_speed, rel=0.0, abs=1.0e-15)


def test_jacobi_constant_at_l4() -> None:
    """The L4 Jacobi constant matches its analytic nondimensional value."""
    mu = 0.0121505856
    l4_state = np.array(
        [0.5 - mu, np.sqrt(3.0) / 2.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )

    C = jacobi_constant(l4_state, mu)

    assert C == pytest.approx(3.0 - mu + mu**2, rel=0.0, abs=2.0e-15)

