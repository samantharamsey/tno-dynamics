import numpy as np


def eom(t, state, mu):
    '''nondimensional cr3bp equations of motion'''

    # state vector
    x, y, z, vx, vy, vz = state

    # distances from the third body to the larger and smaller primaries
    d = np.sqrt((x + mu)**2 + y**2 + z**2)
    r = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)

    # accelerations
    ax =  2*vy + x - (1 - mu)*(x + mu)/d**3 - mu*(x - 1 + mu)/r**3
    ay = -2*vx + y - (1 - mu)*y/d**3        - mu*y/r**3
    az =           - (1 - mu)*z/d**3        - mu*z/r**3

    # time derivative of the state vector
    dstate = np.array([vx, vy, vz, ax, ay, az])

    return dstate


def gamma1(g, mu):
    # polynomial and derivative for l1
    f = g**5 - (3 - mu)*g**4 + (3 - 2*mu)*g**3 - mu*g**2 + 2*mu*g - mu
    df = 5*g**4 - 4*(3 - mu)*g**3 + 3*(3 - 2*mu)*g**2 - 2*mu*g + 2*mu
    return f, df


def gamma2(g, mu):
    # polynomial and derivative for l2
    f = g**5 + (3 - mu)*g**4 + (3 - 2*mu)*g**3 - mu*g**2 - 2*mu*g - mu
    df = 5*g**4 + 4*(3 - mu)*g**3 + 3*(3 - 2*mu)*g**2 - 2*mu*g - 2*mu
    return f, df


def gamma3(g, mu):
    # polynomial and derivative for l3
    f = g**5 + (2 + mu)*g**4 + (1 + 2*mu)*g**3 - (1 - mu)*g**2 - 2*(1 - mu)*g - (1 - mu)
    df = 5*g**4 + 4*(2 + mu)*g**3 + 3*(1 + 2*mu)*g**2 - 2*(1 - mu)*g - 2*(1 - mu)
    return f, df


def newton_method(function, g0, mu, tol=1e-12):
    g = g0
    residual = 1
    while residual > tol:
        f, df = function(g, mu)
        gnew = g - f/df
        residual = abs(gnew - g)
        g = gnew
    return g


def lagrange_points(mu):
    # solve for the collinear points
    g1 = newton_method(gamma1, 0.1, mu)
    g2 = newton_method(gamma2, 0.1, mu)
    g3 = newton_method(gamma3, 1.0, mu)

    # collinear points
    L1 = np.array([1 - mu - g1, 0.0])
    L2 = np.array([1 - mu + g2, 0.0])
    L3 = np.array([-mu - g3,    0.0])

    # equilateral points
    L4 = np.array([0.5 - mu,  np.sqrt(3)/2])
    L5 = np.array([0.5 - mu, -np.sqrt(3)/2])

    return L1, L2, L3, L4, L5


def pseudo_potential(x, y, z, mu):
    # distances to the larger and smaller primaries
    d = np.sqrt((x + mu)**2 + y**2 + z**2)
    r = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)

    # effective potential in the rotating frame
    U = (1 - mu)/d + mu/r + 0.5*(x**2 + y**2)

    return U


def jacobi_constant(state, mu):
    # state vector
    x, y, z, vx, vy, vz = state

    U = pseudo_potential(x, y, z, mu)

    # jacobi constant
    C = 2*U - (vx**2 + vy**2 + vz**2)

    return C


def zero_velocity(x, y, z, C, mu):
    # zero-velocity condition: 2U - C = 0
    U = pseudo_potential(x, y, z, mu)

    return 2*U - C










