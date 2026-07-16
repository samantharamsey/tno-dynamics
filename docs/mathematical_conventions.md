# Mathematical Conventions

This document defines the mathematical conventions used throughout the
`tno_dynamics` package. Scientific implementations must remain consistent
with these conventions unless a module explicitly documents an alternative
model.

## Circular Restricted Three-Body Problem

The initial numerical foundation uses the nondimensional circular restricted
three-body problem.

The model consists of two finite primary bodies with masses \(m_1\) and
\(m_2\), together with a third body whose mass is assumed to be negligible.
The third body therefore responds to the gravity of the primaries but does
not affect their motion.

The primary masses satisfy

\[
m_1 \geq m_2.
\]

The mass parameter is defined as

\[
\mu = \frac{m_2}{m_1 + m_2}.
\]

Consequently,

\[
1-\mu = \frac{m_1}{m_1+m_2}.
\]

The valid mass-parameter interval used by the implementation is

\[
0 < \mu \leq \frac{1}{2}.
\]

The limiting value \(\mu=0\) is excluded from the initial implementation
because it reduces the model to a degenerate two-body limit.

## Nondimensionalization

The following characteristic quantities are normalized to unity:

- the distance between the two primaries;
- the sum of the primary masses;
- the gravitational constant;
- the angular velocity of the rotating frame.

Therefore,

\[
m_1+m_2=1,
\]

and the nondimensional mean motion is

\[
n=1.
\]

The orbital period of the primaries in nondimensional time is therefore

\[
T_{\mathrm{primary}}=2\pi.
\]

All states and integration times used by the initial CR3BP implementation
are nondimensional.

## Reference Frame

The equations are expressed in a barycentric rotating frame, also called the
synodic frame.

The frame:

- has its origin at the barycenter of the two primary bodies;
- rotates with the orbital motion of the primaries;
- has its \(x\)-axis directed from the larger primary toward the smaller
  primary;
- has its \(z\)-axis aligned with the orbital angular momentum of the
  primaries;
- completes a right-handed coordinate system with the \(y\)-axis.

The primary bodies remain fixed in this frame.

The larger primary is located at

\[
\mathbf{r}_{P_1}
=
\begin{bmatrix}
-\mu \\
0 \\
0
\end{bmatrix},
\]

and the smaller primary is located at

\[
\mathbf{r}_{P_2}
=
\begin{bmatrix}
1-\mu \\
0 \\
0
\end{bmatrix}.
\]

## State Convention

The six-dimensional Cartesian state is ordered as

\[
\mathbf{x}
=
\begin{bmatrix}
x &
y &
z &
\dot{x} &
\dot{y} &
\dot{z}
\end{bmatrix}^{T}.
\]

The first three components are position coordinates in the rotating frame.
The final three components are derivatives with respect to nondimensional
time in the rotating frame.

In Python, a single state is represented by a one-dimensional NumPy array
with shape `(6,)`.

The project will not silently accept alternative state orderings.

## Distances from the Primaries

The displacement from the larger primary to the third body is

\[
\mathbf{r}_1
=
\begin{bmatrix}
x+\mu \\
y \\
z
\end{bmatrix},
\]

with magnitude

\[
r_1
=
\sqrt{(x+\mu)^2+y^2+z^2}.
\]

The displacement from the smaller primary to the third body is

\[
\mathbf{r}_2
=
\begin{bmatrix}
x-1+\mu \\
y \\
z
\end{bmatrix},
\]

with magnitude

\[
r_2
=
\sqrt{(x-1+\mu)^2+y^2+z^2}.
\]

The subscripts \(1\) and \(2\) always refer to the larger and smaller
primaries, respectively.

## Effective Potential Convention

The effective potential is defined as

\[
\Omega(x,y,z)
=
\frac{1}{2}\left(x^2+y^2\right)
+
\frac{1-\mu}{r_1}
+
\frac{\mu}{r_2}.
\]

Under this convention, the rotating-frame equations of motion are written as

\[
\ddot{x}
=
2\dot{y}
+
\frac{\partial \Omega}{\partial x},
\]

\[
\ddot{y}
=
-2\dot{x}
+
\frac{\partial \Omega}{\partial y},
\]

\[
\ddot{z}
=
\frac{\partial \Omega}{\partial z}.
\]

The explicit equations will be documented alongside their implementation.

## Jacobi Constant Convention

The Jacobi constant is defined as

\[
C
=
2\Omega(x,y,z)
-
\left(
\dot{x}^{2}
+
\dot{y}^{2}
+
\dot{z}^{2}
\right).
\]

For an exact CR3BP trajectory, \(C\) is constant.

Numerically computed trajectories will exhibit a finite Jacobi-constant
error due to integration tolerances and floating-point arithmetic. The
acceptable error must be stated by each validation experiment rather than
assumed globally.

## State Transition Matrix Convention

The state transition matrix is defined as

\[
\Phi(t,t_0)
=
\frac{\partial \mathbf{x}(t)}
{\partial \mathbf{x}(t_0)}.
\]

It has shape `(6, 6)` and satisfies the initial condition

\[
\Phi(t_0,t_0)=I_6.
\]

When the state and state transition matrix are propagated together, the
augmented state will be ordered as

\[
\mathbf{x}_{\mathrm{aug}}
=
\begin{bmatrix}
\mathbf{x} \\
\operatorname{vec}(\Phi)
\end{bmatrix}.
\]

The six-dimensional state appears first, followed by the 36 state transition
matrix elements.

The state transition matrix will be flattened and reconstructed using
row-major order, corresponding to NumPy's default `order="C"` convention.

Therefore, the augmented state has shape `(42,)`.

This convention must be used consistently by the variational equations,
propagator, targeter, and continuation routines.

## Numerical Precision

The initial implementation will use NumPy double-precision floating-point
values.

Solver tolerances must always be supplied explicitly. Scientific functions
must not rely silently on SciPy's default integration tolerances.

Specific absolute and relative tolerances will be selected and documented
when the integration interface is implemented.

## Event Conventions

Event functions will be implemented separately from the equations of motion.

Each event must explicitly document:

- the scalar quantity whose zero defines the event;
- whether the event terminates integration;
- the permitted crossing direction;
- how an event at the initial time is handled;
- the physical or geometrical meaning of the event.

No universal plane-crossing direction is assumed at the package level.

## Periodic-Orbit Conventions

Orbit-specific symmetry conditions, free variables, constraints, and
continuation parameters will be documented before the periodic-orbit
targeting implementation is introduced.

The package will not assume that all periodic orbits use the same controls,
constraints, symmetry plane, or event condition.

## Current Scope

These conventions define the initial CR3BP foundation only.

Higher-fidelity models, inertial-frame formulations, dimensional models,
elliptic restricted models, and outer Solar System \(N\)-body models will
receive separate documented conventions when they are introduced.