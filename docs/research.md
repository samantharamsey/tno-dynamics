---
title: Research
layout: research
math: true
feature_text: |
  # Research
  A living dissertation on dynamical structures in the outer Solar System
---

## 1. Research overview {#research-overview}

Trans-Neptunian objects preserve information about the formation and evolution of the outer Solar System. This living document records the mathematical development, computational decisions, experiments, and interpretations that support my research into the structures organizing their motion.

Unlike a finished dissertation, this version is intended to evolve. Sections can begin as working notes, expand into complete derivations, and eventually connect directly to figures, numerical results, code, and publications.

<div class="research-note">
  <strong>Document status.</strong> The framework below establishes the structure and demonstrates the mathematical typesetting. Research-specific derivations, results, and references will be expanded as the project develops.
</div>

## 2. System model {#system-model}

The circular restricted three-body problem provides a controlled model for motion under the gravitational influence of two massive bodies. For the Sun–Neptune system, the primary masses move in circular orbits about their barycenter while a third body has negligible mass and does not affect their motion.

Let the nondimensional masses of the primaries be \(1-\mu\) and \(\mu\), with the barycenter at the origin. In the rotating frame, the primaries remain fixed at

\[
\mathbf{r}_1 = (-\mu,0,0),
\qquad
\mathbf{r}_2 = (1-\mu,0,0).
\tag{2.1}
\]

The distances from the third body at \((x,y,z)\) to the primaries are

\[
r_1 = \sqrt{(x+\mu)^2+y^2+z^2},
\qquad
r_2 = \sqrt{(x-1+\mu)^2+y^2+z^2}.
\tag{2.2}
\]

## 3. Mathematical framework {#mathematical-framework}

### 3.1 Rotating frame {#rotating-frame}

A uniformly rotating coordinate system makes the two primaries stationary and exposes the geometry of the problem. The transformation introduces Coriolis and centrifugal terms, while the nondimensional formulation sets the primary separation, total mass, gravitational constant, and rotation rate to unity.

### 3.2 Equations of motion {#equations-of-motion}

Define the pseudo-potential

\[
\Omega(x,y,z)
=
\frac{1}{2}(x^2+y^2)
+
\frac{1-\mu}{r_1}
+
\frac{\mu}{r_2}.
\tag{3.1}
\]

The rotating-frame equations of motion can then be written compactly as

\[
\ddot{x}-2\dot{y}=\frac{\partial\Omega}{\partial x},
\qquad
\ddot{y}+2\dot{x}=\frac{\partial\Omega}{\partial y},
\qquad
\ddot{z}=\frac{\partial\Omega}{\partial z}.
\tag{3.2}
\]

This form will provide the starting point for the detailed derivation, equilibrium-point analysis, variational equations, and numerical propagation methods documented in later revisions.

### 3.3 Jacobi integral {#jacobi-integral}

The system admits the Jacobi integral

\[
C = 2\Omega(x,y,z)
-\left(\dot{x}^{2}+\dot{y}^{2}+\dot{z}^{2}\right).
\tag{3.3}
\]

At zero velocity, the boundary

\[
2\Omega(x,y,z)=C
\tag{3.4}
\]

defines the zero-velocity surface. These surfaces divide configuration space into accessible and forbidden regions and are the subject of the interactive Sun–Neptune visualization.

[Open the zero-velocity surface explorer ↗](https://tno-dynamics.onrender.com/){: .button .button--primary }

## 4. Computational approach {#computational-approach}

The computational work combines numerical integration, root finding, surface extraction, continuation methods, and interactive visualization. This section will record not only the final methods, but also the validation tests, numerical tolerances, failed approaches, and implementation decisions that shape the research.

Planned documentation includes:

- nondimensionalization and unit conversion;
- equilibrium-point computation and verification;
- integration methods and error control;
- periodic-orbit targeting and continuation;
- invariant-manifold computation;
- zero-velocity surface extraction and visualization;
- reproducibility notes linking equations to code and data.

## 5. Research log {#research-log}

The research log will provide dated entries for important decisions, numerical experiments, unexpected results, and changes in direction. Mature material can later move from the log into the formal chapters above while preserving a record of how the work developed.

<div class="research-note research-note--quiet">
  The next step is to replace these scaffolding sections with your existing notes and build the first complete derivation one subsection at a time.
</div>
