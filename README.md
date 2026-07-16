# TNO Dynamics

A scientific Python project for doctoral research in astrodynamics and dynamical systems.

The initial numerical development focuses on:

- circular restricted three-body problem dynamics;
- state transition matrix propagation;
- event-based trajectory integration;
- differential correction of periodic orbits;
- pseudo-arclength continuation;
- periodic-orbit stability;
- Poincaré maps and resonance analysis.

The longer-term research objective is to investigate resonance dynamics,
chaotic transport, resonance sticking, and trans-Neptunian object dynamics.

## Project status

This repository is being developed from first principles. Mathematical
conventions, numerical methods, validation cases, and software interfaces
will be documented before or alongside their implementation.

## Repository structure

- `docs/`: mathematical and numerical documentation
- `src/tno_dynamics/`: reusable Python source code
- `tests/`: automated tests and numerical validation cases
- `notebooks/`: exploratory analysis and demonstrations
- `scripts/`: reproducible command-line research workflows
- `data/`: raw and processed data
- `results/`: generated figures, tables, and orbit-family data