# Project Instructions

## Purpose

This repository supports doctoral research in astrodynamics and dynamical
systems. Scientific correctness, clarity, reproducibility, and traceability
are more important than implementation speed.

## Current Development Stage

The project is being built from scratch.

Do not assume that an algorithm, mathematical convention, software interface,
or validation result has already been accepted unless it is explicitly
documented in the repository.

## General Rules

- Read the relevant files in `docs/` before implementing scientific code.
- Do not invent missing mathematical conventions.
- State assumptions explicitly.
- Prefer small, testable functions.
- Do not introduce classes unless there is a clear need for persistent state.
- Keep reusable algorithms in `src/tno_dynamics/`.
- Keep exploratory notebooks thin.
- Keep plotting separate from core numerical calculations.
- Add tests with each scientific implementation.
- Do not claim that a method is validated unless the validation was run.
- Do not rewrite unrelated files.
- Do not change public interfaces silently.

## Research-Code Style

This is a private engineering research codebase, not a general-purpose
software library.

- Keep scientific functions short and visually close to the equations.
- Use standard mathematical notation such as `mu`, `r1`, `r2`, `xdot`,
  `xddot`, `Phi`, and `A`.
- Do not add extensive defensive input validation to low-level mathematical
  functions.
- Perform basic input validation only at major user-facing boundaries, such
  as propagation or targeting functions.
- Do not repeat the full mathematical model in every function docstring.
  Refer to the documentation in `docs/` for detailed conventions.
- Prefer one-line or short docstrings for straightforward mathematical
  functions.
- Prioritize tests of scientific behavior over tests of artificial input
  restrictions.
- Do not create helper functions, classes, result objects, or abstractions
  unless the current research method requires them.
- A transparent ten-line implementation is preferable to a generalized
  fifty-line implementation.

## Python Conventions

- Use Python 3.12 or later.
- Use NumPy arrays for states and matrices.
- Add type hints to public functions.
- Add scientific docstrings describing state order, units, reference frame,
  inputs, outputs, and assumptions.
- Avoid explicit matrix inversion when solving numerical systems.
- Use standard mathematical names when they correspond directly to the
  documented equations. Avoid unexplained indices or abbreviations.

## Required Checks

Before declaring an implementation complete, run:

- `pytest`
- `ruff check .`

Report which checks were run and whether they passed.

## Communication

For each completed task, report:

1. Files changed.
2. Mathematical assumptions.
3. Implementation decisions.
4. Tests or validations performed.
5. Remaining limitations.