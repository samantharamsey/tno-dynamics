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

## Python Conventions

- Use Python 3.12 or later.
- Use NumPy arrays for states and matrices.
- Add type hints to public functions.
- Add scientific docstrings describing state order, units, reference frame,
  inputs, outputs, and assumptions.
- Avoid explicit matrix inversion when solving numerical systems.
- Use descriptive names rather than unexplained numerical indices.

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