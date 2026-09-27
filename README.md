# TNO Dynamics

Research code for studying dynamical-systems methods applied to trans-Neptunian object dynamics.

## Repository layout

- `neptune_zvs.py` — Dash/Plotly Sun–Neptune zero-velocity-surface explorer
- `src/cr3bp.py` — CR3BP equations, potential, and Lagrange-point utilities
- `assets/preview_bank.js` — loads browser-resident slider preview meshes
- `assets/preview_frames.dat` — generated preview mesh bank used while dragging
- `generate_preview_bank.py` — regenerates `preview_frames.dat` after mesh changes
- `docs/index.html` — redirects the GitHub Pages address to the Render app
- `notebooks/` — research notebooks

Render starts the Dash application through `server = app.server`. GitHub Pages publishes the `docs` directory.
