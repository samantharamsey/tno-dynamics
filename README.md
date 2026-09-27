# TNO Dynamics

Research code for studying dynamical-systems methods applied to trans-Neptunian object dynamics.

## Repository layout

- `apps/neptune_zvs/` — Sun–Neptune zero-velocity-surface interactive app
  - `app.py` — Dash/Plotly application and mesh endpoints
  - `assets/` — browser loader and precomputed slider-preview mesh bank
  - `generate_preview_bank.py` — regenerates the preview mesh bank after mesh changes
- `neptune_zvs.py` — compatibility launcher used by the current Render service
- `src/cr3bp.py` — shared CR3BP equations, potential, and Lagrange-point utilities
- `docs/index.html` — redirects the GitHub Pages address to the Render app
- `notebooks/` — research notebooks

Render can continue using `gunicorn neptune_zvs:server`. Regenerate the moving-preview meshes from the repository root with:

```powershell
python -m apps.neptune_zvs.generate_preview_bank
```
