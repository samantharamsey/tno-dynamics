'''generate planar zero-velocity curves for the research journal'''

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from cr3bp import jacobi_constant, lagrange_points, pseudo_potential


# use the same parameters and palette as the interactive explorer
M_SUN = 1.9890e30
M_NEPTUNE = 1.0241e26
MU = M_NEPTUNE / (M_SUN + M_NEPTUNE)
POINTS = lagrange_points(MU)
OUTPUT = ROOT / 'docs' / 'assets' / 'images' / 'research' / 'cr3bp'

VIOLET = '#6e63c7'
VIOLET_DARK = '#4d438f'
VIOLET_SURFACE = '#d8b4fe'
INK = '#172235'
MUTED = '#5d6b7e'
LINE = '#dfe4ec'
SUN = '#e0a526'
NEPTUNE = '#4f8cff'

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 170,
    'savefig.dpi': 260,
    'savefig.facecolor': 'white',
})


def critical_constants():
    '''return the jacobi constants at the five equilibrium points'''
    constants = []
    for x, y in POINTS:
        constants.append(jacobi_constant([x, y, 0, 0, 0, 0], MU))
    return np.asarray(constants)


def make_zvc_figure():
    '''plot four topology regimes of the planar zero-velocity curves'''
    constants = critical_constants()
    c1, c2, c3, c4, c5 = constants
    levels = [
        c1 + 0.001,
        0.5 * (c1 + c2),
        c3 + 0.25 * (c2 - c3),
        c4 + 0.90 * (c3 - c4),
    ]
    titles = [
        rf'(a) $C={levels[0]:.5f}>C_1$',
        rf'(b) $C_1>C={levels[1]:.5f}>C_2$',
        rf'(c) $C_2>C={levels[2]:.5f}>C_3$',
        rf'(d) $C_3>C={levels[3]:.5f}>C_{{4,5}}$',
    ]

    x = np.linspace(-1.55, 1.55, 1100)
    y = np.linspace(-1.45, 1.45, 1000)
    x_grid, y_grid = np.meshgrid(x, y)
    twice_omega = 2 * pseudo_potential(x_grid, y_grid, 0, MU)

    # resolve the narrow necks around Neptune on a separate fine grid
    x_neptune = np.linspace(0.955, 1.045, 900)
    y_neptune = np.linspace(-0.055, 0.055, 800)
    xn_grid, yn_grid = np.meshgrid(x_neptune, y_neptune)
    twice_omega_neptune = 2 * pseudo_potential(
        xn_grid, yn_grid, 0, MU
    )

    fig, axes = plt.subplots(
        2, 2, figsize=(11.8, 9.7), sharex=True, sharey=True,
        constrained_layout=True
    )

    orbit_angle = np.linspace(0, 2 * np.pi, 500)
    for panel_index, (ax, c_value, title) in enumerate(
            zip(axes.flat, levels, titles)):
        field = twice_omega - c_value
        ax.contourf(
            x_grid, y_grid, field,
            levels=[float(np.nanmin(field)), 0],
            colors=[VIOLET_SURFACE], alpha=0.58, antialiased=True
        )
        ax.contour(
            x_grid, y_grid, field, levels=[0],
            colors=[VIOLET_DARK], linewidths=1.35
        )
        ax.plot(
            (1 - MU) * np.cos(orbit_angle),
            (1 - MU) * np.sin(orbit_angle),
            color=LINE, linestyle=':', linewidth=0.9, zorder=1
        )
        ax.scatter(-MU, 0, s=72, color=SUN, edgecolor=INK,
                   linewidth=0.7, zorder=5)
        ax.scatter(1 - MU, 0, s=34, color=NEPTUNE, edgecolor=INK,
                   linewidth=0.7, zorder=5)
        ax.annotate('Sun', (-MU, 0), xytext=(0, -17),
                    textcoords='offset points', ha='center', color=INK,
                    fontsize=8)
        ax.annotate('Neptune', (1 - MU, 0), xytext=(0, -17),
                    textcoords='offset points', ha='center', color=INK,
                    fontsize=8)

        if panel_index < 2:
            detail = inset_axes(
                ax, width='39%', height='38%', loc='upper right',
                borderpad=0.8
            )
            detail_field = twice_omega_neptune - c_value
            detail.contourf(
                xn_grid, yn_grid, detail_field,
                levels=[float(np.nanmin(detail_field)), 0],
                colors=[VIOLET_SURFACE], alpha=0.58, antialiased=True
            )
            detail.contour(
                xn_grid, yn_grid, detail_field, levels=[0],
                colors=[VIOLET_DARK], linewidths=1.1
            )
            detail.scatter(1 - MU, 0, s=18, color=NEPTUNE,
                           edgecolor=INK, linewidth=0.5, zorder=5)
            detail.set_xlim(x_neptune[0], x_neptune[-1])
            detail.set_ylim(y_neptune[0], y_neptune[-1])
            detail.set_aspect('equal', adjustable='box')
            detail.set_title(
                'Neptune close-up', color=MUTED, fontsize=7, pad=3
            )
            detail.tick_params(colors=MUTED, labelsize=5.5, length=2)
            detail.grid(color=LINE, linewidth=0.4, alpha=0.55)
            for spine in detail.spines.values():
                spine.set_color(LINE)

        ax.set_xlim(-1.55, 1.55)
        ax.set_ylim(-1.45, 1.45)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title, color=INK, pad=9)
        ax.set_xlabel(r'$x$', color=INK)
        ax.set_ylabel(r'$y$', color=INK, rotation=0, labelpad=10)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(color=LINE, linewidth=0.55, alpha=0.55)
        for spine in ax.spines.values():
            spine.set_color(LINE)

    legend_handles = [
        Patch(facecolor=VIOLET_SURFACE, edgecolor='none', alpha=0.58,
              label=r'forbidden region, $2\Omega<C$'),
        Line2D([0], [0], color=VIOLET_DARK, linewidth=1.35,
               label='zero-velocity curve'),
    ]
    fig.legend(
        handles=legend_handles, loc='outside lower center', ncol=2,
        frameon=False, labelcolor=INK, fontsize=10
    )
    fig.suptitle(
        'Planar zero-velocity curves across the critical Jacobi levels',
        color=INK, fontsize=15
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT / 'zero-velocity-curves-levels.png',
        bbox_inches='tight', facecolor='white'
    )
    fig.savefig(
        OUTPUT / 'zero-velocity-curves-levels.svg',
        bbox_inches='tight', facecolor='white'
    )
    plt.close(fig)

    for index, value in enumerate(constants, start=1):
        print(f'C{index}={value:.10f}')
    for label, value in zip('abcd', levels):
        print(f'{label}: C={value:.10f}')


if __name__ == '__main__':
    make_zvc_figure()
