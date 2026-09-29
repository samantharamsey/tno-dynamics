'''generate equilibrium-point figures for the research site'''

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LightSource
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from cr3bp import lagrange_points, pseudo_potential


# use the same physical parameters and violet family as the zvs explorer
M_SUN = 1.9890e30
M_NEPTUNE = 1.0241e26
MU = M_NEPTUNE / (M_SUN + M_NEPTUNE)
POINTS = lagrange_points(MU)
OUTPUT = ROOT / 'docs' / 'assets' / 'images' / 'research' / 'cr3bp'

VIOLET = '#6e63c7'
VIOLET_DARK = '#4d438f'
VIOLET_SURFACE = '#d8b4fe'
VIOLET_LIGHT = '#f3e8ff'
INK = '#172235'
MUTED = '#5d6b7e'
LINE = '#dfe4ec'
SUN = '#e0a526'
NEPTUNE = '#4f8cff'

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 170,
    'savefig.dpi': 260,
    'savefig.facecolor': 'white',
})


def save_figure(fig, stem):
    '''save both web and vector versions of a figure'''
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f'{stem}.png', bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT / f'{stem}.svg', bbox_inches='tight', facecolor='white')


def clean_axis(ax):
    '''apply document styling to a two-dimensional axis'''
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=LINE, linewidth=0.7, alpha=0.55)
    ax.set_xlabel(r'$x$', color=INK)
    ax.set_ylabel(r'$y$', color=INK, rotation=0, labelpad=10)


def draw_planar_points(ax, labels=True, compact=False):
    '''draw the primaries and five equilibrium points'''
    l1, l2, l3, l4, l5 = POINTS
    points = {'$L_1$': l1, '$L_2$': l2, '$L_3$': l3, '$L_4$': l4, '$L_5$': l5}

    if not compact:
        triangle_x = [-MU, l4[0], 1 - MU, l5[0], -MU]
        triangle_y = [0, l4[1], 0, l5[1], 0]
        ax.plot(triangle_x, triangle_y, color=VIOLET, linestyle=(0, (5, 4)),
                linewidth=1.25, alpha=0.65, zorder=1)
        theta = np.linspace(0, 2 * np.pi, 500)
        ax.plot((1 - MU) * np.cos(theta), (1 - MU) * np.sin(theta),
                color=LINE, linestyle=':', linewidth=1.0, zorder=0)

    for name, point in points.items():
        ax.scatter(*point, s=34 if not compact else 44, color=VIOLET_DARK,
                   edgecolor='white', linewidth=0.9, zorder=5)

    ax.scatter(-MU, 0, s=210 if not compact else 110, color=SUN,
               edgecolor=INK, linewidth=0.9, zorder=4)
    ax.scatter(1 - MU, 0, s=70 if not compact else 90, color=NEPTUNE,
               edgecolor=INK, linewidth=0.9, zorder=6)

    if labels and not compact:
        ax.annotate('$L_3$', l3, xytext=(-2, 10), textcoords='offset points',
                    ha='center', color=INK)
        ax.annotate('$L_4$', l4, xytext=(7, 7), textcoords='offset points', color=INK)
        ax.annotate('$L_5$', l5, xytext=(7, -14), textcoords='offset points', color=INK)
        ax.annotate('Sun', (-MU, 0), xytext=(0, -19), textcoords='offset points',
                    ha='center', color=INK, fontsize=9)
        ax.annotate('Neptune region', (1 - MU, 0), xytext=(-7, 13),
                    textcoords='offset points', ha='right', color=INK, fontsize=9)


def make_planar_figure():
    '''create side-by-side full-system and neptune equilibrium maps'''
    l1, l2, _, _, _ = POINTS
    fig, (full, near) = plt.subplots(
        1, 2, figsize=(12.2, 5.7), constrained_layout=True,
        gridspec_kw={'width_ratios': (1.65, 1.0)}
    )

    clean_axis(full)
    draw_planar_points(full)
    full.axhline(0, color=INK, linewidth=0.9, zorder=0)
    full.axvline(0, color=INK, linewidth=0.9, zorder=0)
    full.set_xlim(-1.16, 1.16)
    full.set_ylim(-1.02, 1.02)
    full.set_aspect('equal', adjustable='box')
    full.set_title('(a) Full Sun–Neptune system', color=INK, pad=10)

    clean_axis(near)
    draw_planar_points(near, labels=False, compact=True)
    near.axhline(0, color=INK, linewidth=0.9, zorder=0)
    near.set_xlim(l1[0] - 0.007, l2[0] + 0.007)
    near.set_ylim(-0.012, 0.012)
    near.set_yticks([-0.01, 0, 0.01])
    near.set_title('(b) Neptune close-up', color=INK, pad=10)
    near.annotate('$L_1$', POINTS[0], xytext=(0, 12), textcoords='offset points',
                  ha='center', color=VIOLET_DARK, fontsize=11)
    near.annotate('Neptune', (1 - MU, 0), xytext=(0, -20),
                  textcoords='offset points', ha='center', color=INK, fontsize=9)
    near.annotate('$L_2$', POINTS[1], xytext=(0, 12), textcoords='offset points',
                  ha='center', color=VIOLET_DARK, fontsize=11)

    fig.suptitle('Sun–Neptune equilibrium points in the rotating frame',
                 color=INK, fontsize=15)
    save_figure(fig, 'equilibrium-points-planar')
    plt.close(fig)


def energy_surface_grid(xlim, ylim, nx, ny):
    '''evaluate the inverted effective potential and open the singular cores'''
    x = np.linspace(*xlim, nx)
    y = np.linspace(*ylim, ny)
    x_grid, y_grid = np.meshgrid(x, y)
    energy = -pseudo_potential(x_grid, y_grid, 0, MU)
    return x_grid, y_grid, energy


def surface_point(ax, point, label, z_offset, text_offset=(0, 0)):
    '''place an equilibrium point directly on an energy surface'''
    z = -pseudo_potential(point[0], point[1], 0, MU)
    ax.scatter(point[0], point[1], z + z_offset, s=31, color=INK,
               edgecolor='white', linewidth=0.65, depthshade=False, zorder=20)
    vertical_alignment = 'top' if text_offset[1] < 0 else 'bottom'
    ax.text(point[0] + text_offset[0], point[1],
            z + z_offset + text_offset[1], label,
            color=INK, fontsize=10, ha='center',
            va=vertical_alignment, zorder=21)


def draw_energy_surface(ax, xlim, ylim, zlim, grid_shape, core_radii,
                        title, elevation, azimuth, point_labels, wells,
                        box_aspect):
    '''draw a smooth, clipped surface of the inverted effective potential'''
    x_grid, y_grid, energy = energy_surface_grid(
        xlim, ylim, grid_shape[0], grid_shape[1]
    )
    z_span = zlim[1] - zlim[0]
    surface_floor = zlim[0] + 0.045 * z_span
    r_sun = np.hypot(x_grid + MU, y_grid)
    r_neptune = np.hypot(x_grid - 1 + MU, y_grid)
    well_region = ((r_sun < core_radii[0])
                   | (r_neptune < core_radii[1]))
    outer_cutoff = (energy < surface_floor) & ~well_region
    energy_display = np.ma.masked_where(
        outer_cutoff, np.clip(energy, surface_floor, zlim[1])
    )
    surface_map = LinearSegmentedColormap.from_list(
        'zvs_surface_violet',
        ['#302641', VIOLET_DARK, '#8f72bd', VIOLET_SURFACE, '#faf7ff']
    )
    light = LightSource(azdeg=320, altdeg=38)
    filled = np.ma.filled(energy_display, zlim[0])
    facecolors = light.shade(
        filled, cmap=surface_map, vert_exag=2.0,
        vmin=zlim[0], vmax=zlim[1], blend_mode='soft'
    )
    facecolors[..., 3] = np.where(np.ma.getmaskarray(energy_display), 0, 1)
    ax.plot_surface(
        x_grid, y_grid, energy_display, facecolors=facecolors,
        rcount=grid_shape[1], ccount=grid_shape[0], linewidth=0,
        antialiased=True, shade=False, rasterized=True, zorder=1
    )

    contour_levels = np.linspace(surface_floor + 0.02 * z_span,
                                 zlim[1] - 0.01 * z_span, 13)
    ax.contour(
        x_grid, y_grid, energy_display, levels=contour_levels,
        zdir='z', offset=zlim[0], colors=VIOLET_DARK,
        linewidths=0.8, alpha=0.82, zorder=0
    )

    for index, label, x_offset, text_height in point_labels:
        surface_point(ax, POINTS[index], label, 0.004 * z_span,
                      text_offset=(x_offset, text_height * z_span))

    for x, y, label, height_fraction in wells:
        ax.text(x, y, zlim[0] + height_fraction * z_span, label, color=INK,
                fontsize=9, ha='center', va='bottom', zorder=30)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_xlabel('$x$', color=INK, labelpad=5)
    ax.set_ylabel('$y$', color=INK, labelpad=5)
    ax.set_zlabel(r'$-\Omega(x,y,0)$', color=INK, labelpad=5)
    ax.set_title(title, color=INK, pad=8)
    ax.set_proj_type('ortho')
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_box_aspect(box_aspect)
    ax.tick_params(colors=MUTED, labelsize=7, pad=0)
    ax.set_facecolor('white')
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor(LINE)
        axis._axinfo['grid']['color'] = LINE
        axis._axinfo['grid']['linewidth'] = 0.45
        axis._axinfo['grid']['linestyle'] = ':'


def make_energy_surface_figure():
    '''create the effective-potential landscape and neptune close-up'''
    fig = plt.figure(figsize=(12.7, 6.35), constrained_layout=True)
    full = fig.add_subplot(1, 2, 1, projection='3d', computed_zorder=False)
    near = fig.add_subplot(1, 2, 2, projection='3d', computed_zorder=False)

    draw_energy_surface(
        full,
        (-1.22, 1.22), (-1.08, 1.08), (-1.72, -1.492),
        (390, 350), (0.68, 0.06),
        '(a) Full Sun–Neptune system', 34, -58,
        [(0, '$L_1$', -0.085, 0.020),
         (1, '$L_2$', 0.085, 0.015),
         (2, '$L_3$', 0, 0.018), (3, '$L_4$', 0, 0.018),
         (4, '$L_5$', 0, -0.045)],
        [(-MU, 0, 'Sun well', 0.76)],
        (1.15, 1.0, 0.58),
    )
    draw_energy_surface(
        near,
        (0.958, 1.043), (-0.04, 0.04), (-1.515, -1.4995),
        (410, 390), (0.0, 0.05),
        '(b) Neptune close-up', 31, -60,
        [(0, '$L_1$', 0, 0.035), (1, '$L_2$', 0, 0.035)],
        [(1 - MU, 0, 'Neptune well', 0.70)],
        (1.05, 1.0, 0.62),
    )

    fig.suptitle('Sun–Neptune effective-potential landscape',
                 color=INK, fontsize=15)
    save_figure(fig, 'effective-potential-surface')
    plt.close(fig)

def verify_points():
    '''report equilibrium residuals for the coordinates shown in the figures'''
    scale = 1e-6
    for index, point in enumerate(POINTS, start=1):
        x, y = point
        omega_x = (pseudo_potential(x + scale, y, 0, MU)
                   - pseudo_potential(x - scale, y, 0, MU)) / (2 * scale)
        omega_y = (pseudo_potential(x, y + scale, 0, MU)
                   - pseudo_potential(x, y - scale, 0, MU)) / (2 * scale)
        print(f'L{index}: x={x:.10f}, y={y:.10f}, |grad Omega|={np.hypot(omega_x, omega_y):.3e}')


if __name__ == '__main__':
    verify_points()
    make_planar_figure()
    make_energy_surface_figure()
