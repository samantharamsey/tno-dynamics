import gzip
import struct
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from flask import Response, request
from plotly.subplots import make_subplots
from skimage.measure import marching_cubes

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from cr3bp import lagrange_points, jacobi_constant, zero_velocity


# sun-neptune system
m_sun = 1.9890e30
m_neptune = 1.0241e26
mu = m_neptune / (m_sun + m_neptune)
a_neptune = 4.49833729e9
R_neptune = 24622.0
R_neptune_nd = R_neptune / a_neptune
R_neptune_plot = 0.002

L1, L2, L3, L4, L5 = lagrange_points(mu)
C1 = jacobi_constant([L1[0], L1[1], 0, 0, 0, 0], mu)
C2 = jacobi_constant([L2[0], L2[1], 0, 0, 0, 0], mu)
Ccenter = (C1 + C2)/2
Cmin = 2.99
Cmax = 3.02
scale = 3


def slider_to_C(s):
    '''convert slider position to jacobi constant'''
    f = (10**(scale*abs(s)) - 1)/(10**scale - 1)
    if s < 0:
        return Ccenter - (Ccenter - Cmin)*f
    return Ccenter + (Cmax - Ccenter)*f


def C_to_slider(C):
    '''convert jacobi constant to slider position'''
    if C < Ccenter:
        f = (Ccenter - C)/(Ccenter - Cmin)
        return -np.log10(1 + f*(10**scale - 1))/scale
    f = (C - Ccenter)/(Cmax - Ccenter)
    return np.log10(1 + f*(10**scale - 1))/scale


# use separate grids, with extra samples near neptune in the overview
xmin, xmax = L1[0] - 0.03, L2[0] + 0.03
ymin, ymax = -0.08, 0.08
zmin, zmax = -0.08, 0.08
ngrid = 80


def axis_with_detail(low, high, count, extra):
    '''combine uniform samples with additional local detail'''
    return np.unique(np.concatenate((np.linspace(low, high, count), extra)))


def local_axis(low, high, center, count, extra):
    '''put more samples near neptune without increasing the overall grid size'''
    t = np.linspace(-1, 1, count)
    offsets = np.sinh(2*t)/np.sinh(2)
    values = center + np.where(t < 0, center - low, high - center)*offsets
    return np.unique(np.r_[values, extra])


local_axes = (
    local_axis(xmin, xmax, 1 - mu, ngrid, [L1[0], 1 - mu, L2[0]]),
    local_axis(ymin, ymax, 0, ngrid, [0]),
    local_axis(zmin, zmax, 0, ngrid, [0])
)
system_axes = (
    axis_with_detail(0, 1.5, 64, [L1[0], 1 - mu, 1, L2[0]]),
    np.linspace(0, 2*np.pi, 97),
    local_axis(-1, 1, 0, 40, [0])
)


def physical_points(points, cylindrical=False):
    '''convert overview radius-angle-height coordinates to rotating cartesian coordinates'''
    if not cylindrical:
        return points
    r, theta, z = points
    theta = np.where(theta == 2*np.pi, 0, theta)
    return r*np.cos(theta), r*np.sin(theta), z


def prepare_grid(axes, cylindrical=False):
    '''precompute the potential once, keeping precision near the critical values'''
    X, Y, Z = physical_points(np.meshgrid(*axes, indexing='ij', sparse=True), cylindrical)
    with np.errstate(divide='ignore', invalid='ignore'):
        field = zero_velocity(X, Y, Z, Ccenter, mu)
    # primary singularities lie inside the allowed region throughout the slider range
    field = np.nan_to_num(field, nan=1e6, posinf=1e6, neginf=-1e6)
    return axes, np.asarray(field, dtype=np.float32), cylindrical


# a cylindrical overview grid follows the nearly circular waist instead of cutting across it
grids = (prepare_grid(system_axes, cylindrical=True), prepare_grid(local_axes))

# lighter grids provide responsive feedback while the slider is moving
preview_local_axes = (
    local_axis(xmin, xmax, 1 - mu, 36, [L1[0], 1 - mu, L2[0]]),
    local_axis(ymin, ymax, 0, 36, [0]),
    local_axis(zmin, zmax, 0, 36, [0])
)
preview_system_axes = (
    axis_with_detail(0, 1.5, 36, [L1[0], 1 - mu, 1, L2[0]]),
    np.linspace(0, 2*np.pi, 49),
    local_axis(-1, 1, 0, 22, [0])
)
preview_grids = (prepare_grid(preview_system_axes, cylindrical=True),
                 prepare_grid(preview_local_axes))


def refine_vertices(vertices, axes, C, cylindrical=False):
    '''solve the true potential along crossed grid edges instead of linear interpolation'''
    points = np.array([np.interp(vertices[:, j], np.arange(len(axis)), axis)
                       for j, axis in enumerate(axes)])
    fractions = np.abs(vertices - np.rint(vertices))
    direction = fractions.argmax(axis=1)
    # rare interior vertices in ambiguous cells retain the marching-cubes position
    on_edge = (fractions > 1e-5).sum(axis=1) <= 1
    lower, upper = points.copy(), points.copy()
    for j, axis in enumerate(axes):
        mask = direction == j
        lower[j, mask] = axis[np.floor(vertices[mask, j]).astype(int)]
        upper[j, mask] = axis[np.ceil(vertices[mask, j]).astype(int)]
    with np.errstate(divide='ignore', invalid='ignore'):
        f_lower = zero_velocity(*physical_points(lower, cylindrical), C, mu)
        f_upper = zero_velocity(*physical_points(upper, cylindrical), C, mu)
    valid = on_edge & (np.signbit(f_lower) != np.signbit(f_upper))
    selected = np.flatnonzero(valid)
    if selected.size:
        lo, hi, flo = lower[:, selected], upper[:, selected], f_lower[selected]
        for _ in range(12):
            mid = (lo + hi)/2
            fmid = zero_velocity(*physical_points(mid, cylindrical), C, mu)
            same_side = np.signbit(fmid) == np.signbit(flo)
            lo[:, same_side] = mid[:, same_side]
            flo[same_side] = fmid[same_side]
            hi[:, ~same_side] = mid[:, ~same_side]
        # a final interpolation inside the tiny bracket improves precision cheaply
        fhi = zero_velocity(*physical_points(hi, cylindrical), C, mu)
        weight = np.clip(flo/(flo - fhi), 0, 1)
        points[:, selected] = lo + (hi - lo)*weight
    return [np.ascontiguousarray(p, dtype='<f4') for p in physical_points(points, cylindrical)]


def surface_mesh(grid, C):
    '''extract 2u = c without smoothing away physical necks or changing the level'''
    axes, field, cylindrical = grid
    level = C - Ccenter
    if not field.min() < level < field.max():
        return [np.empty(0, dtype='<f4') for _ in range(3)] + [
            np.empty(0, dtype='<u4') for _ in range(3)]
    vertices, faces, _, _ = marching_cubes(field, level=level, allow_degenerate=False)
    # retain connectivity but correct the edge intersections against the actual potential
    xyz = refine_vertices(vertices, axes, C, cylindrical)
    if cylindrical:
        # weld the periodic seam and polar axis so smooth normals have no artificial crease
        points, inverse = np.unique(np.array(xyz).T, axis=0, return_inverse=True)
        faces = inverse[faces]
        faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2])
                      & (faces[:, 0] != faces[:, 2])]
        xyz = [np.ascontiguousarray(points[:, j], dtype='<f4') for j in range(3)]
    if len(xyz[0]) > np.iinfo(np.uint16).max:
        raise ValueError('mesh has too many vertices for compact indices')
    ijk = [np.ascontiguousarray(faces[:, j], dtype='<u2') for j in range(3)]
    return xyz + ijk


def mesh_trace(mesh):
    '''display an already triangulated surface with smooth lighting'''
    return go.Mesh3d(
        **dict(zip(('x', 'y', 'z', 'i', 'j', 'k'), mesh)),
        color='#d8b4fe', opacity=0.35, flatshading=False,
        lighting=dict(ambient=0.72, diffuse=0.65, specular=0.08, roughness=0.9),
        lightposition=dict(x=2, y=1, z=3),
        name='zero-velocity surface', showlegend=False, hoverinfo='skip'
    )


def sphere(center, radius, nu=60, nv=30):
    '''generate a sphere mesh'''
    U, V = np.meshgrid(np.linspace(0, 2*np.pi, nu), np.linspace(0, np.pi, nv))
    return (center[0] + radius*np.cos(U)*np.sin(V),
            center[1] + radius*np.sin(U)*np.sin(V),
            center[2] + radius*np.cos(V), U, V)


fig = make_subplots(
    rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]],
    horizontal_spacing=0.035,
    subplot_titles=('sun-neptune system', 'neptune close-up')
)

# keep both changing traces first; all other traces remain static
for col, grid in enumerate(grids, start=1):
    fig.add_trace(mesh_trace(surface_mesh(grid, Ccenter)), row=1, col=col)

# overview markers use display sizes so both primaries remain visible
fig.add_trace(go.Scatter3d(
    x=[-mu, 1 - mu], y=[0, 0], z=[0, 0], mode='markers+text',
    text=['sun', 'neptune'], textposition='top center',
    marker=dict(size=[9, 5], color=['#ffd166', '#4f8cff']),
    textfont=dict(color='white', size=12), showlegend=False,
    hovertemplate='%{text}<br>x = %{x:.6f}<extra></extra>'
), row=1, col=1)

# a guide to neptune's orbital radius in the rotating coordinate system
theta = np.linspace(0, 2*np.pi, 240)
fig.add_trace(go.Scatter3d(
    x=(1 - mu)*np.cos(theta), y=(1 - mu)*np.sin(theta), z=np.zeros_like(theta),
    mode='lines', line=dict(color='rgba(160,160,160,0.4)', width=2, dash='dot'),
    showlegend=False, hoverinfo='skip'
), row=1, col=1)

# neptune in the detailed view
xN, yN, zN, UN, VN = sphere([1 - mu, 0, 0], R_neptune_plot)
fig.add_trace(go.Surface(
    x=xN, y=yN, z=zN,
    surfacecolor=0.55 + 0.25*np.sin(8*VN) + 0.08*np.cos(2*UN),
    colorscale=[[0, '#123b7a'], [0.25, '#2456c3'], [0.5, '#4f8cff'],
                [0.75, '#88d3ff'], [1, '#d6f4ff']],
    showscale=False, name='neptune', hoverinfo='skip',
    lighting=dict(ambient=0.55, diffuse=0.8, specular=0.35, roughness=0.6, fresnel=0.1),
    lightposition=dict(x=2, y=1, z=1)
), row=1, col=2)
fig.add_trace(go.Scatter3d(
    x=[1 - mu], y=[0], z=[0], mode='text', text=['neptune'],
    textposition='top center', textfont=dict(color='white', size=12), showlegend=False
), row=1, col=2)
fig.add_trace(go.Scatter3d(
    x=[L1[0], L2[0]], y=[0, 0], z=[0, 0], mode='markers+text',
    text=['L1', 'L2'], textposition='top center', textfont=dict(color='white', size=12),
    marker=dict(size=3, color='gold', symbol='diamond'),
    name='lagrange points', showlegend=False
), row=1, col=2)


def scene_layout(ranges):
    '''keep fixed axis ranges and independent camera controls'''
    spans = np.array([high - low for low, high in ranges])
    ratios = spans / spans.max()
    # fixed physical proportions also hold when the local surface disappears
    scene = dict(aspectmode='manual', aspectratio=dict(zip(('x', 'y', 'z'), ratios)),
                 uirevision='keep',
                 camera=dict(eye=dict(x=0, y=-2.2, z=0),
                             up=dict(x=0, y=0, z=1)))
    for name, limits in zip(('x', 'y', 'z'), ranges):
        scene[name + 'axis'] = dict(
            title=name, range=limits, color='rgb(160, 160, 160)',
            backgroundcolor='black', gridcolor='rgb(30, 30, 30)',
            zerolinecolor='rgb(35, 35, 35)', linecolor='rgb(35, 35, 35)',
            showbackground=False, showgrid=False, showline=False,
            zeroline=False, showspikes=False, ticks=''
        )
    return scene


fig.update_layout(
    title=dict(text=f'sun-neptune zero-velocity surfaces, C = {Ccenter:.8f}', x=0.5),
    paper_bgcolor='black', plot_bgcolor='black', font=dict(color='white'),
    uirevision='keep',
    scene=scene_layout([[-1.5, 1.5], [-1.5, 1.5], [-1, 1]]),
    scene2=scene_layout([[xmin, xmax], [ymin, ymax], [zmin, zmax]]),
    margin=dict(l=0, r=0, t=70, b=0), autosize=True
)

app = Dash(__name__)
server = app.server


@lru_cache(maxsize=48)
def mesh_payload(s, quality='full'):
    '''cache recent surfaces and send compact compressed binary arrays'''
    C = slider_to_C(s)
    selected_grids = preview_grids if quality == 'preview' else grids
    meshes = [surface_mesh(grid, C) for grid in selected_grids]
    # header: c, vertex and face counts for each of the two meshes
    header = struct.pack('<dIIII', C, len(meshes[0][0]), len(meshes[0][3]),
                         len(meshes[1][0]), len(meshes[1][3]))
    body = header + b''.join(a.tobytes() for mesh in meshes for a in mesh)
    return gzip.compress(body, compresslevel=1)


@server.route('/zvs-mesh')
def get_mesh():
    '''return both surfaces at precisely the requested slider position'''
    try:
        s = float(request.args['s'])
        if not np.isfinite(s) or not -1 <= s <= 1:
            raise ValueError
    except (KeyError, ValueError, TypeError):
        return Response('invalid slider position', status=400)
    quality = request.args.get('q', 'full')
    if quality not in ('preview', 'full'):
        return Response('invalid mesh quality', status=400)
    payload = mesh_payload(s, quality)
    headers = {'Cache-Control': 'no-store', 'Vary': 'Accept-Encoding'}
    if 'gzip' in request.headers.get('Accept-Encoding', ''):
        headers['Content-Encoding'] = 'gzip'
    else:
        payload = gzip.decompress(payload)
    return Response(payload, mimetype='application/octet-stream', headers=headers)


app.layout = html.Div([
    dcc.Store(id='camera-state', data={}),
    html.H3('sun-neptune zero-velocity surfaces'),
    dcc.Graph(id='zvc-plot', figure=fig, style={'height': '80vh', 'width': '100%'},
              config={'responsive': True, 'displaylogo': False}),
    html.Div([
        html.Label('jacobi constant'),
        html.Div(id='C-value', children=f'C = {Ccenter:.8f}',
                 style={'textAlign': 'center', 'marginBottom': '5px'}),
        dcc.Slider(
            id='C-slider', min=-1, max=1, step=0.001, value=0, updatemode='drag',
            marks={-1: f'{Cmin:.5f}', C_to_slider(C2): 'C2',
                   C_to_slider(C1): 'C1', 1: f'{Cmax:.5f}'}
        ),
        html.Div(id='render-status', children='both views ready',
                 style={'textAlign': 'center', 'fontSize': '12px', 'marginTop': '8px'}),
        html.Div('logarithmic jacobi scale; rotate and zoom each view independently; primary sizes are exaggerated for visibility',
                 style={'textAlign': 'center', 'fontSize': '12px', 'marginTop': '6px'})
    ], style={'padding': '0 40px 20px 40px'})
])


# remember the actual user-selected cameras, including moves made before the first update
app.clientside_callback(
    '''
    function(event) {
        if (!event) return dash_clientside.no_update;
        const cameras = window.neptuneZvsCameras || (window.neptuneZvsCameras = {});
        for (const scene of ['scene', 'scene2']) {
            if (event[scene + '.camera']) cameras[scene] = event[scene + '.camera'];
        }
        return Object.assign({}, cameras);
    }
    ''',
    Output('camera-state', 'data'), Input('zvc-plot', 'relayoutData'),
    prevent_initial_call=True
)


# update the existing plot directly so dash never replaces its camera state
# only one request/render runs at a time; newer slider positions replace pending work
app.clientside_callback(
    '''
    function(s) {
        const state = window.neptuneZvs || (window.neptuneZvs = {
            latest: 0, serial: 0, busy: false, cache: new Map(),
            requested: null, settleTimer: null
        });
        state.latest = s;
        state.serial += 1;
        const serial = state.serial;
        state.requested = {serial: serial, value: s, quality: 'preview'};
        clearTimeout(state.settleTimer);
        state.settleTimer = setTimeout(function() {
            if (serial !== state.serial) return;
            state.requested = {serial: serial, value: state.latest, quality: 'full'};
            drawLatest();
        }, 250);

        function status(text) {
            dash_clientside.set_props('render-status', {children: text});
        }

        function decode(buffer) {
            const header = new DataView(buffer);
            const C = header.getFloat64(0, true);
            const meshes = [];
            let offset = 24;
            for (let m = 0; m < 2; m++) {
                const nv = header.getUint32(8 + m*8, true);
                const nf = header.getUint32(12 + m*8, true);
                const mesh = {};
                for (const key of ['x', 'y', 'z', 'i', 'j', 'k']) {
                    const position = ['x', 'y', 'z'].includes(key);
                    const count = position ? nv : nf;
                    mesh[key] = position ? new Float32Array(buffer, offset, count)
                                         : new Uint16Array(buffer, offset, count);
                    offset += count*(position ? 4 : 2);
                }
                meshes.push(mesh);
            }
            return {C: C, meshes: meshes};
        }

        async function drawLatest() {
            if (state.busy) return;
            state.busy = true;
            try {
                while (state.requested) {
                    const target = state.requested;
                    state.requested = null;
                    const start = performance.now();
                    status(target.quality === 'preview' ? 'updating preview...' : 'refining both views...');
                    const cacheKey = target.quality + ':' + target.value;
                    let frame = state.cache.get(cacheKey);
                    if (!frame) {
                        const response = await fetch('zvs-mesh?s=' + encodeURIComponent(target.value)
                            + '&q=' + target.quality);
                        if (!response.ok) throw new Error('surface request failed');
                        frame = decode(await response.arrayBuffer());
                        state.cache.set(cacheKey, frame);
                        if (state.cache.size > 18) state.cache.delete(state.cache.keys().next().value);
                    }
                    if (target.serial !== state.serial) continue;
                    const plot = document.querySelector('#zvc-plot .js-plotly-plot');
                    if (!plot || !window.Plotly) throw new Error('plot is not ready');
                    const update = {};
                    for (const key of ['x', 'y', 'z', 'i', 'j', 'k']) {
                        update[key] = frame.meshes.map(mesh => mesh[key]);
                    }
                    const layout = {
                        'title.text': 'sun-neptune zero-velocity surfaces, C = ' + frame.C.toFixed(8)
                    };
                    for (const [scene, camera] of Object.entries(window.neptuneZvsCameras || {})) {
                        layout[scene + '.camera'] = camera;
                    }
                    await Plotly.update(plot, update, layout, [0, 1]);
                    // keep dash's figure in sync so a camera event cannot restore old data
                    dash_clientside.set_props('zvc-plot', {
                        figure: {data: plot.data, layout: plot.layout}
                    });
                    if (target.serial === state.serial) {
                        dash_clientside.set_props('C-value', {children: 'C = ' + frame.C.toFixed(8)});
                        const label = target.quality === 'preview' ? 'preview' : 'full quality';
                        status(label + ' updated in ' + Math.round(performance.now() - start) + ' ms');
                    }
                }
            } catch (error) {
                status('could not update: ' + error.message + '; move the slider to retry');
            } finally {
                state.busy = false;
                if (state.requested) drawLatest();
            }
        }

        drawLatest();
        return dash_clientside.no_update;
    }
    ''',
    Output('C-value', 'children'), Input('C-slider', 'value'), prevent_initial_call=True
)


if __name__ == '__main__':
    app.run(debug=True)
