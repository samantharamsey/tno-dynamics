import numpy as np
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, State

import sys
sys.path.append('src')

from cr3bp import lagrange_points, jacobi_constant, zero_velocity


# sun-neptune system
m_sun = 1.9890e30
m_neptune = 1.0241e26
mu = m_neptune / (m_sun + m_neptune)

# neptune radius
a_neptune = 4.49833729e9      # km
R_neptune = 24622.0           # km

# normalize neptune radius to the distance between sun and neptune
R_neptune_nd = R_neptune / a_neptune

# use the true radius if you want strict physical scale
# use a larger display radius if you want it to actually be visible
R_neptune_plot = 0.002
#R_neptune_plot = R_neptune_nd

def sphere(center, radius, nu=60, nv=30):
    '''generate a sphere mesh'''
    u = np.linspace(0, 2*np.pi, nu)
    v = np.linspace(0, np.pi, nv)
    U, V = np.meshgrid(u, v)
    x = center[0] + radius*np.cos(U)*np.sin(V)
    y = center[1] + radius*np.sin(U)*np.sin(V)
    z = center[2] + radius*np.cos(V)
    return x, y, z, U, V


# lagrange points
L1, L2, L3, L4, L5 = lagrange_points(mu)
# jacobi constants at l1 and l2
C1 = jacobi_constant([L1[0], L1[1], 0, 0, 0, 0], mu)
C2 = jacobi_constant([L2[0], L2[1], 0, 0, 0, 0], mu)

# jacobi constant slider range
Ccenter = (C1 + C2)/2

# jacobi constant range for slider
Cmin = 2.99
Cmax = 3.02
scale = 3

def slider_to_C(s):
    '''convert slider position to jacobi constant'''
    if s < 0:
        f = (10**(-scale*s) - 1)/(10**scale - 1)
        C = Ccenter - (Ccenter - Cmin)*f
    else:
        f = (10**(scale*s) - 1)/(10**scale - 1)
        C = Ccenter + (Cmax - Ccenter)*f
    return C

def C_to_slider(C):
    '''convert jacobi constant to slider position'''
    if C < Ccenter:
        f = (Ccenter - C)/(Ccenter - Cmin)
        s = -np.log10(1 + f*(10**scale - 1))/scale
    else:
        f = (C - Ccenter)/(Cmax - Ccenter)
        s = np.log10(1 + f*(10**scale - 1))/scale
    return s

C1_slider = C_to_slider(C1)
C2_slider = C_to_slider(C2)


# region around neptune and l1/l2
xmin = L1[0] - 0.03
xmax = L2[0] + 0.03

ymin = -0.08
ymax =  0.08

zmin = -0.08
zmax =  0.08


# grid
x = np.linspace(xmin, xmax, 70)
y = np.linspace(ymin, ymax, 70)
z = np.linspace(zmin, zmax, 70)

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')


app = Dash(__name__)


app.layout = html.Div([

    html.H3('sun-neptune zero-velocity surface'),

    dcc.Graph(
        id='zvc-plot',
        style={
            'height': '85vh',
            'width': '100%'},
        config={
            'responsive': True
        }),

    html.Div([

        html.Label('jacobi constant'),

        html.Div(
            id='C-value',
            style={
                'textAlign': 'center',
                'marginBottom': '5px'
            }
        ),

        dcc.Slider(
            id='C-slider',
            min=-1,
            max=1,
            step=0.001,
            value=0,

            marks={
                -1: f'{Cmin:.5f}',
                C2_slider: 'C2',
                0: 'center',
                C1_slider: 'C1',
                1: f'{Cmax:.5f}'
            }
        )

    ],
    style={
        'padding': '0 40px 20px 40px'
    })])


@app.callback(
    Output('zvc-plot', 'figure'),
    Output('C-value', 'children'),
    Input('C-slider', 'value'),
    State('zvc-plot', 'relayoutData')
)
def update_surface(s, relayout_data):

    # convert slider position to jacobi constant
    C = slider_to_C(s)

    # preserve the current camera position
    camera = None

    if relayout_data is not None:
        if 'scene.camera' in relayout_data:
            camera = relayout_data['scene.camera']

    F = zero_velocity(X, Y, Z, C, mu)

    fig = go.Figure()

    fig.add_trace(go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=F.flatten(),

        isomin=-1e-3,
        isomax=1e-3,
        surface_count=1,

        caps=dict(
            x_show=False,
            y_show=False,
            z_show=False
        ),

        showscale=False,
        opacity=0.35,

        colorscale=[
            [0.0, '#f3e8ff'],
            [0.5, '#d8b4fe'],
            [1.0, '#c084fc']
        ],

        name=f'C = {C:.6f}'
    ))

    # neptune
    xN, yN, zN, UN, VN = sphere([1 - mu, 0, 0], R_neptune_plot)

    # simple banded surface coloring to make neptune look nicer
    surfacecolor = 0.55 + 0.25*np.sin(8*VN) + 0.08*np.cos(2*UN)

    fig.add_trace(go.Surface(
        x=xN,
        y=yN,
        z=zN,
        surfacecolor=surfacecolor,

        colorscale=[
            [0.0, '#123b7a'],
            [0.25, '#2456c3'],
            [0.5, '#4f8cff'],
            [0.75, '#88d3ff'],
            [1.0, '#d6f4ff']
        ],

        showscale=False,
        name='neptune',
        hoverinfo='skip',

        lighting=dict(
            ambient=0.55,
            diffuse=0.8,
            specular=0.35,
            roughness=0.6,
            fresnel=0.1
        ),

        lightposition=dict(
            x=2,
            y=1,
            z=1
        )
    ))

    fig.add_trace(go.Scatter3d(
        x=[1 - mu],
        y=[0],
        z=[0],
        mode='text',
        text=['neptune'],
        textposition='top center',
        textfont=dict(
            color='white',
            size=12
        ),
        showlegend=False
    ))

    # l1 and l2 markers
    fig.add_trace(go.Scatter3d(
        x=[L1[0], L2[0]],
        y=[0, 0],
        z=[0, 0],
        mode='markers',
        marker=dict(
            size=3,
            color='gold',
            symbol='diamond'
        ),
        name='lagrange points'
    ))

    # l1 and l2 labels
    fig.add_trace(go.Scatter3d(
        x=[L1[0], L2[0]],
        y=[0, 0],
        z=[0, 0],
        mode='text',
        text=['L1', 'L2'],
        textposition='top center',
        textfont=dict(
            color='white',
            size=12
        ),
        showlegend=False
    ))

    fig.update_layout(
        title=f'zero-velocity surface near neptune, C = {C:.6f}',

        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white'),

        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        ),

        uirevision='keep',
        scene_uirevision='keep',

        scene=dict(
            xaxis=dict(
                title='x',
                range=[xmin, xmax],
                color='white',
                backgroundcolor='black',
                gridcolor='rgb(40, 40, 40)',
                zerolinecolor='rgb(70, 70, 70)',
                linecolor='rgb(60, 60, 60)',
                showbackground=True,
                showgrid=False,
                zeroline=True
            ),

            yaxis=dict(
                title='y',
                range=[ymin, ymax],
                color='white',
                backgroundcolor='black',
                gridcolor='rgb(40, 40, 40)',
                zerolinecolor='rgb(70, 70, 70)',
                linecolor='rgb(60, 60, 60)',
                showbackground=True,
                showgrid=False,
                zeroline=True
            ),

            zaxis=dict(
                title='z',
                range=[zmin, zmax],
                color='white',
                backgroundcolor='black',
                gridcolor='rgb(40, 40, 40)',
                zerolinecolor='rgb(70, 70, 70)',
                linecolor='rgb(60, 60, 60)',
                showbackground=True,
                showgrid=False,
                zeroline=True
            ),

            aspectmode='data',
            camera=camera
        ),

        margin=dict(
            l=0,
            r=0,
            t=50,
            b=0
        ),

        autosize=True
    )

    return fig, f'C = {C:.8f}'


if __name__ == '__main__':
    app.run(debug=True)