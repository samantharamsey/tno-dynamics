'''build the browser-resident moving-preview mesh bank'''

import gzip
import struct
from pathlib import Path

import numpy as np

import neptune_zvs as app


output = bytearray(struct.pack('<4sII', b'ZVSB', 1, 161))
for slider in np.linspace(-1, 1, 161):
    C = app.slider_to_C(float(slider))
    meshes = [app.surface_mesh(grid, C) for grid in app.preview_grids]
    output.extend(struct.pack('<fdIIII', slider, C,
                              len(meshes[0][0]), len(meshes[0][3]),
                              len(meshes[1][0]), len(meshes[1][3])))
    for mesh in meshes:
        for array in mesh:
            output.extend(array.tobytes())
        while len(output) % 4:
            output.append(0)
    while len(output) % 4:
        output.append(0)

asset = Path(__file__).resolve().parent / 'assets' / 'preview_frames.dat'
asset.parent.mkdir(exist_ok=True)
asset.write_bytes(gzip.compress(output, compresslevel=9))
print(f'wrote {asset} ({asset.stat().st_size / 1024 / 1024:.2f} MB)')
