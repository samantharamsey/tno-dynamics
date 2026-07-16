"""Event functions for CR3BP trajectory propagation."""

import numpy as np
from numpy.typing import NDArray


def y_plane_crossing_downward(t: float, state: NDArray[np.float64], mu: float) -> float:
    """Detect a downward crossing of the rotating-frame y = 0 plane."""
    del t, mu
    return float(state[1])


y_plane_crossing_downward.terminal = True
y_plane_crossing_downward.direction = -1
