"""Backend-agnostic array helpers (GPU-ready).

UnPaSt's numeric hot paths are written so the same vectorized code can run on
either CPU (numpy) or GPU (cupy). When cupy is importable, ``xp(arr)`` returns
the array module matching ``arr`` (cupy for device arrays, numpy for host
arrays); otherwise it always returns numpy. On a CPU-only box cupy is absent and
everything transparently falls back to numpy.

Typical use::

    from unpast.utils.backend import xp, to_numpy
    mod = xp(arr)            # cupy or numpy
    result = mod.dot(arr, arr.T)
    result = to_numpy(result)  # hand back to CPU code (pandas, sknetwork, ...)
"""

import numpy as np

# Use cupy when available, fall back to numpy otherwise.
# On a CPU-only box `_cp` stays None and `xp(...)` always returns numpy.
try:
    import cupy as _cp
except Exception:
    _cp = None


def get_cupy():
    """Return the imported cupy module, or ``None`` if unavailable."""
    return _cp


def xp(arr):
    """Return the array module (cupy or numpy) appropriate for ``arr``.

    Lets the same vectorized code run on GPU (cupy arrays) or CPU (numpy
    arrays) without branching::

        mod = xp(arr); mod.sort(arr)
    """
    return _cp.get_array_module(arr) if _cp is not None else np


def to_numpy(arr):
    """Return a numpy array for ``arr``, copying off the GPU if needed.

    Safe to call on numpy arrays (returned as-is via ``np.asarray``). Use this
    at the boundary where downstream CPU-only code (pandas, scikit-network)
    consumes a result that may live on the GPU.
    """
    if _cp is not None and isinstance(arr, _cp.ndarray):
        return _cp.asnumpy(arr)
    return np.asarray(arr)
