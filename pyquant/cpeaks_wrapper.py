import os

import numpy as np

if os.environ.get("PYQUANT_DEV", False) == "True":
    try:
        import pyximport

        pyximport.install(
            setup_args={"include_dirs": np.get_include()}, reload_support=True
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        pass

from .cpeaks import (  # noqa: F401
    bigauss_func,
    gauss_func,
    bigauss_ndim,
    gauss_ndim,
    bigauss_jac,
    gauss_jac,
    find_nearest,
    find_nearest_index,
    find_nearest_indices,
    get_ppm,
)
