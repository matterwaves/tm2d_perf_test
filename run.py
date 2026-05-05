import tm2d
import numpy as np

import sys

import tm2d_utils as tu

pixel_sizes = np.arange(1.04, 1.08, 0.0004)
B_factors = np.arange(0, 250, 2.5)

params = tm2d.make_param_set(
    tm2d.make_ctf_set(
        tu.ctf_like_krios(
            defocus = 12890,
            B = None,
            Cs = 2.7e7
        ),
        B = B_factors
    ),
    rotations=np.array([[188.84183,  78.82107, 326]]),
    #rotations_weights=np.array([1.0]),
    pixel_sizes=pixel_sizes,
)

template_type = sys.argv[1]

micrograph_side_len = 512

micrographs= np.ones(
    (1, micrograph_side_len, micrograph_side_len),
    dtype=np.complex64
)

size_len = 576

if template_type == "atomic":
    results = tu.run_tm2d_atomic_pixels(
        micrographs=micrographs,
        param_set=params,
        template_box_size=(size_len, size_len),
        atomic_coords=tu.load_coords_from_npz("data/atoms.npz"),
        enable_progress_bar=True
    )
elif template_type == "density":
    density = tu.file_loading.DensityData(
        density=np.ones((size_len, size_len, size_len), dtype=np.complex64),
        pixel_size=1.06
    )

    results = tu.run_tm2d_density_pixels(
        micrographs=micrographs,
        param_set=params,
        density=density,
        enable_progress_bar=True
    )
else:
    raise ValueError(f"Unknown template type: {template_type}")
