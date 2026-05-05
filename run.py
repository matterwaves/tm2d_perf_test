import tm2d
import numpy as np

import sys

import vkdispatch as vd
import tm2d_utils as tu

vd.make_context(multi_device=True, multi_queue=True)

pixel_sizes = np.arange(1.04, 1.09, 0.02)
defoci = np.arange(12000, 12800, 25)
rotations = tu.get_orientations_healpix(
    angular_step_size=2,
    psi_step_size=2,
    region=tu.OrientationRegion(symmetry=sys.argv[4])
)

print(f"Pixel size count: {len(pixel_sizes)}")
print(f"Defocus count: {len(defoci)}")
print(f"Rotation count: {len(rotations)}")

params = tm2d.make_param_set(
    tm2d.make_ctf_set(
        tu.ctf_like_krios(
            defocus = None
        ),
        defocus = defoci
    ),
    rotations=rotations,
    pixel_sizes=pixel_sizes,
)

template_type = sys.argv[1]

micrograph_side_len = int(sys.argv[3])

micrographs= np.ones(
    (1, micrograph_side_len, micrograph_side_len),
    dtype=np.complex64
)

size_len = int(sys.argv[2])

if template_type == "atomic":
    results = tu.run_tm2d_atomic_pixels(
        micrographs=micrographs,
        param_set=params,
        template_box_size=(size_len, size_len),
        atomic_coords=tu.load_coords_from_npz("data/atoms.npz"),
        #template_batch_size=1,
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
        #template_batch_size=1,
        enable_progress_bar=True
    )
else:
    raise ValueError(f"Unknown template type: {template_type}")
