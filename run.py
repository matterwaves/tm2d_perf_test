import tm2d
import numpy as np

import sys

import vkdispatch as vd
import tm2d_utils as tu

vd.make_context(multi_device=True, multi_queue=True)

atomic_coords = tu.load_coords_from_pdb("PDBs/6z6u_apoferritin.pdb")

print("Running TM2D with the following parameters:")

print("Sys args: {}".format(sys.argv))

print("Atom count: {}".format(len(atomic_coords)))

D_protein = tu.optics_functions.get_protein_radius(atomic_coords) * 2
ang_step = tu.crowther_ang_step_from_resolution(3, D_protein)

print(f"Protein diameter: {D_protein:.2f} A")
print(f"Crowther angular step: {ang_step:.2f} degrees")

defocus_count = int(sys.argv[6])

pixel_sizes = np.array([1.0])
defoci = np.arange(12000, 12000 + 25 * defocus_count, 25)
rotations = tu.get_orientations_healpix(
    angular_step_size=ang_step,
    psi_step_size=ang_step,
    region=tu.OrientationRegion(symmetry=sys.argv[5])
)

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
    (int(sys.argv[4]), micrograph_side_len, micrograph_side_len),
    dtype=np.float32
)

size_len = int(sys.argv[2])

if template_type == "atomic":
    results = tu.run_tm2d_atomic_pixels(
        micrographs=micrographs,
        param_set=params,
        template_box_size=(size_len, size_len),
        atomic_coords=atomic_coords,
        template_batch_size=min(4, defocus_count),
        enable_progress_bar=False
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
        template_batch_size=min(4, defocus_count),
        enable_progress_bar=False
    )
else:
    raise ValueError(f"Unknown template type: {template_type}")
