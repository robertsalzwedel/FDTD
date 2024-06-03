import numpy as np
import scipy.constants as constants
from collections import namedtuple

Sc = 0.5  # Courant number

"Fundamental units"
nm = constants.nano
c = constants.c
tera = constants.tera
hbar = constants.hbar
eC = constants.e
h_planck = constants.h

# macroscopic dimensions (real space)
Dimensions = namedtuple(
    "Dimensions",
    (
        "x",
        "y",
        "z",
    ),
)

# microscopic dimensions (k space)
Dimensions_k = namedtuple(
    "Dimensions_k",
    (
        "k",
        "phi",
        "theta",
    ),
)

# 3d box definition for various limits
box = namedtuple(
    "box",
    (
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "z_min",
        "z_max",
    ),
)


def define_boxes_arrays(tfsf_dist, dims, ddx, radius):
    # TFSF boundary conditions
    tfsf = box(
        x_min=tfsf_dist,
        x_max=dims.x - tfsf_dist - 1,
        y_min=tfsf_dist,
        y_max=dims.y - tfsf_dist - 1,
        z_min=tfsf_dist,
        z_max=dims.z - tfsf_dist - 1,
    )

    "Cross sectionparameters"
    # Scattering box
    scat = box(
        x_min=tfsf.x_min - 3,
        x_max=tfsf.x_max + 2,
        y_min=tfsf.y_min - 3,
        y_max=tfsf.y_max + 2,
        z_min=tfsf.z_min - 3,
        z_max=tfsf.z_max + 2,
    )
    # Absorption box
    abs = box(
        x_min=tfsf.x_min + 2,
        x_max=tfsf.x_max - 3,
        y_min=tfsf.y_min + 2,
        y_max=tfsf.y_max - 3,
        z_min=tfsf.z_min + 2,
        z_max=tfsf.z_max - 3,
    )

    "Spatial domain"
    # length scales
    length = Dimensions(
        x=ddx * dims.x,  # (args.v[0]*dim)*nm,
        y=ddx * dims.y,  # (args.v[0]*dim)*nm,
        z=ddx * dims.z,  # (args.v[0]*dim)*nm
    )

    array = Dimensions(
        x=np.arange(0, length.x - nm, ddx),
        y=np.arange(0, length.y - nm, ddx),
        z=np.arange(0, length.z - nm, ddx),
    )

    "Object parameters"
    # Sphere parameters
    sphere = Sphere(
        R=radius,  # args.v[3]*nm                 #radius of sphere
        x=length.x / 2,  # center in x direction
        y=length.y / 2,  # center in y direction
        z=length.z / 2 + 0.5 * ddx,  # center in z direction
    )

    offset = int((sphere.x - sphere.R) / ddx) - 1
    diameter = int(2 * sphere.R / ddx) + 3

    return tfsf, scat, abs, length, array, sphere, offset, diameter


# sphere
Sphere = namedtuple("Sphere", ("R", "x", "y", "z"))


# 3d box definition for various limits
FLAGS = namedtuple(
    "flags",
    (
        "PML",  # PML:                      0: off,         1: on
        "OBJECT",  # object choice             0: no object    1: sphere               2: rectangle
        "MATERIAL",  # choice of material        1: Drude        2: DrudeLorentz         3: Etchegoin Model
        "TFSF",  # Plane wave                0: off,         1: TFSF on,             2: Periodic Boundary condition
        "POINT",  # Point source              0: off,         1: on
        "CROSS",  # x sections (scat, abs)    0: off,         1: on
        "DFT3D",  # 3D DFT monitor:           0: off,         1: on
        "DFT2D",  # 2D DFT monitor            0: off,         1: on
        "FFT",  # FFT point monitor         0: off,         1: on
        "ANIMATION",  # Animation                 0: off,         1: on
        "MICRO",  # Microscopic code          0: disabled,    1: enabled          # should be merged with the DRUDE/EPS FLAG at some point
        "ELPHO",  # 0: call matrices from memory, 1: calculate matrices, 2: without electron phonon interaction, 3: with interaction, does not save the matrices
        "LM",  # 0: call matrices from memory, 1: calculate matrices, 2: without light matter interaction, 3: with interaction, does not save the matrices
        "STENCIL",  # 3: Three-point stencil, 5: Five-point stencil, 7: Seven-point stencil
        "E_FIELD",  # 0: Use of the field calculated and updated by Roberts FDTD Maxwell solver; 1: Field is decoupled from electron dynamics (e.g. Gauß-shaped pulse)
        "INIT",  # 5,6: IC for delta f: 5: T_electron=T_phonon, 6: T_electron= args temperature, 7: Fermi distributed dumbbell-shaped, 8: Gauss distributed dumbbell-shaped
    ),
)
