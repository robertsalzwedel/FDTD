import numpy as np
import scipy.constants as constants
from collections import namedtuple
from modules.fundamentals import *

"Adjustable parameters"
ddx = 10 * nm  # args.v[0]*nm           # spatial step size
dt = ddx / c * Sc  # time step, from dx fulfilling stability criteria
radius = 150 * nm  # radius of sphere
tfsf_dist = 12  # args.v[4]           # TFSF distance from computational boundary
npml = 8  # number of PML layers
dim = 50  # args.v[1]                 # number of spatial steps

"Time parameters"
tsteps = 5000  # args.v[2]             # number of time steps
cycle = 10  # selection of visualized frames
time_pause = 0.1  # pause time for individual frames, limited by computation time
time_micro_factor = 100

# DFT parameters
iw_dim = 100
e_min = 1.9
e_max = 3.2


FLAG = FLAGS(
    # boundary
    PML=1,  # PML:                      0: off,         1: on
    # material
    OBJECT=1,  # object choice             0: no object    1: sphere               2: rectangle
    MATERIAL=1,  # choice of material        1: Drude        2: DrudeLorentz         3: Etchegoin Model
    # source
    TFSF=1,  # Plane wave                0: off,         1: TFSF on,             2: Periodic Boundary condition
    POINT=0,  # Point source              0: off,         1: on
    # monitors
    CROSS=0,  # x sections (scat, abs)    0: off,         1: on
    DFT3D=0,  # 3D DFT monitor:           0: off,         1: on
    DFT2D=0,  # 2D DFT monitor            0: off,         1: on
    FFT=0,  # FFT point monitor         0: off,         1: on
    ANIMATION=1,  # Animation                 0: off,         1: on
    # micro
    MICRO=0,  # Microscopic code          0: disabled,    1: enabled          # should be merged with the DRUDE/EPS FLAG at some point
    ELPHO=0,  # 0: call matrices from memory, 1: calculate matrices, 2: without electron phonon interaction, 3: with interaction, does not save the matrices
    LM=3,  # 0: call matrices from memory, 1: calculate matrices, 2: without light matter interaction, 3: with interaction, does not save the matrices
    STENCIL=3,  # 3: Three-point stencil, 5: Five-point stencil, 7: Seven-point stencil
    E_FIELD=0,  # 0: Use of the field calculated and updated by Roberts FDTD Maxwell solver; 1: Field is decoupled from electron dynamics (e.g. Gauß-shaped pulse)
    INIT=6,  # 5,6: IC for delta f: 5: T_electron=T_phonon, 6: T_electron= args temperature, 7: Fermi distributed dumbbell-shaped, 8: Gauss distributed dumbbell-shaped
)
