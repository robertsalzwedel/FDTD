import numpy as np
from numba.experimental import jitclass

# from input import *
from modules.parameters import *
from modules.classes import Field

# define fields
e = Field(dims, 0)  # electric field E
e1 = Field(dims, 0)  # electric field E memory
h = Field(dims, 0)  # magnetic field strength H
d = Field(dims, 0)  # electric displacement field D
p = Field(dims, 0)  # polarization field (macroscopic)
p_drude = Field(dims, 0)  # polarization field (macroscopic)
p_lorentz = Field(dims, 0)  # polarization field (macroscopic)
ga = Field(dims, 1)  # inverse permittivity =1/eps
id = Field(dims, 0)  # accumulated curls for PML calculation
ih = Field(dims, 0)  # accumulated curls for PML calculation


# define fields for specific material
if FLAG.OBJECT != 0:
    # Drude - this uses a lot of memory can this be more efficient?
    p_tmp_drude = Field(dims, 0)  # polarization help field
    d1 = Field(dims, 0)  # prefactor in auxilliary equation
    d2 = Field(dims, 0)  # prefactor in auxilliary equation
    d3 = Field(dims, 0)  # prefactor in auxilliary equation#

    if FLAG.MATERIAL == 2:
        # Lorentz
        p_tmp_lorentz = Field(dims, 0)  # polarization help field
        l1 = Field(dims, 0)  # prefactor in auxilliary equation
        l2 = Field(dims, 0)  # prefactor in auxilliary equation
        l3 = Field(dims, 0)  # prefactor in auxilliary equation

    if FLAG.MATERIAL == 3:
        p_et1 = Field(dims, 0)  # polarization at timestep before
        p_tmp_et1 = Field(dims, 0)  # polarization help field
        p_et2 = Field(dims, 0)  # polarization at timestep before
        p_tmp_et2 = Field(dims, 0)  # polarization help field
        f1_et1 = Field(dims, 0)  # prefactor in auxilliary equation
        f2_et1 = Field(dims, 0)  # prefactor in auxilliary equation
        f3_et1 = Field(dims, 0)  # prefactor in auxilliary equation
        f4_et1 = Field(dims, 0)  # prefactor in auxilliary equation
        f1_et2 = Field(dims, 0)  # prefactor in auxilliary equation
        f2_et2 = Field(dims, 0)  # prefactor in auxilliary equation
        f3_et2 = Field(dims, 0)  # prefactor in auxilliary equation
        f4_et2 = Field(dims, 0)  # prefactor in auxilliary equation


# fields for interaction with microscopic code
if FLAG.MICRO == 1:
    j = Field(dims, 0)  # current density for microscopic code
    j_tmp = Field(dims, 0)  # current density temporary (helper

####################################################
# Sources
####################################################

# 1D plane wave buffer for TFSF simulation
if FLAG.TFSF == 1 or FLAG.TFSF == 2:
    ez_inc = np.zeros(dims.y, float)  # 1d buffer field for plane wave
    hx_inc = np.zeros(dims.y, float)  # 1d buffer field for plane wave
    boundary_low = [0, 0]  # lower absorbing boundary condition
    boundary_high = [0, 0]  # upper absorbing boundary condition
    pulse_t = np.zeros(tsteps, float)  # pulse monitor for FFT

# Point dipole source
if FLAG.POINT == 1:
    source = Dimensions(
        x=int(dims.x / 2), y=int(dims.y / 2), z=int(dims.z / 2)
    )  # point source position
    pulsemon_t = np.zeros(tsteps, float)  # pulse monitor (source)
    ez_source_t = np.zeros(tsteps, float)  # electric field monitor
