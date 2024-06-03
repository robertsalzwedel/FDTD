import numpy as np
import scipy.constants as constants
from collections import namedtuple
from modules.fundamentals import *
from input import *

"computational domain specification"
# real space
dims = Dimensions(x=dim, y=dim, z=dim)

# momentum space
# dims_k = Dimensions_k(k=grid.n_kmax, phi=grid.n_phimax, theta=grid.n_thetamax)


# # TFSF boundary conditions
# tfsf = box(
#     x_min=tfsf_dist,
#     x_max=dims.x - tfsf_dist - 1,
#     y_min=tfsf_dist,
#     y_max=dims.y - tfsf_dist - 1,
#     z_min=tfsf_dist,
#     z_max=dims.z - tfsf_dist - 1,
# )

# "Cross sectionparameters"
# # Scattering box
# scat = box(
#     x_min=tfsf.x_min - 3,
#     x_max=tfsf.x_max + 2,
#     y_min=tfsf.y_min - 3,
#     y_max=tfsf.y_max + 2,
#     z_min=tfsf.z_min - 3,
#     z_max=tfsf.z_max + 2,
# )
# # Absorption box
# abs = box(
#     x_min=tfsf.x_min + 2,
#     x_max=tfsf.x_max - 3,
#     y_min=tfsf.y_min + 2,
#     y_max=tfsf.y_max - 3,
#     z_min=tfsf.z_min + 2,
#     z_max=tfsf.z_max - 3,
# )

# "Spatial domain"
# # length scales
# length = Dimensions(
#     x=ddx * dim,  # (args.v[0]*dim)*nm,
#     y=ddx * dim,  # (args.v[0]*dim)*nm,
#     z=ddx * dim,  # (args.v[0]*dim)*nm
# )

# array = Dimensions(
#     x=np.arange(0, length.x - nm, ddx),
#     y=np.arange(0, length.y - nm, ddx),
#     z=np.arange(0, length.z - nm, ddx),
# )

# "Object parameters"
# # Sphere parameters
# sphere = Sphere(
#     R=radius,  # args.v[3]*nm                 #radius of sphere
#     x=length.x / 2,  # center in x direction
#     y=length.y / 2,  # center in y direction
#     z=length.z / 2 + 0.5 * ddx,  # center in z direction
# )

# offset = int((sphere.x - sphere.R) / ddx) - 1
# diameter = int(2 * sphere.R / ddx) + 3

# subgridding at interface
nsub = 5  # number of subgridding steps for material surface (to reduce staircasing)
eps_out = 1.0  # permittivity of surrounding medium

"Gold parameters"
if FLAG.MATERIAL != 0 and FLAG.MICRO != 0:
    eps_in = 1.0
    print("FLAGs for Microscopic and Macroscopic codes are enabled. Exitting.")
    exit()
if FLAG.MATERIAL == 1:
    eps_in = 1.0
# Drude model
if FLAG.MATERIAL == 1:  # Steve data atm, Vial data provided
    eps_in = 9.0  # background permittivity eps_inf   # Vial paper: 9.0685
    wp = 1.26e16  # gold plasma frequency             # Vial paper: 1.35e16
    gamma = 1.4e14  # gold damping                      # Vial paper: 1.15e14

# DrudeLorentz model from Vial
if FLAG.MATERIAL == 2:
    eps_in = 5.9673
    wp = 1.328e16
    gamma = 1e14
    wl = 4.08e15
    gamma_l = 6.59e14
    delta_eps = 1.09

# Etchegoin model from Etchegoin paper
if FLAG.MATERIAL == 3:
    eps_in = 1.53
    wp = 1.299e16
    gamma = 1.108e14
    w1 = 4.02489654142e15
    w2 = 5.69079026014e15
    gamma1 = 8.1897896143e14
    gamma2 = 2.00388466613e15
    A1 = 0.94
    A2 = 1.36
    c1 = np.sqrt(2) * A1 * w1
    c2 = np.sqrt(2) * A2 * w2


# def print_parameters():
#     print("dx =", int(ddx / nm), " nm")
#     print("dt =", np.round(dt * 1e18, 2), " as")
#     print("xlength =", int(length.x / nm), " nm")
#     print("Full Simulation time: ", dt * tsteps * 1e15, " fs")
