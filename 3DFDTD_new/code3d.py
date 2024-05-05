"""
3D FDTD
Plane Wave in Free Space
"""

import numpy as np
from math import exp
from matplotlib import pyplot as plt

import numba
from numba import int32, float32  # import the types
from numba.experimental import jitclass

import timeit
import scipy.constants as constants
from scipy.fft import rfft, rfftfreq
from collections import namedtuple

# data handling
import pandas as pd
import pyarrow as pals
import pyarrow.parquet as pq
import json
import os  # for making directory

# Robert imports
from fundamentals import *
from input import *
from parameters import *
import fdtd, pml, object
import monitors as mnt
import data
import plotfile as plots
from classes import DFT, Pulse, DFT_Field_3D, DFT_Field_2D, Field
from fields import *

# package for the comparison to the Mie solution case for spherical particle
import miepython

# import command line based parameters
from argparse import ArgumentParser

import sys

np.set_printoptions(threshold=sys.maxsize)

# Parameter handling - could be introduced at some point

# parser = ArgumentParser()
# parser.add_argument('--v', type=int,nargs=5)
# args = parser.parse_args()

# current parser arguments (subject to change within the optimization process)
# FLAG.OBJECT = args.v[0]
# FLAG.MATERIAL = args.v[1]
# ddx = args.v[2]*nm
# dim = args.v[3]
# tsteps = args.v[4]
# R = args.v[4]*nm


####################################################
# Beginning of Code
####################################################

# prints all parameters to console
print_parameters()

# create pulse - choose optical/THz pulse
pulse = Pulse(
    width=2, delay=5, energy=2.3, dt=dt, ddx=ddx, eps_in=eps_in
)  # Optical pulse
# pulse = Pulse(width=1.5*1e3,delay =1, energy=4*1e-3,dt = dt,ddx = ddx, eps_in = eps_in)      # THz Pulse
pulse.print_parameters()

# set global DFT parameters (DFT = discrete Fourier transform or Running Fourier Transform)
dft = DFT(dt=dt, iwdim=100, pulse_spread=pulse.spread, emin=1.9, emax=3.2)

####################################################
# Monitors
####################################################
# comments: for more speedy coding, one could only comment in the ones that one uses.

# DFT Source monitors for 1d buffer
SourceReDFT = np.zeros([dft.iwdim + 1], float)
SourceImDFT = np.zeros([dft.iwdim + 1], float)

# 3D DFT arrays
if FLAG.DFT3D == 1:
    e_dft = DFT_Field_3D(dims, 0, dft.iwdim + 1)
    h_dft = DFT_Field_3D(dims, 0, dft.iwdim + 1)

# 2D DFT arrays
if FLAG.DFT2D == 1:
    # Positions of 2d monitors
    x_DFT = int(dims.x / 2)
    y_DFT = int(dims.y / 2)
    z_DFT = int(dims.z / 2)

    # xnormal
    e_dft_xnormal = DFT_Field_2D(dims.y, dims.z, 0, dft.iwdim + 1)
    h_dft_xnormal = DFT_Field_2D(dims.y, dims.z, 0, dft.iwdim + 1)

    # ynormal
    e_dft_ynormal = DFT_Field_2D(dims.x, dims.z, 0, dft.iwdim + 1)
    h_dft_ynormal = DFT_Field_2D(dims.x, dims.z, 0, dft.iwdim + 1)

    # znormal
    e_dft_znormal = DFT_Field_2D(dims.x, dims.y, 0, dft.iwdim + 1)
    h_dft_znormal = DFT_Field_2D(dims.x, dims.y, 0, dft.iwdim + 1)

if FLAG.TFSF == 2:

    # spatial position equal to absorption box for simplicity
    y_ref = scat.y_min
    y_trans = abs.y_max

    # ynormal
    e_ref = DFT_Field_2D(dims.x, dims.z, 0, dft.iwdim)
    e_trans = DFT_Field_2D(dims.x, dims.z, 0, dft.iwdim)

"Scattering and absorption arrays"
# might reduce the size of the array as I only store the monitor and not the value in the adjacent region/PML
if FLAG.CROSS == 1:

    ## Scattering
    # e_scat = {
    #     "x_min": DFT_Field_2D(dims.y, dims.z, 0, dft.iwdim + 1),
    #     "x_max": DFT_Field_2D(dims.y, dims.z, 0, dft.iwdim + 1),
    #     "y_min": DFT_Field_2D(dims.x, dims.z, 0, dft.iwdim + 1),
    #     "y_max": DFT_Field_2D(dims.x, dims.z, 0, dft.iwdim + 1),
    #     "z_min": DFT_Field_2D(dims.x, dims.y, 0, dft.iwdim + 1),
    #     "z_max": DFT_Field_2D(dims.x, dims.y, 0, dft.iwdim + 1),
    # }

    # h_scat = {
    #     "x_min": DFT_Field_2D(dims.y, dims.z, 0, dft.iwdim + 1),
    #     "x_max": DFT_Field_2D(dims.y, dims.z, 0, dft.iwdim + 1),
    #     "y_min": DFT_Field_2D(dims.x, dims.z, 0, dft.iwdim + 1),
    #     "y_max": DFT_Field_2D(dims.x, dims.z, 0, dft.iwdim + 1),
    #     "z_min": DFT_Field_2D(dims.x, dims.y, 0, dft.iwdim + 1),
    #     "z_max": DFT_Field_2D(dims.x, dims.y, 0, dft.iwdim + 1),
    # }

    # xnormal
    e_scat_x_min = DFT_Field_2D(dims.y, dims.z, 0, dft.iwdim + 1)
    h_scat_x_min = DFT_Field_2D(dims.y, dims.z, 0, dft.iwdim + 1)
    e_scat_x_max = DFT_Field_2D(dims.y, dims.z, 0, dft.iwdim + 1)
    h_scat_x_max = DFT_Field_2D(dims.y, dims.z, 0, dft.iwdim + 1)

    # ynormal
    e_scat_y_min = DFT_Field_2D(dims.x, dims.z, 0, dft.iwdim + 1)
    h_scat_y_min = DFT_Field_2D(dims.x, dims.z, 0, dft.iwdim + 1)
    e_scat_y_max = DFT_Field_2D(dims.x, dims.z, 0, dft.iwdim + 1)
    h_scat_y_max = DFT_Field_2D(dims.x, dims.z, 0, dft.iwdim + 1)

    # znormal
    e_scat_z_min = DFT_Field_2D(dims.x, dims.y, 0, dft.iwdim + 1)
    h_scat_z_min = DFT_Field_2D(dims.x, dims.y, 0, dft.iwdim + 1)
    e_scat_z_max = DFT_Field_2D(dims.x, dims.y, 0, dft.iwdim + 1)
    h_scat_z_max = DFT_Field_2D(dims.x, dims.y, 0, dft.iwdim + 1)

    S_scat_DFT = np.zeros([6, dft.iwdim + 1])
    S_scat_total = np.zeros([dft.iwdim + 1])

    ## Absorption
    # e_abs = {
    #     "x_min": DFT_Field_2D(dims.y, dims.z, 0, dft.iwdim + 1),
    #     "x_max": DFT_Field_2D(dims.y, dims.z, 0, dft.iwdim + 1),
    #     "y_min": DFT_Field_2D(dims.x, dims.z, 0, dft.iwdim + 1),
    #     "y_max": DFT_Field_2D(dims.x, dims.z, 0, dft.iwdim + 1),
    #     "z_min": DFT_Field_2D(dims.x, dims.y, 0, dft.iwdim + 1),
    #     "z_max": DFT_Field_2D(dims.x, dims.y, 0, dft.iwdim + 1),
    # }

    # h_abs = {
    #     "x_min": DFT_Field_2D(dims.y, dims.z, 0, dft.iwdim + 1),
    #     "x_max": DFT_Field_2D(dims.y, dims.z, 0, dft.iwdim + 1),
    #     "y_min": DFT_Field_2D(dims.x, dims.z, 0, dft.iwdim + 1),
    #     "y_max": DFT_Field_2D(dims.x, dims.z, 0, dft.iwdim + 1),
    #     "z_min": DFT_Field_2D(dims.x, dims.y, 0, dft.iwdim + 1),
    #     "z_max": DFT_Field_2D(dims.x, dims.y, 0, dft.iwdim + 1),
    # }
    # xnormal
    e_abs_x_min = DFT_Field_2D(dims.y, dims.z, 0, dft.iwdim + 1)
    h_abs_x_min = DFT_Field_2D(dims.y, dims.z, 0, dft.iwdim + 1)
    e_abs_x_max = DFT_Field_2D(dims.y, dims.z, 0, dft.iwdim + 1)
    h_abs_x_max = DFT_Field_2D(dims.y, dims.z, 0, dft.iwdim + 1)

    # ynormal
    e_abs_y_min = DFT_Field_2D(dims.x, dims.z, 0, dft.iwdim + 1)
    h_abs_y_min = DFT_Field_2D(dims.x, dims.z, 0, dft.iwdim + 1)
    e_abs_y_max = DFT_Field_2D(dims.x, dims.z, 0, dft.iwdim + 1)
    h_abs_y_max = DFT_Field_2D(dims.x, dims.z, 0, dft.iwdim + 1)

    # znormal
    e_abs_z_min = DFT_Field_2D(dims.x, dims.y, 0, dft.iwdim + 1)
    h_abs_z_min = DFT_Field_2D(dims.x, dims.y, 0, dft.iwdim + 1)
    e_abs_z_max = DFT_Field_2D(dims.x, dims.y, 0, dft.iwdim + 1)
    h_abs_z_max = DFT_Field_2D(dims.x, dims.y, 0, dft.iwdim + 1)

    S_abs_DFT = np.zeros([6, dft.iwdim + 1])
    S_abs_total = np.zeros([dft.iwdim + 1])


# FFT Monitors
if FLAG.FFT == 1:
    # location of FFT monitors
    n_mon = 6
    loc_monitors = [
        (tfsf.x_min - 3, int(dims.y / 2), int(dims.z / 2)),
        (int(dims.x / 2), tfsf.y_min - 3, int(dims.z / 2)),
        (int(dims.x / 2), int(dims.y / 2), tfsf.z_min - 3),
        (tfsf.x_max + 2, int(dims.y / 2), int(dims.z / 2)),
        (int(dims.x / 2), tfsf.y_max + 2, int(dims.z / 2)),
        (int(dims.x / 2), int(dims.y / 2), tfsf.z_max + 2),
    ]
    # Definition of FFT monitors
    ex_mon = np.zeros([n_mon, tsteps])
    ey_mon = np.zeros([n_mon, tsteps])
    ez_mon = np.zeros([n_mon, tsteps])
    hx_mon = np.zeros([n_mon, tsteps])
    hy_mon = np.zeros([n_mon, tsteps])
    hz_mon = np.zeros([n_mon, tsteps])


# ------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------

PML = pml.calculate_pml_params(
    dims, npml=8, TFSF_FLAG=FLAG.TFSF
)  # set PML parameters/en/latest/10_basic_tests.html

##########################################
# Object creation depending on flags
##########################################

start_time = timeit.default_timer()

if FLAG.OBJECT == 1:  # if circle

    if FLAG.MATERIAL == 1:  # Drude only
        ga, d1, d2, d3 = object.create_sphere_drude_eps(
            sphere, nsub, ddx, dt, eps_in, eps_out, wp, gamma, ga, d1, d2, d3
        )
    if FLAG.MATERIAL == 2:  # DrudeLorentz
        ga, d1, d2, d3 = object.create_sphere_drude_eps(
            sphere, nsub, ddx, dt, eps_in, eps_out, wp, gamma, ga, d1, d2, d3
        )
        ga, l1, l2, l3 = object.create_sphere_lorentz(
            sphere,
            nsub,
            ddx,
            dt,
            eps_in,
            eps_out,
            wl,
            gamma_l,
            delta_eps,
            ga,
            l1,
            l2,
            l3,
        )

    if FLAG.MATERIAL == 3:  # Etchegoin
        ga, d1, d2, d3 = object.create_sphere_drude_eps(
            sphere, nsub, ddx, dt, eps_in, eps_out, wp, gamma, ga, d1, d2, d3
        )
        f1_et1, f2_et1, f3_et1, f4_et1, f1_et2, f2_et2, f3_et2, f4_et2 = (
            object.create_sphere_etch(
                sphere,
                nsub,
                ddx,
                dt,
                c1,
                c2,
                w1,
                w2,
                gamma1,
                gamma2,
                f1_et1,
                f2_et1,
                f3_et1,
                f4_et1,
                f1_et2,
                f2_et2,
                f3_et2,
                f4_et2,
            )
        )

elif FLAG.OBJECT == 2 and FLAG.TFSF == 2:

    if FLAG.MATERIAL == 1:  # Drude model
        ga, d1, d2, d3 = object.create_rectangle_PBC(
            dims,
            int(dims.y / 2),
            int(dims.y / 2 + sphere.R / ddx),
            dt,
            eps_in,
            wp,
            gamma,
            ga,
            d1,
            d2,
            d3,
        )  # should derive a different quantity for the diameter instead of using sphere radius

    if FLAG.MATERIAL == 2:  # DrudeLorentz
        ga, d1, d2, d3 = object.create_rectangle_PBC(
            dims,
            int(dims.y / 2),
            int(dims.y / 2 + sphere.R / ddx),
            dt,
            eps_in,
            wp,
            gamma,
            ga,
            d1,
            d2,
            d3,
        )
        ga, l1, l2, l3 = object.create_rectangle_PBC_lorentz(
            dims,
            int(dims.y / 2),
            int(dims.y / 2 + sphere.R / ddx),
            dt,
            eps_in,
            wl,
            gamma_l,
            delta_eps,
            ga,
            l1,
            l2,
            l3,
        )

    if FLAG.MATERIAL == 3:  # Etchegoin
        ga, d1, d2, d3 = object.create_rectangle_PBC(
            dims,
            int(dims.y / 2),
            int(dims.y / 2 + sphere.R / ddx),
            dt,
            eps_in,
            wp,
            gamma,
            ga,
            d1,
            d2,
            d3,
        )
        f1_et1, f2_et1, f3_et1, f4_et1, f1_et2, f2_et2, f3_et2, f4_et2 = (
            object.create_rectangle_PBC_etch(
                dims,
                int(dims.y / 2),
                int(dims.y / 2 + sphere.R / ddx),
                dt,
                c1,
                c2,
                w1,
                gamma1,
                w2,
                gamma2,
                f1_et1,
                f2_et1,
                f3_et1,
                f4_et1,
                f1_et2,
                f2_et2,
                f3_et2,
                f4_et2,
            )
        )

elif FLAG.OBJECT == 0:
    print("No object defined")

else:
    print("Something is wrong with the object definition")


###################
# Plot definitions
###################

# f_plot = np.zeros((grid.n_kmax,grid.n_phimax,grid.n_thetamax))
# #General setting
if FLAG.ANIMATION == 1:
    # plt.rcParams.update({'font.size': 14})
    # fig, ax = plt.subplots(3, 4, figsize=(20, 15))
    # ims, imp, text_tstep, xcut, ycut, zcut, incident_e, incident_h =\
    #      plots.setup(FLAG, fig, ax, ddx, dt, length, array, dims, sphere, pulse, npml, tfsf,e, ez_inc, hx_inc)
    # fig, ax = plt.subplots(3, 3, figsize=(20, 15))
    #    ims, imp, text_tstep, incident_loc,plot_f,plot_f2,plot_f3 =\
    #     plots.setup_GIF(FLAG, fig, ax, ddx, dt, length, array, dims, sphere, pulse, npml, tfsf,e, ez_inc, hx_inc,f_plot)

    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(3, 4, figsize=(20, 15))

    ims, imp, text_tstep, xcut, ycut, zcut, incident_e, incident_h = plots.setup(
        FLAG,
        fig,
        ax,
        ddx,
        dt,
        length,
        array,
        dims,
        sphere,
        pulse,
        npml,
        tfsf,
        e,
        ez_inc,
        hx_inc,
    )


intermediate_time = timeit.default_timer()
print("Time for setting up the problem", intermediate_time - start_time)


###################################
# Start of time loop
###################################

for time_step in range(1, tsteps + 1):
    print(time_step)

    "break statement for auto shutoff, in case that needed"
    # if(time_step > t0 and np.max(ez[npml:Xmax-npml,npml:Ymax-npml,npml:Zmax-npml])<1e-5):
    #     break

    # FDTD loop
    # -------------------------------------------------------------------

    # Compute macroscopic polarization using auxilliary equation"
    if FLAG.OBJECT != 0:
        if FLAG.MATERIAL == 1:  # Drude only
            p_drude, p_tmp_drude = object.calculate_polarization(
                dims, sphere, ddx, p_drude, p_tmp_drude, e, d1, d2, d3, FLAG.OBJECT
            )
            p.x = p_drude.x
            p.y = p_drude.y
            p.z = p_drude.z

        if FLAG.MATERIAL == 2:  # DrudeLorentz:
            p_drude, p_tmp_drude = object.calculate_polarization(
                dims, sphere, ddx, p_drude, p_tmp_drude, e, d1, d2, d3, FLAG.OBJECT
            )
            p_lorentz, p_tmp_lorentz = object.calculate_polarization(
                dims, sphere, ddx, p_lorentz, p_tmp_lorentz, e, l1, l2, l3, FLAG.OBJECT
            )
            p.x = p_drude.x + p_lorentz.x
            p.y = p_drude.y + p_lorentz.y
            p.z = p_drude.z + p_lorentz.z

        if FLAG.MATERIAL == 3:  # Etchegoin
            p_drude, p_tmp_drude = object.calculate_polarization(
                dims, sphere, ddx, p_drude, p_tmp_drude, e, d1, d2, d3, FLAG.OBJECT
            )
            p_et1, p_tmp_et1 = object.calculate_polarization_etch(
                dims,
                sphere,
                ddx,
                p_et1,
                p_tmp_et1,
                e,
                e1,
                f1_et1,
                f2_et1,
                f3_et1,
                f4_et1,
                FLAG.OBJECT,
            )
            p_et2, p_tmp_et2 = object.calculate_polarization_etch(
                dims,
                sphere,
                ddx,
                p_et2,
                p_tmp_et2,
                e,
                e1,
                f1_et2,
                f2_et2,
                f3_et2,
                f4_et2,
                FLAG.OBJECT,
            )
            p.x = p_drude.x + p_et1.x + p_et2.x
            p.y = p_drude.y + p_et1.y + p_et2.y
            p.z = p_drude.z + p_et1.z + p_et2.z

    # -------------------------------------------------------------------
    # main FDTD Code
    # -------------------------------------------------------------------

    # update 1d electric field buffer
    if FLAG.TFSF == 1 or FLAG.TFSF == 2:
        # Update incident electric field
        ez_inc = fdtd.calculate_ez_inc_field(dims.y, ez_inc, hx_inc)

        # Implementation of ABC
        ez_inc[0] = boundary_low.pop(0)
        boundary_low.append(ez_inc[1])
        ez_inc[dims.y - 1] = boundary_high.pop(0)
        boundary_high.append(ez_inc[dims.y - 2])

    # field updates
    d, id = fdtd.D_update(FLAG, dims, d, h, id, PML, tfsf, hx_inc)
    e, e1 = fdtd.calculate_e_fields(dims, e, e1, d, ga, p)
    h, ih = fdtd.H_update(FLAG, dims, h, ih, e, PML, tfsf, ez_inc)

    # pulse updating
    pulse_tmp = pulse.update_value(time_step, dt)

    # Update 1d buffer for plane wave - magnetic field
    if FLAG.TFSF == 1 or FLAG.TFSF == 2:
        hx_inc = fdtd.calculate_hx_inc_field(dims.y, hx_inc, ez_inc)

    # -------------------------------------------------------------------
    # Update Monitors
    # -------------------------------------------------------------------

    # pulse monitor plane wave
    if FLAG.TFSF == 1 or FLAG.TFSF == 2:
        pulse_t[time_step - 1] = pulse_tmp
        ez_inc[tfsf.y_min - 3] = pulse_tmp

    # pulse monitor point source
    if FLAG.POINT == 1:
        e.z[source.x, source.y, source.z] += pulse_tmp
        pulsemon_t, ez_source_t = mnt.update_pulsemonitors(
            time_step,
            e.z,
            source.x,
            source.y,
            source.z,
            pulse_tmp,
            pulsemon_t,
            ez_source_t,
        )

    # 1D FFT monitors
    if FLAG.FFT == 1:
        # Something is wrong here!!
        mnt.update_1Dmonitors(
            time_step,
            loc_monitors,
            ex_mon,
            ey_mon,
            ez_mon,
            hx_mon,
            hy_mon,
            hz_mon,
            e,
            h,
        )

    # Source monitors for TFSF
    SourceReDFT, SourceImDFT = mnt.DFT_incident_update(
        dft.omega, SourceReDFT, SourceImDFT, pulse_tmp, dft.iwdim, time_step
    )

    # 3D DFT monitors
    if FLAG.DFT3D == 1 and time_step > dft.tstart:
        e_dft, h_dft = mnt.DFT3D_update(
            e, h, e_dft, h_dft, dft.iwdim, dft.omega, time_step
        )

    # 2D DFT monitors
    if FLAG.DFT2D == 1 and time_step > dft.tstart:
        e_dft_xnormal, e_dft_ynormal, e_dft_znormal,
        h_dft_xnormal, h_dft_ynormal, h_dft_znormal = mnt.DFT2D_update(
            e,
            h,
            dft.iwdim,
            dft.omega,
            time_step,
            x_DFT,
            y_DFT,
            z_DFT,
            e_dft_xnormal,
            e_dft_ynormal,
            e_dft_znormal,
            h_dft_xnormal,
            h_dft_ynormal,
            h_dft_znormal,
        )

    # Reflection and transmission for periodic boundary condition
    if FLAG.TFSF == 2 and time_step > dft.tstart:
        e_ref, e_trans = mnt.DFT_ref_trans(
            e_ref, e_trans, e, y_ref, y_trans, dft.iwdim, dft.omega, time_step
        )

    # Cross Sections
    if FLAG.CROSS == 1 and time_step > dft.tstart:

        # Scattering
        (
            e_scat_x_min,
            e_scat_x_max,
            h_scat_x_min,
            h_scat_x_max,
            e_scat_y_min,
            e_scat_y_max,
            h_scat_y_min,
            h_scat_y_max,
            e_scat_z_min,
            e_scat_z_max,
            h_scat_z_min,
            h_scat_z_max,
        ) = mnt.DFT_scat_update(
            e_scat_x_min,
            e_scat_x_max,
            h_scat_x_min,
            h_scat_x_max,
            e_scat_y_min,
            e_scat_y_max,
            h_scat_y_min,
            h_scat_y_max,
            e_scat_z_min,
            e_scat_z_max,
            h_scat_z_min,
            h_scat_z_max,
            e,
            h,
            scat,
            dft.iwdim,
            dft.omega,
            time_step,
        )

        # Absorption
        (
            e_abs_x_min,
            e_abs_x_max,
            h_abs_x_min,
            h_abs_x_max,
            e_abs_y_min,
            e_abs_y_max,
            h_abs_y_min,
            h_abs_y_max,
            e_abs_z_min,
            e_abs_z_max,
            h_abs_z_min,
            h_abs_z_max,
        ) = mnt.DFT_abs_update(
            e_abs_x_min,
            e_abs_x_max,
            h_abs_x_min,
            h_abs_x_max,
            e_abs_y_min,
            e_abs_y_max,
            h_abs_y_min,
            h_abs_y_max,
            e_abs_z_min,
            e_abs_z_max,
            h_abs_z_min,
            h_abs_z_max,
            e,
            h,
            abs,
            dft.iwdim,
            dft.omega,
            time_step,
        )

    # # # Animation
    # if time_step % cycle == 0 and FLAG.ANIMATION == 1:
    #     plots.animate(
    #         time_step,
    #         text_tstep,
    #         e,
    #         dims,
    #         ims,
    #         array,
    #         ax,
    #         xcut,
    #         ycut,
    #         zcut,
    #         incident_e,
    #         ez_inc,
    #         hx_inc,
    #         incident_h,
    #         p,
    #         imp,
    #         time_pause,
    #     )
    #     plots.animate_GIF(
    #         time_step,
    #         text_tstep,
    #         e,
    #         dims,
    #         ims,
    #         array,
    #         ax,
    #         ez_inc,
    #         hx_inc,
    #         incident_loc,
    #         pulse,
    #         p,
    #         imp,
    #         time_pause,
    #         f_plot,
    #         plot_f,
    #         plot_f2,
    #         plot_f3,
    #     )

# computation time
stop = timeit.default_timer()
print("Time for full computation", stop - start_time)

plt.show()


# -------------------------------------------------------------------
# Data processing
# -------------------------------------------------------------------

"FT 1Dpoint monitors"
if FLAG.FFT == 1:
    fft_res = 20
    omega, ex_mon_om, ey_mon_om, ez_mon_om, hx_mon_om, hy_mon_om, hz_mon_om = (
        mnt.fft_1Dmonitors(
            dt, tsteps, fft_res, n_mon, ex_mon, ey_mon, ez_mon, hx_mon, hy_mon, hz_mon
        )
    )

# FFT for Point monitor
if FLAG.POINT == 1:
    fft_res = 20
    omega_source, pulsemon_om, ez_source_om = mnt.fft_sourcemonitors(
        dt, tsteps, fft_res, pulsemon_t, ez_source_t
    )
    Mon = np.abs(ez_source_om) ** 2 / np.max(np.abs(pulsemon_om) ** 2)
    Source = np.abs(pulsemon_om) ** 2 / np.max(np.abs(pulsemon_om) ** 2)

    GFT = ez_source_om / pulsemon_om  # E_mon(w) / P(w)            # numerical GFT -
    GFT_an = omega_source**3 / c**3 / (6 * np.pi)  # analytical GFT
    print("POINT FFT running")

# calculate bandwidth
if FLAG.TFSF == 1 or FLAG.TFSF == 2:
    fft_res = 20
    omega_source, pulse_om = mnt.fft_source(dt, tsteps, fft_res, pulse_t)
    Source = np.abs(pulse_om) ** 2 / np.max(np.abs(pulse_om) ** 2)
    print("Bandwidth calculated")

# calculate transmission and reflection
if FLAG.TFSF == 2:
    reflection = e_ref.surface_magnitude()
    transmission = e_trans.surface_magnitude()
    plt.plot(
        dft.omega / dt * hbar / eC,
        reflection**2 / (SourceReDFT**2 + SourceImDFT**2) / dims.x**4,
    )
    plt.plot(
        dft.omega / dt * hbar / eC,
        transmission**2 / (SourceReDFT**2 + SourceImDFT**2) / dims.x**4,
    )
    plt.show()

# calculate Cross Sections via Poynting vector
if FLAG.CROSS == 1:
    S_scat_DFT = mnt.update_scat_DFT(
        S_scat_DFT,
        dft.iwdim,
        scat,
        e_scat_x_min,
        e_scat_x_max,
        h_scat_x_min,
        h_scat_x_max,
        e_scat_y_min,
        e_scat_y_max,
        h_scat_y_min,
        h_scat_y_max,
        e_scat_z_min,
        e_scat_z_max,
        h_scat_z_min,
        h_scat_z_max,
    )

    S_abs_DFT = mnt.update_abs_DFT(
        S_abs_DFT,
        dft.iwdim,
        abs,
        e_abs_x_min,
        e_abs_x_max,
        h_abs_x_min,
        h_abs_x_max,
        e_abs_y_min,
        e_abs_y_max,
        h_abs_y_min,
        h_abs_y_max,
        e_abs_z_min,
        e_abs_z_max,
        h_abs_z_min,
        h_abs_z_max,
    )

    S_scat_total = (
        S_scat_DFT[0, :]
        + S_scat_DFT[1, :]
        + S_scat_DFT[2, :]
        + S_scat_DFT[3, :]
        + S_scat_DFT[4, :]
        + S_scat_DFT[5, :]
    ) * ddx**2
    S_abs_total = (
        S_abs_DFT[0, :]
        + S_abs_DFT[1, :]
        + S_abs_DFT[2, :]
        + S_abs_DFT[3, :]
        + S_abs_DFT[4, :]
        + S_abs_DFT[5, :]
    ) * ddx**2


# -------------------------------------------------------------------
# Data storage
# -------------------------------------------------------------------

if FLAG.POINT == 1:
    data.store_point(
        FLAG,
        sphere,
        ddx,
        dt,
        tsteps,
        dims,
        pulse,
        dft,
        npml,
        eps_in,
        eps_out,
        source,
        pulsemon_t,
        ez_source_t,
        start_time,
        stop,
    )

if FLAG.CROSS == 1:
    data.store_cross(
        FLAG,
        sphere,
        ddx,
        dt,
        tsteps,
        dims,
        pulse,
        dft,
        npml,
        eps_in,
        eps_out,
        start_time,
        stop,
        S_scat_total,
        S_abs_total,
        SourceReDFT,
        SourceImDFT,
        wp,
        gamma,
        tfsf_dist,
    )

if FLAG.TFSF == 2 and FLAG.MICRO == 0:
    data.store_periodic(
        FLAG,
        sphere,
        ddx,
        dt,
        tsteps,
        dims,
        pulse,
        dft,
        npml,
        eps_in,
        eps_out,
        start_time,
        stop,
        SourceReDFT,
        SourceImDFT,
        wp,
        gamma,
        tfsf_dist,
        transmission,
        reflection,
    )

if FLAG.TFSF == 2 and FLAG.MICRO == 1:
    data.store_periodic_micro(
        FLAG,
        sphere,
        ddx,
        dt,
        tsteps,
        dims,
        pulse,
        dft,
        npml,
        eps_in,
        eps_out,
        start_time,
        stop,
        SourceReDFT,
        SourceImDFT,
        tfsf_dist,
        transmission,
        reflection,
    )
