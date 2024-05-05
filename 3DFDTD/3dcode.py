"""
3D FDTD
Plane Wave in Free Space
this is broken
"""

from telnetlib import ECHO
import numpy as np
from math import exp
from matplotlib import pyplot as plt

import numba
from numba import int32, float32    # import the types
from numba.experimental import jitclass

import timeit
import scipy.constants as constants
from scipy.fft import rfft,rfftfreq
from collections import namedtuple

#data handling
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
import os # for making directory

#Robert imports
import fdtd,pml,fields,object
import monitors as mnt
from classes import Pulse, DFT,DFT_Field_2D
#Jonas imports
import physical_functions as pf
import boltzmann
import grid
import initial_conditions

from  parameters import *

#package for the comparison to the Mie solution case for spherical particle
import miepython

#import command line based parameters
from argparse import ArgumentParser

import sys
np.set_printoptions(threshold=sys.maxsize)

"FLAGS for macroscopic code"

#boundary
PML_FLAG = 1            # PML:                      0: off,         1: on 

#material
# DRUDE_FLAG =1           # Drude model (-1/w(w+ig))  0: off,         1: on
# EPSILON_FLAG = 1        # background epsilon        0: off (=1),    1: on (=eps_in)
# RECT_FLAG = 0
OBJECT_FLAG = 2        # object choice             0: no object    1: sphere               2: rectangle
MATERIAL_FLAG = 1       # choice of material        1: Drude        2: DrudeLorentz         3: Etchegoin Model 
#source
TFSF_FLAG = 2           # Plane wave                0: off,         1: TFSF on,             2: Periodic Boundary condition
POINT_FLAG = 0          # Point source              0: off,         1: on

#monitors
CROSS_FLAG = 0          # x sections (scat, abs)    0: off,         1: on
DFT3D_FLAG = 0          # 3D DFT monitor:           0: off,         1: on
DFT2D_FLAG = 0          # 2D DFT monitor            0: off,         1: on
FFT_FLAG = 0            # FFT point monitor         0: off,         1: on 

#visualization
ANIMATION_FLAG = 1      # Animation                 0: off,         1: on

#microscopic code
MICRO_FLAG = 0           # Microscopic code          0: disabled,    1: enabled          # should be merged with the DRUDE/EPS FLAG at some point

#Jonas Flags
ELPHO_FLAG = 0      #0: call matrices from memory, 1: calculate matrices, 2: without electron phonon interaction, 3: with interaction, does not save the matrices
LM_FLAG = 3         #0: call matrices from memory, 1: calculate matrices, 2: without light matter interaction, 3: with interaction, does not save the matrices
STENCIL_FLAG = 3    #3: Three-point stencil, 5: Five-point stencil, 7: Seven-point stencil
E_FIELD_FLAG = 0    #0: Use of the field calculated and updated by Roberts FDTD Maxwell solver; 1: Field is decoupled from electron dynamics (e.g. Gauß-shaped pulse)
INIT_FLAG = 6       #5,6: IC for delta f: 5: T_electron=T_phonon, 6: T_electron= args temperature, 7: Fermi distributed dumbbell-shaped, 8: Gauss distributed dumbbell-shaped

'Parameter handling'
# parser = ArgumentParser()
# parser.add_argument('--v', type=int,nargs=5)
# args = parser.parse_args()

#current parser arguments (subject to change within the optimization process)
# OBJECT_FLAG = args.v[0]
# MATERIAL_FLAG = args.v[1]
#ddx = args.v[2]*nm  
#dim = args.v[3]
#tsteps = args.v[4] 
#R = args.v[4]*nm

"Adjustable parameters"
ddx = 10*nm #args.v[0]*nm           # spatial step size
dt = ddx/c*Sc                       # time step, from dx fulfilling stability criteria
tfsf_dist = 12 #args.v[4]           # TFSF distance from computational boundary
npml = 8                            # number of PML layers
dim = 60 #args.v[1]                 # number of spatial steps

"Time parameters"
tsteps = 5000#args.v[2]             # number of time steps
cycle = 10                          # selection of visualized frames
time_pause = 0.1                    # pause time for individual frames, limited by computation time
time_micro_factor = 100

"computational domain specification"
#real space
dims = Dimensions(x=dim, y=dim, z=dim)

#momentum space   
#dims_k = Dimensions_k(k=grid.n_kmax, phi=grid.n_phimax, theta=grid.n_thetamax)

#TFSF boundary conditions
tfsf = box(
    x_min = tfsf_dist, x_max = dims.x -tfsf_dist - 1,
    y_min = tfsf_dist, y_max = dims.y -tfsf_dist - 1,
    z_min = tfsf_dist, z_max = dims.z -tfsf_dist - 1,
)

"Cross sectionparameters"
#Scattering box
scat = box(
    x_min = tfsf.x_min-3, x_max = tfsf.x_max+2,
    y_min = tfsf.y_min-3, y_max = tfsf.y_max+2,
    z_min = tfsf.z_min-3, z_max = tfsf.z_max+2,
)
#Absorption box
abs = box(
    x_min = tfsf.x_min+2, x_max = tfsf.x_max-3,
    y_min = tfsf.y_min+2, y_max = tfsf.y_max-3,
    z_min = tfsf.z_min+2, z_max = tfsf.z_max-3,
)

"Spatial domain"
#length scales
xlength = ddx*dim#(args.v[0]*dim)*nm
ylength = ddx*dim#(args.v[0]*dim)*nm
zlength = ddx*dim#(args.v[0]*dim)*nm

#spatial array
X=np.arange(0,xlength-nm,ddx)
Y=np.arange(0,ylength-nm,ddx)
Z=np.arange(0,zlength-nm,ddx)

print('dx =',int(ddx/nm),' nm')
print('dt =',dt*1e15,' fs')
print('xlength =',int(xlength/nm),' nm')
print('Full Simulation time: ', dt*tsteps*1e15,' fs')


"Object parameters"
#Sphere parameters
xc = xlength/2                  #center in x direction
yc = ylength/2                  #center in y direction
zc = zlength/2+0.5*ddx          #center in z direction
R = 15
# R = 150*nm#args.v[3]*nm          #radius of sphere
offset = int((xc-R)/ddx)-1
diameter = int(2*R/ddx)+3
# print('offset= ', offset)
# print('diameter= ', diameter)

#dielectric environment
eps_out = 1                     # permittivity of surrounding medium
if MICRO_FLAG ==1 or MATERIAL_FLAG == 0:
    eps_in = 1.  
#subgridding at interface
nsub = 5                        # number of subgridding steps for material surface (to reduce staircasing)


"Gold parameters"
# Drude model
if MATERIAL_FLAG == 1: # Steve data atm, Vial data provided
    eps_in = 9.                     # background permittivity eps_inf   # Vial paper: 9.0685 
    wp = 1.26e16                    # gold plasma frequency             # Vial paper: 1.35e16
    gamma = 1.4e14                  # gold damping                      # Vial paper: 1.15e14

# DrudeLorentz model from Vial
if MATERIAL_FLAG == 2: 
    eps_in = 5.9673
    wp = 1.328e16
    gamma = 1e14
    wl = 4.08e15
    gamma_l = 6.59e14
    delta_eps =  1.09

#Etchegoin model from Etchegoin paper
if MATERIAL_FLAG == 3: 
    eps_in = 1.53
    wp = 1.299e16
    gamma = 1.108e14
    w1 = 4.02489654142e15
    w2 = 5.69079026014e15
    gamma1 = 8.1897896143e14
    gamma2 = 2.00388466613e15
    A1 = 0.94
    A2 = 1.36
    c1 = np.sqrt(2)*A1*w1
    c2 = np.sqrt(2)*A2*w2


########################################
# Define fields
#######################################

"Define material arrays"
e = fields.Field(dims,0)            # electric field E 
e1 = fields.Field(dims,0)            # electric field E memory
h = fields.Field(dims,0)            # magnetic field strength H 
d = fields.Field(dims,0)            # electric displacement field D  
p = fields.Field(dims,0)      # polarization field (macroscopic)
p_drude = fields.Field(dims,0)      # polarization field (macroscopic)
p_lorentz = fields.Field(dims,0)    # polarization field (macroscopic)
ga = fields.Field(dims,1)           # inverse permittivity =1/eps
id = fields.Field(dims,0)           # accumulated curls for PML calculation
ih = fields.Field(dims,0)           # accumulated curls for PML calculation

"Material arrays for Drude model"
if OBJECT_FLAG != 0:
    #Drude - this uses a lot of memory can this be more efficient?
    p_tmp_drude = fields.Field(dims,0)    # polarization help field
    d1 = fields.Field(dims,0)       # prefactor in auxilliary equation
    d2 = fields.Field(dims,0)       # prefactor in auxilliary equation
    d3 = fields.Field(dims,0)       # prefactor in auxilliary equation#

    if MATERIAL_FLAG == 2:
        #Lorentz
        p_tmp_lorentz = fields.Field(dims,0)    # polarization help field
        l1 = fields.Field(dims,0)       # prefactor in auxilliary equation
        l2 = fields.Field(dims,0)       # prefactor in auxilliary equation
        l3 = fields.Field(dims,0)       # prefactor in auxilliary equation

    if MATERIAL_FLAG == 3:
        p_et1 = fields.Field(dims,0)       # polarization at timestep before 
        p_tmp_et1 = fields.Field(dims,0)    # polarization help field
        p_et2 = fields.Field(dims,0)       # polarization at timestep before
        p_tmp_et2 = fields.Field(dims,0)    # polarization help field
        f1_et1 = fields.Field(dims,0)       # prefactor in auxilliary equation
        f2_et1 = fields.Field(dims,0)       # prefactor in auxilliary equation
        f3_et1 = fields.Field(dims,0)       # prefactor in auxilliary equation
        f4_et1 = fields.Field(dims,0)       # prefactor in auxilliary equation
        f1_et2 = fields.Field(dims,0)       # prefactor in auxilliary equation
        f2_et2 = fields.Field(dims,0)       # prefactor in auxilliary equation
        f3_et2 = fields.Field(dims,0)       # prefactor in auxilliary equation
        f4_et2 = fields.Field(dims,0)       # prefactor in auxilliary equation


'Fields for interaction with microscopic code'
if MICRO_FLAG ==1:
    j = fields.Field(dims,0)                                                                            # current density for microscopic code
    j_tmp = fields.Field(dims,0)                                                                        # current density temporary (helper

####################################################
# Sources
####################################################

"1D plane wave buffer for TFSF simulation"
if (TFSF_FLAG==1 or TFSF_FLAG ==2):
    ez_inc = np.zeros(dims.y,float)         # 1d buffer field for plane wave
    hx_inc = np.zeros(dims.y,float)         # 1d buffer field for plane wave
    boundary_low = [0, 0]                   # lower absorbing boundary condition
    boundary_high = [0, 0]                  # upper absorbing boundary condition
    pulse_t = np.zeros(tsteps,float)        # pulse monitor for FFT

"Create Pulse"
#optical pulse

#pulse = fields.Pulse(width=2,energy=0.00414,dt = dt,ddx = ddx, eps_in = eps_in)

pulse = Pulse(width=2,delay = 3*2,energy=0.00414,dt = dt,ddx = ddx, eps_in = eps_in)
#THz Pulse
#pulse = fields.Pulse(width=1.5*1e3,energy=4*1e-3,dt = dt,ddx = ddx, eps_in = eps_in)

pulse.print_parameters()


"Point dipole source"
if(POINT_FLAG==1):
    xs = int(dims.x/2);ys = int(dims.y/2);zs = int(dims.z/2)            # point source position
    pulsemon_t = np.zeros(tsteps,float)                                 # pulse monitor (source)
    ez_source_t = np.zeros(tsteps,float)                                # electric field monitor


####################################################
# Monitors
####################################################

"FFT Monitors"
#Definition of Point monitors
if FFT_FLAG ==1:
    n_mon = 6
    loc_monitors = [
        (tfsf.x_min-3,int(dims.y/2),int(dims.z/2)),
        (int(dims.x/2), tfsf.y_min-3,int(dims.z/2)),
        (int(dims.x/2), int(dims.y/2),tfsf.z_min-3),
        (tfsf.x_max+2,int(dims.y/2),int(dims.z/2)),
        (int(dims.x/2), tfsf.y_max+2,int(dims.z/2)),
        (int(dims.x/2), int(dims.y/2),tfsf.z_max+2)
    ]

    ex_mon = np.zeros([n_mon,tsteps])
    ey_mon = np.zeros([n_mon,tsteps])
    ez_mon = np.zeros([n_mon,tsteps])
    hx_mon = np.zeros([n_mon,tsteps])
    hy_mon = np.zeros([n_mon,tsteps])
    hz_mon = np.zeros([n_mon,tsteps])

"DFT Monitors"
# set global DFT parameters
#dft = fields.DFT(dt = dt,iwdim = 100,pulse_spread = pulse.spread,e_min=1.9,e_max=3.2)
dft = DFT(dt = dt,iwdim = 100,pulse_spread = pulse.spread,emin=1.9,emax=3.2)

# DFT Source monitors for 1d buffer
SourceReDFT=np.zeros([dft.iwdim+1],float)
SourceImDFT=np.zeros([dft.iwdim+1],float)

# 3D DFT arrays
if DFT3D_FLAG ==1:
    e_dft = fields.DFT_Field_3D(dims,0,dft.iwdim+1)
    h_dft = fields.DFT_Field_3D(dims,0,dft.iwdim+1)

# 2D DFT arrays
if DFT2D_FLAG ==1:
    #Positions of 2d monitors
    x_DFT = int(dims.x/2)
    y_DFT = int(dims.y/2)
    z_DFT = int(dims.z/2) 

    # xnormal
    e_dft_xnormal = fields.DFT_Field_2D(dims.y,dims.z,0,dft.iwdim+1)
    h_dft_xnormal = fields.DFT_Field_2D(dims.y,dims.z,0,dft.iwdim+1)

    # ynormal
    e_dft_ynormal = fields.DFT_Field_2D(dims.x,dims.z,0,dft.iwdim+1)
    h_dft_ynormal = fields.DFT_Field_2D(dims.x,dims.z,0,dft.iwdim+1)

    # znormal
    e_dft_znormal = fields.DFT_Field_2D(dims.x,dims.y,0,dft.iwdim+1)
    h_dft_znormal = fields.DFT_Field_2D(dims.x,dims.y,0,dft.iwdim+1)

if TFSF_FLAG == 2:

    #spatial position equal to absorption box for simplicity
    y_ref = scat.y_min
    y_trans = abs.y_max

    # ynormal
    e_ref =DFT_Field_2D(dims.x,dims.z,0,dft.iwdim)
    e_trans =DFT_Field_2D(dims.x,dims.z,0,dft.iwdim)

"Scattering and absorption arrays"
#might reduce the size of the array as I only store the monitor and not the value in the adjacent region/PML
if CROSS_FLAG ==1:

    "Scattering"
    # xnormal
    e_scat_x_min = DFT_Field_2D(dims.y,dims.z,0,dft.iwdim+1)
    h_scat_x_min = DFT_Field_2D(dims.y,dims.z,0,dft.iwdim+1)
    e_scat_x_max = DFT_Field_2D(dims.y,dims.z,0,dft.iwdim+1)
    h_scat_x_max = DFT_Field_2D(dims.y,dims.z,0,dft.iwdim+1)

    # ynormal
    e_scat_y_min = DFT_Field_2D(dims.x,dims.z,0,dft.iwdim+1)
    h_scat_y_min = DFT_Field_2D(dims.x,dims.z,0,dft.iwdim+1)
    e_scat_y_max = DFT_Field_2D(dims.x,dims.z,0,dft.iwdim+1)
    h_scat_y_max = DFT_Field_2D(dims.x,dims.z,0,dft.iwdim+1)

    # znormal
    e_scat_z_min = DFT_Field_2D(dims.x,dims.y,0,dft.iwdim+1)
    h_scat_z_min = DFT_Field_2D(dims.x,dims.y,0,dft.iwdim+1)
    e_scat_z_max = DFT_Field_2D(dims.x,dims.y,0,dft.iwdim+1)
    h_scat_z_max = DFT_Field_2D(dims.x,dims.y,0,dft.iwdim+1)

    S_scat_DFT = np.zeros([6,dft.iwdim+1])
    S_scat_total = np.zeros([dft.iwdim+1])
   
    "Absorption"
    # xnormal
    e_abs_x_min = DFT_Field_2D(dims.y,dims.z,0,dft.iwdim+1)
    h_abs_x_min = DFT_Field_2D(dims.y,dims.z,0,dft.iwdim+1)
    e_abs_x_max = DFT_Field_2D(dims.y,dims.z,0,dft.iwdim+1)
    h_abs_x_max = DFT_Field_2D(dims.y,dims.z,0,dft.iwdim+1)

    #ynormal
    e_abs_y_min = fields.DFT_Field_2D(dims.x,dims.z,0,dft.iwdim+1)
    h_abs_y_min = fields.DFT_Field_2D(dims.x,dims.z,0,dft.iwdim+1)
    e_abs_y_max = fields.DFT_Field_2D(dims.x,dims.z,0,dft.iwdim+1)
    h_abs_y_max = fields.DFT_Field_2D(dims.x,dims.z,0,dft.iwdim+1)

    #znormal
    e_abs_z_min = fields.DFT_Field_2D(dims.x,dims.y,0,dft.iwdim+1)
    h_abs_z_min = fields.DFT_Field_2D(dims.x,dims.y,0,dft.iwdim+1)
    e_abs_z_max = fields.DFT_Field_2D(dims.x,dims.y,0,dft.iwdim+1)
    h_abs_z_max = fields.DFT_Field_2D(dims.x,dims.y,0,dft.iwdim+1)
    
    S_abs_DFT = np.zeros([6,dft.iwdim+1])
    S_abs_total = np.zeros([dft.iwdim+1])


####################################################
# Animation
####################################################

def graph(t):
    """This is a mess"""
# main graph is E(z,y, time snapshops), and a small graph of E(t) as center
    plt.clf() # close each time for new update graph/colormap
    ax = fig.add_axes([.05, .35, .2, .4])   

# 2d plot - several options, two examples below
#    img = ax.imshow(Ez)
    x,y =np.meshgrid(X,Y)
    #img = ax.contourf(x,y,np.transpose(ez[int(Ymax/2),:,:]*10))
    img = ax.contourf(x/nm,y/nm,np.transpose(np.abs(e.z[:,:,int(dims.y/2)])))
#    img = ax.contourf(x,y,np.transpose(np.round(ez[:,:,int(Ymax/2)],10)))
    cbar=plt.colorbar(img, ax=ax)
    cbar.set_label('$Ez$ (arb. units)')

# add labels to axes
    ax.set_xlabel('Grid Cells ($x$)')
    ax.set_ylabel('Grid Cells ($y$)')


    cc = plt.Circle((xc/nm, yc/nm), R/nm, color='r',fill=False)
    ax.set_aspect( 1 ) 
    ax.add_artist( cc ) 
    
    ax.hlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    ax.hlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    ax.vlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    ax.vlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')

    #TFSF
    ax.hlines(tfsf.y_min*ddx/nm,tfsf.x_min*ddx/nm,tfsf.x_max*ddx/nm, 'r')
    ax.hlines(tfsf.y_max*ddx/nm,tfsf.x_min*ddx/nm,tfsf.x_max*ddx/nm, 'r')
    ax.vlines(tfsf.x_min*ddx/nm,tfsf.y_min*ddx/nm,tfsf.y_max*ddx/nm, 'r')
    ax.vlines(tfsf.x_max*ddx/nm,tfsf.y_min*ddx/nm,tfsf.y_max*ddx/nm, 'r')



    if POINT_FLAG ==1:
            # incident field
        ax2 = fig.add_axes([.4, .75, .2, .2])
        ax2.plot(pulsemon_t,label='Source_mon')
        ax2.plot(ez_source_t,label='Ez_mon')

        ax2.set_ylim(-1.1,1.1)
        ax2.set_xlabel('Grid Cells ($y$)')
        ax2.set_ylabel('Fields')
        ax2.set_title('Incident fields')
        ax2.legend()

    if TFSF_FLAG ==1 or TFSF_FLAG ==2:
        # incident field
        ax2 = fig.add_axes([.4, .75, .2, .2])
        ax2.plot(Y/nm,ez_inc,label='Ez_inc')
        ax2.plot(Y/nm,hx_inc,label='Hx_inc')
        #ax2.set_ylim(-1.1,1.1)
        ax2.set_xlabel('Grid Cells ($y$)')
        ax2.set_ylabel('Fields')
        ax2.set_title('Incident fields')
        ax2.legend()

    ax01 = fig.add_axes([.4, .35, .2, .4])   

# 2d plot - several options, two examples below
#    img = ax.imshow(Ez)
    z,y =np.meshgrid(X,Y)
    #img = ax.contourf(x,y,np.transpose(ez[int(Ymax/2),:,:]*10))
    img = ax01.contourf(y/nm,z/nm,np.abs(e.z[int(dims.x/2),:,:]))
    cbar=plt.colorbar(img, ax=ax01)
    cbar.set_label('$Ez$ (arb. units)')

# add labels to axes
    ax01.set_xlabel('Grid Cells ($y$)')
    ax01.set_ylabel('Grid Cells ($z$)')
    
    cc = plt.Circle((yc/nm, zc/nm), R/nm, color='r',fill=False)
    ax01.set_aspect( 1 ) 
    ax01.add_artist( cc ) 

    #PML layers
    ax01.hlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    ax01.hlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    ax01.vlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    ax01.vlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')

    #TFSF
    ax01.hlines(tfsf.x_min*ddx/nm,tfsf.z_min*ddx/nm,tfsf.z_max*ddx/nm, 'r')
    ax01.hlines(tfsf.x_max*ddx/nm,tfsf.z_min*ddx/nm,tfsf.z_max*ddx/nm, 'r')
    ax01.vlines(tfsf.z_min*ddx/nm,tfsf.x_min*ddx/nm,tfsf.x_max*ddx/nm, 'r')
    ax01.vlines(tfsf.z_max*ddx/nm,tfsf.x_min*ddx/nm,tfsf.x_max*ddx/nm, 'r')

    if (MICRO_FLAG ==1):
        axmicro = fig.add_axes([.75, .75, .2, .2])   
        axmicro.plot(grid.k_grid(grid.n_kmax),f_plot,label='Fermi distribution')
        axmicro.set_xlabel('Grid Cells ($k$)')
        axmicro.set_ylabel('Fermi')
        axmicro.set_title('Fermi dist')
        axmicro.legend()


    ax02 = fig.add_axes([.75, .35, .2, .4])   

# 2d plot - several options, two examples below
#    img = ax.imshow(Ez)
    x,z =np.meshgrid(X,Z)
    #img = ax.contourf(x,y,np.transpose(ez[int(Ymax/2),:,:]*10))
    img = ax02.contourf(z/nm,x/nm,np.transpose(np.abs(e.z[:,int(dims.y/2),:])))
    cbar=plt.colorbar(img, ax=ax02)
    cbar.set_label('$Ez$ (arb. units)')

# add labels to axes
    ax02.set_xlabel('Grid Cells ($z$)')
    ax02.set_ylabel('Grid Cells ($x$)')

    cc = plt.Circle((zc/nm, xc/nm), R/nm, color='r',fill=False)
    ax02.set_aspect( 1 ) 
    ax02.add_artist( cc ) 

    ax02.hlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    ax02.hlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    ax02.vlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    ax02.vlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')

    #TFSF
    ax02.hlines(tfsf.x_min*ddx/nm,tfsf.z_min*ddx/nm,tfsf.z_max*ddx/nm, 'r')
    ax02.hlines(tfsf.x_max*ddx/nm,tfsf.z_min*ddx/nm,tfsf.z_max*ddx/nm, 'r')
    ax02.vlines(tfsf.z_min*ddx/nm,tfsf.x_min*ddx/nm,tfsf.x_max*ddx/nm, 'r')
    ax02.vlines(tfsf.z_max*ddx/nm,tfsf.x_min*ddx/nm,tfsf.x_max*ddx/nm, 'r')
# add title with current simulation time step
    ax.set_title("frame time {}".format(t))
    '''
# incident field
    ax2 = fig.add_axes([.05, .1, .2, .2])
    ax2.plot(Z,ez_inc,label='Ez_inc')
    ax2.plot(Z,hx_inc,label='Hx_inc')
    #ax2.set_ylim(-1.1,1.1)
    ax2.set_xlabel('Grid Cells ($y$)')
    ax2.set_ylabel('Fields')
    ax2.set_title('Incident fields')
    ax2.legend()
    '''
# plot calculated field shortly after source position
    axx = fig.add_axes([.05, .1, .2, .2])
    axx.plot(X/nm,np.abs(e.z[:,int(dims.y/2),int(dims.z/2)]),label='Ez_inc')
    #ax3.plot(Z,ez[:,ja,int(Zmax/2)]*10,label='Ez_inc')
    axx.set_xlabel('Grid Cells ($x$)')
    axx.set_ylabel('Ez field')
    axx.set_title('X profile')

    #ax2.plot(Z,hx_inc,label='Hx_inc')


# plot calculated field shortly after source position
    axy = fig.add_axes([.4, .1, .2, .2])
    axy.plot(Y/nm,np.abs(e.z[int(dims.x/2),:,int(dims.z/2)]),label='Ez_inc')
    axy.set_xlabel('Grid Cells ($y$)')
    axy.set_ylabel('Ez field')
    axy.set_title('Y profile')


# plot calculated field shortly after source position
    axz = fig.add_axes([.75, .1, .2, .2])
    axz.plot(Z/nm,np.abs(e.z[int(dims.x/2),int(dims.y/2),:]),label='Ez_inc')
    axz.set_xlabel('Grid Cells ($z$)')
    axz.set_ylabel('Ez field')
    axz.set_title('Z profile')
    #plt.tight_layout()
    #plt.savefig('Animation/frametime{}'.format(int(t/10)))
    path = 'Plots/Plots_nkmax{}_dx{}nm_dt{}as'.format(grid.n_kmax,int(ddx/nm),np.round(dt*1e18,2))
    if not os.path.exists(path):
        os.makedirs(path)
    save_name = path+'/animation_time{}.png'.format(int(time_step/cycle))
    plt.savefig(save_name)
    plt.pause(time_pause) # pause sensible value to watch what is happening

def graph_new(t):
    text_tstep.set_text('Time Step: ' + str(t))
    max = np.max(np.abs(e.z[:,:,int(dims.y/2)]))
    #print('max',max)
    for im in ims:
        im.set_clim(vmin=0, vmax=max)
    x,y =np.meshgrid(X,Y)
    ims[0].set_data(np.transpose(np.abs(e.z[:,:,int(dims.y/2)])))
    ims[1].set_data(np.transpose(np.abs(e.z[:,int(dims.y/2),:])))
    ims[2].set_data(np.transpose(np.abs(e.z[int(dims.y/2),:,:])))

    # 1d plots
    ax[0,2].set_ylim(-max,max)
    xcut.set_data(X/nm,np.abs(e.z[:,int(dims.y/2),int(dims.z/2)]))

    ax[1,2].set_ylim(-max,max)
    ycut.set_data(Y/nm,np.abs(e.z[int(dims.x/2),:,int(dims.z/2)]))
    
    ax[2,2].set_ylim(-max,max)
    zcut.set_data(Z/nm,np.abs(e.z[int(dims.x/2),int(dims.y/2),:]))

    #incident field
    ax[1,0].set_ylim(-max,max)
    incident_e.set_data(Y/nm,ez_inc)
    incident_h.set_data(Y/nm,hx_inc)

    max_p = np.max(np.abs(p.z[:,:,int(dims.y/2)]))
    for im in imp:
        im.set_clim(vmin=0, vmax=max_p)
    imp[0].set_data(np.transpose(np.abs(p.z[:,:,int(dims.z/2)])))
    imp[1].set_data(np.transpose(np.abs(p.z[:,int(dims.y/2),:])))
    imp[2].set_data(np.transpose(np.abs(p.z[int(dims.x/2),:,:])))

# main graph is E(z,y, time snapshops), and a small graph of E(t) as center
    #plt.clf() # close each time for new update graph/colormap
    #fig,ax = plt.subplots(3, 4, figsize=(10, 6))

# 2d plot - several options, two examples below
#    img = ax.imshow(Ez)
#     x,y =np.meshgrid(X,Y)
#     #img = ax.contourf(x,y,np.transpose(ez[int(Ymax/2),:,:]*10))
#     #ax[0,1].contourf(x/nm,y/nm,np.transpose(np.abs(e.z[:,:,int(dims.y/2)])))
#     axs = ax[0, 1]
#     img = axs.contourf(x/nm,y/nm,np.transpose(np.abs(e.z[:,:,int(dims.y/2)])))
# #    img = ax.contourf(x,y,np.transpose(np.round(ez[:,:,int(Ymax/2)],10)))
#     cbar=plt.colorbar(img, ax=axs)
#     cbar.set_label('$Ez$ (arb. units)')
    # ax = axs[0,1]
    # pcm = ax.pcolormesh(np.random.random((20, 20)))
    # fig.colorbar(pcm, ax=ax, shrink=0.6)

# # add labels to axes
#     axs.set_xlabel('Grid Cells ($x$)')
#     axs.set_ylabel('Grid Cells ($y$)')


    # cc = plt.Circle((xc/nm, yc/nm), R/nm, color='r',fill=False)
    # ax[0,1].set_aspect( 1 ) 
    # ax[0,1].add_artist( cc ) 
    
    # ax[0,1].hlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    # ax[0,1].hlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    # ax[0,1].vlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    # ax[0,1].vlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')

    # #TFSF
    # ax[0,1].hlines(tfsf.y_min*ddx/nm,tfsf.x_min*ddx/nm,tfsf.x_max*ddx/nm, 'r')
    # ax[0,1].hlines(tfsf.y_max*ddx/nm,tfsf.x_min*ddx/nm,tfsf.x_max*ddx/nm, 'r')
    # ax[0,1].vlines(tfsf.x_min*ddx/nm,tfsf.y_min*ddx/nm,tfsf.y_max*ddx/nm, 'r')
    # ax[0,1].vlines(tfsf.x_max*ddx/nm,tfsf.y_min*ddx/nm,tfsf.y_max*ddx/nm, 'r')



    # incident field
    # axs[1,0].plot(Y/nm,ez_inc,label='Ez_inc')
    # axs[1,0].plot(Y/nm,hx_inc,label='Hx_inc')
    # #ax2.set_ylim(-1.1,1.1)
    # axs[1,0].set_xlabel('Grid Cells ($y$)')
    # axs[1,0].set_ylabel('Fields')
    # axs[1,0].set_title('Incident fields')
    # axs[1,0].legend()
    '''
    ax01 = fig.add_axes([.4, .35, .2, .4])   

# 2d plot - several options, two examples below
#    img = ax.imshow(Ez)
    z,y =np.meshgrid(X,Y)
    #img = ax.contourf(x,y,np.transpose(ez[int(Ymax/2),:,:]*10))
    img = ax01.contourf(y/nm,z/nm,np.abs(e.z[int(dims.x/2),:,:]))
    cbar=plt.colorbar(img, ax=ax01)
    cbar.set_label('$Ez$ (arb. units)')

# add labels to axes
    ax01.set_xlabel('Grid Cells ($y$)')
    ax01.set_ylabel('Grid Cells ($z$)')
    
    cc = plt.Circle((yc/nm, zc/nm), R/nm, color='r',fill=False)
    ax01.set_aspect( 1 ) 
    ax01.add_artist( cc ) 

    #PML layers
    ax01.hlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    ax01.hlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    ax01.vlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    ax01.vlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')

    #TFSF
    ax01.hlines(tfsf.x_min*ddx/nm,tfsf.z_min*ddx/nm,tfsf.z_max*ddx/nm, 'r')
    ax01.hlines(tfsf.x_max*ddx/nm,tfsf.z_min*ddx/nm,tfsf.z_max*ddx/nm, 'r')
    ax01.vlines(tfsf.z_min*ddx/nm,tfsf.x_min*ddx/nm,tfsf.x_max*ddx/nm, 'r')
    ax01.vlines(tfsf.z_max*ddx/nm,tfsf.x_min*ddx/nm,tfsf.x_max*ddx/nm, 'r')

    axmicro = fig.add_axes([.75, .75, .2, .2])   
    axmicro.plot(grid.k_grid(grid.n_kmax),f_plot,label='Fermi distribution')
    axmicro.set_xlabel('Grid Cells ($k$)')
    axmicro.set_ylabel('Fermi')
    axmicro.set_title('Fermi dist')
    axmicro.legend()


    ax02 = fig.add_axes([.75, .35, .2, .4])   

# 2d plot - several options, two examples below
#    img = ax.imshow(Ez)
    x,z =np.meshgrid(X,Z)
    #img = ax.contourf(x,y,np.transpose(ez[int(Ymax/2),:,:]*10))
    img = ax02.contourf(z/nm,x/nm,np.transpose(np.abs(e.z[:,int(dims.y/2),:])))
    cbar=plt.colorbar(img, ax=ax02)
    cbar.set_label('$Ez$ (arb. units)')

# add labels to axes
    ax02.set_xlabel('Grid Cells ($z$)')
    ax02.set_ylabel('Grid Cells ($x$)')

    cc = plt.Circle((zc/nm, xc/nm), R/nm, color='r',fill=False)
    ax02.set_aspect( 1 ) 
    ax02.add_artist( cc ) 

    ax02.hlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    ax02.hlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    ax02.vlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    ax02.vlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')

    #TFSF
    ax02.hlines(tfsf.x_min*ddx/nm,tfsf.z_min*ddx/nm,tfsf.z_max*ddx/nm, 'r')
    ax02.hlines(tfsf.x_max*ddx/nm,tfsf.z_min*ddx/nm,tfsf.z_max*ddx/nm, 'r')
    ax02.vlines(tfsf.z_min*ddx/nm,tfsf.x_min*ddx/nm,tfsf.x_max*ddx/nm, 'r')
    ax02.vlines(tfsf.z_max*ddx/nm,tfsf.x_min*ddx/nm,tfsf.x_max*ddx/nm, 'r')
# add title with current simulation time step
    ax.set_title("frame time {}".format(t))
    '''
# # incident field
#     ax2 = fig.add_axes([.05, .1, .2, .2])
#     ax2.plot(Z,ez_inc,label='Ez_inc')
#     ax2.plot(Z,hx_inc,label='Hx_inc')
#     #ax2.set_ylim(-1.1,1.1)
#     ax2.set_xlabel('Grid Cells ($y$)')
#     ax2.set_ylabel('Fields')
#     ax2.set_title('Incident fields')
#     ax2.legend()
    '''
# plot calculated field shortly after source position
    axx = fig.add_axes([.05, .1, .2, .2])
    axx.plot(X/nm,np.abs(e.z[:,int(dims.y/2),int(dims.z/2)]),label='Ez_inc')
    #ax3.plot(Z,ez[:,ja,int(Zmax/2)]*10,label='Ez_inc')
    axx.set_xlabel('Grid Cells ($x$)')
    axx.set_ylabel('Ez field')
    axx.set_title('X profile')

    #ax2.plot(Z,hx_inc,label='Hx_inc')


# plot calculated field shortly after source position
    axy = fig.add_axes([.4, .1, .2, .2])
    axy.plot(Y/nm,np.abs(e.z[int(dims.x/2),:,int(dims.z/2)]),label='Ez_inc')
    axy.set_xlabel('Grid Cells ($y$)')
    axy.set_ylabel('Ez field')
    axy.set_title('Y profile')


# plot calculated field shortly after source position
    axz = fig.add_axes([.75, .1, .2, .2])
    axz.plot(Z/nm,np.abs(e.z[int(dims.x/2),int(dims.y/2),:]),label='Ez_inc')
    axz.set_xlabel('Grid Cells ($z$)')
    axz.set_ylabel('Ez field')
    axz.set_title('Z profile')
    #plt.tight_layout()
    #plt.savefig('Animation/frametime{}'.format(int(t/10)))
    path = 'Plots/Plots_nkmax{}_dx{}nm_dt{}as'.format(grid.n_kmax,int(ddx/nm),np.round(dt*1e18,2))
    if not os.path.exists(path):
        os.makedirs(path)
    save_name = path+'/animation_time{}.png'.format(int(time_step/cycle))
    '''
    #plt.savefig(save_name)
    plt.pause(time_pause) # pause sensible value to watch what is happening


#------------------------------------------------------------------------
"Start main FDTD loop"
#------------------------------------------------------------------------

#set PML parameters/en/latest/10_basic_tests.html
PML = pml.calculate_pml_params(dims, npml=8,TFSF_FLAG=TFSF_FLAG)

if MICRO_FLAG ==1:
    '''
    Microscopic initial conditions (Jonas)
    '''
    grid.print_grid_parameters()

    E_field = np.full(3,0.)

    #Initial Wigner distribution
    f_global = np.zeros((diameter,diameter,diameter, grid.n_kmax, grid.n_phimax, grid.n_thetamax), dtype=np.float64) 
    f_global[:,:,:] = np.reshape(initial_conditions.load_initial_conditions(INIT_FLAG, grid.E_F, grid.T_ele), (grid.n_kmax, grid.n_phimax, grid.n_thetamax))

    total_electron_number_start = pf.calculate_total_electron_number(f_global, ddx*1e9)
    print('Total elctron number at t=0: ', total_electron_number_start)

    '''
    Electron-phonon matrices
    '''
    in_scattering_matrix_abs, in_scattering_matrix_em, out_scattering_matrix_abs, out_scattering_matrix_em, kp_matrix = boltzmann.initialize_scattering_matrices()

    in_scattering_matrix_abs, in_scattering_matrix_em, out_scattering_matrix_abs, out_scattering_matrix_em, kp_matrix = boltzmann.calculate_scattering_matrices(in_scattering_matrix_abs, in_scattering_matrix_em, out_scattering_matrix_abs, out_scattering_matrix_em, kp_matrix, ELPHO_FLAG)

    ep_in_a, ep_in_e, ep_out_a, ep_out_e, kp = in_scattering_matrix_abs, in_scattering_matrix_em, out_scattering_matrix_abs, out_scattering_matrix_em, kp_matrix

    '''
    Light-matter matrices
    '''
    grad_matrix_k, grad_matrix_phi, grad_matrix_theta, unitvector_k, unitvector_phi, unitvector_theta, funcdet_phi, funcdet_theta = boltzmann.initialize_light_matter_matrices()

    grad_matrix_k, grad_matrix_phi, grad_matrix_theta, unitvector_k, unitvector_phi, unitvector_theta, funcdet_phi, funcdet_theta = boltzmann.calculate_light_matter_matrices(grad_matrix_k, grad_matrix_phi, grad_matrix_theta, unitvector_k, unitvector_phi, unitvector_theta, funcdet_phi, funcdet_theta, LM_FLAG, STENCIL_FLAG)

    lm_k, lm_p, lm_t, e_k, e_p, e_t, fd_p, fd_t = grad_matrix_k, grad_matrix_phi, grad_matrix_theta, unitvector_k, unitvector_phi, unitvector_theta, funcdet_phi, funcdet_theta

##############################################

# computation time
start = timeit.default_timer()    

"Object definition depending on flags"
if OBJECT_FLAG == 1: # if circle
    
    if MATERIAL_FLAG == 1: # Drude only
        ga, d1, d2, d3 = object.create_sphere_drude_eps(
            xc, yc, zc, R, nsub, ddx, dt, eps_in, eps_out, wp, gamma, ga, d1, d2, d3
        )
    if MATERIAL_FLAG == 2: # DrudeLorentz
        ga, d1, d2, d3 = object.create_sphere_drude_eps(
            xc, yc, zc, R, nsub, ddx, dt, eps_in, eps_out, wp, gamma, ga, d1, d2, d3
        )
        ga, l1, l2, l3 = object.create_sphere_lorentz(xc,yc,zc,R,nsub,ddx,dt,eps_in,eps_out,wl,gamma_l,delta_eps,ga,l1,l2,l3)
    
    if MATERIAL_FLAG == 3: # Etchegoin 
        ga, d1, d2, d3 = object.create_sphere_drude_eps(
            xc, yc, zc, R, nsub, ddx, dt, eps_in, eps_out, wp, gamma, ga, d1, d2, d3
        )
        f1_et1, f2_et1, f3_et1, f4_et1, f1_et2, f2_et2, f3_et2,f4_et2 = object.create_sphere_etch(xc,yc,zc,R,nsub,ddx,dt,c1,c2,w1,w2,gamma1,gamma2,f1_et1,f2_et1,f3_et1,f4_et1,f1_et2,f2_et2,f3_et2,f4_et2)

elif OBJECT_FLAG ==2 and TFSF_FLAG ==2:

    if MATERIAL_FLAG == 1: # Drude model
        ga, d1, d2, d3 = object.create_rectangle_PBC(dims,int(dims.y/2),int(dims.y/2+R/ddx),dt,eps_in,wp,gamma,ga,d1,d2,d3)
    
    if MATERIAL_FLAG == 2: # DrudeLorentz
        ga, d1, d2, d3 = object.create_rectangle_PBC(dims,int(dims.y/2),int(dims.y/2+R/ddx),dt,eps_in,wp,gamma,ga,d1,d2,d3)
        ga, l1, l2, l3 = object.create_rectangle_PBC_lorentz(dims,int(dims.y/2),int(dims.y/2+R/ddx),dt,eps_in,wl,gamma_l,delta_eps,ga,l1,l2,l3)
    
    if MATERIAL_FLAG == 3: # Etchegoin
        ga, d1, d2, d3 = object.create_rectangle_PBC(dims,int(dims.y/2),int(dims.y/2+R/ddx),dt,eps_in,wp,gamma,ga,d1,d2,d3)
        f1_et1, f2_et1, f3_et1, f4_et1, f1_et2, f2_et2, f3_et2, f4_et2 = object.create_rectangle_PBC_etch(dims,int(dims.y/2),int(dims.y/2+R/ddx),dt,c1,c2,w1,gamma1,w2,gamma2,f1_et1,f2_et1,f3_et1,f4_et1,f1_et2,f2_et2,f3_et2,f4_et2)
    
# else:
#     print('Something is wrong with the object definition')
#     exit()

# computation time 
intermediate = timeit.default_timer()    
print ("Time for object creation", intermediate - start)

# # Animation
# if(ANIMATION_FLAG ==1):
#     fig,axs = plt.subplots(3, 4, figsize=(16, 8))

# Animation
if(ANIMATION_FLAG ==1):

    #General setting
    plt.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots(3, 4, figsize=(20, 15))  # animation fig
    
    '2d field plots'
    ims = []
    x,z =np.meshgrid(X,Z)
    ims.append(ax[0,1].imshow(np.zeros((dims.x, dims.y))))
    ims.append(ax[1,1].imshow(np.zeros((dims.x, 1)), cmap='viridis',
                                interpolation='quadric', origin='lower'))
    ims.append(ax[2,1].imshow(np.zeros((1, 1)), cmap='viridis',
                                interpolation='quadric', origin='lower'))

    #Labels
    ax[0,1].set_xlabel('Grid Cells ($x$)')
    ax[0,1].set_ylabel('Grid Cells ($y$)')

    ax[1,1].set_xlabel('Grid Cells ($x$)')
    ax[1,1].set_ylabel('Grid Cells ($z$)')

    ax[2,1].set_xlabel('Grid Cells ($y$)')
    ax[2,1].set_ylabel('Grid Cells ($z$)')
    
    #2d fields    
    for im in ims:
        im.set_clim(vmin=0, vmax=10**5)
    ims[0].set_extent((0, dims.x, 0, dims.x))
    ims[1].set_extent((0, dims.x, 0, dims.x))
    ims[2].set_extent((0, dims.x, 0, dims.x))
    cbaxes = fig.add_axes([0.35, 0.95, 0.12, 0.01])
    cbar = plt.colorbar(ims[0], cax=cbaxes,orientation = 'horizontal')
    cbar.ax.set_title('Field [arb. units]')

    ax[0,0].set_axis_off()
    field_component = 'Ez'



    'Information'
    text_tstep = ax[0,0].annotate(
        'Time Step: 0', (0.5, 1), xycoords='axes fraction', va='center', ha='center', weight='bold')
    # boundary condition
    if TFSF_FLAG == 1: 
        plot_text = '\n' +'TFSF on'
    elif TFSF_FLAG == 2: 
        plot_text = '\n' +'Periodic boundary condtions'

    # object information
    if OBJECT_FLAG == 0:
        plot_text += '\nNo object implemented'
    elif OBJECT_FLAG == 1:
        plot_text += '\n Sphere of radius' +str(int(R/nm)) + 'nm implemented'
    if OBJECT_FLAG == 2:
        plot_text += '\nRectangle of thickness'+ str(int(R/nm)) + 'nm implemented'

    # material information
    if MATERIAL_FLAG == 0:
        plot_text += '\nNo object implemented'
    elif MATERIAL_FLAG == 1:
        plot_text += '\nDrude model implemented'
    if MATERIAL_FLAG == 3:
        plot_text += '\nEtchegoin model used'

    #pulse information
    plot_text += '\nPulse information\nAmplitude: ' + str(pulse.amplitude) \
        + '\n center energy: {:.4e}'.format(pulse.energy) \
        + '\nwavelength: {:06.4f} nm'.format(pulse.lam_0) \
        + '\ndx: {}'.format(ddx*1e9) + ' nm' \
        + '\ndt: {:.4e}'.format(dt) + ' s'

    plot_text += '\n\nGrid Dimensions: \nx: ' + \
        str(dims.x) + ', y: ' + str(dims.y) + ', z: ' + str(dims.z)
    #plot_text += '\nScatter Field: ' + str(scat_field)
    if PML_FLAG == 1:
        plot_text += ', PML: ' + str(npml)
    else:
        plot_text += ', PML: Off'
    
    ax[0,0].annotate(plot_text, (0, 0.9),
                    xycoords='axes fraction', va='top', ha='left',)


    ax[0,2].set_ylabel('Field [arb. units]')
    ax[0,2].set_xlabel('Grid Cells ($x$)')
    ax[0,2].set_ylabel('Ez field')
    ax[0,2].set_title('X profile')

    ax[1,2].set_ylabel('Field [arb. units]')
    ax[1,2].set_xlabel('Grid Cells ($y$)')
    ax[1,2].set_ylabel('Ez field')
    ax[1,2].set_title('Y profile')

    ax[2,2].set_ylabel('Field [arb. units]')
    ax[2,2].set_xlabel('Grid Cells ($z$)')
    ax[2,2].set_ylabel('Ez field')
    ax[2,2].set_title('Z profile')
    xcut, = ax[0,2].plot(X/nm,np.abs(e.z[:,int(dims.y/2),int(dims.z/2)]), label='X cut')
    ycut, = ax[1,2].plot(X/nm,np.abs(e.z[int(dims.x/2),:,int(dims.z/2)]), label='Y cut')
    zcut, = ax[2,2].plot(X/nm,np.abs(e.z[int(dims.x/2),int(dims.y/2),:]), label='Z cut')

    #incident field
    incident_e, = ax[1,0].plot(Y/nm,ez_inc,label='Ez_inc')
    incident_h, = ax[1,0].plot(Y/nm,hx_inc,label='Hx_inc')
    ax[1,0].set_xlabel('Grid Cells ($y$)')
    ax[1,0].set_ylabel('Fields')
    ax[1,0].set_title('Incident fields')
    ax[1,0].legend()


    # polarization

    imp = []

    imp.append(ax[0,3].imshow(np.zeros((dims.x, dims.y))))

    imp.append(ax[1,3].imshow(np.zeros((dims.x, 1)), cmap='viridis',
                                interpolation='quadric', origin='lower'))

    imp.append(ax[2,3].imshow(np.zeros((1, 1)), cmap='viridis',
                                interpolation='quadric', origin='lower'))

    #Labels
    ax[0,3].set_xlabel('Grid Cells ($x$)')
    ax[0,3].set_ylabel('Grid Cells ($y$)')

    ax[1,3].set_xlabel('Grid Cells ($x$)')
    ax[1,3].set_ylabel('Grid Cells ($z$)')

    ax[2,3].set_xlabel('Grid Cells ($y$)')
    ax[2,3].set_ylabel('Grid Cells ($z$)')
    
    
    for im in ims:
        im.set_clim(vmin=0, vmax=10**5)
    imp[0].set_extent((0, dims.x, 0, dims.x))
    imp[1].set_extent((0, dims.x, 0, dims.x))
    imp[2].set_extent((0, dims.x, 0, dims.x))
    cbaxes_p = fig.add_axes([0.75, 0.95, 0.12, 0.01])
    cbar_p = plt.colorbar(imp[0], cax=cbaxes_p,orientation = 'horizontal')
    cbar_p.ax.set_title('Polarization [arb. units]')



    'add lines and circles'

    cc = plt.Circle((xc/nm, yc/nm), R/nm, color='r',fill=False)
    #fields
    ax[0,1].set_aspect( 1 ) 
    ax[0,1].add_artist( cc ) 
    ax[1,1].set_aspect( 1 ) 
    #ax[1,1].add_artist( cc ) 
    ax[2,1].set_aspect( 1 ) 
    #ax[2,1].add_artist( cc ) 
    #polarization
    ax[0,3].set_aspect( 1 ) 
    #ax[0,3].add_artist( cc ) 
    ax[1,3].set_aspect( 1 ) 
    #ax[1,3].add_artist( cc ) 
    ax[2,3].set_aspect( 1 ) 
    #ax[2,3].add_artist( cc ) 

    #PML layers
    ax[0,1].hlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    ax[0,1].hlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    ax[0,1].vlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    ax[0,1].vlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')

    #TFSF
    ax[0,1].hlines(tfsf.x_min*ddx/nm,tfsf.z_min*ddx/nm,tfsf.z_max*ddx/nm, 'r')
    ax[0,1].hlines(tfsf.x_max*ddx/nm,tfsf.z_min*ddx/nm,tfsf.z_max*ddx/nm, 'r')
    ax[0,1].vlines(tfsf.z_min*ddx/nm,tfsf.x_min*ddx/nm,tfsf.x_max*ddx/nm, 'r')
    ax[0,1].vlines(tfsf.z_max*ddx/nm,tfsf.x_min*ddx/nm,tfsf.x_max*ddx/nm, 'r')

#------------------------------------------------------------------------
"Start of time loop"
#------------------------------------------------------------------------
print('t_step_micro = ', time_micro_factor*dt, ' fs')

for time_step in range(1,tsteps+1):
    
    #print('current timestep: ',time_step)
    "break statement for auto shutoff, in case that needed"
    # if(time_step > t0 and np.max(ez[npml:Xmax-npml,npml:Ymax-npml,npml:Zmax-npml])<1e-5):
    #     break

    "Update microscopic equation (Jonas)"
    if MICRO_FLAG == 1 and (time_step-1) % time_micro_factor == 0:
        print('current timestep: ',time_step)
        
        # store old current
        j_tmp = j   
        # update microscopic equations and compute macroscopic current density
        vor_solver = timeit.default_timer() 
        
        j, f_global = object.update_micro(ddx,time_micro_factor*dt,offset,diameter,j,f_global,e,xc,yc,zc,R, E_FIELD_FLAG, ep_in_a, ep_in_e, ep_out_a, ep_out_e, kp, lm_k, lm_p, lm_t, e_k, e_p, e_t, fd_p, fd_t)  

        nach_solver = timeit.default_timer() 
        print('Time for Micro code:', nach_solver-vor_solver)

        f_plot = f_global[int(diameter/2),int(diameter/2),int(diameter/2),:,0,int(grid.n_thetamax/2)]    #Wigner distribution on the positve k-space x axis

        p = object.update_polarization_micro(offset,diameter,dt,p,j,j_tmp)
    "Compute macroscopic polarization using auxilliary equation"
    if(OBJECT_FLAG!=0 and MICRO_FLAG ==0):
        if MATERIAL_FLAG == 1: # Drude only
            p_drude, p_tmp_drude = object.calculate_polarization(dims,sphere,ddx,p_drude,p_tmp_drude,e,d1,d2,d3,OBJECT_FLAG)
            p.x = p_drude.x
            p.y = p_drude.y 
            p.z = p_drude.z

        if MATERIAL_FLAG == 2: # DrudeLorentz: 
            p_drude, p_tmp_drude = object.calculate_polarization(dims,xc,yc,zc,R,ddx,p_drude,p_tmp_drude,e,d1,d2,d3,OBJECT_FLAG)
            p_lorentz, p_tmp_lorentz = object.calculate_polarization(dims,xc,yc,zc,R,ddx,p_lorentz,p_tmp_lorentz,e,l1,l2,l3,OBJECT_FLAG)
            p.x = p_drude.x+p_lorentz.x 
            p.y = p_drude.y+p_lorentz.y 
            p.z = p_drude.z+p_lorentz.z

        if MATERIAL_FLAG == 3:  # Etchegoin 
            p_drude, p_tmp_drude = object.calculate_polarization(dims,xc,yc,zc,R,ddx,p_drude,p_tmp_drude,e,d1,d2,d3,OBJECT_FLAG)
            p_et1, p_tmp_et1 = object.calculate_polarization_etch(dims,xc,yc,zc,R,ddx,p_et1,p_tmp_et1,e,e1,f1_et1,f2_et1,f3_et1,f4_et1,OBJECT_FLAG)
            p_et2, p_tmp_et2 = object.calculate_polarization_etch(dims,xc,yc,zc,R,ddx,p_et2,p_tmp_et2,e,e1,f1_et2,f2_et2,f3_et2,f4_et2,OBJECT_FLAG)
            p.x = p_drude.x + p_et1.x + p_et2.x
            p.y = p_drude.y + p_et1.y + p_et2.y
            p.z = p_drude.z + p_et1.z + p_et2.z
        
    "Update 1d buffer for plane wave"
    if(TFSF_FLAG==1 or TFSF_FLAG ==2):
        # Update incident electric field
        ez_inc = fdtd.calculate_ez_inc_field(dims.y,ez_inc,hx_inc)

        #Implementation of ABC
        ez_inc[0] = boundary_low.pop(0)
        boundary_low.append(ez_inc[1])
        ez_inc[dims.y - 1] = boundary_high.pop(0)
        boundary_high.append(ez_inc[dims.y - 2])

    "Update D Fields"
    # Standard FDTD update for D field
    d.x, id.x = fdtd.calculate_dx_field(dims,d.x,h,id.x,PML)
    d.y, id.y = fdtd.calculate_dy_field(dims,d.y,h,id.y,PML)
    d.z, id.z = fdtd.calculate_dz_field(dims,d.z,h,id.z,PML)

    # Implementation of PBC
    if TFSF_FLAG ==2:
        d.x, id.x = fdtd.calculate_dx_field_PBC(dims,d.x,h,id.x,PML)
        d.y, id.y = fdtd.calculate_dy_field_PBC(dims,d.y,h,id.y,PML)
        d.z, id.z = fdtd.calculate_dz_field_PBC(dims,d.z,h,id.z,PML)
        d.z = fdtd.calculate_dz_TFSF_PBC(dims,tfsf,d.z,hx_inc)
  
    # TFSF corrections for D fields
    if(TFSF_FLAG==1):
        d.y = fdtd.calculate_dy_inc_TFSF(tfsf,d.y,hx_inc)
        d.z = fdtd.calculate_dz_inc_TFSF(tfsf,d.z,hx_inc)

    "Update pulse value"
    pulse_tmp = pulse.update_value(time_step,dt)

    # Update pulse monitor
    if(TFSF_FLAG==1 or TFSF_FLAG ==2):
        pulse_t[time_step-1] = pulse_tmp
        ez_inc[tfsf.y_min-3] = pulse_tmp

    "Update E Fields"
    e, e1 = fdtd.calculate_e_fields(dims,e,e1,d,ga,p)

    'Update pulse monitors'
    if(POINT_FLAG==1):
        e.z[xs, ys, zs] += pulse_tmp
        pulsemon_t, ez_source_t = mnt.update_pulsemonitors(
            time_step, e.z, xs, ys, zs, pulse_tmp, pulsemon_t, ez_source_t
        )
    'Update 1d buffer for plane wave '
    if(TFSF_FLAG==1 or TFSF_FLAG ==2):
        hx_inc = fdtd.calculate_hx_inc_field(dims.y,hx_inc,ez_inc)
    
    'Update H fields'
    # Standard FDTD update for H field
    h.x,ih.x = fdtd.calculate_hx_field(dims,h.x,e,ih.x,PML)
    h.y,ih.y = fdtd.calculate_hy_field(dims,h.y,e,ih.y,PML)
    h.z,ih.z = fdtd.calculate_hz_field(dims,h.z,e,ih.z,PML)

    # Implementation of PBC
    if TFSF_FLAG ==2:
        h.x, ih.x = fdtd.calculate_hx_field_PBC(dims,h.x,e,ih.x,PML)
        h.y, ih.y = fdtd.calculate_hy_field_PBC(dims,h.y,e,ih.y,PML)
        h.z, ih.z = fdtd.calculate_hz_field_PBC(dims,h.z,e,ih.z,PML)
        h.x = fdtd.calculate_hx_TFSF_PBC(dims,tfsf,h.x,ez_inc)

    # TFSF corrections for D fields
    if(TFSF_FLAG==1):
        h.x = fdtd.calculate_hx_inc_TFSF(tfsf,h.x,ez_inc)
        h.y = fdtd.calculate_hy_inc_TFSF(tfsf,h.y,ez_inc)

    'Update Monitors'
    # 1D FFT monitors
    if(FFT_FLAG==1):
        #Something is wrong here!!
        mnt.update_1Dmonitors(time_step,loc_monitors,ex_mon,ey_mon,ez_mon,hx_mon,hy_mon,hz_mon,e,h)

    # Source monitors for TFSF
    SourceReDFT, SourceImDFT = mnt.DFT_incident_update(
        dft.omega, SourceReDFT, SourceImDFT, pulse_tmp, dft.iwdim, time_step
    )
    # 3D DFT monitors
    if DFT3D_FLAG == 1 and time_step > dft.tstart:
        e_dft, h_dft = mnt.DFT3D_update(
            e, h, e_dft, h_dft, dft.iwdim, dft.omega, time_step
        )
    # 2D DFT monitors 
    if DFT2D_FLAG == 1 and time_step > dft.tstart:
        ExReDFT_xnormal,ExImDFT_xnormal,EyReDFT_xnormal,EyImDFT_xnormal,EzReDFT_xnormal,EzImDFT_xnormal,\
        HxReDFT_xnormal,HxImDFT_xnormal,HyReDFT_xnormal,HyImDFT_xnormal,HzReDFT_xnormal,HzImDFT_xnormal,\
        ExReDFT_ynormal,ExImDFT_ynormal,EyReDFT_ynormal,EyImDFT_ynormal,EzReDFT_ynormal,EzImDFT_ynormal,\
        HxReDFT_ynormal,HxImDFT_ynormal,HyReDFT_ynormal,HyImDFT_ynormal,HzReDFT_ynormal,HzImDFT_ynormal,\
        ExReDFT_znormal,ExImDFT_znormal,EyReDFT_znormal,EyImDFT_znormal,EzReDFT_znormal,EzImDFT_znormal,\
        HxReDFT_znormal,HxImDFT_znormal,HyReDFT_znormal,HyImDFT_znormal,HzReDFT_znormal,HzImDFT_znormal = mnt.DFT2D_update(e,h,dft.iwdim,dft.omega,time_step,x_DFT,y_DFT,z_DFT,\
                ExReDFT_xnormal,ExImDFT_xnormal,EyReDFT_xnormal,EyImDFT_xnormal,EzReDFT_xnormal,EzImDFT_xnormal,\
                HxReDFT_xnormal,HxImDFT_xnormal,HyReDFT_xnormal,HyImDFT_xnormal,HzReDFT_xnormal,HzImDFT_xnormal,\
                ExReDFT_ynormal,ExImDFT_ynormal,EyReDFT_ynormal,EyImDFT_ynormal,EzReDFT_ynormal,EzImDFT_ynormal,\
                HxReDFT_ynormal,HxImDFT_ynormal,HyReDFT_ynormal,HyImDFT_ynormal,HzReDFT_ynormal,HzImDFT_ynormal,\
                ExReDFT_znormal,ExImDFT_znormal,EyReDFT_znormal,EyImDFT_znormal,EzReDFT_znormal,EzImDFT_znormal,\
                HxReDFT_znormal,HxImDFT_znormal,HyReDFT_znormal,HyImDFT_znormal,HzReDFT_znormal,HzImDFT_znormal)

    # Reflection and transmission for periodic boundary condition
    if(TFSF_FLAG ==2 and time_step > dft.tstart):
        e_ref, e_trans = mnt.DFT_ref_trans(e_ref, e_trans,e,y_ref,y_trans,dft.iwdim,dft.omega,time_step)

    # Scattering Cross Section
    if(CROSS_FLAG == 1 and time_step > dft.tstart):
        e_scat_x_min,e_scat_x_max,h_scat_x_min,h_scat_x_max,\
        e_scat_y_min,e_scat_y_max,h_scat_y_min,h_scat_y_max,\
        e_scat_z_min,e_scat_z_max,h_scat_z_min,h_scat_z_max = mnt.DFT_scat_update(
                                e_scat_x_min,e_scat_x_max,h_scat_x_min,h_scat_x_max,
                                e_scat_y_min,e_scat_y_max,h_scat_y_min,h_scat_y_max,
                                e_scat_z_min,e_scat_z_max,h_scat_z_min,h_scat_z_max,
                                e,h,scat,dft.iwdim,dft.omega,time_step)

    # Absorption Cross Section
    if(CROSS_FLAG == 1 and time_step > dft.tstart):
        e_abs_x_min,e_abs_x_max,h_abs_x_min,h_abs_x_max,\
        e_abs_y_min,e_abs_y_max,h_abs_y_min,h_abs_y_max,\
        e_abs_z_min,e_abs_z_max,h_abs_z_min,h_abs_z_max = mnt.DFT_abs_update(
                                e_abs_x_min,e_abs_x_max,h_abs_x_min,h_abs_x_max,
                                e_abs_y_min,e_abs_y_max,h_abs_y_min,h_abs_y_max,
                                e_abs_z_min,e_abs_z_max,h_abs_z_min,h_abs_z_max,
                                e,h,abs,dft.iwdim,dft.omega,time_step)

    'Animation '
    if(time_step % cycle == 0 and ANIMATION_FLAG == 1):
        graph_new(time_step)

# computation time        
stop = timeit.default_timer()    
print ("Time for full computation", stop - start)

total_electron_number_ende = pf.calculate_total_electron_number(f_global, ddx*1e9)
print('Total elctron number at t=t_max: ', total_electron_number_ende, ' Difference: ', total_electron_number_start - total_electron_number_ende)

plt.show()

#-------------------------------------------------------------------
# Data storage
#-------------------------------------------------------------------

if POINT_FLAG ==1:
    filename = 'point_object{}_r{}_dx{}_T{}_X{}_lam{}_wdt{}_nfreq{}_xs{}_ys{}_zs{}_npml{}_eps{}_final_new'\
        .format(OBJECT_FLAG,int(R/nm),int(ddx/nm),tsteps,dims.x,int(pulse.lam_0/nm),int(pulse.width),dft.iwdim,xs,ys,zs,npml,eps_in)

    time = np.arange(0,tsteps*dt,dt)
    pointmonitors = pd.DataFrame(
                    columns = ['time','pulse','field']
                    )

    pointmonitors["time"] = time
    pointmonitors["pulse"] = pulsemon_t
    pointmonitors["field"] = ez_source_t
    pointmonitors.to_pickle('Results/'+filename+'.pkl')


    custom_meta_content = {
        'object':OBJECT_FLAG,
        'sphere': [R,xc,yc,zc],
        'dx': ddx,
        'dt': dt,
        'timesteps': tsteps,
        'grid': dims.x,
        'eps_in':eps_in,
        'eps_out':eps_out,
        'pulsewidth': pulse.width,
        'delay': pulse.t0,
        'lambda':pulse.lam/nm,
        'nfreq':dft.iwdim,
        'source_loc':[xs,ys,zs],
        'npml':npml,
        'runtime':stop-start
    }

    custom_meta_key = 'pointsource.iot'
    table = pa.Table.from_pandas(pointmonitors)
    custom_meta_json = json.dumps(custom_meta_content)
    existing_meta = table.schema.metadata
    combined_meta = {
        custom_meta_key.encode() : custom_meta_json.encode(),
        **existing_meta
    }
    table = table.replace_schema_metadata(combined_meta)
    pq.write_table(table,'Results/'+filename+'.parquet', compression='GZIP')

#-------------------------------------------------------------------
# Data processing
#-------------------------------------------------------------------

"FT 1Dpoint monitors"
if FFT_FLAG==1:
    fft_res=20
    omega,ex_mon_om,ey_mon_om,ez_mon_om,hx_mon_om,hy_mon_om,hz_mon_om = mnt.fft_1Dmonitors(dt,tsteps,fft_res,n_mon,ex_mon,ey_mon,ez_mon,hx_mon,hy_mon,hz_mon)

if POINT_FLAG==1:
    fft_res=20
    omega_source, pulsemon_om, ez_source_om = mnt.fft_sourcemonitors(
        dt, tsteps, fft_res, pulsemon_t, ez_source_t
    )
    Mon = np.abs(ez_source_om)**2/np.max(np.abs(pulsemon_om)**2)
    Source = np.abs(pulsemon_om)**2/np.max(np.abs(pulsemon_om)**2)
    # numerical GFT - 
    GFT = ez_source_om/pulsemon_om # E_mon(w) / P(w)

    #analytical GFT
    GFT_an = omega_source**3/c**3/(6*np.pi)
    print("POINT FFT running")

if(TFSF_FLAG==1 or TFSF_FLAG ==2):
    fft_res=20
    omega_source,pulse_om = mnt.fft_source(dt,tsteps,fft_res,pulse_t)
    Source = np.abs(pulse_om)**2/np.max(np.abs(pulse_om)**2)
    print("Bandwidth calculated")

if(TFSF_FLAG == 2):
    reflection = e_ref.surface_magnitude()
    transmission = e_trans.surface_magnitude()
    plt.plot(dft.omega/dt*hbar/eC,reflection**2/(SourceReDFT**2+SourceImDFT**2)/dims.x**4)
    plt.plot(dft.omega/dt*hbar/eC,transmission**2/(SourceReDFT**2+SourceImDFT**2)/dims.x**4)
    plt.show()

#calculate Poynting vectors 
if(CROSS_FLAG == 1): 
    S_scat_DFT = mnt.update_scat_DFT(S_scat_DFT,dft.iwdim,scat,\
                    e_scat_x_min,e_scat_x_max,h_scat_x_min,h_scat_x_max,\
                    e_scat_y_min,e_scat_y_max,h_scat_y_min,h_scat_y_max,\
                    e_scat_z_min,e_scat_z_max,h_scat_z_min,h_scat_z_max)

    S_abs_DFT = mnt.update_abs_DFT(S_abs_DFT,dft.iwdim,abs,
                    e_abs_x_min,e_abs_x_max,h_abs_x_min,h_abs_x_max,\
                    e_abs_y_min,e_abs_y_max,h_abs_y_min,h_abs_y_max,\
                    e_abs_z_min,e_abs_z_max,h_abs_z_min,h_abs_z_max)

    S_scat_total = (S_scat_DFT[0,:]+S_scat_DFT[1,:]+S_scat_DFT[2,:]+S_scat_DFT[3,:]+S_scat_DFT[4,:]+S_scat_DFT[5,:])*ddx**2
    S_abs_total = (S_abs_DFT[0,:]+S_abs_DFT[1,:]+S_abs_DFT[2,:]+S_abs_DFT[3,:]+S_abs_DFT[4,:]+S_abs_DFT[5,:])*ddx**2


if CROSS_FLAG ==1:
    filename = 'TFSF_object{}_r{}_dx{}_T{}_X{}_lam{}_wdt{}_nfreq{}_npml{}_eps{}_tfsf{}'\
        .format(OBJECT_FLAG,int(R/nm),int(ddx/nm),tsteps,dims.x,int(pulse.lam_0/nm),int(pulse.width),dft.iwdim,npml,eps_in,tfsf_dist)

    crosssections = pd.DataFrame(
                    columns = ['omega','sigma_scat','sigma_abs']
                    )

    crosssections["omega"] = dft.omega
    crosssections["lambda"] = dft.lam
    crosssections["nu"] = dft.nu
    crosssections["sigma_scat"] = S_scat_total
    crosssections["sigma_abs"] = S_abs_total
    #crosssections["bandwidth"] = Source
    crosssections["source_re"] = SourceReDFT
    crosssections["source_im"] = SourceImDFT
    #only if we want a pure pickle file
    #crosssections.to_pickle('Results/'+filename+'.pkl')


    custom_meta_content = {
        'object':OBJECT_FLAG,
        'sphere': [R,xc,yc,zc],
        'dx': ddx,
        'dt': dt,
        'timesteps': tsteps,
        'grid': dims.x,
        'eps_in':eps_in,
        'eps_out':eps_out,
        'wp':wp,
        'gamma':gamma,
        'pulsewidth': pulse.width,
        'delay': pulse.t0,
        'lambda':pulse.lam_0/nm,
        'nfreq':dft.iwdim,
        'npml':npml,
        'tfsf_dist':tfsf_dist,
        'runtime':stop-start
    }

    custom_meta_key = 'TFSFsource.iot'
    table = pa.Table.from_pandas(crosssections)
    custom_meta_json = json.dumps(custom_meta_content)
    existing_meta = table.schema.metadata
    combined_meta = {
        custom_meta_key.encode() : custom_meta_json.encode(),
        **existing_meta
    }
    table = table.replace_schema_metadata(combined_meta)
    pq.write_table(table,'Results/'+filename+'.parquet', compression='GZIP')

if TFSF_FLAG ==2 and MICRO_FLAG ==0:
    filename = 'periodic_object{}_material{}_r{}_dx{}_T{}_X{}_lam{}_wdt{}_nfreq{}_npml{}_tfsf{}'\
        .format(OBJECT_FLAG,MATERIAL_FLAG,int(R/nm),int(ddx/nm),tsteps,dims.x,int(pulse.lam_0/nm),int(pulse.width),dft.iwdim,npml,tfsf_dist)

    periodic = pd.DataFrame(
                    columns = ['omega','ref','trans']
                    )

    periodic["omega"] = dft.omega
    periodic["lambda"] = dft.lam
    periodic["nu"] = dft.nu
    periodic["trans"] = transmission
    periodic["ref"] = reflection
    periodic["source_re"] = SourceReDFT
    periodic["source_im"] = SourceImDFT
    #only if we want a pure pickle file
    #periodic.to_pickle('Results/'+filename+'.pkl')


    custom_meta_content = {
        'object': OBJECT_FLAG,
        'sphere': [R,xc,yc,zc],
        'dx': ddx,
        'dt': dt,
        'timesteps': tsteps,
        'grid': dims.x,
        'eps_in':eps_in,
        'eps_out':eps_out,
        'wp':wp,
        'gamma':gamma,
        'pulsewidth': pulse.width,
        'delay': pulse.t0,
        'lambda':pulse.lam_0/nm,
        'nfreq':dft.iwdim,
        'npml':npml,
        'tfsf_dist':tfsf_dist,
        'runtime':stop-start
    }

    custom_meta_key = 'periodicsource.iot'
    table = pa.Table.from_pandas(periodic)
    custom_meta_json = json.dumps(custom_meta_content)
    existing_meta = table.schema.metadata
    combined_meta = {
        custom_meta_key.encode() : custom_meta_json.encode(),
        **existing_meta
    }
    table = table.replace_schema_metadata(combined_meta)
    pq.write_table(table,'Results/'+filename+'.parquet', compression='GZIP')

if TFSF_FLAG ==2 and MICRO_FLAG ==1:
    filename = 'periodic_micro_object{}_material{}_r{}_dx{}_T{}_X{}_lam{}_wdt{}_nfreq{}_npml{}_tfsf{}'\
        .format(OBJECT_FLAG,MATERIAL_FLAG,int(R/nm),int(ddx/nm),tsteps,dims.x,int(pulse.lam_0/nm),int(pulse.width),dft.iwdim,npml,tfsf_dist)

    periodic = pd.DataFrame(
                    columns = ['omega','ref','trans']
                    )

    periodic["omega"] = dft.omega
    periodic["lambda"] = dft.lam
    periodic["nu"] = dft.nu
    periodic["trans"] = transmission
    periodic["ref"] = reflection
    periodic["source_re"] = SourceReDFT
    periodic["source_im"] = SourceImDFT
    #only if we want a pure pickle file
    #periodic.to_pickle('Results/'+filename+'.pkl')


    custom_meta_content = {
        'object': OBJECT_FLAG,
        'sphere': [R,xc,yc,zc],
        'dx': ddx,
        'dt': dt,
        'timesteps': tsteps,
        'grid': dims.x,
        'eps_in':eps_in,
        'eps_out':eps_out,
        'pulsewidth': pulse.width,
        'delay': pulse.t0,
        'lambda':pulse.lam_0/nm,
        'nfreq':dft.iwdim,
        'npml':npml,
        'tfsf_dist':tfsf_dist,
        'runtime':stop-start
    }

    custom_meta_key = 'periodicsource.iot'
    table = pa.Table.from_pandas(periodic)
    custom_meta_json = json.dumps(custom_meta_content)
    existing_meta = table.schema.metadata
    combined_meta = {
        custom_meta_key.encode() : custom_meta_json.encode(),
        **existing_meta
    }
    table = table.replace_schema_metadata(combined_meta)
    pq.write_table(table,'Results/'+filename+'.parquet', compression='GZIP')

# "Scattering cross section"
# r = R*1e6  #radius in microns
# geometric_cross_section = np.pi * r**2

# #Johnson and Christy data
# name = "../materialdata/Johnson_Au.txt"
# au = np.genfromtxt(name, delimiter='\t')
# NNN = len(au)//2 # data is stacked so need to rearrange
# au_lam = au[1:NNN,0]
# au_mre = au[1:NNN,1]
# au_mim = au[NNN+1:,1]
# #calculate JohnsonChristy solution
# x = 2*np.pi*r/au_lam;m = au_mre - 1.0j * au_mim
# qext, qsca, qback, g = miepython.mie(m,x)
# scatt   = qsca * geometric_cross_section
# absorb  = (qext - qsca) * geometric_cross_section

# #Mie solution
# lam_mie = dft.lam*1e6
# # data is stacked so need to rearrange
# eps = eps_in - wp**2/(dft.omega/dt*(dft.omega/dt + 1j*gamma))
# epsR = np.real(eps)
# epsI = np.imag(eps)
# au_mre_mie = np.sqrt((np.sqrt(epsR**2+epsI**2)+epsR)/2)
# au_mim_mie = np.sqrt((np.sqrt(epsR**2+epsI**2)-epsR)/2)
# #calculate Mie solution
# x_mie = 2*np.pi*r/lam_mie;m_mie = au_mre_mie - 1.0j * au_mim_mie
# qext_mie, qsca_mie, qback_mie, g = miepython.mie(m_mie,x_mie)
# scatt_mie   = qsca_mie * geometric_cross_section
# absorb_mie  = (qext_mie - qsca_mie) * geometric_cross_section


# if(CROSS_FLAG==1):
#     "Plot time point like frequency monitors"
#     fig,ax = plt.subplots(figsize=(9,6))

#     #individual monitors
#     # for i in range(6):
#     #     ax.plot(dft.lam*1e9,S_scat_DFT[i,:]*ddx**2*1e12/(0.5*(SourceReDFT**2+SourceImDFT**2)),label="%.0f"% i)

#     ax.plot(dft.lam*1e9,S_scat_total*1e12/(0.5*(SourceReDFT**2+SourceImDFT**2)),label="FDTD") # FDTD solution
#     ax.plot(au_lam*1000,scatt,label="Johnson") #MiePython using Johnson
#     ax.plot(lam_mie*1000,scatt_mie,label='MiePython') #MiePython using Drude
#     #ax.plot(2*np.pi*c/omega_source*1e9,Source,label='Bandwidth') #Bandwidth
#     ax.legend()
#     ax.set_xlabel("$\lambda$ [nm]")
#     ax.set_ylabel("Cross Section (microns$^2$)")
#     ax.set_title('Scattering Cross Section')
#     ax.set_xlim(400,800)
#     plt.show()

#     "Plot absorption cross section"
#     fig,ax = plt.subplots(figsize=(9,6))

#     #individual monitors
#     # for i in range(6):
#     #     ax.plot(dft.lam*1e9,S_abs_DFT[i,:]*ddx**2*1e12/(0.5*(SourceReDFT**2+SourceImDFT**2)),label="%.0f"% i)

#     ax.plot(dft.lam*1e9,S_abs_total*1e12/(0.5*(SourceReDFT**2+SourceImDFT**2)),label="FDTD") # FDTD solution
#     ax.plot(au_lam*1000,absorb,label="Johnson") #MiePython using Johnson
#     ax.plot(lam_mie*1000,absorb_mie,label='MiePython') #MiePython using Drude
#     #ax.plot(2*np.pi*c/omega_source*1e9,Source,label='Bandwidth') #Bandwidth
#     ax.legend()
#     ax.set_xlabel("$\lambda$ [nm]")
#     ax.set_ylabel("Cross Section (microns$^2$)")
#     ax.set_xlim(400,800)
#     ax.set_title('Absorption Cross Section')
#     plt.show()


if(FFT_FLAG==1):
    "Plot frequency dependent 1D monitor"
    fig = plt.figure(figsize=(14, 6))
    for i in range(n_mon):
        ax = fig.add_subplot(2, 3,i+1)
        ax.plot(hbar*omega/eC,np.abs(ez_mon_om[i,:]))
        ax.set_title("Pos: x = {0}, y= {1} ,z= {2}".format(loc_monitors[i][0],loc_monitors[i][1],loc_monitors[i][2]))
        ax.set_xlim(0,4)
        ax.set_xlabel('$\hbar\omega$ [eV]')
        ax.set_ylabel('$|E_z(\omega)|$')
    #plt.subplots_adjust(bottom=0.05, left=0.05)
    plt.tight_layout()
    plt.show()

if(POINT_FLAG==1):

    "Plot frequency dependent 1D monitor"
    fig,ax = plt.subplots(figsize=(9, 6))

    ax.plot(hbar*omega_source/eC,np.imag(GFT)*1/ddx**3*1,label='GFT FDTD')
    ax.plot(hbar*omega_source/eC,GFT_an,label='GFT freespace')
    ax.plot(hbar*omega_source/eC,Mon*1e21,label='bandwidth')
    ax.set_title('Free space Green function')
    ax.set_xlim(2,3)
    ax.set_ylim((0, 1e20))
    ax.set_xlabel('$\hbar\omega$ [eV]')
    ax.set_ylabel('Green fct [m$^{-3}$]')
    ax.legend()
    plt.tight_layout()
    #plt.savefig('Results/greenfct_benchmark.pdf')
    plt.show()



    "Plot frequency dependent 1D monitor"
    fig,ax = plt.subplots(2,2,figsize=(9, 6))

    #time dependent  values
    ax[0,0].plot(pulsemon_t,label='pulse')
    ax[0,0].plot(ez_source_t,label='field')
    ax[0,0].set_title('Time domain')
    ax[0,0].set_xlabel('Timestep')
    ax[0,0].set_ylabel('Electric field')
    ax[0,0].legend()

    # #frequency domain
    # ax[0,1].plot(hbar*omega_source/eC,np.real(pulsemon_om),label='pulse (real)')
    # ax[0,1].plot(hbar*omega_source/eC,np.real(ez_source_om),label='field (real)')
    # ax[0,1].plot(hbar*omega_source/eC,Mon*20000,label='bandwidth')
    # ax[0,1].set_title('Freq domain, real part')
    # ax[0,1].set_xlim(0,4)
    # ax[0,1].set_xlabel('$\hbar\omega$ [eV]')
    # ax[0,1].set_ylabel('Electric field')
    # ax[0,1].legend()

    #relative values to incident field
    ax[1,0].plot(hbar*omega_source/eC,np.imag(pulsemon_om),label='pulse (imag)')
    ax[1,0].plot(hbar*omega_source/eC,np.imag(ez_source_om),label='field (imag)')
    ax[1,0].plot(hbar*omega_source/eC,Mon*20000,label='bandwidth')
    ax[1,0].set_title('Freq domain, imag part')
    ax[1,0].set_xlim(1.5,3.5)
    ax[1,0].set_xlabel('$\hbar\omega$ [eV]')
    ax[1,0].set_ylabel('Electric field')
    ax[1,0].legend()

    #Green function
    ax[0,1].plot(hbar*omega_source/eC,np.imag(GFT)*1/ddx**3*1,label='GFT FDTD')
    ax[0,1].plot(hbar*omega_source/eC,GFT_an,label='GFT freespace')
    ax[0,1].plot(hbar*omega_source/eC,Mon*1e22,label='bandwidth')
    ax[0,1].set_title('Green function')
    ax[0,1].set_xlim(2,3)
    ax[0,1].set_ylim((0, 6e21))
    ax[0,1].set_xlabel('$\hbar\omega$ [eV]')
    ax[0,1].set_ylabel('Green fct [m$^{-3}$]')

    ax[0,1].legend()
 
    #LDOS
    ax[1,1].plot(hbar*omega_source/eC,np.imag(GFT)/ddx**3/GFT_an,label='LDOS')
    #ax[1,1].plot(hbar*omega_source/eC,Mon*1e21,label='bandwidth')
    ax[1,1].set_title('LDOS/Purcell factor, not sure')
    ax[1,1].set_xlim(2,3)
    ax[1,1].set_ylim((0, 50))
    ax[1,1].set_xlabel('$\hbar\omega$ [eV]')

    ax[1,1].legend()
    plt.tight_layout()
    #plt.savefig('Results/greenfct_ldos.pdf')
    plt.show()



"Plot DFT solution"
# "Plot DFT Fields if graph flag is set"
# if (DFT_FLAG == 1):

# # test one of these - can adjust as apporpriate
#     EzReDFT2=np.abs(EzReDFT[1,:,:,:]+1j*EzImDFT[1,:,:,:])
#     S = (EzReDFT2)/np.max(EzReDFT2)

#     fig = plt.figure(figsize=(8,8))  
#     ax = fig.add_axes([.16, .16, .6, .6])
# # quick example of second one - add and set w as required from LDOS picture
#     cf=ax.contourf(S[npml:Xmax-npml,jb+2,npml:Zmax-npml])     
#     fig.colorbar(cf, ax=ax, shrink=0.8)
#     ax.set_aspect('equal')
#     im=ax.contour(S[npml:Xmax-npml,jb+2,npml:Zmax-npml]) # add more contour line
# #    CS2 = plt.contour(im, levels=im.levels[::2],colors='r')
#     ax.set_xlabel('Grid Cells ($y$)')
#     ax.set_ylabel('Grid Cells ($x$)')   
#     ax.set_title('Example DFT plot of $E_x(x,y,\omega_2)$',y=1.05)

# # dielectric box
#     plt.vlines(X1,Y1,Y2,colors='b',lw=2)
# #set_linewidth(2)
#     plt.vlines(X2,Y1,Y2,colors='b',lw=2)
#     plt.hlines(Y1,X1,X2,colors='b',lw=2)
#     plt.hlines(Y2,X1,X2,colors='b',lw=2)

# PML box
    # plt.vlines(npml,npml,Ymax-npml,colors='r',lw=2)
    # plt.vlines(Xmax-npml,npml,Ymax-npml,colors='r',lw=2)
    # plt.hlines(npml,npml,Xmax-npml,colors='r',lw=2)
    # plt.hlines(Ymax-npml,npml,Xmax-npml,colors='r',lw=2)
    # plt.show()


"1D monitors"
# fig = plt.figure(figsize=(14, 6))
# for i in range(n_mon):
#     ax = fig.add_subplot(2, 3,i+1)
#     ax.plot(ez_mon[i,:])
# plt.subplots_adjust(bottom=0.05, left=0.05)
# plt.show()

# "FFT"
# fft_res = 20 # pads with zeros for better resolution
# N = tsteps*fft_res
# ex_mon_om = np.zeros([n_mon,int(N/2+1)])
# ey_mon_om = np.zeros([n_mon,int(N/2+1)])
# ez_mon_om = np.zeros([n_mon,int(N/2+1)])
# hx_mon_om = np.zeros([n_mon,int(N/2+1)])
# hy_mon_om = np.zeros([n_mon,int(N/2+1)])
# hz_mon_om = np.zeros([n_mon,int(N/2+1)])
# for i in range(n_mon):
#     ez_mon_om[i,:] = rfft(ez_mon[i,:],n=N)
# nu = rfftfreq(N, dt)
# omega = 2*np.pi*nu/tera

# fig = plt.figure(figsize=(14, 6))
# for i in range(n_mon):
#     ax = fig.add_subplot(2, 3,i+1)
#     ax.plot(np.abs(ez_mon_om[i,:]))
#     ax.set_xlim(0,250)
# plt.subplots_adjust(bottom=0.05, left=0.05)
# plt.show()
