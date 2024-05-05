import numpy as np
from collections import namedtuple

from modules.user_input import get_user_input
from modules.classes import Pulse, DFT, Field
from modules.fundamentals import c, Sc, nm, Dimensions, box, Sphere
import modules.fdtd as fdtd
from modules.classes import DFT_Field_3D, DFT_Field_2D
import modules.pml as pml
import modules.object as object


def to_namedtuple(classname="argparse_to_namedtuple", **kwargs):
    return namedtuple(classname, tuple(kwargs))(**kwargs)


def main():

    arguments = get_user_input()
    args = to_namedtuple(**vars(arguments))

    # constant parameters
    ddx = args.ddx * nm
    dt = ddx / c * Sc
    dims = Dimensions(x=args.dim, y=args.dim, z=args.dim)
    tsteps = args.tsteps

    # Declaring namedtuple()
    Constants = namedtuple("Constants", ["ddx", "dt", "tsteps"])

    # Adding values
    constants = Constants(ddx, dt, tsteps)

    npml = args.npml
    tfsf_dist = npml + 4  # TFSF distance from computational boundary

    radius = args.radius * nm  # radius of sphere
    eps_out = args.eps_out

    tfsf, scat, abs, length, array, sphere, offset, diameter = define_boxes_arrays(
        tfsf_dist, dims, ddx, radius
    )

    # setup
    eps_in = 1.0  # this has to be changed!!
    pulse = define_pulse(args, dt, ddx, eps_in)
    dft = DFT(dt=dt, iwdim=100, pulse_spread=pulse.spread, e_min=1.9, e_max=3.2)

    ####################################################
    # Monitors
    ####################################################
    # comments: for more speedy coding, one could only comment in the ones that one uses.

    # DFT Source monitors for 1d buffer
    SourceReDFT = np.zeros([dft.iwdim + 1], float)
    SourceImDFT = np.zeros([dft.iwdim + 1], float)

    # 3D DFT arrays
    if args.dft3d:
        e_dft = DFT_Field_3D(dims, 0, dft.iwdim + 1)
        h_dft = DFT_Field_3D(dims, 0, dft.iwdim + 1)

    # 2D DFT arrays
    if args.dft2d:
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

    if args.boundary == "PBC":

        # spatial position equal to absorption box for simplicity
        y_ref = scat.y_min
        y_trans = abs.y_max

        # ynormal
        e_ref = DFT_Field_2D(dims.x, dims.z, 0, dft.iwdim)
        e_trans = DFT_Field_2D(dims.x, dims.z, 0, dft.iwdim)

    "Scattering and absorption arrays"
    # might reduce the size of the array as I only store the monitor and not the value in the adjacent region/PML
    if args.cross:

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
    if args.fft:
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
        dims, npml=8, TFSF_FLAG=args.boundary
    )  # set PML parameters/en/latest/10_basic_tests.html

    # setup()
    ga, *args_object = setup_object(constants, args, dims, sphere)

    # if args.o != "Empty":
    #     if args.m == "Drude":  # Drude only
    #         update_polarization = update_polarization_drude

    #     elif args.m == "DrudeLorentz":
    #         update_polarization = update_polarization_drudelorentz

    #     elif args.m == "Etchegoin":
    #         update_polarization = update_polarization_etchegoin

    # print_parameters()

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

    # 1D plane wave buffer for TFSF simulation
    if args.source == "TFSF":
        ez_inc = np.zeros(dims.y, float)  # 1d buffer field for plane wave
        hx_inc = np.zeros(dims.y, float)  # 1d buffer field for plane wave
        boundary_low = [0, 0]  # lower absorbing boundary condition
        boundary_high = [0, 0]  # upper absorbing boundary condition

    if args.source == "Point":
        pulse_t = np.zeros(tsteps, float)  # pulse monitor for FFT

    ########
    # time loop
    ########

    for time_step in range(1, tsteps + 1):

        # update 1d electric field buffer
        if not args.boundary == "None":
            ez_inc = update_buffer_ez_inc(
                ez_inc, hx_inc, dims, boundary_low, boundary_high
            )

        # field updates
        d, id = fdtd.D_update(args, dims, d, h, id, PML, tfsf, hx_inc)
        e, e1 = fdtd.calculate_e_fields(dims, e, e1, d, ga, p)
        h, ih = fdtd.H_update(args, dims, h, ih, e, PML, tfsf, ez_inc)

        # pulse updating
        pulse_tmp = pulse.update_value(time_step, dt)

        # Update 1d buffer for plane wave - magnetic field
        if not args.boundary == "None":
            hx_inc = fdtd.calculate_hx_inc_field(dims.y, hx_inc, ez_inc)

    #     update buffer fields
    #     update_physical fields
    #     update pulse
    #     update monitors
    #     animate()
    # if time_step % cycle == 0 and FLAG.ANIMATION == 1:
    #     # plots.animate(time_step,text_tstep,e,dims,ims,array,ax,xcut,ycut,zcut,incident_e,ez_inc,hx_inc,incident_h,p,imp,time_pause)
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

    # process_data()
    print(args.pulse)


def update_buffer_ez_inc(ez_inc, hx_inc, dims, boundary_low, boundary_high):
    ez_inc = fdtd.calculate_ez_inc_field(dims.y, ez_inc, hx_inc)

    # Implementation of ABC
    ez_inc[0] = boundary_low.pop(0)
    boundary_low.append(ez_inc[1])
    ez_inc[dims.y - 1] = boundary_high.pop(0)
    boundary_high.append(ez_inc[dims.y - 2])

    return ez_inc


def define_pulse(args, dt, ddx, eps_in):
    if args.pulse == "Optical":
        return Pulse(width=2, delay=5, energy=2.3, dt=dt, ddx=ddx, eps_in=eps_in)
    elif args.pulse == "THz":
        return Pulse(
            width=1.5 * 1e3, delay=1, energy=4 * 1e-3, dt=dt, ddx=ddx, eps_in=eps_in
        )
    else:
        raise ValueError("You have chosen a field that is not implemented.")


def setup():
    print("test")


def setup_polarization():
    return "Polarization missing"


def setup_object(constants, args, dims, sphere):

    if args.object == "Sphere":

        if args.material == "Drude":

            ga = Field(dims, 1)  # inverse permittivity =1/eps
            d1 = Field(dims, 0)  # prefactor in auxilliary equation
            d2 = Field(dims, 0)  # prefactor in auxilliary equation
            d3 = Field(dims, 0)  # prefactor in auxilliary equation

            ga, d1, d2, d3 = object.create_sphere_drude_eps(
                constants, args, sphere, ga, d1, d2, d3
            )

            return ga, d1, d2, d3

        if args.material == "DrudeLorentz":

            ga = Field(dims, 1)  # inverse permittivity =1/eps
            d1 = Field(dims, 0)  # prefactor in auxilliary equation
            d2 = Field(dims, 0)  # prefactor in auxilliary equation
            d3 = Field(dims, 0)  # prefactor in auxilliary equation
            l1 = Field(dims, 0)  # prefactor in auxilliary equation
            l2 = Field(dims, 0)  # prefactor in auxilliary equation
            l3 = Field(dims, 0)  # prefactor in auxilliary equation

            ga, d1, d2, d3 = object.create_sphere_drude_eps(
                constants, args, sphere, ga, d1, d2, d3
            )
            # Now we are calculating ga twicem should be optimized.
            ga, l1, l2, l3 = object.create_sphere_lorentz(
                constants, args, sphere, ga, l1, l2, l3
            )

            return ga, d1, d2, d3, l1, l2, l3

        if args.material == "Etchegoin":

            ga = Field(dims, 1)  # inverse permittivity =1/eps
            d1 = Field(dims, 0)  # prefactor in auxilliary equation
            d2 = Field(dims, 0)  # prefactor in auxilliary equation
            d3 = Field(dims, 0)  # prefactor in auxilliary equation

            f1_et1 = Field(dims, 0)  # prefactor in auxilliary equation
            f2_et1 = Field(dims, 0)  # prefactor in auxilliary equation
            f3_et1 = Field(dims, 0)  # prefactor in auxilliary equation
            f4_et1 = Field(dims, 0)  # prefactor in auxilliary equation
            f1_et2 = Field(dims, 0)  # prefactor in auxilliary equation
            f2_et2 = Field(dims, 0)  # prefactor in auxilliary equation
            f3_et2 = Field(dims, 0)  # prefactor in auxilliary equation
            f4_et2 = Field(dims, 0)  # prefactor in auxilliary equation

            ga, d1, d2, d3 = object.create_sphere_drude_eps(
                constants, args, sphere, ga, d1, d2, d3
            )
            f1_et1, f2_et1, f3_et1, f4_et1, f1_et2, f2_et2, f3_et2, f4_et2 = (
                object.create_sphere_etch(
                    constants,
                    args,
                    sphere,
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
            return (
                ga,
                d1,
                d2,
                d3,
                f1_et1,
                f2_et1,
                f3_et1,
                f4_et1,
                f1_et2,
                f2_et2,
                f3_et2,
                f4_et2,
            )

    elif args.object == "Rectangle" and args.boundary == "PBC":

        y_low = int(dims.y / 2)
        y_high = int(dims.y / 2 + sphere.R / constants.ddx)

        if args.material == "Drude":

            ga = Field(dims, 1)  # inverse permittivity =1/eps
            d1 = Field(dims, 0)  # prefactor in auxilliary equation
            d2 = Field(dims, 0)  # prefactor in auxilliary equation
            d3 = Field(dims, 0)  # prefactor in auxilliary equation

            ga, d1, d2, d3 = object.create_rectangle_drude_PBC(
                constants, args, dims, y_low, y_high, ga, d1, d2, d3
            )  # should derive a different quantity for the diameter instead of using sphere radius
            return ga, d1, d2, d3

        if args.material == "DrudeLorentz":

            ga = Field(dims, 1)  # inverse permittivity =1/eps
            d1 = Field(dims, 0)  # prefactor in auxilliary equation
            d2 = Field(dims, 0)  # prefactor in auxilliary equation
            d3 = Field(dims, 0)  # prefactor in auxilliary equation
            l1 = Field(dims, 0)  # prefactor in auxilliary equation
            l2 = Field(dims, 0)  # prefactor in auxilliary equation
            l3 = Field(dims, 0)  # prefactor in auxilliary equation

            ga, d1, d2, d3 = object.create_rectangle_drude_PBC(
                constants, args, dims, y_low, y_high, ga, d1, d2, d3
            )
            ga, l1, l2, l3 = object.create_rectangle_PBC_lorentz(
                dims,
                y_low,
                y_high,
                dt,
                ga,
                l1,
                l2,
                l3,
            )
            return ga, list(d1, d2, d3, l1, l2, l3)

        if args.material == "Etchegoin":

            ga = Field(dims, 1)  # inverse permittivity =1/eps
            d1 = Field(dims, 0)  # prefactor in auxilliary equation
            d2 = Field(dims, 0)  # prefactor in auxilliary equation
            d3 = Field(dims, 0)  # prefactor in auxilliary equation

            f1_et1 = Field(dims, 0)  # prefactor in auxilliary equation
            f2_et1 = Field(dims, 0)  # prefactor in auxilliary equation
            f3_et1 = Field(dims, 0)  # prefactor in auxilliary equation
            f4_et1 = Field(dims, 0)  # prefactor in auxilliary equation
            f1_et2 = Field(dims, 0)  # prefactor in auxilliary equation
            f2_et2 = Field(dims, 0)  # prefactor in auxilliary equation
            f3_et2 = Field(dims, 0)  # prefactor in auxilliary equation
            f4_et2 = Field(dims, 0)  # prefactor in auxilliary equation

            ga, d1, d2, d3 = object.create_rectangle_PBC(
                dims,
                int(dims.y / 2),
                int(dims.y / 2 + sphere.R / ddx),
                dt,
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

            return ga, list(
                d1,
                d2,
                d3,
                f1_et1,
                f2_et1,
                f3_et1,
                f4_et1,
                f1_et2,
                f2_et2,
                f3_et2,
                f4_et2,
            )

    elif args.object == "None":
        print("No object defined")

    else:
        raise ValueError("Something is wrong with the object definition")


def update_polarization_drude_with_object(p):
    return "test"
    # p_drude, p_tmp_drude = object.calculate_polarization(
    #     dims, sphere, ddx, p_drude, p_tmp_drude, e, *args_object
    # )
    # p.x = p_drude.x
    # p.y = p_drude.y
    # p.z = p_drude.z

    # return p

    # if FLAG.MATERIAL == "DrudeLorentz":  # DrudeLorentz:
    #     p_drude, p_tmp_drude = object.calculate_polarization(
    #         dims, sphere, ddx, p_drude, p_tmp_drude, e, d1, d2, d3, FLAG.OBJECT
    #     )
    #     p_lorentz, p_tmp_lorentz = object.calculate_polarization(
    #         dims, sphere, ddx, p_lorentz, p_tmp_lorentz, e, l1, l2, l3, FLAG.OBJECT
    #     )
    #     p.x = p_drude.x + p_lorentz.x
    #     p.y = p_drude.y + p_lorentz.y
    #     p.z = p_drude.z + p_lorentz.z

    # if FLAG.MATERIAL == 3:  # Etchegoin
    #     p_drude, p_tmp_drude = object.calculate_polarization(
    #         dims, sphere, ddx, p_drude, p_tmp_drude, e, d1, d2, d3, FLAG.OBJECT
    #     )
    #     p_et1, p_tmp_et1 = object.calculate_polarization_etch(
    #         dims,
    #         sphere,
    #         ddx,
    #         p_et1,
    #         p_tmp_et1,
    #         e,
    #         e1,
    #         f1_et1,
    #         f2_et1,
    #         f3_et1,
    #         f4_et1,
    #         FLAG.OBJECT,
    #     )
    #     p_et2, p_tmp_et2 = object.calculate_polarization_etch(
    #         dims,
    #         sphere,
    #         ddx,
    #         p_et2,
    #         p_tmp_et2,
    #         e,
    #         e1,
    #         f1_et2,
    #         f2_et2,
    #         f3_et2,
    #         f4_et2,
    #         FLAG.OBJECT,
    #     )
    #     p.x = p_drude.x + p_et1.x + p_et2.x
    #     p.y = p_drude.y + p_et1.y + p_et2.y
    #     p.z = p_drude.z + p_et1.z + p_et2.z


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


if __name__ == "__main__":
    main()
