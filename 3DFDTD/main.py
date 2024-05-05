from argparse import ArgumentParser
from classes import Pulse, DFT
from fundamentals import c, Sc, nm


def main():
    # constant parameters
    ddx = 10*nm #args.v[0]*nm           # spatial step size
    dt = ddx/c*Sc                       # time step, from dx fulfilling stability criteria
    radius = 150*nm                     # radius of sphere
    tfsf_dist = 12 #args.v[4]           # TFSF distance from computational boundary
    npml = 8                            # number of PML layers
    dim = 60 #args.v[1]                 # number of spatial steps
    args = get_user_input()
    
    # setup
    pulse = define_pulse(args)
    dft = DFT(dt=dt, iwdim=100, pulse_spread=pulse.spread, emin=1.9, emax=3.2)
    
    
    
    # setup()
    ga, args_object = setup_object()
    
    
    
    
    # print_parameters()
 

    # time_loop()
    #     update buffer fields
    #     update_physical fields
    #     update pulse
    #     update monitors
    #     animate()

    # process_data()
    print(args.pulse)


def get_user_input():
    parser = ArgumentParser()
    parser.add_argument(
        "--pulse",
        help="Choice: [THz, Optical]",
        choices=["THz", "Optical"],
        default="Optical",
    )
    return parser.parse_args()


def define_pulse(args):
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


def update_polarization():
    
    
def setup_object(FLAG):
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


if __name__ == "__main__":
    main()
