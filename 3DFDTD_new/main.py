from argparse import ArgumentParser
from classes import Pulse, DFT, Field
from fundamentals import c, Sc, nm
import fdtd
from parameters import Dimensions



def main():
    
    args = get_user_input()
    
    # constant parameters
    ddx = args.ddx*nm                               
    dt = ddx/c*Sc                                   
    dims = Dimensions(x = args.dim, y = args.dim, z= args.dim)
    
    npml = args.npml                                
    tfsf_dist = npml + 4                            # TFSF distance from computational boundary  
    
    radius = args.r*nm                     # radius of sphere
    eps_out = args.eps_out

    # setup
    pulse = define_pulse(args)
    dft = DFT(dt=dt, iwdim=100, pulse_spread=pulse.spread, emin=1.9, emax=3.2)
    
    
    
    # setup()
    ga, args_object = setup_object(FLAG,dims,sphere,args.nsub,ddx,dt,eps_out)
    

    if FLAG.OBJECT != 0:
        if FLAG.MATERIAL == "Drude":  # Drude only
            update_polarization = update_polarization_drude

        elif FLAG.MATERIAL == "DrudeLorentz":
            update_polarization = update_polarization_drudelorentz
            
        elif FLAG.MATERIAL = "Etchegoin": 
            update_polarization = update_polarization_etchegoin
        
        


    
    # print_parameters()
 
########
    # time loop
########


    for time_step in range(1, tsteps + 1):
    #     
    
            # update 1d electric field buffer
        if FLAG.TFSF == 1 or FLAG.TFSF == 2:
            ez_inc = update_buffer_ez_inc(ez_inc, hx_inc, dims, boundary_low,boundary_high)
                
            # Update incident electric field


        # field updates
        d, id = fdtd.D_update(FLAG, dims, d, h, id, PML, tfsf, hx_inc)
        e, e1 = fdtd.calculate_e_fields(dims, e, e1, d, ga, p)
        h, ih = fdtd.H_update(FLAG, dims, h, ih, e, PML, tfsf, ez_inc)

        # pulse updating
        pulse_tmp = pulse.update_value(time_step, dt)

        # Update 1d buffer for plane wave - magnetic field
        if FLAG.TFSF == 1 or FLAG.TFSF == 2:
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
    
    
    
    
    
    
    
    
    
    

def update_buffer_ez_inc(ez_inc, hx_inc, dims, boundary_low,boundary_high): 
    ez_inc = fdtd.calculate_ez_inc_field(dims.y, ez_inc, hx_inc)

    # Implementation of ABC
    ez_inc[0] = boundary_low.pop(0)
    boundary_low.append(ez_inc[1])
    ez_inc[dims.y - 1] = boundary_high.pop(0)
    boundary_high.append(ez_inc[dims.y - 2])
    
    return ez_inc
    















def get_user_input():
    parser = ArgumentParser()
    parser.add_argument(
        "--pulse","-p",
        help="Choice: [THz, Optical]",
        choices=["THz", "Optical"],
        default="Optical",
    )
    
    parser.add_argument(
        "-ddx",
        help="Provide spatial resolution in nm.",
        type=int,
        default=10,
    )
    
    parser.add_argument(
        "-o","--object",
        help="Choose object.",
        choices=["Sphere", "Rectangle","Empty"],
        default= "Rectangle",
    )
        
    parser.add_argument(
        "-r","--radius",
        help="Define Radius of Sphere or Diameter of Rectangle",
        type = int,
        default= 150,
    )
    
    parser.add_argument(
        "-dim",
        help="Number of Grid Cells",
        type = int,
        default= 50
    )
    
    parser.add_argument(
        "-npml",
        help="Number of PML Layers",
        choices = [8,10,12],
        type = int,
        default= 8
    )
    
    parser.add_argument(
        "-eps_out",
        help="Permittivity surrounding the structure",
        type = float,
        default= 1.
    )
    
    parser.add_argument(
        "-nsub",
        help="Number of Subgridding Cells",
        type = int,
        choices = [5,7,9]
        default= 5
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


def setup_polarization():
    
    if 
    
    
def setup_object(FLAG,dims,sphere,nsub,ddx,dt,eps_out):
    
    if FLAG.OBJECT == "Sphere": 

        if FLAG.MATERIAL == "Drude":  
            
            ga = Field(dims,1)                      # inverse permittivity =1/eps
            d1 = Field(dims,0)                      # prefactor in auxilliary equation
            d2 = Field(dims,0)                      # prefactor in auxilliary equation
            d3 = Field(dims,0)                      # prefactor in auxilliary equation
            
            ga, d1, d2, d3 = object.create_sphere_drude_eps(sphere, nsub, ddx, dt, eps_out, ga, d1, d2, d3)
            
            return ga, list(d1, d2, d3)
        
        
        if FLAG.MATERIAL == "DrudeLorentz":  
            
            ga = Field(dims,1)                      # inverse permittivity =1/eps
            d1 = Field(dims,0)                      # prefactor in auxilliary equation
            d2 = Field(dims,0)                      # prefactor in auxilliary equation
            d3 = Field(dims,0)                      # prefactor in auxilliary equation
            l1 = Field(dims,0)                  # prefactor in auxilliary equation
            l2 = Field(dims,0)                  # prefactor in auxilliary equation
            l3 = Field(dims,0)                  # prefactor in auxilliary equation
                
            
            ga, d1, d2, d3 = object.create_sphere_drude_eps(
                sphere, nsub, ddx, dt, eps_out, ga, d1, d2, d3
            )
            ga, l1, l2, l3 = object.create_sphere_lorentz(
                sphere, nsub, ddx, dt, eps_out, ga, l1, l2, l3,
            )
        
            return ga, list(d1, d2, d3, l1, l2, l3)

        if FLAG.MATERIAL == "Etchegoin": 
            
            ga = Field(dims,1)                      # inverse permittivity =1/eps
            d1 = Field(dims,0)                      # prefactor in auxilliary equation
            d2 = Field(dims,0)                      # prefactor in auxilliary equation
            d3 = Field(dims,0)                      # prefactor in auxilliary equation
            
            f1_et1 = Field(dims,0)              # prefactor in auxilliary equation
            f2_et1 = Field(dims,0)              # prefactor in auxilliary equation
            f3_et1 = Field(dims,0)              # prefactor in auxilliary equation
            f4_et1 = Field(dims,0)              # prefactor in auxilliary equation
            f1_et2 = Field(dims,0)              # prefactor in auxilliary equation
            f2_et2 = Field(dims,0)              # prefactor in auxilliary equation
            f3_et2 = Field(dims,0)              # prefactor in auxilliary equation
            f4_et2 = Field(dims,0)              # prefactor in auxilliary equation
            
            
            ga, d1, d2, d3 = object.create_sphere_drude_eps(
                sphere, nsub, ddx, dt, eps_out, ga, d1, d2, d3
            )
            f1_et1, f2_et1, f3_et1, f4_et1, f1_et2, f2_et2, f3_et2, f4_et2 = (
                object.create_sphere_etch(
                    sphere, nsub, ddx, dt, f1_et1, f2_et1, f3_et1, f4_et1, f1_et2, f2_et2, f3_et2, f4_et2,
                ))
                
            return ga, list(d1, d2, d3, f1_et1, f2_et1, f3_et1, f4_et1, f1_et2, f2_et2, f3_et2, f4_et2)

    elif FLAG.OBJECT == "Rectangle" and FLAG.TFSF == 2:

        if FLAG.MATERIAL == "Drude": 
            
            ga = Field(dims,1)                      # inverse permittivity =1/eps
            d1 = Field(dims,0)                      # prefactor in auxilliary equation
            d2 = Field(dims,0)                      # prefactor in auxilliary equation
            d3 = Field(dims,0)                      # prefactor in auxilliary equation
            
            ga, d1, d2, d3 = object.create_rectangle_PBC(
                dims, int(dims.y / 2), int(dims.y / 2 + sphere.R / ddx), dt, ga, d1, d2, d3,
            )  # should derive a different quantity for the diameter instead of using sphere radius
            return ga, list(d1, d2, d3)
        

        if FLAG.MATERIAL == "DrudeLorentz":  
            
            ga = Field(dims,1)                      # inverse permittivity =1/eps
            d1 = Field(dims,0)                      # prefactor in auxilliary equation
            d2 = Field(dims,0)                      # prefactor in auxilliary equation
            d3 = Field(dims,0)                      # prefactor in auxilliary equation
            l1 = Field(dims,0)                      # prefactor in auxilliary equation
            l2 = Field(dims,0)                      # prefactor in auxilliary equation
            l3 = Field(dims,0)                      # prefactor in auxilliary equation
            
            ga, d1, d2, d3 = object.create_rectangle_PBC(
                dims, int(dims.y / 2), int(dims.y / 2 + sphere.R / ddx), dt, ga, d1, d2, d3,
            )
            ga, l1, l2, l3 = object.create_rectangle_PBC_lorentz(
                dims, int(dims.y / 2), int(dims.y / 2 + sphere.R / ddx), dt, ga, l1, l2, l3,
            )
            return ga, list(d1, d2, d3, l1, l2, l3)

        if FLAG.MATERIAL == "Etchegoin": 

            ga = Field(dims,1)                      # inverse permittivity =1/eps
            d1 = Field(dims,0)                      # prefactor in auxilliary equation
            d2 = Field(dims,0)                      # prefactor in auxilliary equation
            d3 = Field(dims,0)                      # prefactor in auxilliary equation
            
            f1_et1 = Field(dims,0)              # prefactor in auxilliary equation
            f2_et1 = Field(dims,0)              # prefactor in auxilliary equation
            f3_et1 = Field(dims,0)              # prefactor in auxilliary equation
            f4_et1 = Field(dims,0)              # prefactor in auxilliary equation
            f1_et2 = Field(dims,0)              # prefactor in auxilliary equation
            f2_et2 = Field(dims,0)              # prefactor in auxilliary equation
            f3_et2 = Field(dims,0)              # prefactor in auxilliary equation
            f4_et2 = Field(dims,0)              # prefactor in auxilliary equation
            
            ga, d1, d2, d3 = object.create_rectangle_PBC(
                dims, int(dims.y / 2), int(dims.y / 2 + sphere.R / ddx), dt, ga, d1, d2, d3,
            )
            f1_et1, f2_et1, f3_et1, f4_et1, f1_et2, f2_et2, f3_et2, f4_et2 = (
                object.create_rectangle_PBC_etch(
                    dims, int(dims.y / 2), int(dims.y / 2 + sphere.R / ddx), dt, f1_et1, f2_et1, f3_et1, f4_et1, f1_et2, f2_et2, f3_et2, f4_et2,
                )
            )
            
            return ga, list(d1, d2, d3, f1_et1, f2_et1, f3_et1, f4_et1, f1_et2, f2_et2, f3_et2, f4_et2)

    elif FLAG.OBJECT == 0:
        print("No object defined")

    else:
        raise ValueError("Something is wrong with the object definition")




def update_polarization_drude_with_object(p):

        p_drude, p_tmp_drude = object.calculate_polarization(
            dims, sphere, ddx, p_drude, p_tmp_drude, e, *args_object 
        )
        p.x = p_drude.x
        p.y = p_drude.y
        p.z = p_drude.z
        
        return p 

        if FLAG.MATERIAL == "DrudeLorentz":  # DrudeLorentz:
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


if __name__ == "__main__":
    main()
