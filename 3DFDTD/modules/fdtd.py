import numpy as np
import numba

#########################
# d fields for PML case 
#########################

@numba.jit(nopython=True)
def calculate_dx_field(dims,dx,h,idx,pml):
    """ calculates dx field for standard PML case """
    for i in range(0,dims.x):
        for j in range (1,dims.y):
            for k in range (1,dims.z):
                curlH = h.z[i,j,k] - h.z[i,j-1,k] - \
                        h.y[i,j,k] + h.y[i,j,k-1]
                idx[i,j,k] = curlH + idx[i,j,k]
                dx[i,j,k] = pml.gj3[j]*pml.gk3[k]*dx[i,j,k] +\
                            pml.gj2[j]*pml.gk2[k]*(0.5 * curlH + pml.gi1[i]*idx[i,j,k])
    return dx,idx

@numba.jit(nopython=True)
def calculate_dy_field(dims,dy,h,idy,pml):
    """ calculates dy field for standard PML case """
    for i in range(1,dims.x):
        for j in range (0,dims.y):
            for k in range (1,dims.z):
                curlH = h.x[i,j,k] - h.x[i,j,k-1] - \
                        h.z[i,j,k] + h.z[i-1,j,k]
                idy[i,j,k] = curlH + idy[i,j,k]
                dy[i,j,k] = pml.gi3[i]*pml.gk3[k]*dy[i,j,k] +\
                            pml.gi2[i]*pml.gk2[k]*(0.5 * curlH + pml.gj1[j]*idy[i,j,k])
    return dy,idy

@numba.jit(nopython=True)
def calculate_dz_field(dims,dz,h,idz,pml):
    """ calculates dz field for standard PML case """
    for i in range(1,dims.x):
        for j in range (1,dims.y):
            for k in range (0,dims.z):
                curlH = h.y[i,j,k] - h.y[i-1,j,k] - \
                        h.x[i,j,k] + h.x[i,j-1,k]
                idz[i,j,k] = curlH + idz[i,j,k]
                dz[i,j,k] = pml.gi3[i]*pml.gj3[j]*dz[i,j,k] +\
                            pml.gi2[i]*pml.gj2[j]*(0.5 * curlH + pml.gk1[k]*idz[i,j,k])
    return dz,idz

#########################
# d fields for PBC case 
#########################

@numba.jit(nopython=True)
def calculate_dx_field_PBC(dims,dx,h,idx,pml):
    """ calculates dx field for periodic boundary case """
    for i in range(0,dims.x):
        for j in range (1,dims.y):
            curlH = h.z[i,j,0] - h.z[i,j-1,0] - \
                    h.y[i,j,0] + h.y[i,j,-1]
            idx[i,j,0] = curlH + idx[i,j,0]
            dx[i,j,0] = pml.gj3[j]*pml.gk3[0]*dx[i,j,0] +\
                        pml.gj2[j]*pml.gk2[0]*(0.5 * curlH + pml.gi1[i]*idx[i,j,0])
    return dx,idx


@numba.jit(nopython=True)
def calculate_dy_field_PBC(dims,dy,h,idy,pml):
    """ calculates dy field for periodic boundary case """
    for j in range (0,dims.y):
        for i in range(1,dims.x):
                curlH = h.x[i,j,0] - h.x[i,j,-1] - \
                        h.z[i,j,0] + h.z[i-1,j,0]
                idy[i,j,0] = curlH + idy[i,j,0]
                dy[i,j,0] = pml.gi3[i]*pml.gk3[0]*dy[i,j,0] +\
                            pml.gi2[i]*pml.gk2[0]*(0.5 * curlH + pml.gj1[j]*idy[i,j,0])
        for k in range (1,dims.z):
            curlH = h.x[0,j,k] - h.x[0,j,k-1] - \
                    h.z[0,j,k] + h.z[-1,j,k]
            idy[0,j,k] = curlH + idy[0,j,k]
            dy[0,j,k] = pml.gi3[0]*pml.gk3[k]*dy[0,j,k] +\
                        pml.gi2[0]*pml.gk2[k]*(0.5 * curlH + pml.gj1[j]*idy[0,j,k])
    return dy,idy


@numba.jit(nopython=True)
def calculate_dz_field_PBC(dims,dz,h,idz,pml):
    """ calculates dz field for periodic boundary case """
    for j in range (1,dims.y):
        for k in range (0,dims.z):
            curlH = h.y[0,j,k] - h.y[-1,j,k] - \
                    h.x[0,j,k] + h.x[0,j-1,k]
            idz[0,j,k] = curlH + idz[0,j,k]
            dz[0,j,k] = pml.gi3[0]*pml.gj3[j]*dz[0,j,k] +\
                        pml.gi2[0]*pml.gj2[j]*(0.5 * curlH + pml.gk1[k]*idz[0,j,k])
    return dz,idz
            
#########################
# e fields 
#########################

@numba.jit(nopython=True)
def calculate_e_fields(dims,e,e1,d,ga,p):
    """ calculates all e fields from d fields """
    e1.x = e.x; e1.y = e.y; e1.z = e.z
    for i in range(0,dims.x):
        for j in range (0,dims.y):
            for k in range (0,dims.z):
                e.x[i,j,k]=ga.x[i,j,k]*(d.x[i,j,k]-p.x[i,j,k])
                e.y[i,j,k]=ga.y[i,j,k]*(d.y[i,j,k]-p.y[i,j,k])
                e.z[i,j,k]=ga.z[i,j,k]*(d.z[i,j,k]-p.z[i,j,k])
    return e, e1

#########################
# h fields for PML case 
#########################

@numba.jit(nopython=True)
def calculate_hx_field(dims,hx,e,ihx,pml):
    """ calculates hx field for standard PML case """
    for i in range(0,dims.x):
        for j in range (0,dims.y-1):
            for k in range (0,dims.z-1):
                curlE = (e.y[i,j,k+1] - e.y[i,j,k] -\
                        e.z[i,j+1,k] + e.z[i,j,k])
                ihx[i,j,k] = curlE + ihx[i,j,k]
                hx[i,j,k] = pml.fj3[j]*pml.fk3[k]*hx[i,j,k] +\
                            pml.fj2[j]*pml.fk2[k]*(0.5 * curlE + pml.fi1[i]* ihx[i,j,k])
    return hx,ihx

@numba.jit(nopython=True)
def calculate_hy_field(dims,hy,e,ihy,pml):
    for i in range(0,dims.x-1):
        for j in range (0,dims.y):
            for k in range (0,dims.z-1):
                curlE = (e.z[i+1,j,k] - e.z[i,j,k] -\
                        e.x[i,j,k+1] + e.x[i,j,k])
                ihy[i,j,k] = curlE + ihy[i,j,k]
                hy[i,j,k] = pml.fi3[i]*pml.fk3[k]*hy[i,j,k] +\
                            pml.fi2[i]*pml.fk2[k]*(0.5 * curlE + pml.fj1[j]* ihy[i,j,k])
    return hy,ihy

@numba.jit(nopython=True)
def calculate_hz_field(dims,hz,e,ihz,pml):
    """ calculates hx field for periodic boundary case """
    for i in range(0,dims.x-1):
        for j in range (0,dims.y-1):
            for k in range (0,dims.z):
                curlE = (e.x[i,j+1,k] - e.x[i,j,k] -\
                        e.y[i+1,j,k] + e.y[i,j,k])
                ihz[i,j,k] = curlE + ihz[i,j,k]
                hz[i,j,k] = pml.fi3[i]*pml.fj3[j]*hz[i,j,k] +\
                            pml.fi2[i]*pml.fj2[j]*(0.5 * curlE + pml.fk1[k]* ihz[i,j,k])
    return hz,ihz

#########################
# h fields for PBC case 
#########################

@numba.jit(nopython=True)
def calculate_hx_field_PBC(dims,hx,e,ihx,pml):
    """ calculates hx field for periodic boundary case """
    for i in range(0,dims.x):
        for j in range (0,dims.y-1):
                curlE = (e.y[i,j,0] - e.y[i,j,dims.z-1] -\
                        e.z[i,j+1,dims.z-1] + e.z[i,j,dims.z-1])
                ihx[i,j,dims.z-1] = curlE + ihx[i,j,dims.z-1]
                hx[i,j,dims.z-1] = pml.fj3[j]*pml.fk3[dims.z-1]*hx[i,j,dims.z-1] +\
                                pml.fj2[j]*pml.fk2[dims.z-1]*(0.5 * curlE + pml.fi1[i]* ihx[i,j,dims.z-1])
    return hx,ihx


@numba.jit(nopython=True)
def calculate_hy_field_PBC(dims,hy,e,ihy,pml):
    """ calculates hy field for periodic boundary case """
    for j in range (0,dims.y):
        for i in range(0,dims.x-1):
            curlE = (e.z[i+1,j,dims.z-1] - e.z[i,j,dims.z-1] -\
                    e.x[i,j,0] + e.x[i,j,dims.z-1])
            ihy[i,j,dims.z-1] = curlE + ihy[i,j,dims.z-1]
            hy[i,j,dims.z-1] = pml.fi3[i]*pml.fk3[dims.z-1]*hy[i,j,dims.z-1] +\
                        pml.fi2[i]*pml.fk2[dims.z-1]*(0.5 * curlE + pml.fj1[j]* ihy[i,j,dims.z-1])
        for k in range (0,dims.z-1):
            curlE = (e.z[0,j,k] - e.z[dims.x-1,j,k] -\
                    e.x[dims.x-1,j,k+1] + e.x[dims.x-1,j,k])
            ihy[dims.x-1,j,k] = curlE + ihy[dims.x-1,j,k]
            hy[dims.x-1,j,k] = pml.fi3[dims.x-1]*pml.fk3[k]*hy[dims.x-1,j,k] +\
                        pml.fi2[dims.x-1]*pml.fk2[k]*(0.5 * curlE + pml.fj1[j]* ihy[dims.x-1,j,k])
    return hy,ihy


@numba.jit(nopython=True)
def calculate_hz_field_PBC(dims,hz,e,ihz,pml):
    """ calculates hz field for periodic boundary case """
    for j in range (0,dims.y-1):
        for k in range (0,dims.z):
            curlE = (e.x[dims.x-1,j+1,k] - e.x[dims.x-1,j,k] -\
                    e.y[0,j,k] + e.y[dims.x-1,j,k])
            ihz[dims.x-1,j,k] = curlE + ihz[dims.x-1,j,k]
            hz[dims.x-1,j,k] = pml.fi3[dims.x-1]*pml.fj3[j]*hz[dims.x-1,j,k] +\
                        pml.fi2[dims.x-1]*pml.fj2[j]*(0.5 * curlE + pml.fk1[k]* ihz[dims.x-1,j,k])
    return hz,ihz


##############################
# Incident fields - Standard
##############################

def calculate_ez_inc_field(ymax,ez_inc,hx_inc):
    """ Calculates the dynamics of incident fields """
    for j in range(1,ymax): ## Steve made changes to dims.y-1
        ez_inc[j] = ez_inc[j] +0.5*(hx_inc[j-1]-hx_inc[j]) 
    return ez_inc

def calculate_hx_inc_field(ymax,hx_inc,ez_inc):
    """Calculates the dynamics of incident fields"""
    for j in range(0,ymax-1):
        hx_inc[j] = hx_inc[j] +0.5*(ez_inc[j]-ez_inc[j+1]) 
    return hx_inc

#####################################
# Incident fields - TFSF correction 
#####################################

@numba.jit(nopython=True)
def calculate_dy_inc_TFSF(tfsf,dy,hx_inc):
    """ Corrects the Dy field for TFSF BC """
    for i in range(tfsf.x_min,tfsf.x_max+1):
        for j in range(tfsf.y_min,tfsf.y_max):
            dy[i,j,tfsf.z_min]= dy[i,j,tfsf.z_min]-0.5*hx_inc[j]
#            dy[i,j,tfsf.z_max]= dy[i,j,tfsf.z_max]+0.5*hx_inc[j]
            dy[i,j,tfsf.z_max+1]= dy[i,j,tfsf.z_max+1]+0.5*hx_inc[j]
    return dy

@numba.jit(nopython=True)
def calculate_dz_inc_TFSF(tfsf,dz,hx_inc):
    """ Corrects the Dz field for TFSF BC """
    for i in range(tfsf.x_min,tfsf.x_max+1):
#        for k in range(tfsf.z_min,tfsf.z_max):
        for k in range(tfsf.z_min,tfsf.z_max+1):
            dz[i,tfsf.y_min,k]= dz[i,tfsf.y_min,k]+0.5*hx_inc[tfsf.y_min-1]
            dz[i,tfsf.y_max,k]= dz[i,tfsf.y_max,k]-0.5*hx_inc[tfsf.y_max]
    return dz

@numba.jit(nopython=True)
def calculate_hx_inc_TFSF(tfsf,hx,ez_inc):
    """ Corrects the Hx field for TFSF BC """
    for i in range(tfsf.x_min,tfsf.x_max+1):
#        for k in range(tfsf.z_min,tfsf.z_max):
        for k in range(tfsf.z_min,tfsf.z_max+1):
            hx[i,tfsf.y_min-1,k]= hx[i,tfsf.y_min-1,k]+0.5*ez_inc[tfsf.y_min]
            hx[i,tfsf.y_max,k]= hx[i,tfsf.y_max,k]-0.5*ez_inc[tfsf.y_max]
    return hx

@numba.jit(nopython=True)
def calculate_hy_inc_TFSF(tfsf,hy,ez_inc):
    """ Corrects the Hy field for TFSF BC """
#    for j in range(tfsf.y_min,tfsf.y_max):
    for j in range(tfsf.y_min,tfsf.y_max+1):
#        for k in range(tfsf.z_min,tfsf.z_max):
        for k in range(tfsf.z_min,tfsf.z_max+1):            
            hy[tfsf.x_min-1,j,k]= hy[tfsf.x_min-1,j,k]-0.5*ez_inc[j]
            hy[tfsf.x_max,j,k]= hy[tfsf.x_max,j,k]+0.5*ez_inc[j]
    return hy

##############################################################3
#Periodic boundary conditions
###############################################################

@numba.jit(nopython=True)
def calculate_dz_TFSF_PBC(dims,tfsf,dz,hx_inc):
    """ Corrects the Dz field for TFSF BC """
    for i in range(0,dims.x):
#        for k in range(tfsf.z_min,tfsf.z_max):
        for k in range(0,dims.z):
            dz[i,tfsf.y_min,k] += 0.5*hx_inc[tfsf.y_min-1]
            dz[i,tfsf.y_max,k] -= 0.5*hx_inc[tfsf.y_max]
    return dz

@numba.jit(nopython=True)
def calculate_hx_TFSF_PBC(dims,tfsf,hx,ez_inc):
    """ Corrects the Hx field for TFSF BC """
    for i in range(0,dims.x):
#        for k in range(tfsf.z_min,tfsf.z_max):
        for k in range(0,dims.z):
            hx[i,tfsf.y_min-1,k] += 0.5*ez_inc[tfsf.y_min]
            hx[i,tfsf.y_max,k] -= 0.5*ez_inc[tfsf.y_max]
    return hx

##########################
# Wrapper functions
##########################

def D_update(FLAG,dims,d,h,id,PML,tfsf,hx_inc):
    """ Update D Fields """
    # Standard FDTD update for D field
    d.x, id.x = calculate_dx_field(dims,d.x,h,id.x,PML)
    d.y, id.y = calculate_dy_field(dims,d.y,h,id.y,PML)
    d.z, id.z = calculate_dz_field(dims,d.z,h,id.z,PML)

    # Implementation of PBC
    if FLAG.TFSF ==2:
        d.x, id.x = calculate_dx_field_PBC(dims,d.x,h,id.x,PML)
        d.y, id.y = calculate_dy_field_PBC(dims,d.y,h,id.y,PML)
        d.z, id.z = calculate_dz_field_PBC(dims,d.z,h,id.z,PML)
        d.z = calculate_dz_TFSF_PBC(dims,tfsf,d.z,hx_inc)

    # TFSF corrections for D fields
    if(FLAG.TFSF==1):
        d.y = calculate_dy_inc_TFSF(tfsf,d.y,hx_inc)
        d.z = calculate_dz_inc_TFSF(tfsf,d.z,hx_inc)
        
    return d, id

def H_update(FLAG,dims,h,ih,e,PML,tfsf,ez_inc):
    """ Update H fields """
    # Standard FDTD update for H field
    h.x,ih.x = calculate_hx_field(dims,h.x,e,ih.x,PML)
    h.y,ih.y = calculate_hy_field(dims,h.y,e,ih.y,PML)
    h.z,ih.z = calculate_hz_field(dims,h.z,e,ih.z,PML)

    # Implementation of PBC
    if FLAG.TFSF ==2:
        h.x, ih.x = calculate_hx_field_PBC(dims,h.x,e,ih.x,PML)
        h.y, ih.y = calculate_hy_field_PBC(dims,h.y,e,ih.y,PML)
        h.z, ih.z = calculate_hz_field_PBC(dims,h.z,e,ih.z,PML)
        h.x = calculate_hx_TFSF_PBC(dims,tfsf,h.x,ez_inc)

    # TFSF corrections for D fields
    if(FLAG.TFSF==1):
        h.x = calculate_hx_inc_TFSF(tfsf,h.x,ez_inc)
        h.y = calculate_hy_inc_TFSF(tfsf,h.y,ez_inc)

    return h, ih