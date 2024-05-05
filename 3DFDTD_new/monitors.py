import numpy as np
import numba
from scipy.fft import rfft,rfftfreq

def DFT_incident_update(omegaDFT,SourceReDFT,SourceImDFT,pulse,iwdim,t):
    for om in range (0,iwdim+1):
        SourceReDFT[om] = SourceReDFT[om]+np.cos(omegaDFT[om]*t)*pulse
        SourceImDFT[om] = SourceImDFT[om]-np.sin(omegaDFT[om]*t)*pulse
    return SourceReDFT, SourceImDFT

@numba.jit(nopython=True)
def DFT3D_update(e,h,e_dft,h_dft,iwdim,omegaDFT,t):
    """ updated the 3D monitors"""
    for om in range (0,iwdim+1):
        e_dft.x[om,:,:,:] += np.exp(-1j*omegaDFT[om]*t)*e.x[:,:,:]
        e_dft.y[om,:,:,:] += np.exp(-1j*omegaDFT[om]*t)*e.y[:,:,:]
        e_dft.z[om,:,:,:] += np.exp(-1j*omegaDFT[om]*t)*e.z[:,:,:]
        h_dft.x[om,:,:,:] += np.exp(-1j*omegaDFT[om]*t)*h.x[:,:,:]
        h_dft.y[om,:,:,:] += np.exp(-1j*omegaDFT[om]*t)*h.y[:,:,:]
        h_dft.z[om,:,:,:] += np.exp(-1j*omegaDFT[om]*t)*h.z[:,:,:]
    return e_dft,h_dft

@numba.jit(nopython=True)
def DFT2D_update(e,h,iwdim,omegaDFT,t,x_DFT,y_DFT,z_DFT,\
    e_dft_xnormal, e_dft_ynormal, e_dft_znormal, h_dft_xnormal, h_dft_ynormal, h_dft_znormal):
    for om in range (0,iwdim+1):
        e_dft_xnormal.x[om,:,:] += np.exp(-1j*omegaDFT[om]*t)*e.x[x_DFT,:,:]
        e_dft_xnormal.y[om,:,:] += np.exp(-1j*omegaDFT[om]*t)*e.y[x_DFT,:,:]
        e_dft_xnormal.z[om,:,:] += np.exp(-1j*omegaDFT[om]*t)*e.z[x_DFT,:,:]
        h_dft_xnormal.x[om,:,:] += np.exp(-1j*omegaDFT[om]*t)*h.x[x_DFT,:,:]
        h_dft_xnormal.y[om,:,:] += np.exp(-1j*omegaDFT[om]*t)*h.y[x_DFT,:,:]
        h_dft_xnormal.z[om,:,:] += np.exp(-1j*omegaDFT[om]*t)*h.z[x_DFT,:,:]

        e_dft_ynormal.x[om,:,:] += np.exp(-1j*omegaDFT[om]*t)*e.x[:,y_DFT,:]
        e_dft_ynormal.y[om,:,:] += np.exp(-1j*omegaDFT[om]*t)*e.y[:,y_DFT,:]
        e_dft_ynormal.z[om,:,:] += np.exp(-1j*omegaDFT[om]*t)*e.z[:,y_DFT,:]
        h_dft_ynormal.x[om,:,:] += np.exp(-1j*omegaDFT[om]*t)*h.x[:,y_DFT,:]
        h_dft_ynormal.y[om,:,:] += np.exp(-1j*omegaDFT[om]*t)*h.y[:,y_DFT,:]
        h_dft_ynormal.z[om,:,:] += np.exp(-1j*omegaDFT[om]*t)*h.z[:,y_DFT,:]

        e_dft_znormal.x[om,:,:] += np.exp(-1j*omegaDFT[om]*t)*e.x[:,:,z_DFT]
        e_dft_znormal.y[om,:,:] += np.exp(-1j*omegaDFT[om]*t)*e.y[:,:,z_DFT]
        e_dft_znormal.z[om,:,:] += np.exp(-1j*omegaDFT[om]*t)*e.z[:,:,z_DFT]
        h_dft_znormal.x[om,:,:] += np.exp(-1j*omegaDFT[om]*t)*h.x[:,:,z_DFT]
        h_dft_znormal.y[om,:,:] += np.exp(-1j*omegaDFT[om]*t)*h.y[:,:,z_DFT]
        h_dft_znormal.z[om,:,:] += np.exp(-1j*omegaDFT[om]*t)*h.z[:,:,z_DFT]

    return e_dft_xnormal, e_dft_ynormal, e_dft_znormal, h_dft_xnormal, h_dft_ynormal, h_dft_znormal


@numba.jit(nopython=True)
def DFT_ref_trans(e_ref, e_trans,e,y_ref,y_trans,iwdim,omegaDFT,t):
    for om in range (0,iwdim+1):
        exponent = np.exp(-1j*omegaDFT[om]*t)

        #reflection
        e_ref.x[om,:,:] += exponent*e.x[:,y_ref,:]
        e_ref.y[om,:,:] += exponent*e.y[:,y_ref,:]
        e_ref.z[om,:,:] += exponent*e.z[:,y_ref,:]

        #transmission
        e_trans.x[om,:,:] += exponent*e.x[:,y_trans,:]
        e_trans.y[om,:,:] += exponent*e.y[:,y_trans,:]
        e_trans.z[om,:,:] += exponent*e.z[:,y_trans,:]
        
    return  e_ref, e_trans

@numba.jit(nopython=True)
def DFT_scat_update(e_scat_x_min,e_scat_x_max,h_scat_x_min,h_scat_x_max,
                    e_scat_y_min,e_scat_y_max,h_scat_y_min,h_scat_y_max,
                    e_scat_z_min,e_scat_z_max,h_scat_z_min,h_scat_z_max,
                    e,h,scat,iwdim,omegaDFT,t):
#   below has been faster, do not really understand why
    for om in range (0,iwdim+1):
        exponent = np.exp(-1j*omegaDFT[om]*t)
        #xnormal
        e_scat_x_min.y[om,:,:] += exponent*e.y[scat.x_min,:,:]
        e_scat_x_min.z[om,:,:] += exponent*e.z[scat.x_min,:,:]
        h_scat_x_min.y[om,:,:] += exponent*h.y[scat.x_min,:,:]
        h_scat_x_min.z[om,:,:] += exponent*h.z[scat.x_min,:,:]

        e_scat_x_max.y[om,:,:] += exponent*e.y[scat.x_max,:,:]
        e_scat_x_max.z[om,:,:] += exponent*e.z[scat.x_max,:,:]
        h_scat_x_max.y[om,:,:] += exponent*h.y[scat.x_max,:,:]
        h_scat_x_max.z[om,:,:] += exponent*h.z[scat.x_max,:,:]

        #ynormal
        e_scat_y_min.x[om,:,:] += exponent*e.x[:,scat.y_min,:]
        e_scat_y_min.z[om,:,:] += exponent*e.z[:,scat.y_min,:]
        h_scat_y_min.x[om,:,:] += exponent*h.x[:,scat.y_min,:]
        h_scat_y_min.z[om,:,:] += exponent*h.z[:,scat.y_min,:]

        e_scat_y_max.x[om,:,:] += exponent*e.x[:,scat.y_max,:]
        e_scat_y_max.z[om,:,:] += exponent*e.z[:,scat.y_max,:]
        h_scat_y_max.x[om,:,:] += exponent*h.x[:,scat.y_max,:]
        h_scat_y_max.z[om,:,:] += exponent*h.z[:,scat.y_max,:]

        #znormal
        e_scat_z_min.x[om,:,:] += exponent*e.x[:,:,scat.z_min]
        e_scat_z_min.y[om,:,:] += exponent*e.y[:,:,scat.z_min]
        h_scat_z_min.x[om,:,:] += exponent*h.x[:,:,scat.z_min]
        h_scat_z_min.y[om,:,:] += exponent*h.y[:,:,scat.z_min]

        e_scat_z_max.x[om,:,:] += exponent*e.x[:,:,scat.z_max]
        e_scat_z_max.y[om,:,:] += exponent*e.y[:,:,scat.z_max]
        h_scat_z_max.x[om,:,:] += exponent*h.x[:,:,scat.z_max]
        h_scat_z_max.y[om,:,:] += exponent*h.y[:,:,scat.z_max]
        
    return  e_scat_x_min,e_scat_x_max,h_scat_x_min,h_scat_x_max,\
            e_scat_y_min,e_scat_y_max,h_scat_y_min,h_scat_y_max,\
            e_scat_z_min,e_scat_z_max,h_scat_z_min,h_scat_z_max

# @numba.jit(nopython=True)
# def DFT_scat_update(e_scat,h_scat, e,h,scat,iwdim,omegaDFT,t):
# #   below has been faster, do not really understand why
#     for om in range (0,iwdim+1):
#         exponent = np.exp(-1j*omegaDFT[om]*t)
#         #xnormal
#         e_scat["x_min"].y[om,:,:] += exponent*e.y[scat.x_min,:,:]
#         e_scat["x_min"].z[om,:,:] += exponent*e.z[scat.x_min,:,:]
#         h_scat["x_min"].y[om,:,:] += exponent*h.y[scat.x_min,:,:]
#         h_scat["x_min"].z[om,:,:] += exponent*h.z[scat.x_min,:,:]

#         e_scat["x_max"].y[om,:,:] += exponent*e.y[scat.x_max,:,:]
#         e_scat["x_max"].z[om,:,:] += exponent*e.z[scat.x_max,:,:]
#         h_scat["x_max"].y[om,:,:] += exponent*h.y[scat.x_max,:,:]
#         h_scat["x_max"].z[om,:,:] += exponent*h.z[scat.x_max,:,:]

#         #ynormal
#         e_scat["y_min"].x[om,:,:] += exponent*e.x[:,scat.y_min,:]
#         e_scat["y_min"].z[om,:,:] += exponent*e.z[:,scat.y_min,:]
#         h_scat["y_min"].x[om,:,:] += exponent*h.x[:,scat.y_min,:]
#         h_scat["y_min"].z[om,:,:] += exponent*h.z[:,scat.y_min,:]

#         e_scat["y_max"].x[om,:,:] += exponent*e.x[:,scat.y_max,:]
#         e_scat["y_max"].z[om,:,:] += exponent*e.z[:,scat.y_max,:]
#         h_scat["y_max"].x[om,:,:] += exponent*h.x[:,scat.y_max,:]
#         h_scat["y_max"].z[om,:,:] += exponent*h.z[:,scat.y_max,:]

#         #znormal
#         e_scat["y_min"].x[om,:,:] += exponent*e.x[:,:,scat.z_min]
#         e_scat["y_min"].y[om,:,:] += exponent*e.y[:,:,scat.z_min]
#         h_scat["y_min"].x[om,:,:] += exponent*h.x[:,:,scat.z_min]
#         h_scat["y_min"].y[om,:,:] += exponent*h.y[:,:,scat.z_min]

#         e_scat["y_max"].x[om,:,:] += exponent*e.x[:,:,scat.z_max]
#         e_scat["y_max"].y[om,:,:] += exponent*e.y[:,:,scat.z_max]
#         h_scat["y_max"].x[om,:,:] += exponent*h.x[:,:,scat.z_max]
#         h_scat["y_max"].y[om,:,:] += exponent*h.y[:,:,scat.z_max]
        
#     return  e_scat, h_scat

@numba.jit(nopython=True)
def DFT_abs_update(e_abs_x_min,e_abs_x_max,h_abs_x_min,h_abs_x_max,
                    e_abs_y_min,e_abs_y_max,h_abs_y_min,h_abs_y_max,
                    e_abs_z_min,e_abs_z_max,h_abs_z_min,h_abs_z_max,
                    e,h,abs,iwdim,omegaDFT,t):

    for om in range (0,iwdim+1):
        exponent = np.exp(-1j*omegaDFT[om]*t)
        #xnormal
        e_abs_x_min.y[om,:,:] += exponent*e.y[abs.x_min,:,:]
        e_abs_x_min.z[om,:,:] += exponent*e.z[abs.x_min,:,:]
        h_abs_x_min.y[om,:,:] += exponent*h.y[abs.x_min,:,:]
        h_abs_x_min.z[om,:,:] += exponent*h.z[abs.x_min,:,:]

        e_abs_x_max.y[om,:,:] += exponent*e.y[abs.x_max,:,:]
        e_abs_x_max.z[om,:,:] += exponent*e.z[abs.x_max,:,:]
        h_abs_x_max.y[om,:,:] += exponent*h.y[abs.x_max,:,:]
        h_abs_x_max.z[om,:,:] += exponent*h.z[abs.x_max,:,:]

        #ynormal
        e_abs_y_min.x[om,:,:] += exponent*e.x[:,abs.y_min,:]
        e_abs_y_min.z[om,:,:] += exponent*e.z[:,abs.y_min,:]
        h_abs_y_min.x[om,:,:] += exponent*h.x[:,abs.y_min,:]
        h_abs_y_min.z[om,:,:] += exponent*h.z[:,abs.y_min,:]

        e_abs_y_max.x[om,:,:] += exponent*e.x[:,abs.y_max,:]
        e_abs_y_max.z[om,:,:] += exponent*e.z[:,abs.y_max,:]
        h_abs_y_max.x[om,:,:] += exponent*h.x[:,abs.y_max,:]
        h_abs_y_max.z[om,:,:] += exponent*h.z[:,abs.y_max,:]

        #znormal
        e_abs_z_min.x[om,:,:] += exponent*e.x[:,:,abs.z_min]
        e_abs_z_min.y[om,:,:] += exponent*e.y[:,:,abs.z_min]
        h_abs_z_min.x[om,:,:] += exponent*h.x[:,:,abs.z_min]
        h_abs_z_min.y[om,:,:] += exponent*h.y[:,:,abs.z_min]

        e_abs_z_max.x[om,:,:] += exponent*e.x[:,:,abs.z_max]
        e_abs_z_max.y[om,:,:] += exponent*e.y[:,:,abs.z_max]
        h_abs_z_max.x[om,:,:] += exponent*h.x[:,:,abs.z_max]
        h_abs_z_max.y[om,:,:] += exponent*h.y[:,:,abs.z_max]
        return e_abs_x_min,e_abs_x_max,h_abs_x_min,h_abs_x_max,\
            e_abs_y_min,e_abs_y_max,h_abs_y_min,h_abs_y_max,\
            e_abs_z_min,e_abs_z_max,h_abs_z_min,h_abs_z_max

# this does not work due to numba does not understand the dictionary
# @numba.jit(nopython=True)
# def DFT_abs_update(e_abs, h_abs, e, h, abs, iwdim, omegaDFT, t):

#     for om in range(0, iwdim + 1):
#         exponent = np.exp(-1j * omegaDFT[om] * t)
#         # xnormal
#         e_abs["x_min"].y[om, :, :] += exponent * e.y[abs.x_min, :, :]
#         e_abs["x_min"].z[om, :, :] += exponent * e.z[abs.x_min, :, :]
#         h_abs["x_min"].y[om, :, :] += exponent * h.y[abs.x_min, :, :]
#         h_abs["x_min"].z[om, :, :] += exponent * h.z[abs.x_min, :, :]

#         e_abs["x_max"].y[om, :, :] += exponent * e.y[abs.x_max, :, :]
#         e_abs["x_max"].z[om, :, :] += exponent * e.z[abs.x_max, :, :]
#         h_abs["x_max"].y[om, :, :] += exponent * h.y[abs.x_max, :, :]
#         h_abs["x_max"].z[om, :, :] += exponent * h.z[abs.x_max, :, :]

#         # ynormal
#         e_abs["y_min"].x[om, :, :] += exponent * e.x[:, abs.y_min, :]
#         e_abs["y_min"].z[om, :, :] += exponent * e.z[:, abs.y_min, :]
#         h_abs["y_min"].x[om, :, :] += exponent * h.x[:, abs.y_min, :]
#         h_abs["y_min"].z[om, :, :] += exponent * h.z[:, abs.y_min, :]

#         e_abs["y_max"].x[om, :, :] += exponent * e.x[:, abs.y_max, :]
#         e_abs["y_max"].z[om, :, :] += exponent * e.z[:, abs.y_max, :]
#         h_abs["y_max"].x[om, :, :] += exponent * h.x[:, abs.y_max, :]
#         h_abs["y_max"].z[om, :, :] += exponent * h.z[:, abs.y_max, :]

#         # znormal
#         e_abs["z_min"].x[om, :, :] += exponent * e.x[:, :, abs.z_min]
#         e_abs["z_min"].y[om, :, :] += exponent * e.y[:, :, abs.z_min]
#         h_abs["z_min"].x[om, :, :] += exponent * h.x[:, :, abs.z_min]
#         h_abs["z_min"].y[om, :, :] += exponent * h.y[:, :, abs.z_min]

#         e_abs["z_max"].x[om, :, :] += exponent * e.x[:, :, abs.z_max]
#         e_abs["z_max"].y[om, :, :] += exponent * e.y[:, :, abs.z_max]
#         h_abs["z_max"].x[om, :, :] += exponent * h.x[:, :, abs.z_max]
#         h_abs["z_max"].y[om, :, :] += exponent * h.y[:, :, abs.z_max]
#         return e_abs, h_abs


def update_1Dmonitors(t,loc_monitors,ex_mon,ey_mon,ez_mon,hx_mon,hy_mon,hz_mon,e,h):
    '''updates point monitors'''
    count = 0
    for monitor in loc_monitors:
        ex_mon[count,t-1] = e.x[monitor[0],monitor[1],monitor[2]]
        ey_mon[count,t-1] = e.y[monitor[0],monitor[1],monitor[2]]
        ez_mon[count,t-1] = e.z[monitor[0],monitor[1],monitor[2]]
        hx_mon[count,t-1] = h.x[monitor[0],monitor[1],monitor[2]]
        hy_mon[count,t-1] = h.y[monitor[0],monitor[1],monitor[2]]
        hz_mon[count,t-1] = h.z[monitor[0],monitor[1],monitor[2]]
        count +=1


def fft_1Dmonitors(dt,tsteps,fft_res,n_mon,ex_mon,ey_mon,ez_mon,hx_mon,hy_mon,hz_mon):
    '''Fourier transforms the point monitors'''
    N = tsteps*fft_res
    ex_mon_om = np.zeros([n_mon,int(N/2+1)])
    ey_mon_om = np.zeros([n_mon,int(N/2+1)])
    ez_mon_om = np.zeros([n_mon,int(N/2+1)])
    hx_mon_om = np.zeros([n_mon,int(N/2+1)])
    hy_mon_om = np.zeros([n_mon,int(N/2+1)])
    hz_mon_om = np.zeros([n_mon,int(N/2+1)])
    for i in range(n_mon):
        ex_mon_om[i,:] = rfft(ex_mon[i,:],n=N)
        ey_mon_om[i,:] = rfft(ey_mon[i,:],n=N)
        ez_mon_om[i,:] = rfft(ez_mon[i,:],n=N)
        hx_mon_om[i,:] = rfft(hx_mon[i,:],n=N)
        hy_mon_om[i,:] = rfft(hy_mon[i,:],n=N)
        hz_mon_om[i,:] = rfft(hz_mon[i,:],n=N)
    nu = rfftfreq(N, dt)
    omega = 2*np.pi*nu
    return omega,ex_mon_om,ey_mon_om,ez_mon_om,hx_mon_om,hy_mon_om,hz_mon_om



def fft_source(dt,tsteps,fft_res,pulsemon_t):
    '''Fourier transforms the source monitors for Green function'''
    N = tsteps*fft_res
    pulsemon_om = np.zeros([int(N/2+1)],complex)
    
    pulsemon_om = rfft(pulsemon_t,n=N)
    omega_source = 2*np.pi*rfftfreq(N, dt)
    return omega_source,pulsemon_om

def fft_sourcemonitors(dt,tsteps,fft_res,pulsemon_t,ez_source_t):
    '''Fourier transforms the source monitors for Green function'''
    N = tsteps*fft_res
    pulsemon_om = np.zeros([int(N/2+1)],complex)
    ez_source_om = np.zeros([int(N/2+1)],complex)
    
    pulsemon_om = rfft(pulsemon_t,n=N)
    ez_source_om = rfft(ez_source_t,n=N)
    omega_source = 2*np.pi*rfftfreq(N, dt)
    return omega_source,pulsemon_om,ez_source_om


def update_pulsemonitors(t,ez,xs,ys,zs,pulse,pulsemon_t,ez_source_t):
    pulsemon_t[t-1] = pulse
    ez_source_t[t-1] = ez[xs,ys,zs]
    return pulsemon_t,ez_source_t

        
@numba.jit(nopython=True)
def update_scat_DFT(S_scat_DFT,iwdim,scat,\
                    e_scat_x_min,e_scat_x_max,h_scat_x_min,h_scat_x_max,\
                    e_scat_y_min,e_scat_y_max,h_scat_y_min,h_scat_y_max,\
                    e_scat_z_min,e_scat_z_max,h_scat_z_min,h_scat_z_max):
    '''Calculate the Poynting flow thorugh the six 2d monitors at each DFT frequency'''
    for om in range(iwdim+1):
        #xnormal
        for y in range(scat.y_min,scat.y_max+1):
            for z in range(scat.z_min,scat.z_max+1):
                S_scat_DFT[0,om] -= 0.5*(np.real(e_scat_x_min.y[om,y,z]*np.conjugate(h_scat_x_min.z[om,y,z]))
                                        -np.real(e_scat_x_min.z[om,y,z]*np.conjugate(h_scat_x_min.y[om,y,z])))
                S_scat_DFT[1,om] += 0.5*(np.real(e_scat_x_max.y[om,y,z]*np.conjugate(h_scat_x_max.z[om,y,z]))
                                        -np.real(e_scat_x_max.z[om,y,z]*np.conjugate(h_scat_x_max.y[om,y,z])))


                # S_scat_DFT[0,om] -= 0.5*((EyReDFT_scat_xnormal[0,om,y,z]*HzReDFT_scat_xnormal[0,om,y,z]+EyImDFT_scat_xnormal[0,om,y,z]*HzImDFT_scat_xnormal[0,om,y,z])
                #                             -(EzReDFT_scat_xnormal[0,om,y,z]*HyReDFT_scat_xnormal[0,om,y,z]+EzImDFT_scat_xnormal[0,om,y,z]*HyImDFT_scat_xnormal[0,om,y,z]))
                # S_scat_DFT[1,om] += 0.5*((EyReDFT_scat_xnormal[1,om,y,z]*HzReDFT_scat_xnormal[1,om,y,z]+EyImDFT_scat_xnormal[1,om,y,z]*HzImDFT_scat_xnormal[1,om,y,z])
                #                             -(EzReDFT_scat_xnormal[1,om,y,z]*HyReDFT_scat_xnormal[1,om,y,z]+EzImDFT_scat_xnormal[1,om,y,z]*HyImDFT_scat_xnormal[1,om,y,z]))

        # ylow and yhigh
        for x in range(scat.x_min,scat.x_max+1):
            for z in range(scat.z_min,scat.z_max+1):
                S_scat_DFT[2,om] -= 0.5*(np.real(e_scat_y_min.z[om,x,z]*np.conjugate(h_scat_y_min.x[om,x,z]))
                                        -np.real(e_scat_y_min.x[om,x,z]*np.conjugate(h_scat_y_min.z[om,x,z])))
                S_scat_DFT[3,om] += 0.5*(np.real(e_scat_y_max.z[om,x,z]*np.conjugate(h_scat_y_max.x[om,x,z]))
                                        -np.real(e_scat_y_max.x[om,x,z]*np.conjugate(h_scat_y_max.z[om,x,z])))
                # S_scat_DFT[2,om] -= 0.5*((EzReDFT_scat_ynormal[0,om,x,z]*HxReDFT_scat_ynormal[0,om,x,z]+EzImDFT_scat_ynormal[0,om,x,z]*HxImDFT_scat_ynormal[0,om,x,z])
                #                             -(ExReDFT_scat_ynormal[0,om,x,z]*HzReDFT_scat_ynormal[0,om,x,z]+ExImDFT_scat_ynormal[0,om,x,z]*HzImDFT_scat_ynormal[0,om,x,z]))
                # S_scat_DFT[3,om] += 0.5*((EzReDFT_scat_ynormal[1,om,x,z]*HxReDFT_scat_ynormal[1,om,x,z]+EzImDFT_scat_ynormal[1,om,x,z]*HxImDFT_scat_ynormal[1,om,x,z])
                #                             -(ExReDFT_scat_ynormal[1,om,x,z]*HzReDFT_scat_ynormal[1,om,x,z]+ExImDFT_scat_ynormal[1,om,x,z]*HzImDFT_scat_ynormal[1,om,x,z]))

        # zlow and zhigh
        for x in range(scat.x_min,scat.x_max+1):
            for y in range(scat.y_min,scat.y_max+1):
                S_scat_DFT[4,om] -= 0.5*(np.real(e_scat_z_min.x[om,x,y]*np.conjugate(h_scat_z_min.y[om,x,y]))
                                        -np.real(e_scat_z_min.y[om,x,y]*np.conjugate(h_scat_z_min.x[om,x,y])))
                S_scat_DFT[5,om] += 0.5*(np.real(e_scat_z_max.x[om,x,y]*np.conjugate(h_scat_z_max.y[om,x,y]))
                                        -np.real(e_scat_z_max.y[om,x,y]*np.conjugate(h_scat_z_max.x[om,x,y])))

                # S_scat_DFT[4,om] -= 0.5*((ExReDFT_scat_znormal[0,om,x,y]*HyReDFT_scat_znormal[0,om,x,y]+ExImDFT_scat_znormal[0,om,x,y]*HyImDFT_scat_znormal[0,om,x,y])
                #                             -(EyReDFT_scat_znormal[0,om,x,y]*HxReDFT_scat_znormal[0,om,x,y]+EyImDFT_scat_znormal[0,om,x,y]*HxImDFT_scat_znormal[0,om,x,y]))
                # S_scat_DFT[5,om] += 0.5*((ExReDFT_scat_znormal[1,om,x,y]*HyReDFT_scat_znormal[1,om,x,y]+ExImDFT_scat_znormal[1,om,x,y]*HyImDFT_scat_znormal[1,om,x,y])
                #                             -(EyReDFT_scat_znormal[1,om,x,y]*HxReDFT_scat_znormal[1,om,x,y]+EyImDFT_scat_znormal[1,om,x,y]*HxImDFT_scat_znormal[1,om,x,y]))
    return S_scat_DFT

@numba.jit(nopython=True)
def update_abs_DFT(S_abs_DFT,iwdim,abs,
                    e_abs_x_min,e_abs_x_max,h_abs_x_min,h_abs_x_max,\
                    e_abs_y_min,e_abs_y_max,h_abs_y_min,h_abs_y_max,\
                    e_abs_z_min,e_abs_z_max,h_abs_z_min,h_abs_z_max):
    '''Calculate the Poynting flow thorugh the six 2d monitors at each DFT frequency'''
    for om in range(iwdim+1):

         # xlow and xhigh
        for y in range(abs.y_min,abs.y_max+1):
            for z in range(abs.z_min,abs.z_max+1):
                S_abs_DFT[0,om] += 0.5*(np.real(e_abs_x_min.y[om,y,z]*np.conjugate(h_abs_x_min.z[om,y,z])
                                                -e_abs_x_min.z[om,y,z]*np.conjugate(h_abs_x_min.y[om,y,z])))
                S_abs_DFT[1,om] -= 0.5*(np.real(e_abs_x_max.y[om,y,z]*np.conjugate(h_abs_x_max.z[om,y,z])
                                                -e_abs_x_max.z[om,y,z]*np.conjugate(h_abs_x_max.y[om,y,z])))


                # S_abs_DFT[0,om] += 0.5*((EyReDFT_abs_xnormal[0,om,y,z]*HzReDFT_abs_xnormal[0,om,y,z]+EyImDFT_abs_xnormal[0,om,y,z]*HzImDFT_abs_xnormal[0,om,y,z])
                #                             -(EzReDFT_abs_xnormal[0,om,y,z]*HyReDFT_abs_xnormal[0,om,y,z]+EzImDFT_abs_xnormal[0,om,y,z]*HyImDFT_abs_xnormal[0,om,y,z]))
                # S_abs_DFT[1,om] -= 0.5*((EyReDFT_abs_xnormal[1,om,y,z]*HzReDFT_abs_xnormal[1,om,y,z]+EyImDFT_abs_xnormal[1,om,y,z]*HzImDFT_abs_xnormal[1,om,y,z])
                #                             -(EzReDFT_abs_xnormal[1,om,y,z]*HyReDFT_abs_xnormal[1,om,y,z]+EzImDFT_abs_xnormal[1,om,y,z]*HyImDFT_abs_xnormal[1,om,y,z]))

        # ylow and yhigh
        for x in range(abs.x_min,abs.x_max+1):
            for z in range(abs.z_min,abs.z_max+1):
                S_abs_DFT[2,om] += 0.5*(np.real(e_abs_y_min.z[om,x,z]*np.conjugate(h_abs_y_min.x[om,x,z])
                                                -e_abs_y_min.x[om,x,z]*np.conjugate(h_abs_y_min.z[om,x,z])))
                S_abs_DFT[3,om] -= 0.5*(np.real(e_abs_y_max.z[om,x,z]*np.conjugate(h_abs_y_max.x[om,x,z])
                                                -e_abs_y_max.x[om,x,z]*np.conjugate(h_abs_y_max.z[om,x,z])))
 
                # S_abs_DFT[2,om] += 0.5*((EzReDFT_abs_ynormal[0,om,x,z]*HxReDFT_abs_ynormal[0,om,x,z]+EzImDFT_abs_ynormal[0,om,x,z]*HxImDFT_abs_ynormal[0,om,x,z])
                #                             -(ExReDFT_abs_ynormal[0,om,x,z]*HzReDFT_abs_ynormal[0,om,x,z]+ExImDFT_abs_ynormal[0,om,x,z]*HzImDFT_abs_ynormal[0,om,x,z]))
                # S_abs_DFT[3,om] -= 0.5*((EzReDFT_abs_ynormal[1,om,x,z]*HxReDFT_abs_ynormal[1,om,x,z]+EzImDFT_abs_ynormal[1,om,x,z]*HxImDFT_abs_ynormal[1,om,x,z])
                #                             -(ExReDFT_abs_ynormal[1,om,x,z]*HzReDFT_abs_ynormal[1,om,x,z]+ExImDFT_abs_ynormal[1,om,x,z]*HzImDFT_abs_ynormal[1,om,x,z]))

        # zlow and zhigh
        for x in range(abs.x_min,abs.x_max+1):
            for y in range(abs.y_min,abs.y_max+1):
                S_abs_DFT[4,om] += 0.5*(np.real(e_abs_z_min.x[om,x,y]*np.conjugate(h_abs_z_min.y[om,x,y])
                                                -e_abs_z_min.y[om,x,y]*np.conjugate(h_abs_z_min.x[om,x,y])))
                S_abs_DFT[5,om] -= 0.5*(np.real(e_abs_z_max.x[om,x,y]*np.conjugate(h_abs_z_max.y[om,x,y])
                                                -e_abs_z_max.y[om,x,y]*np.conjugate(h_abs_z_max.x[om,x,y])))

                # S_abs_DFT[4,om] += 0.5*((ExReDFT_abs_znormal[0,om,x,y]*HyReDFT_abs_znormal[0,om,x,y]+ExImDFT_abs_znormal[0,om,x,y]*HyImDFT_abs_znormal[0,om,x,y])
                #                             -(EyReDFT_abs_znormal[0,om,x,y]*HxReDFT_abs_znormal[0,om,x,y]+EyImDFT_abs_znormal[0,om,x,y]*HxImDFT_abs_znormal[0,om,x,y]))
                # S_abs_DFT[5,om] -= 0.5*((ExReDFT_abs_znormal[1,om,x,y]*HyReDFT_abs_znormal[1,om,x,y]+ExImDFT_abs_znormal[1,om,x,y]*HyImDFT_abs_znormal[1,om,x,y])
                #                             -(EyReDFT_abs_znormal[1,om,x,y]*HxReDFT_abs_znormal[1,om,x,y]+EyImDFT_abs_znormal[1,om,x,y]*HxImDFT_abs_znormal[1,om,x,y]))
    return S_abs_DFT

"original version "
# @numba.jit(nopython=True)
# def update_2dmonitors_DFT(S_DFT,iwdim,ia,ib,ja,jb,ka,kb,ExReDFT,ExImDFT,EyReDFT,EyImDFT,EzReDFT,EzImDFT,HxReDFT,HxImDFT,HyReDFT,HyImDFT,HzReDFT,HzImDFT):
#     '''Calculate the Poynting flow thorugh the six 2d monitors at each DFT frequency'''
#     for om in range(iwdim+1):

#          # xlow and xhigh
#         for y in range(ja-3,jb+3):
#             for z in range(ka-3,kb+3):
#                 S_DFT[0,om] -= 0.5*((EyReDFT[om,ia-3,y,z]*HzReDFT[om,ia-3,y,z]+EyImDFT[om,ia-3,y,z]*HzImDFT[om,ia-3,y,z])
#                                             -(EzReDFT[om,ia-3,y,z]*HyReDFT[om,ia-3,y,z]+EzImDFT[om,ia-3,y,z]*HyImDFT[om,ia-3,y,z]))
#                 S_DFT[1,om] += 0.5*((EyReDFT[om,ib+2,y,z]*HzReDFT[om,ib+2,y,z]+EyImDFT[om,ib+2,y,z]*HzImDFT[om,ib+2,y,z])
#                                             -(EzReDFT[om,ib+2,y,z]*HyReDFT[om,ib+2,y,z]+EzImDFT[om,ib+2,y,z]*HyImDFT[om,ib+2,y,z]))

#         # ylow and yhigh
#         for x in range(ia-3,ib+3):
#             for z in range(ka-3,kb+3):
#                 S_DFT[2,om] -= 0.5*((EzReDFT[om,x,ja-3,z]*HxReDFT[om,x,ja-3,z]+EzImDFT[om,x,ja-3,z]*HxImDFT[om,x,ja-3,z])
#                                             -(ExReDFT[om,x,ja-3,z]*HzReDFT[om,x,ja-3,z]+ExImDFT[om,x,ja-3,z]*HzImDFT[om,x,ja-3,z]))
#                 S_DFT[3,om] += 0.5*((EzReDFT[om,x,jb+2,z]*HxReDFT[om,x,jb+2,z]+EzImDFT[om,x,jb+2,z]*HxImDFT[om,x,jb+2,z])
#                                             -(ExReDFT[om,x,jb+2,z]*HzReDFT[om,x,jb+2,z]+ExImDFT[om,x,jb+2,z]*HzImDFT[om,x,jb+2,z]))

#         # zlow and zhigh
#         for x in range(ia-3,ib+3):
#             for y in range(ja-3,jb+3):
#                 S_DFT[4,om] -= 0.5*((ExReDFT[om,x,y,ka-3]*HyReDFT[om,x,y,ka-3]+ExImDFT[om,x,y,ka-3]*HyImDFT[om,x,y,ka-3])
#                                             -(EyReDFT[om,x,y,ka-3]*HxReDFT[om,x,y,ka-3]+EyImDFT[om,x,y,ka-3]*HxImDFT[om,x,y,ka-3]))
#                 S_DFT[5,om] += 0.5*((ExReDFT[om,x,y,kb+2]*HyReDFT[om,x,y,kb+2]+ExImDFT[om,x,y,kb+2]*HyImDFT[om,x,y,kb+2])
#                                             -(EyReDFT[om,x,y,kb+2]*HxReDFT[om,x,y,kb+2]+EyImDFT[om,x,y,kb+2]*HxImDFT[om,x,y,kb+2]))
#     return S_DFT

# @numba.jit(nopython=True)
# def update_2dmonitors_abs_DFT(S_abs_DFT,iwdim,ia,ib,ja,jb,ka,kb,ExReDFT,ExImDFT,EyReDFT,EyImDFT,EzReDFT,EzImDFT,HxReDFT,HxImDFT,HyReDFT,HyImDFT,HzReDFT,HzImDFT):
#     '''Calculate the Poynting flow thorugh the six 2d monitors at each DFT frequency'''
#     for om in range(iwdim+1):

#          # xlow and xhigh
#         for y in range(ja+2,jb-2):
#             for z in range(ka+2,kb-2):
#                 S_abs_DFT[0,om] -= 0.5*((EyReDFT[om,ia+2,y,z]*HzReDFT[om,ia+2,y,z]+EyImDFT[om,ia+2,y,z]*HzImDFT[om,ia+2,y,z])
#                                             -(EzReDFT[om,ia+2,y,z]*HyReDFT[om,ia+2,y,z]+EzImDFT[om,ia+2,y,z]*HyImDFT[om,ia+2,y,z]))
#                 S_abs_DFT[1,om] += 0.5*((EyReDFT[om,ib-3,y,z]*HzReDFT[om,ib-3,y,z]+EyImDFT[om,ib-3,y,z]*HzImDFT[om,ib-3,y,z])
#                                             -(EzReDFT[om,ib-3,y,z]*HyReDFT[om,ib-3,y,z]+EzImDFT[om,ib-3,y,z]*HyImDFT[om,ib-3,y,z]))

#         # ylow and yhigh
#         for x in range(ia+2,ib-2):
#             for z in range(ka+2,kb-2):
#                 S_abs_DFT[2,om] -= 0.5*((EzReDFT[om,x,ja+2,z]*HxReDFT[om,x,ja+2,z]+EzImDFT[om,x,ja+2,z]*HxImDFT[om,x,ja+2,z])
#                                             -(ExReDFT[om,x,ja+2,z]*HzReDFT[om,x,ja+2,z]+ExImDFT[om,x,ja+2,z]*HzImDFT[om,x,ja+2,z]))
#                 S_abs_DFT[3,om] += 0.5*((EzReDFT[om,x,jb-3,z]*HxReDFT[om,x,jb-3,z]+EzImDFT[om,x,jb-3,z]*HxImDFT[om,x,jb-3,z])
#                                             -(ExReDFT[om,x,jb-3,z]*HzReDFT[om,x,jb-3,z]+ExImDFT[om,x,jb-3,z]*HzImDFT[om,x,jb-3,z]))

#         # zlow and zhigh
#         for x in range(ia+2,ib-2):
#             for y in range(ja+2,jb-2):
#                 S_abs_DFT[4,om] -= 0.5*((ExReDFT[om,x,y,ka+2]*HyReDFT[om,x,y,ka+2]+ExImDFT[om,x,y,ka+2]*HyImDFT[om,x,y,ka+2])
#                                             -(EyReDFT[om,x,y,ka+2]*HxReDFT[om,x,y,ka+2]+EyImDFT[om,x,y,ka+2]*HxImDFT[om,x,y,ka+2]))
#                 S_abs_DFT[5,om] += 0.5*((ExReDFT[om,x,y,kb-3]*HyReDFT[om,x,y,kb-3]+ExImDFT[om,x,y,kb-3]*HyImDFT[om,x,y,kb-3])
#                                             -(EyReDFT[om,x,y,kb-3]*HxReDFT[om,x,y,kb-3]+EyImDFT[om,x,y,kb-3]*HxImDFT[om,x,y,kb-3]))
#     return -S_abs_DFT
