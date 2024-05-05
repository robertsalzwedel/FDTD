import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
import scipy.constants as constants
import numpy as np
from scipy.fft import rfft,rfftfreq
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
custom_meta_key = 'TFSFsource.iot'
custom_meta_key_periodic = 'periodicsource.iot'
import miepython
from parameters import *

# from argparse import ArgumentParser
# parser = ArgumentParser()
# parser.add_argument('--v', type=float)
# args = parser.parse_args()
# rad = args.v
# #file1

#Font stuff
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['axes.linewidth'] = 1

def plotter(rad,dx,tsteps,Xmax,lam,wth,freqnum,npml,eps_in,obj,tfsf_dist,i):

    filename = 'TFSF_object{}_r{}_dx{}_T{}_X{}_lam{}_wdt{}_nfreq{}_npml{}_eps{}_tfsf{}'\
        .format(obj,rad,dx,tsteps,Xmax,lam,wth,freqnum,npml,eps_in,tfsf_dist)

    restored_table = pq.read_table('Results/'+filename+'.parquet')
    data = restored_table.to_pandas()
    restored_meta_json = restored_table.schema.metadata[custom_meta_key.encode()]
    meta = json.loads(restored_meta_json)

    'Data for Mie Python calculation'

    "Scattering cross section"
    r = meta['sphere'][0]*1e6  #radius in microns
    geometric_cross_section = np.pi * r**2

    #Johnson and Christy data
    name = "../materialdata/Johnson_Au.txt"
    au = np.genfromtxt(name, delimiter='\t')
    NNN = len(au)//2 # data is stacked so need to rearrange
    au_lam = au[1:NNN,0]
    au_mre = au[1:NNN,1]
    au_mim = au[NNN+1:,1]
    #calculate JohnsonChristy solution
    x = 2*np.pi*r/au_lam;m = au_mre - 1.0j * au_mim
    qext, qsca, qback, g = miepython.mie(m,x)
    scatt   = qsca * geometric_cross_section
    absorb  = (qext - qsca) * geometric_cross_section

    #Mie solution
    lam_mie = data['lambda']*1e6
    # data is stacked so need to rearrange
    eps = eps_in - meta['wp']**2/(data['omega']/meta['dt']*(data['omega']/meta['dt'] + 1j*meta['gamma']))
    epsR = np.real(eps)
    epsI = np.imag(eps)
    au_mre_mie = np.sqrt((np.sqrt(epsR**2+epsI**2)+epsR)/2)
    au_mim_mie = np.sqrt((np.sqrt(epsR**2+epsI**2)-epsR)/2)
    #calculate Mie solution
    x_mie = 2*np.pi*r/lam_mie;m_mie = au_mre_mie - 1.0j * au_mim_mie
    qext_mie, qsca_mie, qback_mie, g = miepython.mie(m_mie,x_mie)
    scatt_mie   = qsca_mie * geometric_cross_section
    absorb_mie  = (qext_mie - qsca_mie) * geometric_cross_section

    #Green function
    ax.plot(data["lambda"]*1e9,data["sigma_abs"]*1e12/(0.5*(data["source_re"]**2+data["source_im"]**2)),label='FDTD'+str(i))
    #ax.plot(au_lam*1000,scatt,label="Johnson"+str(i)) #MiePython using Johnson
    ax.plot(lam_mie*1000,absorb_mie,label='MiePython'+str(i)) #MiePython using Drude
    i +=1
    return i

def plotter_trans(rad,dx,tsteps,Xmax,lam,wth,freqnum,npml,obj,material,tfsf_dist):

    filename = 'periodic_object{}_material{}_r{}_dx{}_T{}_X{}_lam{}_wdt{}_nfreq{}_npml{}_tfsf{}'\
        .format(obj,material,rad,dx,tsteps,Xmax,lam,wth,freqnum,npml,tfsf_dist)

    restored_table = pq.read_table('Results/'+filename+'.parquet')
    data = restored_table.to_pandas()
    restored_meta_json = restored_table.schema.metadata[custom_meta_key_periodic.encode()]
    meta = json.loads(restored_meta_json)




    #Green function
    ax.plot(data["omega"]/meta['dt']*hbar/eC,data["trans"]**2/((data["source_re"]**2+data["source_im"]**2))/((meta['grid'])**4),label='Transmission')
    ax.plot(data["omega"]/meta['dt']*hbar/eC,data["ref"]**2/((data["source_re"]**2+data["source_im"]**2))/((meta['grid'])**4),label='Reflection')
    
    # ax.plot(data["omega"]/meta['dt']*hbar/eC,data["trans"]**2/((data["source_re"]**2+data["source_im"]**2))/((meta['grid'])**4),'b',linewidth=2.5,label='Transmission')
    # ax.plot(data["omega"]/meta['dt']*hbar/eC,data["ref"]**2/((data["source_re"]**2+data["source_im"]**2))/((meta['grid'])**4),'r',linewidth=2.5,label='Reflection')
    # #ax.plot(data["omega"]/meta['dt']*hbar/eC,1-data["ref"]**2/((data["source_re"]**2+data["source_im"]**2))/((meta['grid'])**4)-data["trans"]**2/((data["source_re"]**2+data["source_im"]**2))/((meta['grid'])**4),color='gold',linewidth=2.5,label='Absorption')

    if obj ==1:
        ax.set_title('3d code periodic boundary condition for nanoparticle with 50 nm radius')
    #ax.plot(data["omega"]/meta['dt']*hbar/eC,(data["source_re"]**2+data["source_im"]**2)/np.max((data["source_re"]**2+data["source_im"]**2)),label='Bandwidth',color='black')

    if obj ==2:
        if material==1:
            f=data['omega']/meta['dt']/c # - this is almost bang on??
            eps_in = 9
            wp = 1.26e16                    # gold plasma frequency
            gamma = 1.4e14                  # gold damping
            n1=1
            nb=(eps_in-wp**2/((data['omega']/meta['dt'])**2+1j*(data['omega']/meta['dt'])*gamma))**0.5
            # from b to 1
            r1 = (n1-nb)/(n1+nb)
            r2 = (nb-n1)/(nb+n1)

        if material ==2: 
            f=data['omega']/meta['dt']/c # - this is almost bang on??
            eps_in = 5.9673
            wp = 1.328e16
            gamma = 1e14
            wl = 4.08e15
            gamma_l = 6.59e14
            delta_eps =  1.09
            n1=1
            nb=(eps_in-wp**2/((data['omega']/meta['dt'])**2+1j*(data['omega']/meta['dt'])*gamma)+delta_eps*wl**2/(wl**2-(data['omega']/meta['dt'])**2-1j*gamma_l*(data['omega']/meta['dt'])))**0.5
            # from b to 1
            r1 = (n1-nb)/(n1+nb)
            r2 = (nb-n1)/(nb+n1)

        if material ==3: 
            f=data['omega']/meta['dt']/c # - this is almost bang on??
            eps_in = 1.53
            wp = 1.299e16
            gamma = 1.108e14
            w1 = 4.02489654142e15
            w2 = 5.69079026014e15
            gamma1 = 8.1897896143e14
            gamma2 = 2.00388466613e15
            A1 = 0.94
            A2 = 1.36
            n1=1
            nb=(eps_in-wp**2/((data['omega']/meta['dt'])**2+1j*(data['omega']/meta['dt'])*gamma)
            +A1*w1*(np.exp(-1j*np.pi/4)/(w1-(data['omega']/meta['dt'])-1j*gamma1) 
            + np.exp(+1j*np.pi/4)/(w1+(data['omega']/meta['dt'])+1j*gamma1))
            +A2*w2*(np.exp(-1j*np.pi/4)/(w2-(data['omega']/meta['dt'])-1j*gamma2) 
            + np.exp(+1j*np.pi/4)/(w2+(data['omega']/meta['dt'])+1j*gamma2)))**0.5
            # +np.sqrt(2)*A1*w1*(w1+gamma1-1j*(data['omega']/meta['dt']))/(w1**2-(data['omega']/meta['dt'])**2-1j*2*gamma1*(data['omega']/meta['dt']))
            # +np.sqrt(2)*A2*w2*(w2+gamma2-1j*(data['omega']/meta['dt']))/(w2**2-(data['omega']/meta['dt'])**2-1j*2*gamma2*(data['omega']/meta['dt'])))**0.5
            # # from b to 1
            r1 = (n1-nb)/(n1+nb)
            r2 = (nb-n1)/(nb+n1)

        # analytical solution
        film = rad*nm
        print(film)
        R1=(r1+r2*np.exp(2*1j*f*film*nb))/(1+r1*r2*np.exp(2*1j*f*film*nb))
        T1=(1+r1)*(1+r2)*np.exp(1*1j*f*film*nb)/(1+r1*r2*np.exp(2*1j*f*film*nb))
        ax.plot(data["omega"]/meta['dt']*hbar/eC,np.abs(T1)**2,'m--',linewidth=1.5,label='an. trans')
        ax.plot(data["omega"]/meta['dt']*hbar/eC,np.abs(R1)**2,'c--',linewidth=1.5,label='an. ref')
        ax.set_title('3d code for 100 nm slab')
'''
"Plot frequency dependent 1D monitor"
fig,ax = plt.subplots(1,1,figsize=(12, 6))
i=1
# i =plotter(rad = 30,dx = 2,tsteps = 10000,Xmax = 80,lam = 499,wth = 2,freqnum = 50,npml = 8,eps_in = 9,obj = 1,i=i)
# i=plotter(rad = 40,dx = 2,tsteps = 10000,Xmax = 80,lam = 499,wth = 2,freqnum = 50,npml = 8,eps_in = 9,obj = 1,i=i)
# i=plotter(rad = 50,dx = 2,tsteps = 10000,Xmax = 100,lam = 499,wth = 2,freqnum = 50,npml = 8,eps_in = 9,obj = 1,i=i)
# i=plotter(rad = 60,dx = 2,tsteps = 10000,Xmax = 100,lam = 499,wth = 2,freqnum = 50,npml = 8,eps_in = 9,obj = 1,i=i)
# i=plotter(rad = 60,dx = 2,tsteps = 10000,Xmax = 120,lam = 499,wth = 2,freqnum = 50,npml = 8,eps_in = 9,obj = 1,i=i)
i=plotter(rad = 100,dx = 10,tsteps = 2000,Xmax = 80,lam = 499,wth = 2,freqnum = 50,npml = 8,eps_in = 9,obj = 1,tfsf_dist =15,i=i)
i=plotter(rad = 100,dx = 5,tsteps = 2000,Xmax = 120,lam = 499,wth = 2,freqnum = 50,npml = 8,eps_in = 9,obj = 1,tfsf_dist =15,i=i)
#i=plotter(rad = 100,dx = 20,tsteps = 2000,Xmax = 100,lam = 499,wth = 2,freqnum = 50,npml = 8,eps_in = 9,obj = 1,tfsf_dist =15,i=i)
#i=plotter(rad = 50,dx = 5,tsteps = 10000,Xmax = 80,lam = 499,wth = 2,freqnum = 50,npml = 8,eps_in = 9,obj = 1,tfsf_dist =25,i=i)
#i=plotter(rad = 50,dx = 5,tsteps = 10000,Xmax = 80,lam = 499,wth = 2,freqnum = 50,npml = 8,eps_in = 9,obj = 1,tfsf_dist =30,i=i)
ax.set_title('Scat Cross section')
ax.set_xlim(450,650)
ax.set_ylim((0, .5))
ax.set_xlabel('$\lambda$ [nm]')
ax.set_ylabel('Scatt Cross section [m$^{-3}$]')

ax.legend()
plt.show()
'''




fig,ax = plt.subplots(1,1,figsize=(3.54, 2.655))
#plotter_trans(rad = 100,dx = 5,tsteps = 1000,Xmax = 120,lam = 499,wth = 2,freqnum = 50,npml = 8,eps_in = 9,obj = 2,tfsf_dist = 15)
#plotter_trans(rad = 100,dx = 5,tsteps = 2000,Xmax = 120,lam = 499,wth = 2,freqnum = 50,npml = 8,eps_in = 9,obj = 2,tfsf_dist = 15)
#plotter_trans(rad = 100,dx = 5,tsteps = 5000,Xmax = 120,lam = 499,wth = 2,freqnum = 50,npml = 8,eps_in = 9,obj = 2,tfsf_dist = 15)
#plotter_trans(rad = 100,dx = 5,tsteps = 10000,Xmax = 120,lam = 499,wth = 2,freqnum = 50,npml = 8,eps_in = 9,obj = 2,tfsf_dist = 15)

#plotter_trans(rad = 50,dx = 2,tsteps = 20000,Xmax = 120,lam = 539,wth = 2,freqnum = 100,npml = 8,eps_in = 9,obj = 2,material =1,tfsf_dist = 15)
plotter_trans(rad = 150,dx = 10,tsteps = 5000,Xmax = 60,lam = 539,wth = 2,freqnum = 100,npml = 8,obj = 2,material =1,tfsf_dist = 12)
#plotter_trans(rad = 50,dx = 2,tsteps = 20000,Xmax = 120,lam = 539,wth = 2,freqnum = 100,npml = 8,eps_in = 9,obj = 1,material =3,tfsf_dist = 15)
#plotter_trans(rad = 50,dx = 2,tsteps = 20000,Xmax = 200,lam = 539,wth = 2,freqnum = 100,npml = 8,eps_in = 9,obj = 1,material =3,tfsf_dist = 15)

#plotter_trans(rad = 100,dx = 2,tsteps = 20000,Xmax = 200,lam = 539,wth = 2,freqnum = 100,npml = 8,eps_in = 9,obj = 1,tfsf_dist = 15)
ax.set_xlim(2,3)
#ax.set_ylim((0, 1.5))
ax.set_xlabel('$\hbar\omega$ [eV]',labelpad=10,fontsize=9)
ax.set_ylabel('$R,T$',labelpad=10,fontsize=9)
#ax.set_ylabel('Reflect [m$^{-3}$]')

ax.legend()
plt.savefig('Results/periodic_100particle.pdf')
plt.show()