import numpy as np
from matplotlib import pyplot as plt
import scipy
from math import e
import math
from scipy import integrate as inte
import time
import os
import argparse
from scipy.optimize import curve_fit

# #Own modules:
# import constants_jonas as cj
# import physical_functions as pf
# import boltzmann as boltz
# import grid

##############################################

parser = argparse.ArgumentParser()                                 
parser.add_argument("varA",  type=int)      #n_kmax
parser.add_argument("varB",  type=int)      #n_phimax
parser.add_argument("varC",  type=int)      #n_thetamax
parser.add_argument("varD",  type=float)    #width of the Energy interval
parser.add_argument("varF",  type=float)    #E_F 
parser.add_argument("varG",  type=float)    #T_electron

args = parser.parse_args() 

E_F = args.varF
T_ele = args.varG

##############################################
'''
Define grids from the grid.py module
'''
k_min=grid.k_min    #Unit is 1/nm
k_max=grid.k_max    #Unit is 1/nm
n_kmax=grid.n_kmax
dk=grid.dk

phi_max=grid.phi_max
n_phimax=grid.n_phimax
dphi=grid.dphi

theta_max=grid.theta_max
n_thetamax=grid.n_thetamax
dtheta=grid.dtheta

t_max=grid.t_max
n_tmax= grid.n_tmax
dt=grid.dt

k_grid = grid.k_grid(n_kmax)

t_grid = grid.t_grid(n_tmax)

##############################################
'''
Load numerical data
'''
path = "data"+str(n_kmax)+'_'+str(n_phimax)+'_'+str(n_thetamax)

path_time=path+'/time_'+str(n_kmax)+'.npy'
path_wignermatrix=path+'/wignermtx_'+str(n_kmax)+'.npy'
path_currentdensity=path+'/currdens_'+str(n_kmax)+'.npy'

t_values=np.load(path_time)
wignermatrix=np.load(path_wignermatrix)
current_density=np.load(path_currentdensity)

wignerpolar, wignerfunction = pf.calculate_wigners(wignermatrix)
energy, occupation = pf.calculate_energy_occupation(wignermatrix)

wignerpolar_0=np.full((n_tmax,n_kmax), 0.)
wignerpolar_pi=np.full((n_tmax,n_kmax), 0.)
wignerpolar_pi2=np.full((n_tmax,n_kmax), 0.)
wignerpolar_3pi2=np.full((n_tmax,n_kmax), 0.)
j_abs=np.full((n_tmax),0.)
j_abssquare=np.full((n_tmax),0.)

for n_t in range(n_tmax):
    for n_kk in range(n_kmax):
        wignerpolar_0[n_t][n_kk]=wignermatrix[n_t][n_kk][0][int(n_thetamax/2)]
        wignerpolar_pi[n_t][n_kk]=wignermatrix[n_t][n_kk][int(cj.pi/dphi)][int(n_thetamax/2)]
        wignerpolar_pi2[n_t][n_kk]=wignermatrix[n_t][n_kk][int(cj.pi/2./dphi)][int(n_thetamax/2)]
        wignerpolar_3pi2[n_t][n_kk]=wignermatrix[n_t][n_kk][int(3.*cj.pi/2./dphi)][int(n_thetamax/2)]
    for i in range(3):
      j_abssquare[n_t] += np.power(current_density[i][n_t],2)
    j_abs[n_t] = np.sqrt(j_abssquare[n_t])

#print(wignerpolar_0[n_tmax-1])
##############################################
'''
Color and time scheme
'''
count_time = [int(0.0*n_tmax),int(0.025*n_tmax),int(0.05*n_tmax),int(0.1*n_tmax),int(0.2*n_tmax),int(0.4*n_tmax),int(0.8*n_tmax)]
color = ['#3675bc','#4e6aa0','#655f84','#7d5468','#95494c','#ac3e30','#c43314']

##############################################
'''
Fitting
'''
if False:
  def fermi_fit(kk, T,mu):
    result=1./(np.exp((pf.electron_dispersion(kk,0.)-mu)/(cj.kB*T))+1)
    return result

  def fermi_fit2(kk,T,mu):
    result=-1./(np.exp((pf.electron_dispersion(kk,0.)-E_F)/(cj.kB*cj.T_cryo))+1)+1./(np.exp((pf.electron_dispersion(kk,0.)-mu)/(cj.kB*T))+1)
    return result

  y_data=np.full(n_kmax,0.)
  y_data=wignerfunction[0][:]

  popt = curve_fit(fermi_fit, k_grid, y_data, bounds=([0.,-2.],[5000.,8.]))
  T1=popt[0][0]
  mu1=popt[0][1]
  print(T1,mu1)

  y_data=np.full(n_kmax,0.)
  y_data=wignerfunction[count_time[6]][:]

  popt = curve_fit(fermi_fit, k_grid, y_data, bounds=([0.,-2.],[5000.,8.]))
  T2=popt[0][0]
  mu2=popt[0][1]
  print(T2,mu2)

##############################################
'''
Plot data
'''
plt.subplots_adjust(wspace=.5, hspace=0.4)
plt.suptitle('Various plots of $f_k$(t) \n Size of grid (k,$\phi$, $\Theta$) = ('+str(n_kmax)+','+str(n_phimax)+','+str(n_thetamax)+')    kmin= '+str(np.round(k_min, 1))+'; kmax= '+str(np.round(k_max, 1))+'    $E_F$ = '+str(E_F)+' eV')

plt.subplot(241)

for j in range(7):
  #plt.plot(pf.electron_dispersion(k_grid,0.),fermi_fit(k_grid,cj.T_cryo, E_F)+wignerfunction[count_time[j]],color = color[j])
  plt.plot(pf.electron_dispersion(k_grid),wignerfunction[count_time[j]],color = color[j])
plt.title('Mean value for all $\Theta, \phi$')
plt.xlabel("Kinetic Energy [eV]")
plt.ylabel("Electron Occupation [a.u.]")

plt.subplot(242)

for j in range(7):
  plt.plot(pf.electron_dispersion(k_grid),wignerpolar_pi[count_time[j]],color = color[j])
plt.title('$\phi$ =$\pi$, $\Theta$ = $\pi$/2')
plt.xlabel("Kinetic Energy [eV]")
plt.ylabel("Electron Occupation [a.u.]")

plt.subplot(243)

for j in range(7):
  plt.plot(pf.electron_dispersion(k_grid),wignerpolar_0[count_time[j]],color = color[j])
plt.title('$\phi$ =0, $\Theta$ = $\pi$/2')
plt.xlabel("Kinetic Energy [eV]")
plt.ylabel("Electron Occupation [a.u.]")
#plt.ylim(-.005,0.005)
#plt.xlim(4.5, 5)

plt.subplot(244)

#plt.plot(t_values/1000,np.transpose(energy),color = 'black')
#plt.title('Total electron energy')
#plt.ylabel("Total Energy [eV]")
plt.plot(t_values/1000,np.transpose(j_abs),color = '#90d743', linewidth=2.5)
plt.plot(t_values/1000,np.transpose(current_density[0]),color = '#ea4044')
plt.plot(t_values/1000,np.transpose(current_density[1]),color = '#e26a42', linestyle='dashed')
plt.plot(t_values/1000,np.transpose(current_density[2]),color = '#db9e3f', linestyle='dotted')
plt.title('Current density')
plt.legend(['| J|', 'J_x', 'J_y', 'J_z'])
plt.ylabel("Current density [eC/nm³]")
plt.xlabel("Time [ps]")
#plt.ylim(-0.00001,0)

plt.subplot(245)

plt.plot(pf.electron_dispersion(k_grid),wignerfunction[count_time[0]],color =color[0])
plt.plot(pf.electron_dispersion(k_grid),wignerfunction[count_time[6]],color =color[6])
#plt.plot(pf.electron_dispersion(k_grid,0.),fermi_fit(k_grid, T1,mu1),'y:')
#plt.plot(pf.electron_dispersion(k_grid,0.),fermi_fit(k_grid, T2,mu2),'b--')
#plt.legend(['f(t=0)', 'f(t='+str(count_time[6]*dt/1000)+')', str('%4.0f' %T1)+' K', str('%4.0f' %T2)+' K'])
plt.xlabel("Kinetic Energy [eV]")
plt.ylabel("Electron Occupation [a.u.]")

plt.subplot(246)

for j in range(7):
  plt.plot(pf.electron_dispersion(k_grid),wignerpolar_3pi2[count_time[j]],color = color[j])
plt.title('$\phi$ =$3\pi/2$, $\Theta$ = $\pi$/2')
plt.xlabel("Kinetic Energy [eV]")
plt.ylabel("Electron Occupation [a.u.]")

plt.subplot(247)

for j in range(7):
  plt.plot(pf.electron_dispersion(k_grid),wignerpolar_pi2[count_time[j]],color = color[j])
plt.title('$\phi$ =$\pi/2$, $\Theta$ = $\pi$/2')
plt.xlabel("Kinetic Energy [eV]")
plt.ylabel("Electron Occupation [a.u.]")

plt.subplot(248)

plt.plot(t_values/1000, np.transpose(occupation),color = 'black')
plt.title('Total electron occupation')
plt.xlabel("Time [ps]")
plt.ylabel("Electron Occupation [a.u.]")

print('Electron number at t=0:', occupation[0])
print('Electron number at t=tmax:', occupation[n_tmax-1])
print('Difference:', occupation[0]-occupation[n_tmax-1])

plt.show()