#from symbol import varargslist
#from tkinter import Variable
import numpy as np
from matplotlib import pyplot as plt
import imageio.v2 as imageio
import argparse
import warnings

import grid
import modules.parameters as par

warnings.filterwarnings('ignore')

##############################################

parser = argparse.ArgumentParser()             
parser.add_argument("varA",  type=int)      #n_kmax
parser.add_argument("varB",  type=int)      #dx
parser.add_argument("varC",  type=float)    #dt

args = parser.parse_args() 

n_kmax = args.varA
dx = args.varB
dt = args.varC

tsteps=500000
cycle=100

##############################################
path = 'Plots/Plots_nkmax{}_dx{}nm_dt{}as'.format(n_kmax,int(dx),np.round(dt,2))

##############################################

images = []

for time_step in range(1,int(tsteps/cycle/10)):
  
  save_name = path+'/animation_time{}.png'.format(int(time_step*10))

  images.append(imageio.imread(save_name))

imageio.mimsave('movie.gif', images)