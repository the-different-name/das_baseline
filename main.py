"""

"""


import sys
import numpy as np

import matplotlib.pyplot as plt
from expspec import *

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['figure.figsize'] = [6.0, 3.2]
plt.style.use('ggplot')
plt.rcParams['axes.edgecolor'] = 'black'



if __name__ == '__main__':
    
    # # graphene test spectrum:
    testspec = np.genfromtxt('test_data/graphene_oxide_Raman_spectrum.txt')
    testspec = ExpSpec(testspec[:,0], testspec[:,1]); testspec.working_range = (400, 3700)
    olam,op = find_both_optimal_parameters(testspec)

    # # aq test spectrum:
    # testspec = np.genfromtxt('test_data/water_hyper_Raman_spectrum.txt')
    # testspec = ExpSpec(testspec[:,0], testspec[:,1])
    # olam,op = find_both_optimal_parameters(testspec)


    # # achitin test spectrum:
    # testspec = np.genfromtxt('test_data/achitin_Raman_spectrum.txt')
    # testspec = ExpSpec(testspec[:,0], testspec[:,1]); testspec.working_range = (200, 3700)    
    # olam,op = find_both_optimal_parameters(testspec)
    


