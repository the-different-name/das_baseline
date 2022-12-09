"""
multipeak fit

- whether lambda should be normalized by pixel width?
- add timing if display > 0
- add flexible lambda with scaling wrt pixel width and number
- make the peaks only positive
"""

import sys
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from datetime import datetime

from scipy.sparse.linalg import spsolve
from copy import deepcopy
from scipy import signal, stats, sparse

sys.path.append('D:/scripts/python/expspec') # at lab
# or / and
sys.path.append('/driveD/scripts/python/expspec') # at Daruma

from spectralfeature import voigt_asym, MultiPeak, CalcPeak
import matplotlib.pyplot as plt
from openpyxl import load_workbook, Workbook
from expspec import *
from itertools import cycle
import colorsys
from numpy.polynomial.polynomial import polyfit, polyval
from scipy.stats import moment
from scipy.optimize import Bounds

import warnings
warnings.simplefilter('ignore',sparse.SparseEfficiencyWarning)
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['figure.figsize'] = [6.0, 3.2]
plt.style.use('ggplot')

from csaps import csaps

# import pywt


def _get_colors(num_colors):
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors


def plot_the_funcamential(testspec, autolam=None, number_of_points_to_scan=49):

    if not autolam:
        olam,op = find_both_optimal_parameters(testspec)
        autolam = olam
        
    thefuncamential = np.zeros((number_of_points_to_scan, number_of_points_to_scan))
    thefuncamential_blpart = np.zeros_like(thefuncamential)
    thefuncamential_ppart = np.zeros_like(thefuncamential)
    
    p_s = np.zeros_like(thefuncamential)
    lam_s = np.zeros_like(thefuncamential)

    lambda_range = np.exp(np.linspace(np.log(autolam/1000), np.log(autolam*1000), number_of_points_to_scan))
    p_range = np.exp(np.linspace(np.log(8e-5), np.log(0.8), number_of_points_to_scan))

    d2bl = spectral_powers(testspec, display=0)
    for i in range(number_of_points_to_scan): # i ~ lam
        for j in range(number_of_points_to_scan): # j ~ p
            current_bl, current_weights = das_baseline(testspec,
                                                       als_lambda=lambda_range[i],
                                                       als_p_weight=p_range[j],
                                                       display=0,
                                                       return_weights=True)

            current_residuals = testspec.y - current_bl
            current_residuals_weighted = (testspec.y - current_bl) * current_weights
            negativeresiduals = current_residuals * (testspec.y < current_bl) #  / sum(testspec.y < current_bl)
            d2bl_current = (np.mean(np.diff(current_bl, 2)**2))**0.5
            thefuncamential_blpart[i,j] = 2 * (d2bl-d2bl_current)**2 / (d2bl**2 + d2bl_current**2)
            neg_residuals_without_positive = negativeresiduals[np.where(testspec.y < current_bl)]
    
            # asymmetry index:
            weighted_positive_residuals_without_negative = current_residuals_weighted[np.where(testspec.y > current_bl)]
            positive_median = np.median(weighted_positive_residuals_without_negative)
            m4lvl_of_negativeresiduals_good = ( np.mean(neg_residuals_without_positive**4) )**0.25
            thefuncamential_ppart[i,j] = 2 * (positive_median-m4lvl_of_negativeresiduals_good)**2 / (positive_median**2+m4lvl_of_negativeresiduals_good**2)

            thefuncamential[i,j] = thefuncamential_blpart[i,j] + thefuncamential_ppart[i,j]

            lam_s[i,j] = lambda_range[i]
            p_s[i,j] = p_range[j]

    theaspect = 0.32 
    extent = np.log([min(p_range), max(p_range), min(lambda_range), max(lambda_range)])
    x_label_list = (np.linspace(np.log(min(p_range)), np.log(max(p_range)), num=5))
    x_label_list_string = np.char.mod('%.1e', np.exp(x_label_list)).tolist()
    y_label_list = (np.linspace(np.log(min(lambda_range)), np.log(max(lambda_range)), num=5))
    y_label_list_string = np.char.mod('%.1e', np.exp(y_label_list)).tolist()


    def plot_results_in_matshow(datname, filename=''):
        save_it_path = 'current_output/thefuncamential'
        Path(save_it_path).mkdir(parents=True, exist_ok=True)
        plt.matshow(np.flipud(datname), cmap=plt.cm.nipy_spectral_r, aspect = theaspect, extent=extent)
        cbar = plt.colorbar(fraction=0.032*1/1.875, pad=0.04)
        cbar.ax.tick_params(labelsize=4) 
        plt.title(filename, fontsize=6)
        plt.xticks(x_label_list, x_label_list_string, fontsize=6)
        plt.yticks(y_label_list, y_label_list_string, fontsize=6)
        plt.xlabel('als p', fontsize=6)
        plt.ylabel('als lambda', fontsize=6)
        plt.savefig(save_it_path + '/' + filename +'.png', bbox_inches='tight', transparent=True)
        plt.savefig(save_it_path + '/' + filename +'.eps', bbox_inches='tight', transparent=True)
        plt.show()
        
    plot_results_in_matshow(thefuncamential_blpart, 'thefuncamential_blpart')
    plot_results_in_matshow(thefuncamential_ppart, 'thefuncamential_ppart')
    plot_results_in_matshow(thefuncamential, 'thefuncamential')    
    



if __name__ == '__main__':
    
    # # graphene test spectrum:
    testspec = np.genfromtxt('test_data/graphene_oxide_Raman_spectrum.txt')
    testspec = ExpSpec(testspec[:,0], testspec[:,1]); testspec.working_range = (400, 3700)
    plot_the_funcamential(testspec, autolam=3.29e9)

    # # aq test spectrum:
    # testspec = np.genfromtxt('test_data/water_hyper_Raman_spectrum.txt')
    # testspec = ExpSpec(testspec[:,0], testspec[:,1])
    # olam,op = find_both_optimal_parameters(testspec)


    # # achitin test spectrum:
    # testspec = np.genfromtxt('test_data/achitin_Raman_spectrum.txt')
    # testspec = ExpSpec(testspec[:,0], testspec[:,1]); testspec.working_range = (200, 3700)    
    # olam,op = find_both_optimal_parameters(testspec)
    


