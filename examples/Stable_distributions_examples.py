# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 21:39:53 2024

@author: Jalal
"""

import numpy as np
import matplotlib.pyplot as plt 
from math import floor,sqrt
import scipy.stats as st
from functions.Stable_distribution import stable_distribution_generator,stable_density


# =============================================================================
# Histograms and densities with varying 
# =============================================================================


def histogram_zoom(array: np.ndarray,minimum_bound: float,maximum_bound:float,nb_bins: int):
    ''' 
    Computes bins and heights of a normalized histogram 
    of an array zooming on values that are inside 
    [infimum_bound, supremum_bound].
    
    Parameters: 
        array: (np.ndarray) array of values
        infimum_bound: (float) lower bound of the zoom box
        supremum_bound: (float) upper bound of the zoom box
        nb_bins: (int) number of bins for the histogram
    
    Returns:
        tuple: (values, bins)
            values: Heights of the histogram bins in [infimum_bound, supremum_bound].
            bins: Edges of the histogram bins in [infimum_bound, supremum_bound].
    '''
    min_array=np.min(array)
    max_array=np.max(array)
    if min_array> minimum_bound:
        raise ValueError('minimum bound not included in the support of the array')
    if max_array<maximum_bound:
        raise ValueError('maximum bound not included in the support of the array')
        
    bins=np.linspace(minimum_bound,maximum_bound,nb_bins) 
    bins=np.concatenate(([min_array],bins,[max_array])) #Adding the bounds as one range value
    values,bins=np.histogram(array,bins=bins,density=True)
    return values[1:-1],bins[1:-1] #dropping the bins and values between the extremums and the desired bounds.


def plot_stable_density_histogram(alpha: float,sigma:float, beta:float,mu:float,nb_sample: int, plot_grid: np.ndarray, nb_bins: int, density=True, histogram=True):
    ''' Plot a stable density and the histogram with nb_bins bins of n_sample drawing of S_alpha(sigma,beta,mu) in a given grid
    
    Parameters: 
    
    nb_sample: (int) number of sample values of S_\alpha(sigma,beta,mu)
    plot_grid: (np.ndarray) grid of display
    nb_bins: (int) number of bins for the histogram
    Density: (bool) 
    
    '''
    #One of the density or the histogram has to appear
    if density and histogram==False:
        raise ValueError('One of the density or the histogram have to be plotted')
    
    #bound of the plot grid
    minimum_bound= plot_grid[0]
    maximum_bound= plot_grid[-1]
    
    plt.figure()
    
    if density==True:
        density=stable_density(plot_grid,alpha,sigma,beta,mu)
        plt.plot(plot_grid,density,label='density')
    if histogram==True:
        sample_array=stable_distribution_generator(alpha, sigma, beta, mu, nb_sample)
        values,bins=histogram_zoom(sample_array, minimum_bound,maximum_bound,nb_bins)
        plt.stairs(values,bins,color='red')
    plt.show()
    


alpha,sigma,beta,mu=1.5,1.0,0.5,0
nb_sample=10000
plot_grid=np.linspace(-5,5,1000)
nb_bins=50
plot_stable_density_histogram(alpha,sigma,beta,mu,nb_sample, plot_grid, nb_bins)