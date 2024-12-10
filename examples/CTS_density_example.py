# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 23:09:53 2024

@author: Jalal
"""


import numpy as np
import matplotlib.pyplot as plt
import time
from lstable.CTS_distribution import CTS_density, CTS_generator_Bauemer_vectorial
import sys


# =============================================================================
# Density plots
# =============================================================================


def plot_CTS_density(x_grid: np.ndarray, alpha: float, P: float, Q: float, A: float, B: float):
    density = CTS_density(x_grid, alpha, P, Q, A, B, adaptive_bound=True)
    plt.figure()
    plt.plot(x_grid, density)
    plt.show()


x_grid = np.linspace(-10, 10, 1000)
alpha, P, Q, A, B = 1.5, 1, 0.5, 10, 1
plot_CTS_density(x_grid, alpha, P, Q, A, B)


# =============================================================================
# Histogram plots
# =============================================================================


def histogram_zoom(array: np.ndarray, minimum_bound: float, maximum_bound: float, nb_bins: int):
    """
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
    """
    min_array = np.min(array)
    max_array = np.max(array)
    if min_array > minimum_bound:
        raise ValueError("minimum bound not included in the support of the array")
    if max_array < maximum_bound:
        raise ValueError("maximum bound not included in the support of the array")

    bins = np.linspace(minimum_bound, maximum_bound, nb_bins)
    bins = np.concatenate(([min_array], bins, [max_array]))  # Adding the bounds as one range value
    values, bins = np.histogram(array, bins=bins, density=True)
    return values[1:-1], bins[1:-1]  # dropping the bins and values between the extremums and the desired bounds.


def plot_stable_density_histogram(
    alpha: float,
    P: float,
    Q: float,
    A: float,
    B: float,
    plot_grid: np.ndarray,
    nb_sample: int,
    nb_bins: int,
    c: float = 0,
    density=True,
    histogram=True,
):
    """Draws samples of CTS distribution and plots the histogram with nb_bins bins of and the corresponding density


    Parameters:
    alpha : float
        Stability index (0 < alpha <= 2).
    P : float
        Positive jump parameter (P >= 0).
    Q : float
        Negative jump parameter (Q >= 0).
    A : float
        Positive jump tempering parameter (A >= 0).
    B : float
        Negative jump tempering parameter (B >= 0).
    c : float
        Approximation parameter (default value is 0)
    nb_sample: int
        Number of sample values of CTS distribution with parameter alpha,P,Q,A,B
    plot_grid: np.ndarray
        Grid of display
    nb_bins:int
        number of bins for the histogram
    Density: bool
        if True, the plot displays the theoretical density of the CTS distribution
    Histogram: bool
        if True, the plot displays the normalized histogram of the random sample of CTS

    """
    # One of the density or the histogram has to appear
    if density and histogram == False:
        raise ValueError("One of the density or the histogram have to be plotted")

    # bound of the plot grid
    minimum_bound = plot_grid[0]
    maximum_bound = plot_grid[-1]

    plt.figure()

    if density == True:
        density = CTS_density(plot_grid, alpha, P, Q, A, B, adaptive_bound=True)
        plt.plot(plot_grid, density, label="density")
    if histogram == True:
        sample_array = CTS_generator_Bauemer_vectorial(alpha, P, Q, A, B, nb_sample, c, verbose=True)
        values, bins = np.histogram(
            sample_array, density=True, bins=nb_bins
        )  # histogram_zoom(sample_array, minimum_bound,maximum_bound,nb_bins)
        plt.stairs(values, bins)
    plt.show()


plot_grid = np.linspace(-5, 5, 1000)
c = 5
nb_sample = 1000
nb_bins = 100
alpha, P, Q, A, B = 0.5, 1, 1, 1, 1
plot_stable_density_histogram(alpha, P, Q, A, B, plot_grid, nb_sample, nb_bins, c)
