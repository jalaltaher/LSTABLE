# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 20:30:26 2024

@author: Jalal
"""

import numpy as np
from scipy.integrate import romb, quad, simpson
import scipy.stats as st
import matplotlib.pyplot as plt
from math import cos, gamma, pi,log,tan,sin,floor,exp,ceil,sqrt
from scipy.special import sici,gamma, gammainc, gammaincc,gammaincinv
from scipy.integrate import quad
import scipy.optimize as so
import mpmath
import timeit
from mpmath import gammainc
import mpmath as mp
from scipy.optimize import brentq
from Non_Parametric_Fourier_Estimation_Adaptive_Methods import*


def valid_stable_parameters(alpha:float, P:float, Q:float, A:float, B:float)->bool:
    '''
    Checks if the parameters are in a valid domain for CTS distributions.
    
    Parameters
    ----------
    alpha : float
        Stability index (0<alpha<=2)
    P : float
        Positive jump parameter (P>=0)
    Q : float
        Negative jump parameters (Q>=0)
    A : float
        Positive jump tempering parameter (A>=0)
    B : float
        Negative jump tempereing parameter (B>=0)

    Raises
    ------
    ValueError
        If one the parameter is not in the valid domain
    Returns
    -------
    bool
        return True if the parameter are valid

    '''
    if not (0 < alpha <= 2):
        raise ValueError("alpha must satisfy 0 < alpha <= 2.")
    if P < 0 or Q < 0 or A < 0 or B < 0:
        raise ValueError("P, Q, A, and B must be non-negative.")
    if P==0 and Q==0:
        raise ValueError("P or Q must be positive.")
    if A==0 and B==0:
        raise ValueError("There is no tempering it is a stable distribution.")
# =============================================================================
# Tempered stable density via Fourier inversion
# =============================================================================


def CTS_characteristic_function(grid: np.ndarray,alpha:float,P:float,Q:float,A:float,B:float)->np.ndarray:
    '''
    Computes the characteristic function of a CTS process at time Delta

    Parameters
    ----------
    grid : np.ndarray
        Grid where the characteristic function is computed

    alpha : float
        Stability index (0<alpha<=2)
    P : float
        Positive jump parameter (P>=0)
    Q : float
        Negative jump parameters (Q>=0)
    A : float
        Positive jump tempering parameter (A>=0)
    B : float
        Negative jump tempereing parameter (B>=0)

    Returns
    -------
    res : np.ndarray
        array with an evaluation of the characteristic function of CTS distribution

    '''
    
    # Input validation
    valid_stable_parameters(alpha, P, Q, A, B)
        
    # Prior computation
    gamma_alpha= gamma(-alpha)
    
    # Initialization the result
    u=grid
    res=np.ones_like(u, dtype=np.complex128)
    
    if P > 0:
        term_P = (A - 1.j * u) ** alpha - A ** alpha + 1.j * u * alpha * A ** (alpha - 1)
        res *= np.exp(P * gamma_alpha * term_P)
    if Q > 0:
        term_Q = (B + 1.j * u) ** alpha - B ** alpha - 1.j * u * alpha * B ** (alpha - 1)
        res *= np.exp(Q * gamma_alpha * term_Q)   
    return res


def adaptive_integration_bound_for_CTS_Fourier_inverse(
    alpha: float, P: float, Q: float, A: float, B: float,
    epsilon: float = 0.01, x_min: float = 0, x_max: float = 1e5, verbose: bool = False
) -> float:
    """
    Finds the root of the equation |CTS_characteristic_function(u)| = epsilon
    for the CTS characteristic function (which vanishes to 0 at infinity).
    Equivalently, determines the bound [-b, b] where most of the mass is concentrated.

    Parameters
    ----------
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
    epsilon : float, optional
        Precision parameter for the characteristic function. Default is 0.01.
    x_min : float, optional
        Minimum value of the search interval. Default is 0.
    x_max : float, optional
        Maximum value of the search interval. Default is 1e5.
    verbose : bool, optional
        If True, prints diagnostic messages. Default is False.

    Returns
    -------
    float or None
        The root of the equation |CTS_characteristic_function(u)| = epsilon,
        or None if no root is found.

    Raises
    ------
    ValueError
        If input parameters are invalid or if no root is found.
    """

    # Validate parameters
    valid_stable_parameters(alpha, P, Q, A, B)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if x_min >= x_max:
        raise ValueError("x_min must be less than x_max.")

    def g(u):
        return np.abs(CTS_characteristic_function(u, alpha, P, Q, A, B)) - epsilon

    try:
        root = brentq(g, x_min, x_max)
        if verbose:
            print(f"Root found at x = {root}")
        return root
    except ValueError as e:
        if verbose:
            print(f"No root found in the interval [{x_min}, {x_max}]: {e}")
        return None

def CTS_density(
    grid: np.ndarray, alpha: float, P: float, Q: float, A: float, B: float,
    adaptive_bound: bool = False, integration_step: float = 0.01, integration_grid: np.ndarray = None
) -> np.ndarray:
    ''' 
    Computes the density of the CTS distribution by Fourier inversion of the characteristic function. 
    If adaptive_bound=True, it uses an adaptive integration bound. 
    If adaptive_bound is False, it uses the provided integration grid.

    Parameters
    ----------
    grid : np.ndarray
        Grid on which the density will be computed.
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
    adaptive_bound : bool, optional
        If True, adaptive integration bounds will be used. Default is False.
    integration_step : float, optional
        Step size for generating the integration grid. Default is 0.01.
    integration_grid : np.ndarray, optional
        Predefined integration grid. Used only if adaptive_bound=False. Default is None.

    Returns
    -------
    np.ndarray
        Array representing the computed density.
    '''
    # Validate parameters (custom validation function or use if already defined)
    valid_stable_parameters(alpha, P, Q, A, B)

    # Adaptive grid generation if needed
    if adaptive_bound:
        integration_bound = adaptive_integration_bound_for_CTS_Fourier_inverse(alpha, P, Q, A, B)
        integration_grid = np.arange(-integration_bound, integration_bound, integration_step)
    
    # Ensure integration_grid is defined
    if integration_grid is None:
        raise ValueError("integration_grid must be provided or adaptive_bound must be True")

    # Compute characteristic function
    CTS_cf = CTS_characteristic_function(integration_grid, alpha, P, Q, A, B)

    # Perform Fourier inversion to get the density
    try:
        res = density_by_fourier_inversion(CTS_cf, grid, integration_grid)
    except Exception as e:
        print(f"Error in Fourier inversion: {e}")
        return None

    return res




# =============================================================================
# Bauemer algorithm
# =============================================================================




# =============================================================================
# Compound Poisson approximation + Gaussian approximation of the small jumps
# =============================================================================




# =============================================================================
# 
# =============================================================================
