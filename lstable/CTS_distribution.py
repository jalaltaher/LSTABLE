# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 20:30:26 2024

@author: Jalal
"""

import numpy as np
import scipy.stats as st
from math import cos, gamma, pi, log, tan, sin, floor, exp, ceil, sqrt
from scipy.integrate import quad
import time
from scipy.optimize import brentq
from tqdm import trange
from .Stable_distribution import stable_distribution_generator
from .Fourier_inversion import density_by_fourier_inversion


def valid_Tstable_parameters(alpha: float, P: float, Q: float, A: float, B: float) -> bool:
    """
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

    """
    if not (0 < alpha <= 2):
        raise ValueError("alpha must satisfy 0 < alpha <= 2.")
    if P < 0 or Q < 0 or A < 0 or B < 0:
        raise ValueError("P, Q, A, and B must be non-negative.")
    if P == 0 and Q == 0:
        raise ValueError("P or Q must be positive.")
    if A == 0 and B == 0:
        raise ValueError("There is no tempering it is a stable distribution.")


# =============================================================================
# Tempered stable density via Fourier inversion
# =============================================================================


def CTS_characteristic_function(grid: np.ndarray, alpha: float, P: float, Q: float, A: float, B: float) -> np.ndarray:
    """
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

    """

    # Input validation
    valid_Tstable_parameters(alpha, P, Q, A, B)

    # Initialization the result
    u = grid
    res = np.ones_like(u, dtype=np.complex128)
    if alpha != 1:
        # Prior computation
        gamma_alpha = gamma(-alpha)

        if P > 0:
            term_P = (A - 1.0j * u) ** alpha - A**alpha + 1.0j * u * alpha * A ** (alpha - 1)
            res *= np.exp(P * gamma_alpha * term_P)
        if Q > 0:
            term_Q = (B + 1.0j * u) ** alpha - B**alpha - 1.0j * u * alpha * B ** (alpha - 1)
            res *= np.exp(Q * gamma_alpha * term_Q)
    else:
        if P > 0:
            term_P = P * (A - 1.0j * u) * np.log(1 - 1.0j * u / A)
            res *= np.exp(term_P)
        if Q > 0:
            term_Q = (B + 1.0j * u) ** alpha - B**alpha - 1.0j * u * alpha * B ** (alpha - 1)
            res *= np.exp(term_Q)
        res *= np.exp(1.0j * u * (P - Q))
    return res


def adaptive_integration_bound_for_CTS_Fourier_inverse(
    alpha: float,
    P: float,
    Q: float,
    A: float,
    B: float,
    epsilon: float = 0.01,
    x_min: float = 0,
    x_max: float = 1e5,
    verbose: bool = False,
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
    valid_Tstable_parameters(alpha, P, Q, A, B)
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
    grid: np.ndarray,
    alpha: float,
    P: float,
    Q: float,
    A: float,
    B: float,
    adaptive_bound: bool = False,
    integration_step: float = 0.01,
    integration_grid: np.ndarray = None,
) -> np.ndarray:
    """
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
    """
    # Validate parameters (custom validation function or use if already defined)
    valid_Tstable_parameters(alpha, P, Q, A, B)

    # Adaptive grid generation if needed
    if adaptive_bound:
        integration_bound = adaptive_integration_bound_for_CTS_Fourier_inverse(alpha, P, Q, A, B)
        if integration_bound==None:
            raise ValueError("adaptive_bound not computed: modify threshold epsilon for the adpative bound or choose a empirical integration grid for the Fourier inverse transform")
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


def totally_skewed_CTS_generator_Bauemer(alpha: float, P: float, A: float, c: float = 0):
    """
    Samples a totally positively skewed tempered stable random variable.
    when alpha>1 the algorithm approximates with the introduction of the parameter $c$
    the higher $c$ the better the approximation but the slower is the acceptance rate

    Parameters
    ----------
    alpha : float
        Stability index (0 < alpha <= 2).
    P : float
        Positive jump parameter (P >= 0).
    A : float
        Positive jump tempering parameter (A >= 0).
    c: float
        Approximation parameter (c>=0)

    """
    U = st.uniform().rvs(1)
    sigma = (P * gamma(1 - alpha) / alpha * cos(pi * alpha / 2)) ** (1 / alpha)
    S = stable_distribution_generator(alpha, sigma, 1, 0, 1)[0]
    c = c * (alpha >= 1)  # if alpha<1 no need to introduce c, the sampling is exact
    while U > exp(-A * (S + c)):
        U = st.uniform().rvs(1)
        S = stable_distribution_generator(alpha, sigma, 1, 0, 1)[0]
    return S - P * gamma(1 - alpha) * A ** (alpha - 1)


def CTS_generator_Bauemer(alpha: float, P: float, Q: float, A: float, B: float, c: float = 0):
    """
    Sample a bilateral tempered stable random variable.
    when alpha>1 the algorithm approximates with the introduction of the parameter $c$
    the higher $c$ the better the approximation but the slower is the acceptance rate

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
    c : float
        Approximation parameter (default value is 0)

    Returns
    -------
    float

    """
    Y_plus, Y_minus = 0, 0
    if P > 0:
        Y_plus = totally_skewed_CTS_generator_Bauemer(alpha, P, A, c)
    if Q > 0:
        Y_minus = totally_skewed_CTS_generator_Bauemer(alpha, Q, B, c)
    return Y_plus - Y_minus


def CTS_generator_Bauemer_vectorial(
    alpha: float, P: float, Q: float, A: float, B: float, n_sample: int, c: float = 0, loading_bar: bool = False
):
    """
    Vectorial version of the CTS generator

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
    c : float
        Approximation parameter (default value is 0)
    n_sample: int
        number of samples
    loading_bar: bool
        additional informations on the computational time are given if True

    Returns
    -------
    res : np.ndarray


    """
    res = np.zeros(n_sample)
    if loading_bar==True:
        iterable=trange(n_sample)
    else:
        iterable=range(n_sample)
    for i in iterable:
        res[i] = CTS_generator_Bauemer(alpha, P, Q, A, B, c)

    return res


