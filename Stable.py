# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 19:26:08 2024

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



# =============================================================================
# 
# =============================================================================

def float_or_int(parameter):
    """Check if a parameter is of type float or int."""
    return isinstance(parameter, (float, int))

def valid_stable_parameters(alpha: float, sigma: float, beta: float, mu: float) -> bool:
    """
    Validates the parameters for the stable distribution.
    - alpha: Stability parameter (0 < alpha < 2)
    - sigma: Scale parameter (sigma > 0)
    - beta: Skewness parameter (-1 <= beta <= 1)
    - mu: Location parameter (any real number)
    """
    alpha_requirement = float_or_int(alpha) and (0 < alpha < 2)
    beta_requirement = float_or_int(beta) and (-1 <= beta <= 1)
    sigma_requirement = float_or_int(sigma) and (sigma > 0)
    mu_requirement = float_or_int(mu)
    return alpha_requirement and beta_requirement and sigma_requirement and mu_requirement


def stable_to_levy_parameter(alpha: float, P: float, Q: float, drift: float):
    """
    Converts the Lévy triplet parameters (alpha, P, Q, drift) to the stable distribution parameters 
    (alpha, sigma, beta, mu).

    Parameters
    ----------
    - alpha (float): Stability parameter (0 < alpha < 2)
    - P (float): Positive jumps parameter (P >= 0)
    - Q (float): Negative jumps parameter (Q >= 0)
    - drift (float): Drift parameter for the linear term (any real number)

    Returns
    -------
    - tuple: (alpha, sigma, beta, mu), where:
        - alpha: Stability parameter (same as input alpha)
        - sigma: Scale parameter
        - beta: Skewness parameter
        - mu: Location parameter
    """
    # Validate inputs
    if not (0 < alpha < 2):
        raise ValueError("alpha must be in the range (0, 2).")
    if P < 0 or Q < 0:
        raise ValueError("P and Q must be non-negative.")
    if P + Q == 0:
        raise ValueError("P + Q must be greater than 0 to define a valid skewness.")

    # Compute beta
    beta = (P - Q) / (P + Q)

    # Handle alpha = 1 case
    if alpha == 1:
        # c is a constant involving special functions and Euler's gamma
        c = 1 + sici(1)[0] - sici(1)[1] + np.euler_gamma
        mu = drift + c * (P - Q)
        sigma = (P + Q) * pi / 2
    else:
        # General case
        mu = drift + (Q - P) / (1 - alpha)
        sigma = ((P + Q) * gamma(1 - alpha) * cos(pi * alpha / 2) / alpha) ** (1 / alpha)

    return alpha, sigma, beta, mu




def stable_distribution_generator(alpha: float, sigma: float, beta: float, mu: float, n_sample: int):
    """
    Generates a sample of size n_sample from a stable distribution S_alpha(sigma, beta, mu).
    
    Parameters:
    - alpha: Stability parameter (0 < alpha < 2)
    - sigma: Scale parameter (sigma > 0)
    - beta: Skewness parameter (-1 <= beta <= 1)
    - mu: Location parameter (any real number)
    - n_sample: Number of samples to generate
    
    Returns:
    - A list of N samples from the stable distribution.
    """
    # Validate parameters
    if not valid_stable_parameters(alpha, sigma, beta, mu):
        raise ValueError("Invalid parameters for the stable distribution.")

    # Generate N draws of Z ~ S_alpha(1, beta, 0)
    z_vector = st.levy_stable.rvs(alpha, beta, size=n_sample)  

    # Rescale and shift Z to X
    if alpha != 1:
        return sigma * z_vector + mu  # When alpha != 1
    else:
        return sigma * z_vector + mu + (2 / pi) * beta * sigma * log(sigma)  # When alpha == 1
    

def convert_to_stable_and_sample(alpha: float,P:float,Q:float,drift: float, n_sample: int):
    '''
    Generate an n sample from a infinite divisible distribution of levy triplet (drift,0,nu) where 
    nu is the stable Levy measure of parameter alpha,P,Q
    '''
    # Convert the Levy parameters
    alpha,sigma,beta,mu = stable_to_levy_parameter(alpha, P, Q, drift)
    return stable_distribution_generator(alpha,sigma,beta,mu,n_sample)

def stable_density(grid ,alpha: float,sigma: float,beta: float,mu: float): 
        """
        Evaluates the density function of a stable distribution 
        \( S_\alpha(\sigma, \beta, \mu) \) at the given grid points.        
        """
        if not valid_stable_parameters(alpha, sigma, beta, mu):
            raise ValueError("Invalid parameters for the stable distribution.")
            
        g=st.levy_stable(alpha,beta).pdf # density of S_alpha(1,beta,0)
        if alpha==1:
            temp=-mu - 2/pi*beta*sigma*log(sigma)
            shifted_grid=(grid-temp)/sigma
            return 1/sigma* g(shifted_grid)
        else:
            shifted_grid=(grid-mu)/sigma
            return 1/sigma*g((grid-mu)/sigma)
        
        
def stable_characteristic_function(grid: np.ndarray ,alpha:float,sigma:float,beta:float,mu:float):
    '''
    Evaluates the characteristic function of a stable distribution 
    \( S_\alpha(\sigma, \beta, \mu) \) at the given grid points.
    
    Returns:
    -------
    np.ndarray
    An array of the same size as `grid`, containing the characteristic function 
    values evaluated at the corresponding grid points.
    '''
    if not valid_stable_parameters(alpha, sigma, beta, mu):
        raise ValueError("Invalid parameters for the stable distribution.")    
    if alpha==1:
        temp = -2/pi*np.log(np.abs(grid))
    else:
        temp = tan(pi*alpha/2)
    return np.exp(1.j*grid*mu - np.abs(sigma*grid)**alpha*(1- 1.j*beta*np.sign(grid)*temp))

def support_density(alpha:float,sigma:float,beta:float,mu:float):
    ''' 
   Determines the support of the probability density function of the 
   \( S_\alpha(\sigma, \beta, \mu) \) stable distribution.
   
   Returns:
   -------
   tuple
   A tuple (lower_bound, upper_bound) representing the lower and upper 
   bounds of the support of the stable distribution.
   - If alpha < 1 and beta = 1: [mu, +inf)
   - If alpha < 1 and beta = -1: (-inf, mu]
   - Otherwise: (-inf, +inf)

    '''
    if not valid_stable_parameters(alpha, sigma, beta, mu):
        raise ValueError("Invalid parameters for the stable distribution.")
    
    # Determine support bounds
    if alpha < 1:
        if beta == 1:
            return mu, float("inf")
        elif beta == -1:
            return float("-inf"), mu
   
    # General case
    return float("-inf"), float("inf")
