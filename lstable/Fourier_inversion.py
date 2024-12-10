# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 22:10:18 2024

@author: Jalal
"""


import numpy as np
from scipy.integrate import romb, quad, simpson
from math import pi




def is_grid_uniform(grid: np.ndarray)-> bool:
    '''
    Returns True if a grid is (approximatively) uniform (see documentation on np.allclose for the numerical precision)
    '''
    increments= grid[1:] - grid[:-1]
    return np.allclose(increments,np.ones(len(grid)-1)*increments[0])



def density_by_fourier_inversion(characteristic_function: np.ndarray,evaluation_grid: np.ndarray,integration_grid: np.ndarray) -> np.ndarray:
    """
    Perform Fourier inversion to retrieve the probability density function from a characteristic function.
    
    Parameters:
    characteristic_function: (nd.ndarray) characteristic function evaluated in the integration grid
    evaluation_grid: (np.ndarray) uniform grid where the density is evaluated 
    integration_grid: (np.ndarray) uniform grid wherer the integration is conducted they
    
    Returns:
    float: Value of the pdf in evaluation_grid
    """
    #Verifies that the grids are uniform
    if is_grid_uniform(integration_grid)==0:
        raise ValueError('Integration grid is not uniform')
        
    d_int= integration_grid[1]-integration_grid[0]
    phi=characteristic_function #evaluation at the integration grid
    """
    Matrix expresion of the integrand
    shape(integrand)= (evaluation_grid,integration_grid)
    """
    x_grid=np.resize(evaluation_grid,(1,len(evaluation_grid)))
    integrand_matrix= np.resize(integration_grid,(1,len(integration_grid)))
    integrand_matrix= np.exp(-np.dot(np.transpose(x_grid),integrand_matrix)*1j)*phi
    res=np.real(1/(2*pi)*simpson(integrand_matrix,dx=d_int,axis=1))
    return res*(res>=0)
