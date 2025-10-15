   


import numpy as np
from scipy.integrate import romb, quad, simpson
import scipy.stats as st
import matplotlib.pyplot as plt
from math import cos, gamma, pi,log,tan,sin,floor,exp,ceil,sqrt
from scipy.special import sici
from scipy.integrate import quad
import scipy.optimize as so

# ==========
# ===================================================================
# Characteristic function estimators 
# =============================================================================

def risk(est, true, dx): 
    return simpson((est-true)**2) *dx / (simpson((true)**2) *dx)
    

#Estimator of the characteristic function

def cf_estimator(incr,u_grid):
    '''
    Parameters
    ----------
    incr : Array
        i.i.d. observation
    u_grid : Array
        grid 
    Returns
        Empirical characteristic function of i.i.d. observations
    -------

    '''
    u_grid0= np.reshape(u_grid,(1,len(u_grid)))
    incr=np.reshape(incr,(1,len(incr)))
    quot=np.exp(1j* np.dot(np.transpose(incr),u_grid0))
    res=np.mean(quot,axis=0)
    return res

def cf_estimator_cutoff(cf_est,n,Delta,kappa):
    """
    Computes the estimator of [ADD]

    Parameters
    ----------
    incr : Array: Increments of the process
    u_grid : Array: Grid where the estimator is computed
    n : int: number of observations
    Delta : float: rate of observations
    kappa : float: Threshold for the cutoff

    Returns: characteristic function estimator with cutoff of the empirical characteristic function
    -------

    """
    kappa_n = (1+kappa*sqrt(log(n))) #No dependency on Delta ?
    res=cf_est* (np.abs(cf_est)>= kappa_n/sqrt(n))
    return res

def cf_estimator_cutoff_part(incr,u_grid,n,Delta,kappa):
    """
    Computes the estimator of [ADD] but cutting off the real and imaginary part separately

    Parameters
    ----------
    incr : Array: Increments of the process
    u_grid : Array: Grid where the estimator is computed
    n : int: number of observations
    Delta : float: rate of observations
    kappa : float: Threshold for the cutoff

    Returns: characteristic function estimator with cutoff of the empirical characteristic function
    -------

    """
    cf_est=cf_estimator(incr,n,Delta,u_grid)
    kappa_n = (1+kappa*sqrt(log(n)))
    res_real=np.real(cf_est)
    res_imag=np.imag(cf_est)
    res=res_real* (np.abs(res_real)>= kappa_n/sqrt(n)) + 1.j*res_imag* (np.abs(res_imag)>=kappa_n/sqrt(n))
    return res

def cf_estimator_restriction(incr,u_grid,n,Delta,m):
    cf_est=cf_estimator(incr,u_grid)
    return cf_est*(np.abs(u_grid)<=m)

#Fourier inversion 
def finv_density(x_grid,incr,u_grid,n,Delta,kappa,cutoff_type='abs cutoff'):
    """
    Computes the Fourier inverse given

    Parameters
    ----------
    x_grid : array
        grid where the density estimator is computed
    incr : Array: Increments of the process
    u_grid : Array: Integration grid where the empirical characteristic function is computed
    n : int: number of observations
    Delta : float: rate of observations
    kappa : float: Threshold for the cutoff
    cutoff_type : Cutoff of the absolute value/real and imag/ direct of the empirical characteristic function, or th


    """
    dx= x_grid[1]-x_grid[0]
    if cutoff_type=="abs cutoff":
        char_fct=cf_estimator_cutoff(incr,n,Delta,u_grid,kappa)
    if cutoff_type=='direct':
        char_fct=cf_estimator(incr,n,Delta,u_grid)
    if cutoff_type=='real/imag cutoff':
        char_fct=cf_estimator_cutoff_part(incr,n,Delta,u_grid,kappa)
        
    d_int= u_grid[1]-u_grid[0]
    """
    """
    
    integrand_matrix=np.resize(u_grid,(1,len(u_grid)))
    x_grid=np.resize(x_grid,(1,len(x_grid)))
    integrand_matrix= np.exp(-np.dot(np.transpose(x_grid),integrand_matrix)*1j)*char_fct
    res=np.real(1/(2*pi)*simpson(integrand_matrix,dx=d_int,axis=1))
    return res*(res>=0)

def f_hat(x_grid,char_fct,u_grid):
    
    d_int= u_grid[1]-u_grid[0]
    integrand_matrix=np.resize(u_grid,(1,len(u_grid)))
    x_grid=np.resize(x_grid,(1,len(x_grid)))
    integrand_matrix= np.exp(-np.dot(np.transpose(x_grid),integrand_matrix)*1j)*char_fct
    res=np.real(1/(2*pi)*simpson(integrand_matrix,dx=d_int,axis=1))
    return res*(res>=0)

