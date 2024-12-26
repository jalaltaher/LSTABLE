# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:35:07 2024

@author: Jalal
"""

import numpy as np
from scipy.special import gamma
import scipy.stats as st
from math import sqrt
from mpmath import gammainc, inf

from tqdm import trange



from . import CTS_distribution
from .CTS_distribution import *



# =============================================================================
# Bauemer-Mershaert 
# =============================================================================

def increments_CTS_generator(
    n_increments: int,
    Delta: float,
    alpha: float,
    P: float,
    Q: float,
    A: float,
    B: float,
    drift: float,
    c: float = 0,
    loading_bar: bool = False,
) -> np.ndarray:
    """
    Generates increments of a CTS process with drift.

    Parameters:
        n_increments (int): Number of increments to generate.
        Delta (float): Sampling rate.
        alpha (float): Parameter of the CTS process.
        P, Q, A, B (float): Parameters of the CTS process.
        drift (float): Drift coefficient.
        c (float, optional): Additional parameter, default is 0.
        verbose (bool, optional): Verbose output giving time remaining, default is False.

    Returns:
        np.ndarray: Array of shape (n_increments,) representing the increments.
    """
    increments = np.zeros(n_increments+1)
    increments[1:]=CTS_generator_Bauemer_vectorial(
        alpha, Delta * P, Delta * Q, A, B, n_increments, c, loading_bar
    ) + Delta * drift
    return increments


def trajectory_CTS_generator(
    n_increments: int,
    n_trajectories: int,
    Delta: float,
    alpha: float,
    P: float,
    Q: float,
    A: float,
    B: float,
    drift: float,
    c: float = 0,
    loading_bar: bool = False,
) -> np.ndarray:
    """
    Generates trajectories of a CTS process with drift.

    Parameters:
        n_increments (int): Number of increments per trajectory.
        n_trajectories (int): Number of trajectories to generate.
        Delta (float): Sampling rate.
        alpha (float): Parameter of the CTS process.
        P, Q, A, B (float): Parameters of the CTS process.
        drift (float): Drift coefficient.
        c (float, optional): Additional parameter, default is 0.
        loading_bar (bool, optional): Verbose output giving time remaining, default is False.

    Returns:
        np.ndarray: Array of shape (n_trajectories, n_increments + 1)
        if `n_trajectories > 1`, else a 1D array of shape (n_increments + 1,).
    """
    increment_matrix = np.array(
        [
            increments_CTS_generator(n_increments, Delta, alpha, P, Q, A, B, drift, c, loading_bar)
            for _ in range(n_trajectories)
        ]
    )


    # Compute the cumulative sum along increments
    res = np.cumsum(increment_matrix, axis=1)

    # Return a 1D array if there's only one trajectory
    if n_trajectories == 1:
        return res[0]
    return res



# =============================================================================
# Compound Poisson approximation
# =============================================================================
def upper_gamma(z: float, x: float):
    '''
    Upper Incomplete Gamma function using the library mpmath, which supports precise computations and complex values.

    Parameters
    ----------
    z : complex or float
        Shape parameter(s).
    x : complex or float
        Lower integration bound.

    Returns
    -------
    float or complex
        Upper incomplete gamma function Gamma(z, x) = int_{x}^infty t^{z-1}e^{-t}dt that can be extended to an entire function.

    '''
    res= complex(gammainc(z,a=x,b=inf, regularized=False))
    if np.imag(res)==0:
        return np.real(res)
    else:
        return res
        
def lower_gamma(z: float, x: float):
    '''
    Lower Incomplete Gamma function using the library mpmath.
    Vectorize version with fixed z
    
    Parameters
    ----------
    z : complex or float
        Shape parameter(s) that is fixed.
    x : np.ndarray
        Upper integration bound.

    Returns
    -------
    np.ndarray
        Lower incomplete gamma function gamma(z, x) = int_{0}^x t^{z-1}e^{-t}dt that can be extended 
    '''
    res = complex(gammainc(z,a=0,b=x,regularized=False))
    if np.imag(res)==0:
        return np.real(res)
    else:
        return res
    

def positive_jumpsize_compound_poisson_approximation(alpha: float,A:float, delta: float)-> float:
    '''
    Computes via rejection sampling the positive jump size part of the compound Poisson approximation

    Parameters
    ----------
    alpha : float
        stability index
    A : float
        positive tempering parameter
    delta : float
        truncation parameter

    Returns
    -------
    float
        return a sample of $$Z_0 \sim f_\delta^0(x)= \frac{1}{A^\alpha \Gamma(-\alpha,A\delta) \frac{e^{-Ax}}{x^{1+\alpha}} \ind_{x>\delta},$$

    '''
    W=np.random.uniform(0,1)
    V=np.random.uniform(0,1)
    Z= delta*W**(-1/alpha)
    T= np.exp(A*(Z-delta))
    while V*T >1:
        W=np.random.uniform(0,1)
        V=np.random.uniform(0,1)
        Z= delta*W**(-1/alpha)
        T= np.exp(A*(Z-delta))
    return Z

def negative_jumpsize_compound_poisson_approximation(alpha:float ,B:float,delta:float)->float:
    '''
    Compute via rejection sampling the negative jump size part of the compound Poisson approximation

    Parameters
    ----------
    alpha : float
        stability index
    A : float
        positive tempering parameter
    delta : float
        truncation parameter

    Returns
    -------
    float
        return a sample of $$ Z_1 \sim f_\delta^1(x) \frac{1}{B^\alpha \Gamma(-\alpha,B\delta) \frac{e^{-B|x|}}{|x|^{1+\alpha}} \ind_{x<-\delta} $$

    '''
    W=np.random.uniform(0,1)
    V=np.random.uniform(0,1)
    Z= -delta*W**(-1/alpha)
    T= np.exp(B*(np.abs(Z)-delta))
    while V*T >1:
        W=np.random.uniform(0,1)
        V=np.random.uniform(0,1)
        Z= -delta*W**(-1/alpha)
        T= np.exp(B*(Z-delta))
    return Z

def jumpsize_compound_poisson_approximation(alpha: float,P:float,Q:float,A:float,B:float,delta:float)->float:
    '''
    Compute via rejection sampling the jumpsize of the compound Poisson approximation of a tempered stable process.

    Parameters
    ----------
    alpha : float
        stability index
    P : float
        positive jump parameter
    Q : float
        negative jump parameter
    A : float
        positive jump tempering parameter
    B : float
        negative jump tempering parameter
    delta : float
        truncation parameter

    Returns
    -------
    float
        realization $$Y \sim  f_\delta(x) = \frac{Pe^{-Ax}}{\lambda(\delta) x^{1+\alpha}} \ind_{x>\delta} + \frac{Qe^{-B|x|}}{\lambda(\delta) |x|^{1+\alpha}} \ind_{x<-\delta},$$ where $
        \lambda(\delta) = PA^{\alpha} \Gamma(-\alpha, A \delta) + Q B^\alpha \Gamma(-\alpha, B\delta)$.

    '''
    #probability of positive and negative jumps
    temp_A= float(upper_gamma(-alpha,A*delta))
    temp_B= float(upper_gamma(-alpha,B*delta))
    proba= P*A**(alpha)*temp_A/(P*A**(alpha)*temp_A+ Q*B**(alpha)*temp_B)
    S=st.bernoulli(proba).rvs(1)
    if S==1:
        return  positive_jumpsize_compound_poisson_approximation(alpha,A,delta)
    else:
        return  negative_jumpsize_compound_poisson_approximation(alpha,B,delta)
    

def jumpsize_compound_poisson_approximation_vectorized(alpha: float,P:float,Q:float,A:float,B:float,delta:float,nb_jumps:int,loading_bar:bool =False)->float:
    '''
    vectorized version of jumpsize of the compound Poisson approximation of a tempered stable process.

    Parameters
    ----------
    alpha : float
        stability index
    P : float
        positive jump parameter
    Q : float
        negative jump parameter
    A : float
        positive jump tempering parameter
    B : float
        negative jump tempering parameter
    delta : float
        truncation parameter
    nb_jumps : int
        number of jumps
    loading_bar: bool
        additional informations on the computational time are given if True

    Returns
    -------
    np.ndarray
        
        array of the size jumps of the compound Poisson approximation.
        
'''

    if loading_bar==True:
        iterable=trange(nb_jumps)
    else:
        iterable=range(nb_jumps)
    res=np.zeros(nb_jumps)
    for i in iterable:
        res[i]=jumpsize_compound_poisson_approximation(alpha,P,Q,A,B,delta)
    return res

def drift_compensation(alpha:float, P:float, Q:float, A:float, B:float, delta:float):
    '''
    drift compensation for the compound Poisson approximation $\int_{\delta<|x|\leq 1} x \nu(dx)$ 
    for tempered stable process using incomplete gamma functions
    Parameters
    ----------
    alpha : float
        stability index
    P : float
        positive jump parameter
    Q : float
        negative jump parameter
    A : float
        positive jump tempering parameter
    B : float
        negative jump tempering parameter
    delta : float
        truncation parameter
    nb_jumps : int
        number of jumps

    Returns
    -------
    float   
        $\int_{\delta<|x|\leq 1} x \nu(dx) =  PA^{\alpha-1} \int_{A\delta}^A z^{1-\alpha-1}e^{-z}dz + QB*^{\alpha-1} \int_{B\Delta}^{B} z^{1-\alpha-1}e^{-z}dz$ 
    '''
    #compute the integral terms 
    I1= float(gammainc(1-alpha, a=A*delta, b=A))
    I2= float(gammainc(1-alpha, a=B*delta, b=B))
    
    return P*A**(alpha-1)* I1 - Q*B**(alpha-1)*I2

def compound_poisson_approximation_direct_algorithm(Delta:float, drift:float,alpha:float ,P:float ,Q:float ,A:float ,B:float ,delta:float)-> float:
    '''
    Computes the increments at time Delta of the compound Poisson approximation of a tempered stable process of

    Parameters
    ----------
    Delta : float
        time step
    drift : float
        drift term 
    alpha : float
        stability index
    P : float
        positive jump parameter
    Q : float
        negative jump parameter
    A : float
        positive jump tempering parameter
    B : float
        negative jump tempering parameter
    delta : float
        truncation parameter
    epsilon : float
        DESCRIPTION.

    Returns
    -------
    float
    
    A single increments $X_\Delta^\delta$ of the compound poisson approximation of a tempered stable process.

    '''
    
    #drift
    gamma = drift - drift_compensation(alpha,P,Q,A,B,delta)
    
    # jumps intensity
    jump_intensity= P*A**(alpha)*upper_gamma(-alpha,A*delta) + Q*B**(alpha)*upper_gamma(-alpha,B*delta)
    jump_intensity = jump_intensity*Delta
    
    # poisson process
    N_Delta=st.poisson(jump_intensity).rvs(1)[0]
    
    # jump sizes
    res= jumpsize_compound_poisson_approximation_vectorized(alpha,P,Q,A,B,delta,N_Delta)
    return np.sum(res)+Delta*gamma
    

=======
def compound_poisson_approximation_direct_algorithm_vectorized(n_increments: int, Delta:float, drift:float,alpha:float ,P:float ,Q:float ,A:float ,B:float ,delta:float,loading_bar:bool =False)-> np.ndarray:
    """
    Vectorized version of compound_poisson_approximation_direct_algorithm

    Parameters
    ----------
    
    Delta : float
        time step
    drift : float
        drift term 
    alpha : float
        stability index
    P : float
        positive jump parameter
    Q : float
        negative jump parameter
    A : float
        positive jump tempering parameter
    B : float
        negative jump tempering parameter
    delta : float
        truncation parameter

    loading_bar: bool
        additional informations on the computational time are given if True
 
     Returns
    -------
    np.ndarray
    
    increments array (0,\tilde{X}_\Delta,\tilde{X}_{2\Delta}-X_\Delta..., \tilde{X}_{n\Delta} - \tilde{X}_{(n-1)\Delta}) of the compound poisson approximation of a tempered stable process


    """

   
    res=[0]
    res+= [compound_poisson_approximation_direct_algorithm(Delta,drift,alpha,P,Q,A,B,delta) for _ in range(n_increments)]
    return np.array(res)
    


def compound_poisson_approximation_sorting_algorithm(n_increments:float, Delta:float,drift:float ,alpha:float ,P:float,Q:float ,A:float, B:float ,delta:float,loading_bar:bool =False )->np.ndarray:
    '''
    Computes the increments at time Delta of the compound Poisson approximation of a tempered stable process 
    by compound Poisson sorting algorithm.


    Parameters
    ----------

    nb_increments: int
        number of increments
    Delta : float
        time step
    drift : float
        drift term 
    alpha : float
        stability index
    P : float
        positive jump parameter
    Q : float
        negative jump parameter
    A : float
        positive jump tempering parameter
    B : float
        negative jump tempering parameter
    delta : float
        truncation parameter

    loading_bar: bool
        additional informations on the computational time are given if True

     Returns
    -------
    res : np.ndarray
        increments (0,\tilde{X}_\Delta,\tilde{X}_{2\Delta}-X_\Delta..., \tilde{X}_{n\Delta} - \tilde{X}_{(n-1)\Delta}) of the compound poisson approximation of a tempered stable process

    '''
    
    n=n_increments
    T=n*Delta #final time
    gamma = drift - drift_compensation(alpha,P,Q,A,B,delta) #drift term
    
    #jump intensiy
    jump_intensity= P*A**(alpha)*upper_gamma(-alpha,A*delta) + Q*B**(alpha)*upper_gamma(-alpha,B*delta)
    jump_intensity = jump_intensity*T
    
    #number of jumps
    N=st.poisson(jump_intensity).rvs(1)[0]
    
    #jump times
    U=T*st.uniform().rvs(N) #jump times
    

    jump_sizes=jumpsize_compound_poisson_approximation_vectorized(alpha,P,Q,A,B,delta,N,loading_bar)
    res=np.zeros(n+1)
    for i in range(1,n+1):
        res[i] = np.sum(jump_sizes *(Delta*(i-1)<U)*(U<=Delta*i)) + Delta*gamma
    return res


# =============================================================================
#  Compound Poisson approximation + Gausian approximation of small jumps
# =============================================================================

def residual_variance(delta:float,alpha:float,P:float,Q:float, A:float, B:float)->float:
    '''
    residual variance $\int_{|x|\leq \delta} x^2 \nu(dx)$ (approximation of the small jump martingale)
    
    Parameters
    ----------
    delta : float
        trunction parameter
    alpha : float
        stability index
    P : float
        positive jump parameter
    Q : float
        negative jump parameter
    A : float
        positive jump tempering parameter
    B : float
        negative jump tempering parameter

    Returns
    -------
    float
        variance term $\int_{|x|\leq \delta} x^2 \nu(dx)$

    '''
    I1 = float(gammainc(alpha, a=0,b=A*delta))
    I2 = float(gammainc(alpha, a=0,b=B*delta))
    return  P*A**(alpha-2)*I1 + Q*B**(alpha-2)*I2


=======
def compound_poisson_gaussian_approximation_tempered_stable(n_increments:float, Delta:float,drift:float ,alpha:float ,P:float,Q:float ,A:float, B:float ,delta:float,loading_bar:bool =False):
    '''
    compound poisson approximation and gaussian approximation of the residual error (small jumps)

    Parameters
    ----------
    n_increments : float
        number of increments
    Delta : float
        sampling rate
    drift : float
        drift term
    alpha : float
        stability index
        stability index
    P : float
        positive jump parameter
    Q : float
        negative jump parameter
    A : float
        positive jump tempering parameter
    B : float
        negative jump tempering parameter
    delta : float
        truncation parameter
    loading_bar: bool
        additional informations on the computational time are given if True
        

    Returns
    -------
    np.ndarray
    array of size (nb_increment +1) of the cp approximation and gaussian approximation of the residual error.
    '''

    cp_approx_incr =  compound_poisson_approximation_sorting_algorithm(n_increments, Delta,drift ,alpha ,P,Q ,A, B ,delta,loading_bar)
    res=np.zeros(n_increments+1)
    res_var= residual_variance(delta,alpha,P,Q,A,B)
    brown_incr = sqrt(Delta)*st.norm().rvs(n_increments)
    res[1:] = cp_approx_incr[1:] + sqrt(res_var)* brown_incr
    return res

def criterion_gaussian_approximation(delta:float,alpha:float,P:float,Q:float, A:float, B:float):
    '''
    Asmussen-Rosinski criterion for validity of Gaussian approxmation of the residual term : $\sigma(\delta)/\delta \rightarrow 0$

    Parameters
    ----------
    delta : float
        trunction parameter
    alpha : float
        stability index
    P : float
        positive jump parameter
    Q : float
        negative jump parameter
    A : float
        positive jump tempering parameter
    B : float
        negative jump tempering parameter

    Returns
    -------
    float

    '''
    sigma_delta= residual_variance(delta,alpha,P,Q,A,B)
    return sqrt(sigma_delta)/delta


# =============================================================================
# General method function
# =============================================================================


=======
def tempered_stable_process_increments(n_increments: int ,Delta: float ,drift: float ,alpha: float,P:float ,Q:float ,A:float ,B:float ,delta:float, c:float = 0, loading_bar:bool = False, method='bm'):
    '''
    wrapper function for all the methods

    Parameters
    ----------
    n_increments : float
        number of increments
    Delta : float
        sampling rate
    drift : float
        drift term
    alpha : float
        stability index
    P : float
        positive jump parameter
    Q : float
        negative jump parameter
    A : float
        positive jump tempering parameter
    B : float
        negative jump tempering parameter
    delta : float
        truncation parameter
        
    c : float, optional
        bauemer merschaert parameter. The default is 0 when alpha<=1.
    loading_bar : bool, optional
        DESCRIPTION. The default is False.
    method : TYPE, optional
        sampling method. The default is 'bm'.

    Returns
    -------
    np.ndarray
        increments of a tempered stable process.

    '''
    
    if method=='bm':
        return increments_CTS_generator(
            n_increments,
            Delta,
            alpha,
            P,
            Q,
            A,
            B,
            drift,
            c,
            loading_bar
        )
    elif method=='cpa':
        return compound_poisson_approximation_sorting_algorithm(
            n_increments,
            Delta, 
            drift,
            alpha,
            P,
            Q,
            A,
            B,
            delta,
            loading_bar
        )
    elif method=='cpga':
        return compound_poisson_gaussian_approximation_tempered_stable(
            n_increments,
            Delta, 
            drift,
            alpha,
            P,
            Q,
            A,
            B,
            delta,
            loading_bar
        )
        

# =============================================================================
# Goodness of fit tests for the increments
# =============================================================================
#To do

