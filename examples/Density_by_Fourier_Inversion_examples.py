# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 12:01:32 2024

@author: Jalal
"""


import numpy as np
from scipy.integrate import romb, quad, simpson
import scipy.stats as st
import matplotlib.pyplot as plt
from math import cos, gamma, pi,log,tan,sin,floor,exp,ceil,sqrt
from scipy.special import sici
from scipy.integrate import quad
import scipy.optimize as so
import mpmath
from scipy.special import gamma, gammainc, gammaincc,gammaincinv
import timeit
from mpmath import gammainc
import mpmath as mp
from functions import Fourier_inversion






# =============================================================================
# Some classical density function
# =============================================================================

def gaussian_pdf(x, mu, sigma):
    """ Compute the PDF of a Gaussian (Normal) distribution of mean mu and std sigma"""
    coeff = 1 / (np.sqrt(2 * np.pi) * sigma)
    exponent = -((x - mu) ** 2) / (2 * sigma**2)
    return coeff * np.exp(exponent)

def bimodal_gaussian_pdf(x, mu1, sigma1, mu2, sigma2, p):
    """Compute the PDF of a bimodal Gaussian distribution."""
    # Gaussian PDF
    pdf1 = (1 / (np.sqrt(2 * np.pi) * sigma1)) * np.exp(-((x - mu1) ** 2) / (2 * sigma1**2))
    pdf2 = (1 / (np.sqrt(2 * np.pi) * sigma2)) * np.exp(-((x - mu2) ** 2) / (2 * sigma2**2))
    return p * pdf1 + (1 - p) * pdf2

def cauchy_pdf(x, x0, gamma):
    """Compute the PDF of a Cauchy distribution"""
    return 1 / (np.pi * gamma * (1 + ((x - x0) / gamma) ** 2))

def stable_pdf(x: np.ndarray ,alpha: float,sigma:float,beta:float,mu:float)-> np.ndarray:
    """ Computes the PDF of a stable distribution \sim S_\alpha(sigma,beta,mu) """
    g= st.levy_stable(alpha,beta).pdf
    return 1/(sigma)*g((x-mu)/(sigma))

def uniform_pdf(x, a, b):
    """Computes the pdf of the Uniform distribution on [a,b] U([a,b)]"""
    return np.where((x >= a) & (x <= b), 1 / (b - a), 0)

def exponential_pdf(x, lam):
    """Computes the pdf of the Uniform distribution on [a,b] U([a,b)]"""
    return np.where(x >= 0, lam * np.exp(-lam * x), 0)


    
# =============================================================================
# Some classical characteristic function
# =============================================================================


def gaussian_cf(t, mu, sigma):
    """Compute the characteristic function of a normal distribution."""
    return np.exp(1j * mu * t - 0.5 * (sigma**2) * t**2)

def bimodal_gaussian_cf(t, mu1, sigma1, mu2, sigma2, p):
    """Compute the characteristic function of a bimodal Gaussian distribution."""
    # Gaussian CF
    cf1 = gaussian_cf(t, mu1, sigma1) #np.exp(1j * mu1 * t - 0.5 * sigma1**2 * t**2)
    cf2 = gaussian_cf(t, mu2, sigma2) #np.exp(1j * mu2 * t - 0.5 * sigma2**2 * t**2)
    return p * cf1 + (1 - p) * cf2

def cauchy_cf(t, x0, gamma):
    """Compute the characteristic function of a Cauchy distribution."""
    return np.exp(1j * x0 * t - gamma * np.abs(t))

def uniform_cf(t, a, b):
    """Compute the characteristic function of a Uniform distribution"""
    return np.where(
        t != 0,
        (np.exp(1j * b * t) - np.exp(1j * a * t)) / (1j * t * (b - a)),
        1.0  # Handle t = 0
    )

def exponential_cf(t, lam):
    """Compute the characteristic function of an Exponential distribution."""
    return lam / (lam - 1j * t)

def stable_cf(u,alpha,sigma,beta,mu):
    """ Characteristic function for a general stable distribution """
    if alpha==1:
        temp = -2/pi*np.log(np.abs(u))
    else:
        temp = tan(pi*alpha/2)
    return np.exp(1.j*u*mu - np.abs(sigma*u)**alpha*(1- 1.j*beta*np.sign(u)*temp))
     


# =============================================================================
# Test of Fourier inversion
# =============================================================================

def plot_density_and_fourierinverse_evaluation(density,cf,evaluation_grid,integration_grid,*args):
    true_density=density(evaluation_grid,*args)
    characteristic_function= cf(integration_grid,*args)
    inversion_density=density_by_fourier_inversion(characteristic_function,evaluation_grid,integration_grid)
    
    plt.plot(evaluation_grid,true_density, label='True density',linestyle='dotted',color='olive')
    plt.plot(evaluation_grid,inversion_density, label='FI density',alpha=0.5,color='red')
    plt.legend()

#Gaussian
M=4
evaluation_grid=np.linspace(-10,10,1000)
integration_grid=np.arange(-M,M,0.1)
mu,sigma=0,1
args=[mu,sigma]
density=gaussian_pdf
cf=gaussian_cf    
plt.subplot(2,1,1)
plot_density_and_fourierinverse_evaluation(density,cf,evaluation_grid,integration_grid,*args)
plt.subplot(2,1,2)
plt.plot(integration_grid,cf(integration_grid,*args))
plt.show()

#Bilateral Gaussian
M=50
evaluation_grid=np.linspace(-10,10,1000)
integration_grid=np.arange(-M,M,0.1)
mu1,sigma1,mu2,sigma2,p=0,0.1,3,0.4,0.5
args=[mu1,sigma1,mu2,sigma2,p]
density=bimodal_gaussian_pdf
cf=bimodal_gaussian_cf
plt.subplot(2,1,1)
plot_density_and_fourierinverse_evaluation(density,cf,evaluation_grid,integration_grid,*args)
plt.subplot(2,1,2)
plt.plot(integration_grid,cf(integration_grid,*args))
plt.show()

#Cauchy
M=50
evaluation_grid=np.linspace(-10,10,1000)
integration_grid=np.arange(-M,M,0.1)
x0,gamma=1,1
args=[x0,gamma]
density=cauchy_pdf
cf=cauchy_cf
plt.subplot(2,1,1)
plot_density_and_fourierinverse_evaluation(density,cf,evaluation_grid,integration_grid,*args)
plt.subplot(2,1,2)
plt.plot(integration_grid,cf(integration_grid,*args))
plt.show()



#Exponential
M=1000
evaluation_grid=np.linspace(-0.1,3,1000)
integration_grid=np.arange(-M,M,1)
lam=1
args=[lam]
density=exponential_pdf
cf=exponential_cf
plt.subplot(2,1,1)
plot_density_and_fourierinverse_evaluation(density,cf,evaluation_grid,integration_grid,*args)
plt.subplot(2,1,2)
plt.plot(integration_grid,cf(integration_grid,args),label='characteristic function')
plt.show()

#Uniform
M=50
evaluation_grid=np.linspace(-1,2,1000)
integration_grid=np.arange(-M,M,0.1)
a,b=0,1
args=[a,b]
density=uniform_pdf
cf=uniform_cf
plt.subplot(2,1,1)
plot_density_and_fourierinverse_evaluation(density,cf,evaluation_grid,integration_grid,*args)
plt.subplot(2,1,2)
plt.plot(integration_grid,np.imag(cf(integration_grid,*args)))
plt.show()


#Stable FV
M=50
evaluation_grid=np.linspace(-10,10,1000)
integration_grid=np.arange(-M,M,0.1)
alpha,sigma,beta,mu=0.5,1,0,0
args=[alpha,sigma,beta,mu]
density=stable_pdf
cf=stable_cf
plt.subplot(2,1,1)
plot_density_and_fourierinverse_evaluation(density,cf,evaluation_grid,integration_grid,*args)
plt.subplot(2,1,2)
plt.plot(integration_grid,np.abs(cf(integration_grid,*args)))
plt.show()


#Stable IV
M=50
evaluation_grid=np.linspace(-10,10,1000)
integration_grid=np.arange(-M,M,0.1)
alpha,sigma,beta,mu=1.5,1,0,0
args=[alpha,sigma,beta,mu]
density=stable_pdf
cf=stable_cf
plt.subplot(2,1,1)
plot_density_and_fourierinverse_evaluation(density,cf,evaluation_grid,integration_grid,*args)
plt.subplot(2,1,2)
plt.plot(integration_grid,cf(integration_grid,*args))
plt.show()
