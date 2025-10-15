# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 00:10:17 2024

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
from lstable.CTS_process import *
from lstable.CTS_distribution import *
from PIL import Image #For gifs
import csv
import time


# =============================================================================
# Increments generators
# =============================================================================

# Tempered Stable
def tempered_stable(n,Delta,alpha,P,Q,A,B,drift,c,N,loading_bar,method='bm'):
    increments = [tempered_stable_process_increments(n,Delta,drift,alpha,P,Q,A,B,c,loading_bar, method) for _ in range(N)]
    return np.array(increments)
    

        
# =============================================================================
# Characteristic function estimators 
# =============================================================================

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

def f_hat(x_grid,char_fct):
    
    d_int= u_grid[1]-u_grid[0]
    integrand_matrix=np.resize(u_grid,(1,len(u_grid)))
    x_grid=np.resize(x_grid,(1,len(x_grid)))
    integrand_matrix= np.exp(-np.dot(np.transpose(x_grid),integrand_matrix)*1j)*char_fct
    res=np.real(1/(2*pi)*simpson(integrand_matrix,dx=d_int,axis=1))
    return res*(res>=0)


# =============================================================================
# Euler characteristic: Selection of the kappa
# =============================================================================


def nb_connected_component_ones(array):
    """
    Computes the number of connected subarray of 1 in a {0,1} array
    Parameters
    ----------
    array : Array of {0,1}

    Returns number of subsequences of 1 that are present in the array
    """
    l = len(array)
    cpt=0
    for i in range(l-1):
        if array[i]==1 and array[i+1]==0:
            cpt+=1
    if array[-1]==1:
        cpt+=1
    return cpt

def nb_connected_component(arr):
    count = 1  # There's at least one component (the first element)
    
    # Iterate over the array starting from the second element
    for i in range(1, len(arr)):
        # If the current element is different from the previous one, it starts a new component
        if arr[i] != arr[i-1]:
            count += 1
    
    return count
    
def euler_char(phiH,n,Delta,kappa):
    """
    Computes the euler function of the empirical characteristic function

    Parameters
    ----------
    phiH : array
        empirical characteristic function
   n : int: number of observations
   Delta : float: rate of observations
   kappa : float: Threshold for the cutoff

    ---------
    """
    start = timeit.default_timer() 
    end= timeit.default_timer() 
    kappa_n = (1+kappa*sqrt(log(n)))
    temp = (np.abs(phiH) >= kappa_n/sqrt(n))
    return nb_connected_component(temp)

def find_equal_subarray(arr, n):
    # Fonction pour vérifier si toutes les valeurs dans un sous-array sont égales
    def all_equal(subarr):
        return all(x == subarr[0] for x in subarr)

    # Boucle pour trouver la sous-séquence de taille n, puis n-1, etc.
    for size in range(n, 1, -1):
        # Boucle à travers le tableau pour vérifier chaque sous-séquence de taille `size`
        for i in range(len(arr) - size + 1):
            subarr = arr[i:i+size]
            if all_equal(subarr):
                return i  # Retourne l'indice de la première sous-séquence trouvée
    return -1  # Retourne -1 si aucune sous-séquence de taille ≥ 2 n'a été trouvée

def kappa_selection(phiH,kappa_grid,n,flag,title,selection_level=1):
    """
    """
    delta=kappa_grid[1]-kappa_grid[0]
    selected_kappa=kappa_grid[-1]
    euler_tab= np.array([euler_char(phiH,n,Delta,kappa) for kappa in kappa_tab])
    j=find_equal_subarray(euler_tab,selection_level)
    selected_kappa=kappa_grid[j+2]
    if (j==-1):
        raise TypeError("No equal sequence for the Euler Characteristic")
    # if flag:
    #     plt.figure()
    #     plt.title('kappa selection n={},Delta={}'.format(n,Delta))
    #     plt.scatter(kappa_grid,euler_tab,marker='x',color='b',linewidth=1)
    #     plt.scatter([selected_kappa],[euler_tab[j+selection_level]],color='r',linewidth=2)
    #     plt.savefig(title+'_euler.png')
    #     plt.show()
    

    return selected_kappa

# =============================================================================
# Penalization
# =============================================================================

def pen(char_fct,u_grid,m_tab,n,kappa):
    du= u_grid[1]-u_grid[0]
    res=np.zeros(len(m_tab))
    for i in range(len(m_tab)):
        m=m_tab[i]
        integrand=np.abs(char_fct*(u_grid>=0)*(u_grid<=m))**2
        integral=simpson(integrand,dx=du)
        res[i]=integral
    gamma_n=1/pi* res
    pen= kappa*m_tab/n
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.title('gamma_n')
    plt.plot(m_tab,-gamma_n)
    plt.xticks([])

    plt.subplot(3,1,2)
    plt.title('pen')
    plt.plot(m_tab,pen)
    plt.xticks([])
    
    plt.subplot(3,1,3)
    plt.title('pen+gamma')
    plt.plot(m_tab,pen - gamma_n)
    plt.show()
    

    return m_tab[np.argmin(-gamma_n+pen)]
    

# =============================================================================
# Error measurement
# =============================================================================

def risk(est, true, dx): 
    return simpson((est-true)**2) *dx / (simpson((true)**2) *dx)
    
# =============================================================================
# Display functions
# =============================================================================

def histogram_pdf(a,b,incr,n,Delta,pdf,pdf_param,title):
    x = np.linspace(a, b, 100)
    plt.figure()
    plt.title(title)
    plt.plot(x, pdf(x,Delta,pdf_param),
            'r-', lw=5, alpha=0.6, label='levy pdf')
    print('n={},Delta={}, a={}, incrmin={}, b={}, incrmax={}'.format(n,Delta,a,np.min(incr),b,np.max(incr)))
    bins = np.concatenate(([np.min(incr)],np.linspace(a, b, 40), [np.max(incr)]))
    plt.hist(incr[0], bins=bins, density=True, histtype='stepfilled', alpha=0.2)
    plt.xlim([x[0], x[-1]])
    plt.legend(loc='best', frameon=False)
    plt.show()


def display_cutoffeffect(incr,n,Delta,u_grid,kappa,text,cf_th=None,reduce=False,bound=(0,0)):
    res=cf_estimator(incr,n,Delta,u_grid)
    res2=cf_estimator_cutoff(incr,n,Delta,u_grid,kappa)
    res3=cf_estimator_cutoff_part(incr,n,Delta,u_grid,kappa)
    plt.figure(figsize=(15, 15))
    
    plt.subplot(3,1,1)
    plt.title('real part')
    plt.plot(u_grid,np.real(res),label='empirical',color='blue',linestyle='dotted')
    plt.axhline((1+kappa*sqrt(log(n)))/sqrt(n),color='black',alpha=0.2)
    plt.plot(u_grid,np.real(res2),color='red',label='cutoff',linestyle='dashdot',alpha=0.6)
    plt.plot(u_grid,np.real(res3),color='olive',label='cutoff separate')
    if cf_th is not None:
        plt.scatter(u_grid,np.real(cf_th),color='grey',label='th')
        #plt.plot(u_grid,np.abs(cf_th),linestyle='-',color='grey',alpha=0.2)
        #plt.plot(u_grid,-np.abs(cf_th),linestyle='-',color='grey',alpha=0.2)
    if reduce:
        plt.xlim([bound[0],bound[1]])
    plt.legend()
    
    
    plt.subplot(3,1,2)
    plt.title('imag part')
    plt.plot(u_grid,np.imag(res),label='empirical',color='blue',linestyle='dotted')
    plt.axhline((1+kappa*sqrt(log(n)))/sqrt(n),color='black',alpha=0.2)
    plt.plot(u_grid,np.imag(res2),label='cutoff',color='red',linestyle='dashdot',alpha=0.6)
    plt.plot(u_grid,np.imag(res3),color='olive',label='cutoff separate')
    if cf_th is not None:
        plt.scatter(u_grid,np.imag(cf_th),label='th',color='grey')
        # plt.plot(u_grid,np.abs(cf_th),linestyle='-',color='grey',alpha=0.2)
        # plt.plot(u_grid,-np.abs(cf_th),linestyle='-',color='grey',alpha=0.2)
    if reduce:
        plt.xlim([bound[0],bound[1]])
    plt.legend()


    
    plt.subplot(3,1,3)
    plt.title('abs')
    plt.plot(u_grid,np.abs(res),label='empirical',color='blue',linestyle='dotted')
    plt.axhline((1+kappa*sqrt(log(n)))/sqrt(n),color='black',alpha=0.2)
    plt.plot(u_grid,np.abs(res2),color='red',label='cutoff',linestyle='dashdot',alpha=0.6)
    plt.plot(u_grid,np.abs(res3),color='olive',label='cutoff separate')
    if cf_th is not None:
        plt.scatter(u_grid,np.abs(cf_th),color='grey',label='th')

    if reduce:
        plt.xlim([bound[0],bound[1]])
    plt.legend()
    plt.figtext(0.55, 0.005,text, ha="center", fontsize=20)
    plt.savefig(text)
    plt.show()
    

def display_density(incr_tab,x_grid,n,Delta,kappa,n_bound,n_int,real_density,title):
    plt.figure()
    plt.title(title)
    i=0
    for incr in incr_tab:
        density_est=finv_density(x_grid,incr,n,Delta,n_bound,n_int,kappa,cutoff='abs cutoff')
        i+=1
        print('Density {}:{}'.format(i,len(incr_tab)))
        plt.plot(x_grid,density_est,color='lime')
    plt.plot(x_grid,real_density,label='True density',color='red')
    plt.legend()
    plt.savefig(title)
    plt.show()
   
    
def display_density_separate(incr_tab,x_grid,n,Delta,kappa,n_bound,n_int,real_density,title):
    plt.figure()
    plt.title(title)
    i=0
    for incr in incr_tab:
        density_est_separate=finv_density(x_grid,incr,n,Delta,n_bound,n_int,kappa,cutoff='real/imag cutoff')
        i+=1
        print('Density {}:{}'.format(i,len(incr_tab)))
        plt.plot(x_grid,density_est_separate,color='lime')
    plt.plot(x_grid,real_density,label='True density',color='red')
    plt.legend()
    plt.savefig(title)
    plt.show()
    
def display_density_direct(incr_tab,x_grid,n,Delta,n_bound,n_int,kappa,real_density,title):
    #n_bound is the m
    plt.figure()
    plt.title(title)
    i=0
    for incr in incr_tab:
        density_est_direct=finv_density(x_grid,incr,n,Delta,n_bound,n_int,kappa,cutoff='direct')
        i+=1
        print('Density {}:{}'.format(i,len(incr_tab)))
        plt.plot(x_grid,density_est_direct,color='lime')
    plt.plot(x_grid,real_density,label='True density',color='red')
    plt.legend()
    plt.savefig(title)
    plt.show()


def display_density_selectedkappa(incr_tab,x_grid,n,Delta,int_bound,n_int,kappa_tab,real_density,title):
    plt.figure()
    plt.title(title)
    for i in range(len(incr_tab)):
        incr=incr_tab[i]
        kappa=kappa_tab[i]
        density_est= finv_density(x_grid,incr,n,Delta,int_bound,n_int,kappa,cutoff='abs cutoff')
        print('Density {}:{}'.format(i,len(incr_tab)))
        plt.plot(x_grid,density_est,color='lime')
    plt.plot(x_grid,real_density,label='True density',color='red')
    plt.legend()
    #plt.savefig(title)
    plt.show()
    

def create_gif(image_paths, output_path, duration_per_frame=10):
    # Open images and store them in a list
    images = [Image.open(image) for image in image_paths]
    
    # Convert duration from milliseconds to seconds
    duration = duration_per_frame  # duration in milliseconds
    
    # Save the images as a GIF
    images[0].save(
        output_path, 
        save_all=True, 
        append_images=images[1:], 
        duration=duration, 
        loop=0
    )
    print(f"GIF saved at {output_path}")



# =============================================================================
# Examples calibration
# =============================================================================

def  integration_bound(Delta,process='default'):
    if process=='default':
        return None 
    elif process=='brownian':
        if Delta==10:
            return 5,1000
        if Delta==1:
            return 10,1000
        if Delta==0.1:
            return 100,1000
        if Delta==0.01:
            return 50,1000 
        
    elif process=='cauchy':
        if Delta==10:
            return 5,1000
        if Delta==1:
            return 10,1000
        if Delta==0.1:
            return 100,1000
        if Delta==0.01:
            return 500,1000
        
    elif process=='levy':
        if Delta==10:
            return 0.01,5000
        if Delta==1:
            return 1,5000#100,1000
        if Delta==0.1:
            return 100,5000
        if Delta==0.01:
            return 5000,10000 #100,1000
        
    elif process=='stable FV':
        if Delta==10:
            return 0.1,5000
        if Delta==1:
            return 1,5000#100,1000
        if Delta==0.1:
            return 50,5000
        if Delta==0.01:
            return 5000,10000 #100,1000
    elif process=='stable IV':
        if Delta==10:
            return 0.5,1000
        if Delta==1:
            return 1,1000#100,1000
        if Delta==0.1:
            return 10,1000
        if Delta==0.01:
            return 5000,10000 #100,1000

def bound_cf(Delta):
    if Delta==0.1:
        return 100,0.01
    elif Delta==1:
        return 10,0.001
    elif Delta==10:
        return 1,0.0001
    elif Delta==5:
    
                                            return 3,0.0002

    
def density_bound(Delta,process='default',P=1):
    if process=='default':
        return None
    elif process=='brownian':
        if Delta==10:
            return -10,10
        if Delta==5:
            return -10,10
        if Delta==1:
            return -3,3
        if Delta==0.1:
            return -1.5,1.5
        if Delta==0.01:
            return -0.5,0.5 #-0.2,.2
        
    elif process=='cauchy':
        if Delta==10:
            return -100,100
        if Delta==5:
            return -20,20
        if Delta==1:
            return -5,5
        if Delta==0.1:
            return -1.5,1.5
        if Delta==0.01:
            return -0.2,0.2
        
    elif process=='levy':
        if Delta==10:
            return 100,10000
        if Delta==5:
            return 0,2000
        if Delta==1:
            return -2*Delta*P,100 #P=1 or mutiply both by P
        if Delta==0.1:
            return -2*Delta*P,1.5
        if Delta==0.01:
            return -2*Delta*P,0.1 
        
    elif process=='stable FV':
        if Delta==10:
            return -2000,2000
        if Delta==5:
            return -500,500
        if Delta==1:
            return -50,50
        if Delta==0.1:
            return -2,2
        if Delta==0.01:
            return -0.25,0.25
    elif process=='stable IV':
        if Delta==10:
            return -100,50
        if Delta==5:
            return -60,60
        if Delta==1:
            return -40,40
        if Delta==0.1:
            return -10,10
        if Delta==0.01:
            return -0.25,0.25
    elif process=='tempered stable FV':
        if Delta==10:
            return -2000,2000
        if Delta==5:
            return -500,500
        if Delta==1:
            return -50,50
        if Delta==0.1:
            return -2,2
        if Delta==0.01:
            return -0.25,0.25
    elif process=='tempered stable Cauchy':
        if Delta==10:
            return -100,100
        if Delta==5:
            return -20,20
        if Delta==1:
            return -5,5
        if Delta==0.1:
            return -1.5,1.5
        if Delta==0.01:
            return -0.2,0.2
    elif process=='tempered stable IV':
        if Delta==10:
            return -100,50
        if Delta==5:
            return -60,60
        if Delta==1:
            return -40,40
        if Delta==0.1:
            return -10,10
        if Delta==0.01:
            return -0.25,0.25
        




def display_function(title,incr_tab,x_grid,true_density,th_cf,u_grid,n,Delta,N,selection_level=1):
    print('########## {} estimation ##############'.format(title))    
    #Memory tabs
    print("Initilizing memory")
    kappa_selected_tab=np.zeros(N)
    density_tab=np.zeros((N,len(x_grid)))
    density_tab_fixedkappa=np.zeros((len(kappa_fix),N,len(x_grid)))
    cf_tab=np.zeros((N,len(u_grid)),dtype=complex)
    risk_tab=np.zeros(N)
    risk_tab_fixedkappa=np.zeros((len(kappa_fix),N))
    emp_estimator_tab=np.zeros((N,len(u_grid)),dtype=complex)
    print("Estimator computation")
    for i in range(N):
        print(" Estimator {}/{}".format(i+1,N))
        print("     Increment computation")
        incr=incr_tab[i]
        print("     empirical cf computation")
        emp_estimator=cf_estimator(incr,u_grid)
        print("     kappa selection")
        kappa=kappa_selection(emp_estimator,kappa_tab,n,(i==0),title,selection_level)
        print("     Fourier inversion")
        emp_estimator_cutoff=cf_estimator_cutoff(emp_estimator,n,Delta,kappa)
        est_density= f_hat(x_grid,emp_estimator_cutoff)
        #
        density_tab[i]=est_density
        emp_estimator_tab[i]=emp_estimator
        kappa_selected_tab[i]=kappa
        cf_tab[i]=emp_estimator_cutoff
        risk_tab[i]=risk(est_density,true_density,x_grid[1]-x_grid[0])
        print("     computation with fixed kappa")
        for j in range(len(kappa_fix)):
            kappa=kappa_fix[j]
            emp_estimator_fixedcutoff=cf_estimator_cutoff(emp_estimator,n,Delta,kappa)
            est_density_fixedkappa=f_hat(x_grid,emp_estimator_fixedcutoff)
            density_tab_fixedkappa[j,i]= est_density_fixedkappa
            risk_tab_fixedkappa[j,i]=risk(est_density_fixedkappa,true_density,x_grid[1]-x_grid[0])
    
    # #display characteristic function
    # print("Display of the characteristic function")
    # plt.figure()
    # plt.title('abs(cf) adaptive estimation {} n={} Delta={}'.format(title,n,Delta))
    # for i in range(N):
    #     print('CF {}:{}'.format(i+1,N))
    #     kappa=kappa_selected_tab[i]
    #     emp_estimator=emp_estimator_tab[i]
    #     cutoff_est=cf_tab[i]
    #     plt.plot(u_grid,np.abs(cutoff_est))
    #     plt.plot(u_grid,np.abs(emp_estimator))
    #     kappa_n = (1+kappa*sqrt(log(n)))/sqrt(n)
    #     plt.axhline(kappa_n)
    # plt.plot(u_grid,np.abs(th_cf),label='True cf',color='red')
    # plt.axhline(1)
    # plt.legend()
    # plt.savefig(title+"_CFEstimationAdaptativen{}Delta{}.png".format(n,Delta))
    # plt.show()

    #DISPLAY
    #display density
    # print("Display of the density")
    # plt.figure()
    # plt.title('Density adaptive estimation {} n={} Delta={}'.format(title,n,Delta))
    # for i in range(N//2):
    #     print('Density {}:{}'.format(i+1,N))
    #     plt.plot(x_grid,density_tab[i],color='lime')
    # plt.plot(x_grid,true_density,label='True density',color='red')
    # plt.legend()
    # plt.savefig(title+"_DensityEstimationAdaptativen{}Delta{}.png".format(n,Delta))
    # plt.show()
    
#    #☺ display density fixed kappa
#     print("Display the density for a fixed kappa")
#     for j in range(len(kappa_fix)):
#         print(  'kappa={}'.format(kappa))
#         kappa=kappa_fix[j]
#         plt.figure()
#         plt.title('Density {} kappa={} n={} Delta={}'.format(title,kappa,n,Delta))
#         for i in range(N):
#             print('Density {}:{}'.format(i+1,N))
#             plt.plot(x_grid,density_tab_fixedkappa[j,i],color='lime')
#         plt.plot(x_grid,true_density,label='True density',color='red')
#         plt.legend()
#         plt.savefig(title+"_DensityEstimationn{}Delta{}Kappa={}.png".format(n,Delta,kappa))
#         plt.show()
    #display selected kappa
    # plt.figure()
    # plt.title('Selected kappa n={} Delta={}'.format(n,Delta))
    # plt.hist(kappa_selected_tab)
    # #plt.savefig(title+"_AdaptiveKappaHistogramn{}Delta{}.png")
    # plt.show()
    
    #Error
    print('Quadratic error for n={}, Delta={}'.format(n,Delta))
    print('         mean error:{0:.4f} / mean std:{1:.4f}'.format(np.mean(risk_tab),np.std(risk_tab)))
    
    # for j in range(len(kappa_fix)):
    #     kappa=kappa_fix[j]
    # #Error fixed kappa
    #     print('Fixed kappa={} Quadratic error for n={}, Delta={}'.format(kappa,n,Delta))
    #     print('         mean error:{0:.4f} / mean std:{1:.4f}'.format(np.mean(risk_tab_fixedkappa[j]),np.std(risk_tab_fixedkappa[j])))
        
    print('----------------------------------------------------')
    # print('Plot of the quadratic error')
    # plt.figure()
    # plt.title('{} Quadratic error : kappa choice n={}, Delta={} '.format(title,n,Delta))
    # kappa_grid=kappa_fix #+ #list(kappa_selected_tab)
    # risk_grid=list(np.mean(risk_tab_fixedkappa,axis=1)) #+ #list(risk_tab)
    # plt.scatter(kappa_grid,risk_grid,label='fixed kappa')
    # plt.scatter(np.mean(kappa_selected_tab), np.mean(risk_tab),color='red',label='adaptive_kappa')

    # plt.savefig(title+"_QuadraticErrorn{}Delta{}.png".format(n,Delta))
    # plt.legend()
    # plt.show()
     
    #SAVE
    return np.mean(risk_tab),np.std(risk_tab),np.mean(risk_tab_fixedkappa,axis=1),np.std(risk_tab_fixedkappa,axis=1),np.mean(kappa_selected_tab),np.std(kappa_selected_tab)




start = time.time()
n_tab=[500]
Delta_tab=[0.1,1]
N=2
# #Tempered Stable FV
title='tempered stable FV'
P=1
Q=2
A,B=1,1
drift=0
c=0.0
selection_level=3
alpha=0.7 #alpha != 1
kappa_tab=np.linspace(0,2,20)
kappa_fix=[]

for n in n_tab:
    for Delta in Delta_tab:
        print("-"*20)
        print("Computation for n={} and Delta={}".format(n,Delta))
        print(" Increments generation")
        incr_tab=tempered_stable(n,Delta,alpha,P,Q,A,B,drift,c,N,True,method='bm')
        a,b=density_bound(Delta,title)
        x_grid=np.linspace(a,b,1000)
        true_density = CTS_density(x_grid,alpha,P*Delta,Q*Delta,A,B, adaptive_bound=True)
        M,delta= bound_cf(Delta)
        u_grid=np.arange(-M,M,delta)#np.linspace(-M,M,nM)
        th_cf=CTS_characteristic_function(u_grid, alpha, Delta*P, Delta*Q, A, B)
        print(" Risk computation")
        adt_risk_mean,adt_risk_std,risk_mean,risk_std,adt_kappa_mean,adt_kappa_std= display_function(title,incr_tab,x_grid,true_density,th_cf,u_grid,n,Delta,N,selection_level)
        print(" CSV table setup")
        with open('RiskTableFV.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows([[title,n,Delta,adt_kappa_mean,adt_kappa_std,adt_risk_mean,adt_risk_std]])
            for j in range(len(kappa_fix)):
                kappa=kappa_fix[j]
                writer.writerows([[title,n,Delta,kappa,0,risk_mean[j],risk_std[j]]])



n_tab=[500]
Delta_tab=[0.1,1]
N=2
# #Tempered Stable Cauchy
title='tempered stable Cauchy'
P=1
Q=2
A,B=1,1
drift=0
c=5
selection_level=3
alpha=1.1 #alpha != 1
kappa_tab=np.linspace(0,2,20)
kappa_fix=[]

for n in n_tab:
    for Delta in Delta_tab:
        print("-"*20)
        print("Computation for n={} and Delta={}".format(n,Delta))
        print(" Increments generation")
        incr_tab=tempered_stable(n,Delta,alpha,P,Q,A,B,drift,c,N,True,method='bm')
        a,b=density_bound(Delta,title)
        x_grid=np.linspace(a,b,1000)
        true_density = CTS_density(x_grid,alpha,P*Delta,Q*Delta,A,B, adaptive_bound=True)
        M,delta= bound_cf(Delta)
        u_grid=np.arange(-M,M,delta)#np.linspace(-M,M,nM)
        th_cf=CTS_characteristic_function(u_grid, alpha, Delta*P, Delta*Q, A, B)
        print(" Risk computation")
        adt_risk_mean,adt_risk_std,risk_mean,risk_std,adt_kappa_mean,adt_kappa_std= display_function(title,incr_tab,x_grid,true_density,th_cf,u_grid,n,Delta,N,selection_level)
        print(" CSV table setup")
        with open('RiskTableCauchy.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows([[title,n,Delta,adt_kappa_mean,adt_kappa_std,adt_risk_mean,adt_risk_std]])
            for j in range(len(kappa_fix)):
                kappa=kappa_fix[j]
                writer.writerows([[title,n,Delta,kappa,0,risk_mean[j],risk_std[j]]])



n_tab=[500]
Delta_tab=[0.1,1]
N=2

# #Tempered Stable IV
title='tempered stable IV'
P=1
Q=2
A,B=1,1
drift=0
c=5
selection_level=3
alpha=1.7 #alpha != 1
kappa_tab=np.linspace(0,2,20)
kappa_fix=[]

for n in n_tab:
    for Delta in Delta_tab:
        print("-"*20)
        print("Computation for n={} and Delta={}".format(n,Delta))
        print(" Increments generation")
        incr_tab=tempered_stable(n,Delta,alpha,P,Q,A,B,drift,c,N,True,method='bm')
        a,b=density_bound(Delta,title)
        x_grid=np.linspace(a,b,1000)
        true_density = CTS_density(x_grid,alpha,P*Delta,Q*Delta,A,B, adaptive_bound=True)
        M,delta= bound_cf(Delta)
        u_grid=np.arange(-M,M,delta)#np.linspace(-M,M,nM)
        th_cf=CTS_characteristic_function(u_grid, alpha, Delta*P, Delta*Q, A, B)
        print(" Risk computation")
        adt_risk_mean,adt_risk_std,risk_mean,risk_std,adt_kappa_mean,adt_kappa_std= display_function(title,incr_tab,x_grid,true_density,th_cf,u_grid,n,Delta,N,selection_level)
        print(" CSV table setup")
        with open('RiskTableIV.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows([[title,n,Delta,adt_kappa_mean,adt_kappa_std,adt_risk_mean,adt_risk_std]])
            for j in range(len(kappa_fix)):
                kappa=kappa_fix[j]
                writer.writerows([[title,n,Delta,kappa,0,risk_mean[j],risk_std[j]]])
end = time.time()
requiered_time = end - start
print("Process took {}".format(requiered_time))

