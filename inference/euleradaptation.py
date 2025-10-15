

import numpy as np
from scipy.integrate import romb, quad, simpson
import scipy.stats as st
import matplotlib.pyplot as plt
from math import cos, gamma, pi,log,tan,sin,floor,exp,ceil,sqrt
import time


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
    
def euler_char(phiH,n,kappa):
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
    euler_tab= np.array([euler_char(phiH,n,kappa) for kappa in kappa_grid])
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
    