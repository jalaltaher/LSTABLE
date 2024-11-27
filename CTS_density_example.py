# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 23:09:53 2024

@author: Jalal
"""


import numpy as np
import matplotlib.pyplot as plt
from CTS_distribution import*

def plot_CTS_density(x_grid: np.ndarray, alpha:float, P:float, Q:float, A:float, B:float):
    density =  CTS_density(x_grid, alpha, P, Q, A, B, adaptive_bound=True)
    plt.figure()
    plt.plot(x_grid,density)
    plt.show()
    
    
x_grid=np.linspace(-10,10,1000)
alpha,P,Q,A,B=0.5,1,0.5,10,1
plot_CTS_density(x_grid,alpha,P,Q,A,B)