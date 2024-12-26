# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 14:37:54 2024

@author: Jalal
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from lstable.CTS_process import *
from lstable.CTS_distribution import *


#Test Upper Gamma function
# z,x=1.j,1
# P1=upper_gamma(z, x)
# P1_l=lower_gamma(z,x)

# z,x=1,5
# P2=upper_gamma(z,x)
# P2_lower=lower_gamma(z,x)
# P3=np.exp(-x)
# P3_lower=1-np.exp(-x)



#Test of the compound Poisson approximation

# alpha,P,Q,A,B=0.6,1,1,1,1
# delta=0.01
# drift=0.0

#print(positive_jumpsize_compound_poisson_approximation(alpha,A,delta))
#print(negative_jumpsize_compound_poisson_approximation(alpha,B,delta))
#print(jumpsize_compound_poisson_approximation(alpha,P,Q,A,B,delta))
# sample=jumpsize_compound_poisson_approximation_vectorized(alpha,P,Q,A,B,delta,1000)
# plt.figure()
# plt.hist(sample,density=True,bins=50)
# plt.show()


# alpha,P,Q,A,B=1.6,1,1,1,1
# delta=0.001
# drift=0.0
# n,Delta=1000,0.001

# time_grid=np.linspace(0,n*Delta,n+1)
# start = time.time()
# increments=compound_poisson_approximation_direct_algorithm_vectorized(n,Delta,drift,alpha,P,Q,A,B,delta)
# end=time.time()
# print('direct cp algorithm:', end-start)
# start= time.time()
# increments_sorting=compound_poisson_approximation_sorting_algorithm(n,Delta,drift,alpha,P,Q,A,B,delta)
# end= time.time()
# print('sorting cp algorithm:', end-start)

# start= time.time()
# increments_w_brownian_approximation=compound_poisson_gaussian_approximation_tempered_stable(n,Delta,drift,alpha,P,Q,A,B,delta)
# end= time.time()
# print('cp + gaussian:', end-start)
# print('Asmussen_criterion: ', criterion_gaussian_approximation(delta,alpha,P,Q,A,B))


# plt.figure()
# plt.plot(time_grid,increments)
# plt.plot(time_grid,increments_sorting)
# plt.plot(time_grid,increments_w_brownian_approximation)
# plt.plot()


# plt.figure()
# #plt.plot(time_grid,np.cumsum(increments),label='d')
# plt.plot(time_grid,np.cumsum(increments_sorting),label='cpa')
# plt.plot(time_grid,np.cumsum(increments_w_brownian_approximation),label='cpga')
# plt.legend()
# plt.plot()



#Testing all methods
alpha,P,Q,A,B=1.6,1,1,1,1
delta=0.001
drift=0.0
n,Delta=1000,0.01
c=1

grid=np.linspace(-3,3,1000)

theoretical_density= CTS_density(
    grid,
    alpha,
    P*Delta,
    Q*Delta,
    A,
    B,
    adaptive_bound=True,
)

plt.figure()
plt.plot(grid,theoretical_density)
plt.show()


time_grid=np.linspace(0,n*Delta,n+1)
increments_bm=tempered_stable_process_increments(n,Delta,drift,alpha,P,Q,A,B,c, method='bm')
increments_cpa=tempered_stable_process_increments(n,Delta,drift,alpha,P,Q,A,B,c, method='cpa')
increment_cpga=tempered_stable_process_increments(n,Delta,drift,alpha,P,Q,A,B,c, method='cpga')

plt.figure()
plt.plot(time_grid,np.cumsum(increments_bm))
plt.plot(time_grid,np.cumsum(increments_cpa))
plt.plot(time_grid,np.cumsum(increments_cpga))
plt.show()
