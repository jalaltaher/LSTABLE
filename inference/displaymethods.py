
import numpy as np
import matplotlib.pyplot as plt
import csv

import sys
import os
print("cwd:", os.getcwd())

from inference.euleradaptation import *
from inference.parameterscalibration import *
from inference.spectralestimator import cf_estimator, cf_estimator_cutoff, risk, f_hat


kappa_fix=[]
print("Import done")
def display_function(title,incr_tab,x_grid,true_density,th_cf,u_grid,n,Delta,N,kappa_tab,selection_level=1):
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
        est_density= f_hat(x_grid,emp_estimator_cutoff,u_grid)

        #
        density_tab[i]=est_density
        emp_estimator_tab[i]=emp_estimator
        kappa_selected_tab[i]=kappa
        cf_tab[i]=emp_estimator_cutoff
        risk_tab[i]=risk(est_density,true_density,x_grid[1]-x_grid[0])
        print("     computation with fixed kappa: Disregarded")
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

print("A")