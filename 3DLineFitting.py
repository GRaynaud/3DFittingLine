# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 11:51:48 2020

@author: G. Raynaud
"""

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt


# Data construction In a real case, just import the real data
N_data = 10

z_data = np.linspace(0.,2.,N_data)
x_data = np.cos(z_data)
y_data = np.sin(z_data)

# Parameters for the fitting
N_courbe = 100
t = np.linspace(0,1,N_courbe)

N_params = 5 # Degree -1 of the polynomia
x0 = np.ones(3*N_params)


def dist_L2(x1,y1,z1,x2,y2,z2):
    return np.square(x2-x1)+np.square(y2-y1) + np.square(z2-z1)

def cost(x,disp=False):
    '''
    x : 1D np.array or list containing your scalar parameters to optimize
    disp : boolean, True to plot data vs fit
    ----
    return cost function defined by the mea square distance of each data point to the curve 
    and the min square distance of each point from the curve to the data
    '''
    # Fitting line
    x_fit = np.sum(np.asarray([x[k]*np.power(t,k) for k in range(N_params)]),axis=0)
    y_fit = np.sum(np.asarray([x[k+N_params]*np.power(t,k) for k in range(N_params)]),axis=0)
    z_fit = np.sum(np.asarray([x[k+2*N_params]*np.power(t,k) for k in range(N_params)]),axis=0)
    
    # Min distance of each data points to the fitting line
    err = 0
    for k in range(N_data):
        err += np.min([dist_L2(x_fit[j],y_fit[j],z_fit[j],x_data[k],y_data[k],z_data[k]) for j in range(N_courbe)])
    err *= 1/N_data
    
    # Min distance of each curve points to the data points
    err2 = 0
    for j in range(N_courbe):
        err2 += np.min([dist_L2(x_fit[j],y_fit[j],z_fit[j],x_data[k],y_data[k],z_data[k]) for k in range(N_data)]) 
    err2 *= 1/N_courbe
    
    # 3D plot of data vs fit
    if disp:
        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        ax.scatter(x_data,y_data,z_data,color='green')
        ax.scatter(x_fit,y_fit,z_fit,marker='.',s=2.,color='blue')
    
    return err + 1e-1*err2



# =============================================================================
# Approximated derivation based method : here an approx of a second order method using low memory
# =============================================================================
res = scipy.optimize.minimize(cost,x0,method='L-BFGS-B')
cost(res.x,True)


# =============================================================================
# Method with no derivatives : differential evolution
# =============================================================================
max_data = np.max(np.abs(np.asarray([x_data,y_data,z_data])))
bounds = [(-3*max_data,3*max_data) for k in range(3*N_params)] # To be tuned

res = scipy.optimize.differential_evolution(cost,bounds,popsize=50,disp=True)
cost(res.x,True)
