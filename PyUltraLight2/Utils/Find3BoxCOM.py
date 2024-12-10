import numpy as np

def __call__():
    return Find3BoxCOM()

def Find3BoxCOM(rho,xGrid, yGrid, zGrid):
    return np.array([np.sum(xGrid * rho),np.sum(yGrid * rho),np.sum(zGrid * rho)])/np.sum(rho)