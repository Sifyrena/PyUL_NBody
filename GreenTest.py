#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 16:10:28 2020

@author: ywan598
"""

import numpy as np

import numexpr as ne

import scipy.fftpack



def safeInverse3D(x):
    # Safe inverse for 3D arrays
    n1,n2,n3 = np.shape(x)
    y = np.zeros((n1, n2, n3))
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                if x[i, j, k] > 0:
                    #x[i, j] = 1.0 / float(x[i, j])
                    y[i,j,k] = np.reciprocal(x[i,j,k])
    return y


def greenArray(xyz):
    # make grid for Green's function
    x2, y2, z2 = np.meshgrid(
        xyz, xyz, xyz, 
        sparse=True, indexing='ij',
    )
    gplane = ne.evaluate("x2 + y2 + z2")
    return(gplane)

# makeDCTGreen

n = 15

x2 = np.arange(0, n + 1)**2

arr = greenArray(x2)

arrR = np.sqrt(arr)

arrRZ = safeInverse3D(arrR)



arrRZ[0,0,0] = 3*np.log(np.sqrt(3) + 2) - np.pi/2 #calculated in https://arxiv.org/pdf/chem-ph/9508002.pdf
# no normalization below matches un-normalization in Mathematica

green =  scipy.fftpack.dctn(arrRZ, type = 1, norm = None)

