#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
PyULN Component: PyULN.CalculateSoliton

Based on Build Fri Sep 25 13:43:24 2020
Original author: Ningyuan Guo

Script to find soliton solutions of Schroedinger-Poisson equation using 4th
order Runge-Kutta shooting algorithm.
Adapted to include multiple nodes, external central potential with mass BHmass,
and axion self-interaction of form Lambda * phi^4.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as si
import time

#Note that the spatial resolution of the profile must match the specification of delta_x in the main code.
dr = .00001
max_radius = 9.0 # This may be too large for lower order states; for ground state up to 2 nodes 9 is enough, up to 5 nodes 18 is enough.
rge = max_radius/dr
BHmass = 1 # give in code units. Mass providing central potential -BHmass / r
Lambda = 0 # axion self-interaction strength. Lambda < 0: attractive.

s = 0. # Note that for large BH masses s may become positive.
nodes = [1] # Include here the number of nodes desired.
tolerance = 1e-9 # how close f(r) must be to 0 at r_max; can be changed

plot_while_calculating = True # If true, a figure will be updated for every attempted solution
verbose_output = True # If true, s value of every attempt will be printed in console.


def g1(r, a, b, c, BHmass = BHmass):
    return -(2/r)*c+2*b*a - 2*(BHmass / r)*a + 2 * Lambda * a **3

def g2(r, a, d):
    return 4*np.pi*a**2-(2/r)*d

for order in nodes:
    optimised = False
    tstart = time.time()

    phi_min = -5 # lower phi-tilde value (i.e. more negative phi value)
    phi_max = s
    draw = 0
    currentflag = 0
    if plot_while_calculating == True:
        plt.figure(1)

    while optimised == False:
        nodenumber = 0

        ai = 1
        bi = s
        ci = 0
        di = 0

        la = []
        lb = []
        lc = []
        ld = []
        lr = []
        intlist = []

        la.append(ai)
        lb.append(bi)
        lc.append(ci)
        ld.append(di)
        lr.append(dr/1000)
        intlist.append(0.)


        # kn lists follow index a, b, c, d, i.e. k1[0] is k1a
        k1 = []
        k2 = []
        k3 = []
        k4 = []

        if verbose_output == True:
            print('0. s = ', s)
        for i in range(int(rge)):
            list1 = []
            list1.append(lc[i]*dr)
            list1.append(ld[i]*dr)
            list1.append(g1(lr[i],la[i],lb[i],lc[i])*dr)
            list1.append(g2(lr[i],la[i],ld[i])*dr)
            k1.append(list1)

            list2 = []
            list2.append((lc[i]+k1[i][2]/2)*dr)
            list2.append((ld[i]+k1[i][3]/2)*dr)
            list2.append(g1(lr[i]+dr/2,la[i]+k1[i][0]/2,lb[i]+k1[i][1]/2,lc[i]+k1[i][2]/2)*dr)
            list2.append(g2(lr[i]+dr/2,la[i]+k1[i][0]/2,ld[i]+k1[i][3]/2)*dr)
            k2.append(list2)

            list3 = []
            list3.append((lc[i]+k2[i][2]/2)*dr)
            list3.append((ld[i]+k2[i][3]/2)*dr)
            list3.append(g1(lr[i]+dr/2,la[i]+k2[i][0]/2,lb[i]+k2[i][1]/2,lc[i]+k2[i][2]/2)*dr)
            list3.append(g2(lr[i]+dr/2,la[i]+k2[i][0]/2,ld[i]+k2[i][3]/2)*dr)
            k3.append(list3)

            list4 = []
            list4.append((lc[i]+k3[i][2])*dr)
            list4.append((ld[i]+k3[i][3])*dr)
            list4.append(g1(lr[i]+dr,la[i]+k3[i][0],lb[i]+k3[i][1],lc[i]+k3[i][2])*dr)
            list4.append(g2(lr[i]+dr,la[i]+k3[i][0],ld[i]+k3[i][3])*dr)
            k4.append(list4)

            la.append(la[i]+(k1[i][0]+2*k2[i][0]+2*k3[i][0]+k4[i][0])/6)
            lb.append(lb[i]+(k1[i][1]+2*k2[i][1]+2*k3[i][1]+k4[i][1])/6)
            lc.append(lc[i]+(k1[i][2]+2*k2[i][2]+2*k3[i][2]+k4[i][2])/6)
            ld.append(ld[i]+(k1[i][3]+2*k2[i][3]+2*k3[i][3]+k4[i][3])/6)
            lr.append(lr[i]+dr)
            intlist.append((la[i]+(k1[i][0]+2*k2[i][0]+2*k3[i][0]+k4[i][0])/6)**2*(lr[i]+dr)**2)

            if la[i]*la[i-1] < 0:
                nodenumber = nodenumber + 1

            if (draw % 10 == 0) and (plot_while_calculating == True):
                plt.clf()

            if nodenumber > order:
                phi_min = s
                s = (phi_min + phi_max)/2
                if verbose_output == True:
                    print('1. ', s)
                if plot_while_calculating == True:
                    plt.plot(la)
                    plt.pause(0.05)
                draw += 1
                break
            elif la[i] > 1.0:
                currentflag = 1.1
                phi_max = s
                s = (phi_min + phi_max)/2
                if verbose_output == True:
                    print('1.1 ', s)
                if plot_while_calculating == True:
                    plt.plot(la)
                    plt.pause(0.05)
                draw += 1
                break
            elif la[i] < -1.0:
                currentflag = 1.2
                phi_max = s
                s = (phi_min + phi_max)/2
                if verbose_output == True:
                    print('1.2 ', s)
                if plot_while_calculating == True:
                    plt.plot(la)
                    plt.pause(0.05)
                draw += 1
                break

            if i == int(rge)-1:
                if nodenumber < order:
                    currentflag = 2
                    phi_max = s
                    s = (phi_min + phi_max)/2
                    if verbose_output == True:
                        print('2. ', s)
                    if plot_while_calculating == True:
                        plt.plot(la)
                        plt.pause(0.05)
                    draw += 1
                    break
                elif ((order%2 == 1) and (la[i] < -tolerance)) or ((order%2 == 0) and (la[i] > tolerance)):
                    currentflag = 4
                    phi_max = s
                    s = (phi_min + phi_max)/2
                    if verbose_output == True:
                        print('4. ', s)
                    if plot_while_calculating == True:
                        plt.plot(la)
                        plt.pause(0.05)
                    draw += 1
                    break
                else:
                    optimised = True
                    print('{}{}'.format('Successfully optimised for s = ', s))
                    timetaken = time.time() - tstart
                    print('{}{}'.format('Time taken = ', timetaken))
                    grad = (lb[i] - lb[i - 1]) / dr
                    const = lr[i] ** 2 * grad
                    beta = lb[i] + const / lr[i]

    #Calculate full width at half maximum density:

    difflist = []
    for i in range(int(rge)):
        difflist.append(abs(la[i]**2 - 0.5))
    fwhm = 2*lr[difflist.index(min(difflist))]

    #Calculate the (dimensionless) mass of the soliton:

    mass = si.simps(intlist,lr)*4*np.pi

    #Calculate the radius containing 90% of the massin

    partial = 0.
    for i in range(int(rge)):
        partial = partial + intlist[i]*4*np.pi*dr
        if partial >= 0.9*mass:
            r90 = lr[i]
            break

    partial = 0.
    for i in range(int(rge)):
        partial = partial + intlist[i]*4*np.pi*dr
        if lr[i] >= 0.5*1.38:
            print ('{}{}'.format('M_core = ', partial))
            break

    print ('{}{}'.format('Full width at half maximum density is ', fwhm))
    print ('{}{}'.format('Beta is ', beta))
    print ('{}{}'.format('mass is ', mass))
    print ('{}{}'.format('Radius at 90% mass is ', r90))
    print ('{}{}'.format('MBH/Msol is ', BHmass/mass))

    #Save the numpy array and plots of the potential and wavefunction profiles.
    psi_array = np.array(la)
    np.save('initial_f-order='+str(order) + ' massratio=' + str(np.round(BHmass/mass, decimals=5)) + 'lambda = '+ str(Lambda), psi_array)

    plt.figure()
    plt.plot(lr,lb, 'brown', label='Potential Profile')
    plt.xlim(0.,5.6)
    plt.ylim(-2.5, 2.0)
    plt.xlabel('r (code units)', fontsize=16)
    locs, labels = plt.xticks()
    plt.xticks(locs, labels, fontsize=14)
    locs, labels = plt.yticks()
    plt.yticks(locs, labels, fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.figure()
    plt.plot(lr,la, label='Wavefunction Profile')
    plt.xlim(0.,5.6)
    locs, labels = plt.xticks()
    plt.xticks(locs, labels, fontsize=14)
    locs, labels = plt.yticks()
    plt.yticks(locs, labels, fontsize=14)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('r (code units)', fontsize=16)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.title('{}{}'.format('Wavefunction profile for s = ', s))
    plt.savefig('{}{}{}{}{}'.format('./Soliton Profile s = ', s, 'massratio = ', np.round(BHmass/mass, decimals=5), '.eps'), format='eps', dpi = 1000)
    plt.xlim(0.,(rge-1)*dr)

    plt.close(1)

print ('Complete.')