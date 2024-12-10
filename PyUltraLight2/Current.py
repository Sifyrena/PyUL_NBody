import builtins

# Housekeeping
import time
from datetime import datetime
import sys
import os
import multiprocessing
import json

# Core Maths
import numpy as np
import numexpr as ne
import numba
import pyfftw

# Solving Stuff
from scipy.interpolate import CubicSpline as CS 
from scipy.special import sph_harm as SPH

# IO
import h5py

# Jupyter
from IPython.core.display import clear_output


################################## DIALOG BOX #################################

from .printU import printU


    

def ULDStepEst(duration,duration_units,length,length_units,resol,step_factor, save_number = -1):
    
    lengthC = convert(length, length_units, 'l')
 
    t = convert(duration, duration_units, 't')
    
    delta_t = (lengthC/float(resol))**2/np.pi

    min_num_steps = np.ceil(t / delta_t)
    MinUS = int(min_num_steps//step_factor)

    #print(f'The required number of ULDM steps is {MinUS}')
    
    if save_number > 0:
        
        if save_number >= MinUS:
            MinUS = int(save_number)
        
        else:
            MinUS = int(save_number * (MinUS // (save_number) + 1))
            
    #print(f'The actual ULDM steps is {MinUS}')
    
    return MinUS

DispN = ULDStepEst


        
# CALCULATE_ENERGIES, NEW VERSION IN 2.16
            

### Toroid or something
        
def Wrap(TMx, TMy, TMz, lengthC):
    
    if TMx > lengthC/2:
        TMx = TMx - lengthC
        
    if TMx < -lengthC/2:
        TMx = TMx + lengthC
        
        
    if TMy > lengthC/2:
        TMy = TMy - lengthC
    
    if TMy < -lengthC/2:
        TMy = TMy + lengthC
        
        
    if TMz > lengthC/2:
        TMz = TMz - lengthC
    
    if TMz < -lengthC/2:
        TMz = TMz + lengthC
        
    return TMx,TMy,TMz

FWrap = Wrap

### For Immediate Interpolation of Field Energy

def QuickInterpolate(Field,lengthC,resol,position):
        #Code Position
                
        RNum = (position*1/lengthC+1/2)*resol

        RPt = np.floor(RNum)
        RRem = RNum - RPt
                
        RX = RRem[0]
        RY = RRem[1]
        RZ = RRem[2]
        
        Interp = 0
        
        # Need special treatment if any of these is zero or close to resol!
        RPtX = int(RPt[0])
        RPtY = int(RPt[1])
        RPtZ = int(RPt[2])

        if (RPtX >= resol-1) or (RPtY >= resol-1) or (RPtZ >= resol-1):
            #raise RuntimeError (f'Particle #{i} reached boundary on the +ve side. Halting.')
            return Interp
            
        if (RPtX <= 0) or (RPtY <= 0) or (RPtZ <= 0):
            #raise RuntimeError (f'Particle #{i} reached boundary on the +ve side. Halting.')
            return Interp

        else:
        
            SPC = Field[RPtX:RPtX+2,RPtY:RPtY+2,RPtZ:RPtZ+2]
            # This monstrosity is actually faster than tensor algebra...
            Interp += (1-RX)*(1-RY)*(1-RZ)*SPC[0,0,0] # Lower Left Near
            Interp += (1-RX)*(1-RY)*(  RZ)*SPC[0,0,1]
            Interp += (1-RX)*(  RY)*(1-RZ)*SPC[0,1,0]
            Interp += (1-RX)*(  RY)*(  RZ)*SPC[0,1,1]
            Interp += (  RX)*(1-RY)*(1-RZ)*SPC[1,0,0]
            Interp += (  RX)*(1-RY)*(  RZ)*SPC[1,0,1]
            Interp += (  RX)*(  RY)*(1-RZ)*SPC[1,1,0]
            Interp += (  RX)*(  RY)*(  RZ)*SPC[1,1,1] # Upper Right Far

            return Interp
            
### Method 3 Interpolation Algorithm




def LpEval(psi,rho,funct,resol,Uarray,Kx,Ky,Kz,ifft_funct):
    
    funct *= 1j
    
    A = np.absolute(psi)

    spacing = Uarray[1]-Uarray[0]
    
    DAx, DAy, DAz = np.gradient(A, spacing)

    DAx = ne.evaluate('A*DAx')
    KGx = ne.evaluate('Kx*funct')
    DPx = ifft_funct(KGx)
    DAx = ne.evaluate('imag(DAx - conj(psi)*DPx)')
    
    DAy = ne.evaluate('A*DAy')
    KGy = ne.evaluate('Ky*funct')
    DPy = ifft_funct(KGy)
    DAy = ne.evaluate('imag(DAy - conj(psi)*DPy)')    

    DAz = ne.evaluate('A*DAz')
    KGz = ne.evaluate('Kz*funct')
    DPz = ifft_funct(KGz)
    DAz = ne.evaluate('imag(DAz - conj(psi)*DPz)')
    
    pOut = -1 * np.array([np.sum(DAx),np.sum(DAy),np.sum(DAz)])*spacing**3
    
    LOut = L_jit(Uarray,DAx,DAy,DAz)

    LOut *= spacing**3

    return pOut, LOut 

def pEval(psi,rho,funct,resol,Uarray,Kx,Ky,Kz,ifft_funct):
    
    funct *= 1j
    
    A = np.absolute(psi)

    spacing = Uarray[1]-Uarray[0]
    
    DAx, DAy, DAz = np.gradient(A, spacing)

    DAx = ne.evaluate('A*DAx')
    KGx = ne.evaluate('Kx*funct')
    DPx = ifft_funct(KGx)
    DAx = ne.evaluate('imag(DAx - conj(psi)*DPx)')*spacing**3
    
    DAy = ne.evaluate('A*DAy')
    KGy = ne.evaluate('Ky*funct')
    DPy = ifft_funct(KGy)
    DAy = ne.evaluate('imag(DAy - conj(psi)*DPy)')*spacing**3    

    DAz = ne.evaluate('A*DAz')
    KGz = ne.evaluate('Kz*funct')
    DPz = ifft_funct(KGz)
    DAz = ne.evaluate('imag(DAz - conj(psi)*DPz)')*spacing**3
    
    return DAx, DAy, DAz

def LUQuick(Uarray,DelTx,DelTy,DelTz):
    
    L = np.zeros(3)

    for ind in np.ndindex(DelTx.shape): 
        R = np.array([Uarray[ind[0]],Uarray[ind[1]],Uarray[ind[2]]])
        P = np.array([DelTx[ind],DelTy[ind],DelTz[ind]])

        Vec = np.cross(R,P)

        L += Vec
    
    return L
    
L_jit = numba.jit(LUQuick)

###
###
###
#### Mid June Addition (Momentum)

def WrapToCircle(Array):
    
    Array[Array<np.pi] += np.pi*2
    Array[Array>np.pi] -= np.pi*2
    
    return Array


def LpEvalFast(psi,rho, funct,resol,Uarray,Kx,Ky,Kz,ifft_funct): # THIS DOES NOT WORK, FOR OBVIOUS REASONS.
 
    Theta = np.angle(psi)
    spacing = Uarray[1]-Uarray[0]
        
    DTx, DTy, DTz = np.gradient(Theta, 1)
    
    DTx = WrapToCircle(DTx)/spacing * rho
    DTy = WrapToCircle(DTy)/spacing * rho
    DTz = WrapToCircle(DTz)/spacing * rho
        
    pOut = -1 * np.array([np.sum(DTx),np.sum(DTy),np.sum(DTz)])*spacing**3
    
    LOut = L_jit(Uarray,DTx,DTy,DTz)

    LOut *= spacing**3

    return pOut, LOut 

######################### Soliton Init Factory Setting!

def LoadDefaultSoliton(Silent = True):
    
    f = np.load('./Soliton Profile Files/initial_f.npy')
    
    if not Silent:
        printU(f"\n{Version} Loaded original PyUL soliton profiles.",'Load Soliton')
    
    return f

######################### The fun kind of Soliton Init.

def BHGuess(Ratio):

    # More to come!
    LowGuess = 0
    HighGuess = 1
    return LowGuess, HighGuess

def BHRatioTester(TargetRatio,Iter,Tol,BHMassGMin,BHMassGMax,Smoo):
    
    # FW Draft
    
    #Note that the spatial resolution of the profile must match the specification of delta_x in the main code.
    dr = .01
    max_radius = 10.0 
    rge = max_radius/dr

    Lambda = 0 # axion self-interaction strength. Lambda < 0: attractive.

    s = 0. # Note that for large BH masses s may become positive.
    nodes = [0] # Include here the number of nodes desired.
    tolerance = 1e-5 # how close f(r) must be to 0 at r_max; can be changed

    plot_while_calculating = False # If true, a figure will be updated for every attempted solution

    verbose_output = False # If true, s value of every attempt will be printed in console.
    
    print(f"Starting experiment between {BHMassGMin} and {BHMassGMax}")
    
   
    BHMassLo = BHMassGMin
    BHMassHi = BHMassGMax
    
    BHMass = (BHMassLo+BHMassHi)/2
    
    BHmass = BHMass # give in code units.
 
    s = 0
    IterInt = 0
    
    while IterInt <= Iter:
        
        order = 0
        
        def g1(r, a, b, c, BHmass = BHMass):
            return -(2/r)*c+2*b*a - 2*(Smoo*BHmass / np.sqrt(1+(Smoo*r)**2))*a + 2 * Lambda * a **3

        def g2(r, a, d):
            return 4*np.pi*a**2-(2/r)*d

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
                        plt.show()
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
                            plt.show()
                        draw += 1
                        break

                    else:
                        optimised = True



        #Calculate the (dimensionless) mass of the soliton:
        import scipy.integrate as si
        mass = si.simps(intlist,lr)*4*np.pi

        # IMPORTANT 
        Ratio = BHMass/mass

                
        if np.abs(Ratio - TargetRatio) <= Tol:
            print(f"Done at #{IterInt}!")
            
            return s, BHmass
            break
            
        if Ratio < TargetRatio:
            print('>', end = "")
            
            BHMassLo = BHMass
            BHMassHi = BHMassHi
            
            BHMass = (BHMassLo+BHMassHi)/2
            
        if Ratio > TargetRatio:
            print('<', end = "")

            BHMassLo = BHMassLo
            BHMassHi = BHMass
            
            BHMass = (BHMassLo+BHMassHi)/2
        
        BHmass = BHMass
        IterInt += 1
        
        if IterInt == Iter:
            
            print("Shooting algorithm failed to converge to given ratio. Type 'Y' to add ten more trials, 'B' to raise the upper bound, or '' to cancel.")
            
            Response = input()
            
            if Response == 'Y':
                Iter += 10
                
            elif Response == 'B':
                Iter += 10
                BHMassHi += 0.5
                
            elif Response == '':
                raise ValueError('Failed to converge to specified mass ratio.')
                return 0,0
        s = 0    

def SolitonProfile(BHMass,s,Smoo,Production):
      
    Save_Folder = './Soliton Profile Files/Custom/'
    
    #Note that the spatial resolution of the profile must match the specification of delta_x in the main code.
    if Production:
        dr = .00001
        max_radius = 15
        tolerance = 1e-9 # how close f(r) must be to 0 at r_max; can be changed
        
    else:
        dr = .001
        max_radius = 12 
        tolerance = 1e-6 # how close f(r) must be to 0 at r_max; can be changed
        
    print(f"Central Potential Mass = {BHMass:.5f} @(a = {Smoo}) and Resolution {dr}")
        
    rge = max_radius/dr
    
    BHmass = BHMass

    Lambda = 0 # axion self-interaction strength. Lambda < 0: attractive.
    
    nodes = [0] # Only consider ground state

    plot_while_calculating = False # If true, a figure will be updated for every attempted solution

    verbose_output = True # If true, s value of every attempt will be printed in console.

    if Smoo == 0:
        def g1(r, a, b, c, BHmass = BHmass):
            return -(2/r)*c+2*b*a - 2*(BHmass / r)*a + 2 * Lambda * a **3
    else:
        def g1(r, a, b, c, BHmass = BHmass):
            return -(2/r)*c+2*b*a - 2*(Smoo*BHmass / np.sqrt(1+(Smoo*r)**2))*a + 2 * Lambda * a **3

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
                        print("◌", end ="")
                    if plot_while_calculating == True:
                        plt.plot(la)
                        plt.pause(0.05)
                        plt.show()
                    draw += 1
                    break

                elif la[i] > 1.0:
                    currentflag = 1.1
                    phi_max = s
                    s = (phi_min + phi_max)/2
                    if verbose_output == True:
                        print("⚬", end ="")
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
                        print("⬤", end ="")
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
                            print("◯", end ="")
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
                            print("☺", end ="")
                        if plot_while_calculating == True:
                            plt.plot(la)
                            plt.pause(0.05)
                            plt.show()
                        draw += 1
                        break

                    else:
                        optimised = True
                        print('{}{}'.format('\n Successfully optimised for s = ', s))
                        timetaken = time.time() - tstart
                        print('{}{}'.format('\n Time taken = ', timetaken))
                        grad = (lb[i] - lb[i - 1]) / dr
                        const = lr[i] ** 2 * grad
                        beta = lb[i] + const / lr[i]

        #Calculate full width at half maximum density:

        difflist = []
        for i in range(int(rge)):
            difflist.append(abs(la[i]**2 - 0.5))

        fwhm = 2*lr[difflist.index(min(difflist))]

        #Calculate the (dimensionless) mass of the soliton:
        import scipy.integrate as si
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
        print ('{}{}'.format('Pre-Alpha (Mass) is ', mass))
        print ('{}{}'.format('Radius at 90% mass is ', r90))
        print ('{}{}'.format('MBH/MSoliton is ', BHmass/mass))

        #Save the numpy array and plots of the potential and wavefunction profiles.
        psi_array = np.array(la)
        
        Ratio = BHmass / mass 
        
        Save_Folder = './Soliton Profile Files/Custom/'
        
        if Production:
            RATIOName = f"f_{Ratio:.4f}_Pro"
        else:
            RATIOName = f"f_{Ratio:.4f}_Dra"
        
        print(RATIOName)
        
        Profile_Config = {}
        
        Profile_Config["Version"] = ({"Short": Version, 
                                      "Long": D_version})

        Profile_Config["Actual Ratio"] = Ratio
        Profile_Config["Resolution"] = dr
        Profile_Config["Alpha"] = mass
        Profile_Config["Beta"] = beta
        Profile_Config["Field Smoothing Factor"] = Smoo
        Profile_Config["Radial Cutoff"] = max_radius
        
        np.save(Save_Folder + RATIOName + '.npy', psi_array)
        
        with open(Save_Folder + RATIOName + '_info.uldm', "w+") as outfile:
            json.dump(Profile_Config, outfile,indent=4)

    print ('Successfully Initiated Soliton Profile.')
    
def LoadSolitonConfig(Ratio): 
    
    Ratio = float(Ratio)
    
    RatioN = f"{Ratio:.4f}"
    
    FileName = './Soliton Profile Files/Custom/f_'+RatioN+'_Pro_info.uldm'
    
    if os.path.isfile(FileName):
        with open(configfile) as json_file:
            config = json.load(json_file)
            
    else:
        FileName = './Soliton Profile Files/Custom/f_'+RatioN+'_Dra_info.uldm'

    try:
        config = json.load(open(FileName))
        
        delta_x = config["Resolution"]
        alpha   = config["Alpha"]
        beta   = config["Beta"]
        CutOff = config["Radial Cutoff"]
        
        return delta_x, alpha, beta, CutOff
    
    except FileNotFoundError:
        raise RuntimeError("This Ratio has not been generated!")
        
        
def LoadSoliton(Ratio):  
    Ratio = float(Ratio)
    RatioN = f"{Ratio:.4f}"
    
    FileName = './Soliton Profile Files/Custom/f_'+RatioN+'_Pro.npy'
    
    if os.path.isfile(FileName):
        return np.load(FileName)
            
    else:
        FileName = './Soliton Profile Files/Custom/f_'+RatioN+'_Dra.npy'
        return np.load(FileName)
    
##########################################################################################
# CREATE THE Just-In-Time Functions (work in progress)

initsoliton_jit = numba.jit(initsoliton)

IP_jit = numba.jit(isolatedPotentialSP)

PC_jit = numba.jit(planeConvolveSP)
 
Lp_jit = LpEval
    
######################### New Version With Built-in I/O Management
######################### Central Function


# New IO Functions

def ULDump(loc,psi,TMState,Status):
    np.save(f'{loc}/{Status}_psi.npy',psi)
    np.save(f'{loc}/{Status}_TM.npy',TMState)
    return 1
    
def ULRead(InitPath):
    psi = np.load(f'{InitPath}_psi.npy')
    return psi

def QuadrupoleFirst(mass_grid, x,y,z,gridvec, r_sq):
    
    # dx = gridvec[1] - gridvec[0]
    nx = len(gridvec)

    # Compute the coordinates using broadcasting

    # Compute r^2 and x_i * x_j terms
    # r_sq = ne.evaluate("x**2 + y**2 + z**2") # This thing is static?!
    
    x_i_x_j = np.array([[x**2, x*y, x*z], [y*x, y**2, y*z], [z*x, z*y, z**2]],dtype=object)

    # Compute the quadrupole moment tensor elements
    quadrupole_moment = np.zeros((3, 3))
    
    for ii in range(3):
        for ij in range(ii+1):
            quadrupole_moment[ii, ij] = np.sum(mass_grid * (3 * x_i_x_j[ii, ij] - (ii == ij) * r_sq)) #* dx ** 3

    return quadrupole_moment


def QuadrupoleSecond(mass_grid, gridvec, COMLoc=[0, 0, 0]):
    nx = len(gridvec)

    # Compute the coordinates using broadcasting
    xG = gridvec - COMLoc[0]
    yG = gridvec - COMLoc[1]
    zG = gridvec - COMLoc[2]

    xA, yA, zA = np.meshgrid(xG, yG, zG, indexing='ij', sparse=True)

    r_sq = ne.evaluate("xA**2 + yA**2 + zA**2")

    x_i_x_j = np.array(
        [[xA**2, xA * yA, xA * zA], [yA * xA, yA**2, yA * zA], [zA * xA, zA * yA, zA**2]],
        dtype=object)

    # Compute the quadrupole moment tensor elements
    quadrupole_moment = np.zeros((3, 4))  # Dipole is attached as a column vector [:,3]

    for ii in range(3):
        for ij in range(ii + 1):
            quadrupole_moment[ii, ij] = np.sum(mass_grid *
                                               (3 * x_i_x_j[ii, ij] -
                                                (ii == ij) * r_sq))  #* dx ** 3

    # Compute the dipole moment vector elements
    quadrupole_moment[0, 3] = np.sum(mass_grid * xA)
    quadrupole_moment[1, 3] = np.sum(mass_grid * yA)
    quadrupole_moment[2, 3] = np.sum(mass_grid * zA)

    return quadrupole_moment


def evolve(save_path,run_folder, EdgeClear = False, DumpInit = False, DumpFinal = False, UseInit = False, IsoP = False, UseDispSponge = False, SelfGravity = True, NBodyInterp = True, NBodyGravity = True, Shift = False, Simpson = False, Silent = False, AutoStop = False, AutoStop2 = False,AutoStop3 = False, KEThreshold = 0.9, WellThreshold = 100, InitPath = '', InitWeight = 1, Message = '', Stream = False, StreamChar = [0], GenerateLog = True, CenterCalc = False, NLM = False, Length_Ratio = 0.5, resolR = 64, PrintEK = True, DR = 1, MassChange = False, MassFunc = "", CorrectDrift = False, CorrectFreq = 6, ComputeQuad = False, ExtPhi = 0):
    
    if run_folder == "":
        printU('Nothing done!','Evolve')
        return

    if CorrectDrift:
        CenterCalc = True
        printU(f"Performing kicks to stop ULDM COM drift every {CorrectFreq} steps. This does NOT yet apply to N body.",'SPCD')
        

    Draft = True

    Method = 3 # Backward Compatibility
    
    loc = save_path + '/' + run_folder
        
    if UseInit and (InitPath == ''):
        raise RuntimeError("UseInit set to true. Must supply initial wavefunction!")
        
    try:
        os.mkdir(str(loc + '/Outputs'))
        
    except(FileExistsError):
        
        if Silent:
            Protect = 'Y'
        else:
            printU(f"{Version}: Folder Contains Outputs. Remove current files and Proceed [Y/n]?", 'IO')

            Protect = str(input())
        
        if Protect == 'n':
            return
        
        elif Protect == 'Y':
            import shutil
            
            print('Pre-existing Output files removed.')
            
            shutil.rmtree(str(loc + '/Outputs'))
            os.mkdir(str(loc + '/Outputs'))
            
        else:
            return
            
                   
    timestamp = run_folder

    file = open('{}{}'.format(save_path, '/latest.uldm'), "w+")
    file.write(run_folder)
    file.close()
    
    
    PyULConfig = {}
    
    PyULConfig["Integrator Version"] = S_version

    PyULConfig["Axion Mass"] = axion_mass
    PyULConfig["m22"] = axion_mass * 10**22 /eV
    
    PyULConfig["Integration Modifiers"] = ({
    "Integration Method": Method,
    "Dispersive Sponge Condition": UseDispSponge,
    "Reflective Sponge Condition": EdgeClear,
    "Wrap Back Particles": False,
    "Shift Particles by Half Grid": Shift,
    "Use Zero Padded Potential": IsoP,
    })
        
    PyULConfig["Built-in I/O Features"] = ({
    "Use Precompiled Initial Conditions": {"Flag": UseInit, "Path": InitPath, "Blend": InitWeight},
    "Dump Final Wavefunction": DumpFinal,
    "Dump Initial Wavefunction": DumpInit,
    })
    
    PyULConfig["Core System Modifiers"] = ({
    "Schroedinger-Poisson Self Gravity": SelfGravity,
    "N body backreaction": NBodyInterp,
    "N body Mutual Gravitation": True,
    "N body Projected Gravitation": NBodyGravity,
    })
    
    PyULConfig["Stopping Conditions"] = ({
    "When body #0 Stops": AutoStop,
    "When body #0 Loses Significant Energy": AutoStop3,
    "Energy Loss Factor": KEThreshold,
    "When Field Exceeds Limit": AutoStop2,
    "Depth Factor" : WellThreshold,
    })
    
    LogLocation = f"{save_path}/{run_folder}/evolve_{GenFromTime()}.log"
    
    PyULConfig["Misc. Settings"] = ({"Custom Soliton Draft Quallity": Draft,
                                    "Calculate Centre of Mass": CenterCalc,
                                    "Soliton Init Stretch Factor":DR,
                                    "Correct COM Drift": CorrectDrift})
    
    
    
    with open(f'{save_path}/{run_folder}/reproducibility.uldm', "w+") as outfile:
        json.dump(PyULConfig, outfile,indent=4)

    NS, length, length_units, resol, duration, duration_units, step_factor, save_number, save_options, save_format, s_mass_unit, s_position_unit, s_velocity_unit, solitons,start_time, m_mass_unit, m_position_unit, m_velocity_unit, particles, embeds, Uniform,Density, density_unit,a, UVel = LoadConfig(loc)

    for SaveName in SaveOptionsCompile(save_options).split():
        os.mkdir(str(loc + '/Outputs/'+SaveName))        
    num_threads = multiprocessing.cpu_count()
    
    if resol < 128:
        num_threads = np.min([num_threads,4])
        
    printU(f"Using {num_threads} CPU Threads for FFT.",'FFT', ToFile= GenerateLog, FilePath= LogLocation)
    # External Credits Print
    PyULCredits(IsoP,UseDispSponge,embeds)
    # Embedded particles are Pre-compiled into the list.
        
    if not Uniform:
        Density = 0
        UVel = [0,0,0]
    
    if a>=1e8:
        printU(f"Smoothing has been turned off!",'NBody')
        a = 0
    
    printU(f"Loaded Parameters from {loc}",'IO', ToFile= GenerateLog, FilePath= LogLocation)
    printU(f"Data to save this run:\n{SaveOptionsCompile(save_options)}",'IO', ToFile= GenerateLog, FilePath= LogLocation)

    NumSol = len(solitons)
    NumTM = len(particles)
            
    if (Method == 3): # 1 = Real Space Interpolation (Orange), 2 = Fourier Sum (White)
        printU(f"Using Linear Interpolation for gravity.",'NBody', ToFile= GenerateLog, FilePath= LogLocation)
    
    printU(f"Simulation grid resolution is {resol}^3.",'FFT', ToFile= GenerateLog, FilePath= LogLocation)
    
    if a == 0:
        printU(f"Using 1/r Point Mass Potential.",'NBody', ToFile= GenerateLog, FilePath= LogLocation)
    

    if EdgeClear:
        print("WARNING: The Wavefunction on the boundary planes will be Auto-Zeroed at every iteration.")

    print('==========================Additional Settings=================================')

    if NBodyGravity:
        print(f"Particle gravity  ON.") 
    else:
        print(f"Particle gravity OFF.")     
        
    if SelfGravity:
        print(f"ULDM self-gravity  ON.")    
    else:
        print(f"ULDM self-gravity OFF.")
        
    if NBodyInterp:
        print(f"NBody response to ULDM  ON.")
    else:
        print(f"NBody response to ULDM OFF.")
    
    if Shift:
        print(f"NBody particle shifted down by half-grid in x,y,z directions.")
        
    
    print('==========================Stopping Conditions=================================')
    
    if AutoStop and Uniform and NumTM == 1:
        print("Integration will automatically halt when test mass stops.")
        
    if AutoStop2:
        print(f"Integration will automatically halt when lowest potential exceeds {WellThreshold}x N body initial.")
    
    TIntegrate = 0
    
    TimeWritten = False
    
    masslist = []
    
    TMState = []

    ##########################################################################################
    #CONVERT INITIAL CONDITIONS TO CODE UNITS

    lengthC = convert(length, length_units, 'l')
    
    t = convert(duration, duration_units, 't')

    t0 = convert(start_time, duration_units, 't')

    Density = convert(Density,density_unit,'d')
    
    Vcell = (lengthC / float(resol)) ** 3
    
    ne.set_num_threads(num_threads)

    ##########################################################################################
    # Backwards Compatibility
    
    NCV = np.array([[0,0,0]])
    NCW = np.array([1])

    save_path = os.path.expanduser(save_path)

    ##########################################################################################
    # SET UP THE REAL SPACE COORDINATES OF THE GRID - FW Revisit

    gridvec = np.linspace(-lengthC / 2.0, lengthC / 2.0, resol, endpoint = False) # careful!
    
    xarray, yarray, zarray = np.meshgrid(
        gridvec, gridvec, gridvec,
        sparse=True, indexing='ij')
    
    WN = 2*np.pi*np.fft.fftfreq(resol, lengthC/(resol)) # 2pi Pre-multiplied
    
    Kx,Ky,Kz = np.meshgrid(WN,WN,WN,sparse=True, indexing='ij',)
     ##########################################################################################
    # SET UP K-SPACE COORDINATES FOR COMPLEX DFT

    kvec = 2 * np.pi * np.fft.fftfreq(resol, lengthC / float(resol))
    
    kxarray, kyarray, kzarray = np.meshgrid(
        kvec, kvec, kvec,
        sparse=True, indexing='ij',
    )
    
    karray2 = ne.evaluate("kxarray**2+kyarray**2+kzarray**2")
    ##########################################################################################
    delta_x = 0.00001 # Needs to match resolution of soliton profile array file. Default = 0.00001

    warn = 0 
    funct = pyfftw.zeros_aligned((resol, resol, resol), dtype='complex128')   
    
    if UseInit:
        psiEx = ULRead(InitPath)
            
        if len(psiEx) != resol:
            raise ValueError('Loaded grid is not currently compatible with run settings!')
        print("======================================================")
        printU(f"Loaded initial wavefunction from {InitPath}",'IO', ToFile= GenerateLog, FilePath= LogLocation)
        
        MassCom = Density*lengthC**3

        UVelocity = convert(np.array(UVel),s_velocity_unit, 'v')

        DensityCom = MassCom / resol**3

        print('========================Dispersive Background====================================')
        printU(f"Loaded a pre-generated wavefunction from {InitPath}_psi.npy.",'Init', ToFile= GenerateLog, FilePath= LogLocation)
        
    # INITIALISE SOLITONS WITH SPECIFIED MASS, POSITION, VELOCITY, PHASE

    psi = pyfftw.zeros_aligned((resol, resol, resol), dtype='complex128')

    MassCom = Density*lengthC**3

    UVelocity = convert(np.array(UVel),s_velocity_unit, 'v')
    
    VTot = np.linalg.norm(UVelocity)
    
    if Stream:
        CreateStream(loc, NS, VTot,StreamChar)
        printU("Created stream file at root folder for variables {StreamChar}.",'NBody', ToFile= GenerateLog, FilePath= LogLocation)

    if AutoStop and NumTM == 1:
        ThresholdVelocity = -1*UVelocity[1]
    else:
        ThresholdVelocity = 0

    DensityCom = MassCom / resol**3
    
    if Uniform:
        print('========================Uniform Background====================================')
        printU(f"Added a uniform wavefunction with no phase.",'Init', ToFile= GenerateLog, FilePath= LogLocation)
        printU(f"Background ULDM mass in domain is {MassCom:.4f}, at {Density:.4f} per grid.",'Init', ToFile= GenerateLog, FilePath= LogLocation)
        printU(f"Background Global velocity is (x,y,z): {UVel[1]},{UVel[0]},{UVel[2]}.",'Init', ToFile= GenerateLog, FilePath= LogLocation)
        print('==============================================================================')
    psi = ne.evaluate("0*psi + sqrt(Density)")

    velx = UVelocity[0]
    vely = UVelocity[1]
    velz = UVelocity[2]
    psi = ne.evaluate("exp(1j*(velx*xarray + vely*yarray + velz*zarray))*psi")
    #psi = ne.evaluate("psi + funct")

    for EI,emb in enumerate(embeds):

        # 0.     1.     2.          3.                   4.
        # [mass,[x,y,z],[vx,vy,vz], BH-Total Mass Ratio, Phase]
        Ratio = emb[3]
        RatioBU = float(Ratio/(1-Ratio))
        try:
            delta_xL, prealphaL, betaL,CutOff = LoadSolitonConfig(RatioBU)
        except RuntimeError:
            print('==============================================================================')
            printU(f'Note that this process will not be required for repeated runs. To remove custom profiles, go to the folder /Soliton Profile Files/Custom.', 'Profiler', ToFile= GenerateLog, FilePath= LogLocation)
            printU(f'Generating profile for Soliton with MBH/MSoliton = {RatioBU:.4f}, Part 1', ToFile= GenerateLog, FilePath= LogLocation)
            GMin,GMax = BHGuess(RatioBU)
            s, BHMass = BHRatioTester(RatioBU,30,1e-6,GMin,GMax,a)
            printU(f'Generating profile for Soliton with MBH/MSoliton = {RatioBU:.4f}, Part 2', ToFile= GenerateLog, FilePath= LogLocation)
            SolitonProfile(BHMass,s,a,not Draft)
        print('==============================================================================')

        delta_xL, prealphaL, betaL,CutOff = LoadSolitonConfig(RatioBU)

        # L stands for Local, as in it's only used once.
        fL = LoadSoliton(RatioBU)

        printU(f"Loaded embedded soliton {EI} with BH-Soliton mass ratio {RatioBU:.4f}.", 'Init', ToFile= GenerateLog, FilePath= LogLocation)

        mass = convert(emb[0], s_mass_unit, 'm')*(1-Ratio)
        position = convert(np.array(emb[1]), s_position_unit, 'l')
        velocity = convert(np.array(emb[2]), s_velocity_unit, 'v')

        # Note that alpha and beta parameters are computed when the initial_f.npy soliton profile file is generated.

        alphaL = (mass / prealphaL) ** 2

        phase = emb[4]

        funct = initsoliton_jit(funct, xarray, yarray, zarray, position, alphaL, fL, delta_xL)

        if(np.isnan(funct).any()):
            print('Something is seriously wrong!')
            raise RuntimeError('Duh')
        ####### Impart velocity to solitons in Galilean invariant way
        velx = velocity[0]
        vely = velocity[1]
        velz = velocity[2]
        funct = ne.evaluate("exp(1j*(alphaL*betaL*t0 + velx*xarray + vely*yarray + velz*zarray -0.5*(velx*velx+vely*vely+velz*velz)*t0  + phase))*funct")
        psi = ne.evaluate("psi + funct")
        EI += 1 # For displaying.

    if solitons != []:
        printU(f"Loaded standard soliton radial profile.",'Init', ToFile= GenerateLog, FilePath= LogLocation)
        f = LoadDefaultSoliton()

    for s in solitons:
        mass = convert(s[0], s_mass_unit, 'm')
        position = convert(np.array(s[1]), s_position_unit, 'l')
        velocity = convert(np.array(s[2]), s_velocity_unit, 'v')
        # Note that alpha and beta parameters are computed when the initial_f.npy soliton profile file is generated.
        alpha = (mass / 3.8827652755822006) ** 2 #3.883
        #alpha = (mass / 3.883) ** 2 #3.883
        beta = 2.4538872760773143 #2.454
        #beta = 2.454 #2.454
        phase = s[3]
        
        funct = InitSolitonF(gridvec, position, resol, alpha, DR = DR)
        # funct = initsoliton_jit(funct, xarray, yarray, zarray, position, alpha, f, delta_x, DR = DR)
        
        printU(f"Using scale {DR}","Scaler")
        ####### Impart velocity to solitons in Galilean invariant way
        velx = velocity[0]
        vely = velocity[1]
        velz = velocity[2]
        funct = ne.evaluate("exp(1j*(alpha*beta*t0 + velx*xarray + vely*yarray + velz*zarray -0.5*(velx*velx+vely*vely+velz*velz)*t0  + phase))*funct")
        psi = ne.evaluate("psi + funct")

    if UseInit:

        if np.abs(InitWeight - 0.5) <= 0.5:

            psi = psi*(1-InitWeight) + psiEx*InitWeight

        elif InitWeight == -1:
            PhaseEx = np.angle(psiEx)
            psi = ne.evaluate("psi*exp(1j*PhaseEx)")

    fft_psi = pyfftw.builders.fftn(psi, axes=(0, 1, 2), threads=num_threads)
    
    funct = fft_psi(psi)
    
    ifft_funct = pyfftw.builders.ifftn(funct, axes=(0, 1, 2), threads=num_threads)       
    
    rho = ne.evaluate("abs(abs(psi)**2)")
        
    rho = rho.real
    
    if CenterCalc or save_options[25]:
        LocCOM = Find3BoxCOM(rho,xarray, yarray, zarray)
        
        if save_options[25]:
            IOSave(loc,'ULDCOM',0,save_format,data = LocCOM)
        
        if CenterCalc:
            printU(LocCOM, "COM", ToScreen = False, ToFile= GenerateLog, FilePath= LogLocation) 
            Resample3Box(psi, LocCOM, gridvec, loc, 0, save_format, Length_Ratio, resolR, Save_Rho = save_options[20], Save_Psi = save_options[21], Save_Rho2 = save_options[26])
            
        if CorrectDrift:
            
            COM0 = LocCOM


    ##########################################################################################
    # COMPUTE SIZE OF TIMESTEP (CAN BE INCREASED WITH step_factor)

    delta_t = (lengthC/float(resol))**2/np.pi

    min_num_steps = t / delta_t
    min_num_steps_int = int(min_num_steps + 1)
    min_num_steps_int = int(min_num_steps_int/step_factor)

    if save_number >= min_num_steps_int:
        actual_num_steps = save_number
        its_per_save = 1
    else:
        rem = min_num_steps_int % save_number
        actual_num_steps = min_num_steps_int + save_number - rem
        its_per_save = actual_num_steps / save_number

    if save_number == -1:
        save_number = actual_num_steps
        its_per_save = 1
        
    h = t / float(actual_num_steps)
    
    its_per_momentum = its_per_save
    
    ##########################################################################################
    # First ULDM Momentum and Angular Momentum Saves 
    
    if save_options[28]: #pEval
        
        pXAr, pYAr, pZAr = pEval(psi,rho,funct,resol,gridvec,Kx,Ky,Kz,ifft_funct)
        momentum_I = 0
        printU('Saving ULDM momentum array.','2Momentum', ToFile= GenerateLog, FilePath= LogLocation)
        IOSave(loc,'2Momentum',momentum_I,save_format,data = np.array([pXAr[:,:,resol//2], 
                                                                       pYAr[:,:,resol//2], 
                                                                       pZAr[:,:,resol//2]]))
        
    
    if save_options[18] or save_options[19]:

        momentum_I = 0
    
        pOut,LOut = Lp_jit(psi,rho,funct,resol,gridvec,Kx,Ky,Kz,ifft_funct)
        
        if save_options[18]:
            printU('Saving ULDM momentum.','Momentum', ToFile= GenerateLog, FilePath= LogLocation)
            IOSave(loc,'Momentum',momentum_I,save_format,data = pOut)

        if save_options[19]:
            printU('Saving ULDM angular momentum with respect to origin.','Momentum', ToFile= GenerateLog, FilePath= LogLocation)
            IOSave(loc,'AngMomentum',momentum_I,save_format,data = LOut)
        
    
    
    ##########################################################################################
    # SETUP PADDED POTENTIAL HERE (From JLZ)
    
    if IsoP:
        rhopad = pyfftw.zeros_aligned((2*resol, resol, resol), dtype='complex128')
        bigplane = pyfftw.zeros_aligned((2*resol, 2*resol), dtype='complex128')

        fft_X = pyfftw.builders.fftn(rhopad, axes=(0, ), threads=num_threads)
        ifft_X = pyfftw.builders.ifftn(rhopad, axes=(0, ), threads=num_threads)

        fft_plane = pyfftw.builders.fftn(bigplane, axes=(0, 1), threads=num_threads)
        ifft_plane = pyfftw.builders.ifftn(bigplane, axes=(0, 1), threads=num_threads)

    phiSP = pyfftw.zeros_aligned((resol, resol, resol), dtype='float64')
    phiTM = pyfftw.zeros_aligned((resol, resol, resol), dtype='float64') # New, separate treatment.
    
    # print(phiSP.shape) OK THIS FAR

    fft_phi = pyfftw.builders.fftn(phiSP, axes=(0, 1, 2), threads=num_threads)
    ##########################################################################################
    # SETUP K-SPACE FOR RHO (REAL)

    rkvec = 2 * np.pi * np.fft.fftfreq(resol, lengthC / float(resol))
    
    krealvec = 2 * np.pi * np.fft.rfftfreq(resol, lengthC / float(resol))
    
    rkxarray, rkyarray, rkzarray = np.meshgrid(
        rkvec, rkvec, krealvec,
        sparse=True, indexing='ij'
    )

    rkarray2 = ne.evaluate("rkxarray**2+rkyarray**2+rkzarray**2")

    rfft_rho = pyfftw.builders.rfftn(rho, axes=(0, 1, 2), threads=num_threads)
    
    phik = rfft_rho(rho.real)  # not actually phik but phik is defined in next line

    phik = ne.evaluate("-4*pi*phik/rkarray2")

    phik[0, 0, 0] = 0
    
    irfft_phi = pyfftw.builders.irfftn(phik, axes=(0, 1, 2), threads=num_threads)
    
    if EdgeClear:
        
        Cutoff = (resol//16)
        
        #x
        psi[ 0:Cutoff,:,:] = np.sqrt(Density) + 0j
        psi[-Cutoff:,:,:] = np.sqrt(Density) + 0j

        #y
        psi[:, 0:Cutoff,:] = np.sqrt(Density) + 0j
        psi[:,-Cutoff:,:] = np.sqrt(Density) + 0j

        #z
        psi[:,:, 0:Cutoff] = np.sqrt(Density) + 0j
        psi[:,:,-Cutoff:] = np.sqrt(Density) + 0j
             

    ##########################################################################################
    # COMPUTE INTIAL VALUE OF POTENTIAL

    if IsoP:
        
        ndx = resol - np.abs(np.arange(-resol, resol))
                             
        try:
            green = np.load(f'./Green Functions/G{resol}.npy')
            printU(f"Using pre-computed Green function for simulation region.",'SP', ToFile= GenerateLog, FilePath= LogLocation)
        except FileNotFoundError:
            if not os.path.exists('./Green Functions/'):
                os.mkdir('./Green Functions/')
            green = makeDCTGreen(resol) #make Green's function ONCE
            printU(f"Generating Green function for simulation region.",'SP', ToFile= GenerateLog, FilePath= LogLocation)
            np.save(f'./Green Functions/G{resol}.npy',green)
            
        #green = makeEvenArray(green)
        phiSP = IP_jit(rho, green, lengthC, fft_X, ifft_X, fft_plane, ifft_plane, resol, ndx)
        
    else:
        printU(f"Poisson Equation Solved Using FFT.",'SP', ToFile= GenerateLog, FilePath= LogLocation)
        phiSP = irfft_phi(phik)
        
    ##########################################################################################
       
    phiSP += ExtPhi
    # NBody
    EGPCM = 0
    for MI, particle in enumerate(particles):
               
        if MassChange:
            mT = particle[0] * MassFunc(0)
        else:
            mT = convert(particle[0], m_mass_unit, 'm')


        masslist.append(mT)

        position = convert(np.array(particle[1]), m_position_unit, 'l')
        
        if mT == 0:
            printU(f"Particle #{MI} loaded as observer.",'NBody', ToFile= GenerateLog, FilePath= LogLocation)
        else:
            
            DensityInit = QuickInterpolate(rho,lengthC,resol,position)
            printU(f"Particle #{MI} mass {mT:.5f} and local density {DensityInit:.5f} (code units).",'NBody', ToFile= GenerateLog, FilePath= LogLocation)
            NBDensity(loc,DensityInit)

        
        if Shift:
            position = GridShift(position, lengthC, resol)
        
        velocity = convert(np.array(particle[2]), m_velocity_unit, 'v')
        
        IND = int(6*MI)
        
        TMx = position[0]
        TMy = position[1]
        TMz = position[2]
        
        Vx = velocity[0]
        Vy = velocity[1]
        Vz = velocity[2]
        
        TMState.append([TMx,TMy,TMz,Vx,Vy,Vz])
        
        if mT != 0:
                       
            distarrayTM = ne.evaluate("((xarray-TMx)**2+(yarray-TMy)**2+(zarray-TMz)**2)**0.5") # Radial coordinates
            if a == 0:
                phiTM = ne.evaluate("phiTM-mT/(distarrayTM)")
            else:
                phiTM = ne.evaluate("phiTM-a*mT/sqrt(1+a**2*distarrayTM**2)")

        
        if (save_options[3]):
            EGPCM += mT*QuickInterpolate(phiSP,lengthC,resol,np.array([TMx,TMy,TMz]))
        MI = int(MI + 1)
        
        if AutoStop and Uniform and len(particles) == 1:
            ThresholdVelocity += Vy
            
        if AutoStop and len(particles) == 1:
            E0 = 1/2 * mT * np.linalg.norm(velocity + UVelocity)**2  
            if E0 == 0:
                E0 = 1
           
    masslist = np.array(masslist)
    TMState = np.array(TMState)
    TMState = TMState.flatten(order='C')
    
    if NBodyGravity:
        if AutoStop2:
            if a == 0:
                phiRef = np.min(phiTM) * WellThreshold # Empirical for now
            else:
                phiRef = - a * np.max(masslist) * WellThreshold
    else:    
        phiTM *= 0
    
    if SelfGravity:
        phi = phiSP + phiTM
    else: 
        phi = phiTM
    
    if NumTM == 1:
        Vinitial = TMState[3:6]
    
    #TMStateDisp = TMState.reshape((-1,6))
    #
    #printU(f"The test mass initial state (vectorised) is:", 'NBody')
    #print(TMStateDisp)
        
    MI = 0
    
    GridMass = [Vcell*np.sum(rho)] # Mass of ULDM in Grid
    
    if (save_options[3]):
        egylist = []
        egpcmlist = []
        egpsilist = []
        ekandqlist = []
        mtotlist = []
        egpcmMlist = [EGPCM]
        
        

        ETotP, EKQP, ESIP = calculate_energies(rho, Vcell, phiSP,phiTM, psi, karray2, fft_psi, ifft_funct, Density,Uniform, egpcmlist, egpsilist, ekandqlist, egylist, mtotlist, resol, (save_options[22] or save_options[23] or save_options[24]))
    
        if save_options[22]:
            IOSave(loc,'2EnergyTot',0,save_format,ETotP)
            
        if save_options[23]:   
            IOSave(loc,'2EnergyKQ',0,save_format,EKQP)
        
        if save_options[24]:
            IOSave(loc,'2EnergySI',0,save_format,ESIP)


    GradientLog = np.zeros(NumTM*3)

    if save_options[10]:
        
        EntropyLog = [-1*np.sum(ne.evaluate('rho*log(rho)'))]
        np.save(os.path.join(os.path.expanduser(loc), "Outputs/Entro.npy"), EntropyLog)

    #######################################
    if np.isnan(rho).any() or np.isnan(psi).any():
        raise RuntimeError("Something is seriously wrong.")
    
    if DumpInit:
        printU(f'Successfully initiated Wavefunction and NBody Initial Conditions. Dumping to file.','IO', ToFile= GenerateLog, FilePath= LogLocation)
    
        ULDump(loc,psi,TMState,'Init')
        
        return
        
    
    else:
        printU(f'Successfully initiated Wavefunction and NBody Initial Conditions.', 'Init', ToFile= GenerateLog, FilePath= LogLocation)

    ##########################################################################################
    # PRE-LOOP SAVE I.E. INITIAL CONFIG
    save_grid(
        rho, psi, resol, 
        TMState, phiSP, phi, GradientLog,
        save_options,
        save_format,
        loc, -1, 1
        )
    
    if ComputeQuad:
        #r_sq = ne.evaluate("xarray**2+yarray**2+zarray**2")
        LocCOM = Find3BoxCOM(rho,xarray, yarray, zarray)
        Qij = QuadrupoleSecond(rho, gridvec, LocCOM)
        IOSave(loc,'Quadrupole',0,save_format,data = Qij)
        
    tBegin = time.time()
    
    tBeginDisp = datetime.fromtimestamp(tBegin).strftime("%d/%m/%Y, %H:%M:%S")
    

    ########################################################################################## 
    # From Yale.
  
    if UseDispSponge:
        
        SpongeRatio = 6/8
        # This works in grid units in Chapel. We make it work in Code Units.
        rn = 1/2*lengthC
        rp = SpongeRatio*rn
        rs = (rn+rp)/2
        invdelta = 1/(rn-rp)
        c0 = 2 - np.tanh(rs*invdelta)
        V0 = 0.6
        distarray = ne.evaluate("((xarray)**2+(yarray)**2+(zarray)**2)**0.5") # Radial coordinates for system
        #Vpot without the potential
        PreMult = 0.5*V0*(c0+np.tanh((distarray-rs)*invdelta))
        
        #High Performance Mask
        # PreMult[distarray<=rp] = 0
        # Commented in 17 Oct 2023
        
        printU(f'Dispersive Sponge Condition Pre Multiplier Ready.','SP', ToFile= GenerateLog, FilePath= LogLocation)
    ##########################################################################################
    # LOOP NOW BEGINS
    if Silent:
        clear_output()
        print(f"{D_version}\nMessage: {Message}")
        
    printU(f"Simulation name is {loc}",'Runtime', ToFile= GenerateLog, FilePath= LogLocation)
    printU(f"{resol} Resolution for {duration:.4g}{duration_units}",'Runtime', ToFile= GenerateLog, FilePath= LogLocation)
    printU(f"Simulation Started at {tBeginDisp}.",'Runtime', ToFile= GenerateLog, FilePath= LogLocation)
            
    HaSt = 1  # 1 for a half step 0 for a full step

    tenth = float(save_number/10) #This parameter is used if energy outputs are saved while code is running.
    if actual_num_steps == save_number:
        printU(f"Taking {int(actual_num_steps)} ULDM steps", 'Runtime', ToFile= GenerateLog, FilePath= LogLocation)
    else:
        printU(f"Taking {int(actual_num_steps)} ULDM steps @ {save_number} snapshots", 'Runtime', ToFile= GenerateLog, FilePath= LogLocation)
    

    tinit = time.time()
    tint = 0
    
    EGPCM = 0
    PBEDisp = ''
    #####################################################################################LOOP
    for ix in range(actual_num_steps):
                
        TIntegrate += h
        prog_bar(actual_num_steps, ix + 1, tint,'FT',PBEDisp)
        if HaSt == 1:
            psi = ne.evaluate("exp(-1j*0.5*h*phi)*psi")
            HaSt = 0

        else:
            psi = ne.evaluate("exp(-1j*h*phi)*psi")
        
        
        funct = fft_psi(psi)
            

        ###### New Momentum Evaluator
        if (ix+1) % its_per_momentum == 0:
            
            if save_options[28]: #pEval
                
                momentum_I += 1
        
                pXAr, pYAr, pZAr = pEval(psi,rho,funct,resol,gridvec,Kx,Ky,Kz,ifft_funct)

                IOSave(loc,'2Momentum',momentum_I,save_format,data = np.array([pXAr[:,:,resol//2], 
                                                                               pYAr[:,:,resol//2], 
                                                                               pZAr[:,:,resol//2]]))
                
                save_options[18] = 0
                save_options[19] = 0
                
        
            if save_options[18] or save_options[19]:
                prog_bar(actual_num_steps, ix + 1, tint,'pL',PBEDisp)

                momentum_I += 1
                
                pOut,LOut = Lp_jit(psi,rho,funct,resol,gridvec,Kx,Ky,Kz,ifft_funct)
        
                if save_options[18]:
                    IOSave(loc,'Momentum',momentum_I,save_format = 'npy',data = pOut)

                if save_options[19]:
                    IOSave(loc,'AngMomentum',momentum_I,save_format = 'npy',data = LOut)

        funct = ne.evaluate("funct*exp(-1j*0.5*h*karray2)")
        psi = ifft_funct(funct)

        if EdgeClear:

            #x
            psi[ 0:Cutoff,:,:] = np.sqrt(Density) + 0j
            psi[-Cutoff:,:,:] = np.sqrt(Density) + 0j

            #y
            psi[:, 0:Cutoff,:] = np.sqrt(Density) + 0j
            psi[:,-Cutoff:,:] = np.sqrt(Density) + 0j

            #z
            psi[:,:, 0:Cutoff] = np.sqrt(Density) + 0j
            psi[:,:,-Cutoff:] = np.sqrt(Density) + 0j

        if UseDispSponge:
            prog_bar(actual_num_steps, ix + 1, tint,'DS',PBEDisp)
            psi *= np.exp(-PreMult*h)
        

        rho = ne.evaluate("abs(abs(psi)**2)").real
        
        if CenterCalc:
            LocCOM = Find3BoxCOM(rho,xarray, yarray, zarray)
            printU(LocCOM, "COM", ToScreen = False, ToFile= GenerateLog, FilePath= LogLocation)
        
        ###### Kick Back COM Here
        
        CorrectFlag = False
        
        if CorrectDrift:

            if ix % CorrectFreq == 2:
            
                COM2 = LocCOM
                CorrectFlag = True

            if ix % CorrectFreq == 0:
                COM0 = LocCOM

            if CorrectFlag:
                
                prog_bar(actual_num_steps, ix + 1, tint,'CD',PBEDisp)
                
                UVelO = (COM0 - COM2) / 2 / h

                velx =  UVelO[0]
                vely =  UVelO[1]
                velz =  UVelO[2]
                psi = ne.evaluate("exp(1j*(velx*xarray + vely*yarray + velz*zarray))*psi")

        phik = rfft_rho(rho)  # not actually phik but phik is defined in next line
            
        phik = ne.evaluate("-4*pi*phik/rkarray2")

        phik[0, 0, 0] = 0        

        prog_bar(actual_num_steps, ix + 1, tint,'SP',PBEDisp)
        # New Green Function Methods
        if not IsoP:
            phiSP = irfft_phi(phik)
        else:
            phiSP = IP_jit(rho, green, lengthC, fft_X, ifft_X, fft_plane, ifft_plane, resol, ndx)
            
        phiSP += ExtPhi # New Handle

        # FW N Body Toy
        prog_bar(actual_num_steps, ix + 1, tint,'RK4',PBEDisp)
        
        if NBodyInterp:

            TMState, GradientLog = NBodyAdvance(TMState,h,masslist,phiSP,a,lengthC,resol,NS, loc, Stream, StreamChar)
            
        else:

            TMState, GradientLog = NBodyAdvance_NI(TMState,h,masslist,phiSP,a,lengthC,resol,NS)
 
        prog_bar(actual_num_steps, ix + 1, tint,'Phi ')
        phiTM = pyfftw.zeros_aligned((resol, resol, resol), dtype='float64') # Reset!   
        
        if NBodyGravity:
            for MI in range(NumTM):

                State = TMState[int(MI*6):int(MI*6+5)]

                TMx = State[0]
                TMy = State[1]
                TMz = State[2]

                if MassChange:
                    masslist[MI] = particle[0] * MassFunc(TIntegrate)

                mT = masslist[MI]

                if mT != 0:

                    distarrayTM = ne.evaluate("((xarray-TMx)**2+(yarray-TMy)**2+(zarray-TMz)**2)**0.5") # Radial coordinates
                    if a == 0:
                        phiTM = ne.evaluate("phiTM-mT/(distarrayTM)")
                    else:
                        phiTM = ne.evaluate("phiTM-a*mT/sqrt(1+a**2*distarrayTM**2)")

                    if (save_options[3]) and ((ix + 1) % its_per_save) == 0:
                        EGPCM += mT*QuickInterpolate(phiSP,lengthC,resol,np.array([TMx,TMy,TMz]))
            
        if AutoStop and len(particles) == 1:
            velocity = TMState[3:6]
            Vdisp = np.linalg.norm(velocity)
            PBEDisp = f'[V={Vdisp:.2f} / Tg.V={VTot:.2f}]'
            
        if SelfGravity:
            phi = ne.evaluate('phiSP + phiTM')
        else: 
            phi = phiTM

        prog_bar(actual_num_steps, ix + 1, tint,'FT',PBEDisp)
        #Next if statement ensures that an extra half step is performed at each save point
        if (((ix + 1) % its_per_save) == 0) and HaSt == 0:
            psi = ne.evaluate("exp(-1j*0.5*h*phi)*psi")

            rho = ne.evaluate("abs(abs(psi)**2)")
            HaSt = 1
            rho = rho.real
            prog_bar(actual_num_steps, ix + 1, tint,'IO',PBEDisp,OutNumber = 1 + int((ix + 1) / its_per_save))
            #Next block calculates the energies at each save, not at each timestep.
            if (save_options[3]):
    
                ETotP, EKQP, ESIP = calculate_energies(rho, Vcell, phiSP,phiTM, psi, karray2, fft_psi, ifft_funct, Density,Uniform, egpcmlist, egpsilist, ekandqlist, egylist, mtotlist, resol, (save_options[22] or save_options[23] or save_options[24]))

                if save_options[22]:
                    IOSave(loc,'2EnergyTot',int((ix + 1) / its_per_save),save_format,ETotP)

                if save_options[23]:   
                    IOSave(loc,'2EnergyKQ',int((ix + 1) / its_per_save),save_format,EKQP)

                if save_options[24]:
                    IOSave(loc,'2EnergySI',int((ix + 1) / its_per_save),save_format,ESIP)
           
################################################################################
# SAVE DESIRED OUTPUTS
        if ((ix + 1) % its_per_save) == 0:
        
            egpcmMlist.append(EGPCM)
            EGPCM = 0
            
            if ComputeQuad: # This Happens At Origin, of Course
                # LocCOM = Find3BoxCOM(rho,xarray, yarray, zarray)
                Qij = QuadrupoleSecond(rho, gridvec, LocCOM)
                IOSave(loc, "Quadrupole", int((ix + 1) / its_per_save),save_format,data = Qij)

            if CenterCalc:
                Resample3Box(psi, LocCOM, gridvec, loc, int((ix + 1) / its_per_save), save_format, Length_Ratio, resolR, Save_Rho = save_options[20], Save_Psi = save_options[21],Save_Rho2 = save_options[26])
            
            save_grid(
                rho, psi, resol, 
                TMState, phiSP, phi, GradientLog,
                save_options,
                save_format,
                loc, ix, its_per_save
                )
            
            GridMass.append(Vcell*np.sum(rho))
            np.save(os.path.join(os.path.expanduser(loc), "Outputs/ULDMass.npy"), GridMass)

            if save_options[25]:

                if not CenterCalc:
                    LocCOM = Find3BoxCOM(rho,xarray, yarray, zarray) # Happen at different freqs.
                
                IOSave(loc,'ULDCOM',int((ix + 1) / its_per_save),save_format,data = LocCOM)
            
            if (save_options[3]):  
                np.save(os.path.join(os.path.expanduser(loc), "Outputs/egylist.npy"), egylist)
                np.save(os.path.join(os.path.expanduser(loc), "Outputs/egpcmlist.npy"), egpcmlist) # Original Method
                np.save(os.path.join(os.path.expanduser(loc), "Outputs/egpcmMlist.npy"), egpcmMlist) # New Method
                np.save(os.path.join(os.path.expanduser(loc), "Outputs/egpsilist.npy"), egpsilist)
                np.save(os.path.join(os.path.expanduser(loc), "Outputs/ekandqlist.npy"), ekandqlist)
                np.save(os.path.join(os.path.expanduser(loc), "Outputs/masseslist.npy"), mtotlist)
                
                if PrintEK:
                    PBEDisp = f"Delta Ek (code units): {ekandqlist[-1]-ekandqlist[-2]:.6g}"
         
            if save_options[10]:
                EntropyLog.append(-1*np.sum(ne.evaluate('rho*log(rho)')))
                np.save(os.path.join(os.path.expanduser(loc), "Outputs/Entro.npy"), EntropyLog)
            if TimeWritten:
                print(f'\nSimulation Concluded at step {ix}!')
                break
                
        if AutoStop and Uniform and len(particles)==1:
            VCur = TMState[4] - UVelocity[1]
            
            if VCur * ThresholdVelocity <= 0:
                
                if not TimeWritten:
                    print(f'\nTest mass has stopped at step {ix}!')
                
                    TIntDisp = convert_back(TIntegrate,duration_units,'t')
                
                    print(f'Integrated Time is {TIntDisp:.5g}{duration_units}')

                    file = open(f'{loc}/StoppingTime.uldm', "w+")
                    file.write(f"{TIntegrate}")
                    file.close()
                    TimeWritten = True
                    
        if AutoStop2:
            if np.min(phi) < phiRef:
                print('\n')
                printU('Gravitational field runaway threshold reached!','Consistency', ToFile= GenerateLog, FilePath= LogLocation)
                print(f'\nSimulation Concluded at step {ix}!')
                break
                

        ################################################################################
        # UPDATE INFORMATION FOR PROGRESS BAR

        tint = time.time() - tinit
        tinit = time.time()
        prog_bar(actual_num_steps, ix + 1, tint,'',PBEDisp)

    ################################################################################
    # LOOP ENDS

    tFinal = time.time()
    
    Time = tFinal - tBegin
    
    day = Time // (24 * 3600)
    Time = Time % (24 * 3600)
    hour = Time // 3600
    Time %= 3600
    minutes = Time // 60
    Time %= 60
    seconds = Time
    print('\n')
    printU(f"Run Complete. Time Elapsed (d:h:m:s): {day:.0f}:{hour:.0f}:{minutes:.0f}:{seconds:.2f}",'Runtime', ToFile= GenerateLog, FilePath= LogLocation)
    if DumpFinal:
        printU(f'Dumped final state to file.','IO', ToFile= GenerateLog, FilePath= LogLocation)
        ULDump(loc,psi,TMState,'Final')
        
    if AutoStop and not TimeWritten:
        file = open(f'{loc}/StoppingTime.uldm', "w+")
        file.write('-1')
        file.close()

################################################################################
################################################################################
################################################################################

def SSEst(SO, save_number, resol):
    
    save_options = SaveOptionsDigest(SO)
    
    save_rho = save_options[0]
    
    save_psi = save_options[1]
    
    save_phi = save_options[6]
    
    save_phi_plane = save_options[7]
    
    save_plane = save_options[2]
    
    save_gradients = save_options[8]
    
    save_testmass = save_options[5]
    
    save_phase_plane = save_options[9]
    
    
    PreMult = 0
    
    if save_rho:
        printU('Saving Mass Density Data (3D)')
        PreMult = PreMult + resol**3
        
    if save_psi:
        printU('Saving Complex Field Data (3D)')
        PreMult = PreMult + resol**3*2
        
    if save_phi:
        printU('Saving Gravitational Field Data (3D)')
        PreMult = PreMult + resol**3
        
    if save_plane:
        printU('Saving Mass Density Data (2D)')
        PreMult = PreMult + resol**2
        
    if save_phase_plane:
        printU('Saving ULD Argument Data (2D)')
        PreMult = PreMult + resol**2
        
    if save_phi_plane:
        printU('Saving Gravitational Field Data (2D)')
        PreMult = PreMult + resol**2
    
    if save_gradients:
        printU('Saving NBody Gradient Data')
    
    if save_testmass:
        printU('Saving NBody Position Data')
    
    return (save_number+1)*(PreMult)*8/(1024**3)
    
    

def DSManagement(save_path, Force = False):
    
    print('[',save_path,']',": The current size of the folder is", round(get_size(save_path)/1024**2,3), 'Mib')

    if get_size(save_path) == 0:
        cleardir = 'N'
    elif not Force:
        print('[',save_path,']',": Do You Wish to Delete All Files Currently Stored In This Folder? [Y] \n")
        cleardir = str(input())
    
    if cleardir == 'Y' or Force:
        import shutil 
        shutil.rmtree(save_path)
        print("Folder Cleaned! \n")
    
    try:
        os.mkdir(save_path)
        print('[',save_path,']',": Save Folder Created.")
    except FileExistsError:
        if cleardir != 'Y':
            print("")
        else:
            print('[',save_path,']',": File Already Exists!")
            

            
def get_size(start_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size



def Load_Latest(save_path):
    
    
    with open('{}{}'.format(save_path, '/latest.uldm'), 'r') as timestamp:
        ts = timestamp.read()
        printU('Loading Folder',ts)
        
        return ts
    

    
def Load_Data(save_path,ts,save_options,save_number):

    data = []
    TMdata = []
    phidata = []
    graddata = []
    phasedata = []
    
    
    save_rho = save_options[0]
    
    save_psi = save_options[1]
    
    save_phi = save_options[6]
    
    save_phi_plane = save_options[7]
    
    save_plane = save_options[2]
    
    save_gradients = save_options[8]
    
    save_phase_plane = save_options[9]
    
    save_testmass = save_options[5]
    
    
    loc = save_path + '/' + ts

    import time   
    import warnings 
    warnings.filterwarnings("ignore")


    EndNum = 0
    
    
    if save_plane:
        printU('Loaded Planar Mass Density Data \n')
    if save_testmass:
        printU('Loaded Test Mass State Data \n')
    if save_phi_plane:
        printU('Loaded Planar Gravitational Field Data \n')
    if save_gradients:
        printU('Loaded Test Mass Gradient Data \n')
    if save_phase_plane:
        printU('Loaded Planar ULD Phase Data \n')
        

    
    for x in np.arange(0,save_number+1,1):
    #for x in np.arange(0,550,1):    
    
        try:
            if save_plane:
                data.append(np.load('{}{}{:03d}{}'.format(loc, '/Outputs/R2D_#', x, '.npy')))
            if save_testmass:

                TMdata.append(np.load('{}{}{:03d}{}'.format(loc, '/Outputs/NTM_#', x, '.npy')))
            if save_phi_plane:

                phidata.append(np.load('{}{}{:03d}{}'.format(loc, '/Outputs/G2D_#', x, '.npy')))
                
            if save_gradients:
                graddata.append(np.load('{}{}{:03d}{}'.format(loc, '/Outputs/DYF_#', x, '.npy')))
                
            if save_phase_plane:
                phasedata.append(np.load('{}{}{:03d}{}'.format(loc, '/Outputs/A2D_#', x, '.npy')))
            
            EndNum += 1
        
        except FileNotFoundError:

            print("WARNING: Run incomplete or the storage is corrupt!")

            break
        
    printU("Loaded", EndNum, "Data Entries")
    return EndNum, data,  TMdata, phidata,    graddata, phasedata


def Load_npys(loc,save_options, LowMem = False, Extension = "npy", Old = False):
    
    if Extension == "hdf5":
        Loader = IOLoad_h5
    else: 
        Loader = IOLoad_npy
    
    if Old:
        if Extension == "hdf5":
            Loader = IOLoad_h5_O
        else: 
            Loader = IOLoad_npy_O

    printU('3D saves are not automatically loaded. Please load them manually.','IO')
    save_options[0] = False
    save_options[1] = False
    save_options[6] = False
    save_options[12] = False
    save_options[20] = False
    save_options[21] = False
    
    if LowMem:
        printU('Skipping 2D data. Please load them manually.','IO')
        save_options[2] = False
        save_options[7] = False
        save_options[13] = False
    
    SaveWordList = SaveOptionsCompile(save_options).split()
    
    Out = {}
    
    print(SaveWordList)
    
    Out['Directory'] = loc
    
    for Word in SaveWordList:
        if (Word != 'Energy') and (Word != 'Entropy'): 
            Out[Word] = []
        
    import time   
    import warnings 
    warnings.filterwarnings("ignore")
    x = 0
    success = True

    while success:
        
        try:
            for Word in SaveWordList:
                if (Word != 'Energy') and (Word != 'Entropy') and (Word != 'Momentum') and (Word != 'AngMomentum'): 
                    Out[Word].append(Loader(loc,Word,x))        
            x += 1
        
        except:
            success = False

    printU(f"Loaded {x} Data Entries from {loc}",'Loader')
    
    return x, Out




def SmoothingReport(a,resol,clength, silent = True):
    
    GridLenFS = clength/(resol)
    
    COMin = resol/2
    COMid = resol/2*np.sqrt(2)
    COMax = resol/2*np.sqrt(3)
    
    
    FS = np.arange(resol)+1
    
    rR = GridLenFS*FS
    
    GOrig = -1/rR
    
    GMod = -a/np.sqrt(1+a**2*rR**2)
    
    GDiff = - GOrig + GMod
    
    GRati = GMod / GOrig
    
    # Two little quantifiers
    BoundaryEn = next(x for x, val in enumerate(GRati) if val > 0.99)
    rS = FS[BoundaryEn]
    
    BoundaryEx = next(x for x, val in enumerate(GMod) if val > -a/2)
    rQ = FS[BoundaryEx]
    
    
    if not silent:
        
        import matplotlib.pyplot as plt

        fig_grav = plt.figure(figsize=(10, 9))

        # Diagnostics For Field Smoothing

        ax1 = plt.subplot(211, xticks = FS, xticklabels = [])
        ax2 = plt.subplot(212, sharex = ax1,xticks = FS, xticklabels = [])

        ax1.plot(FS,GOrig,'k--',label = 'Point Potential')

        ax1.plot(FS,GMod,'g.',label = 'Plummer Potential')

        ax1.set_ylim([-1.5*a,0])
        ax1.set_xlim([0,COMax])

        ax1.vlines(rS,-1.5*a,0,color = 'red',label = '1% difference')
        ax1.vlines(rQ,-1.5*a,0,color = 'blue',label = 'HWHM')

        ax1.vlines([COMin,COMid,COMax],-1.5*a,0,color = 'black')

        ax1.legend()

        ax1.set_ylabel('Potential (C.U.)')
        ax1.grid()

        ax2.plot(FS,(GRati),'g.')
        ax2.set_xlabel('Radial distance from origin (Grids)')
        ax2.set_ylabel('$Φ_{Plummer}*r$')
        ax2.vlines(rS,0,1,color = 'red')
        ax2.vlines(rQ,0,1,color = 'blue')
        ax2.set_ylim(0,1)
        #ax2.set_ylim([np.min(GDiff),np.max(GDiff)])
        ax2.grid()

        plt.show()

        print('Generating Field Smoothing Report:')
        print('  The simulation runs on a %.0f^3 grid with total side length %.1f'%(resol,clength))
        print('  The simulation grid size is %.4f Code Units,\n  ' % ( GridLenFS))

        print('\n==========Grid Counts of Important Features=========\n')
        print("  Radius outside which the fields are practically indistinguishable (Grids): %.0f" % rS)
        print("  Modified Potential HWHM (Grids): %.0f" % rQ)

def MeshSpacing(resol,length,length_units, silent = False):
    clength = convert(length,length_units,'l')
    lengthpc = convert_back(clength,'pc','l')
    if not silent:
        printU(f'Each grid spacing is {length/resol:.3f}{length_units}, this is {lengthpc/resol:.3f}pc.','Mesh')
    return length/resol
    
def GenPlummer(rP,length_units, silent = True, resol = 0,length = 0):
    a = convert_back(1/rP,length_units,'l')
    if not silent:
        clength = convert(length,length_units,'l')

        SmoothingReport(a,resol,clength, silent = silent)
    return a # IN CODE UNITS (LENGTH^-1)
    
def RecPlummer(a,length_units):   
    rP = 1/convert(a,length_units,'l')
    return rP # IN USER UNITS (LENGTH)
    
def EmbedParticle(particles,embeds,solitons):

    EI = 0
    
    embedsIter = embeds.copy()
    
    for Mollusk in embedsIter:
 
        Mass = Mollusk[0]*Mollusk[3]
        

        printU(f"Calculating and loading the mass of embedded particle #{EI}.")
        Pearl = [Mass,Mollusk[1],Mollusk[2]]

        if not (Pearl in particles):
            particles.append(Pearl)
        
        if Mollusk[3] == 0:
            printU(f"Embed structure #{EI} contains no black hole. Moving to regular solitons list.")
            
            Shell = [Mollusk[0],Mollusk[1],Mollusk[2],Mollusk[4]]
            solitons.append(Shell)
            embeds.remove(Mollusk)
            
        if Mollusk[3] == 1:
            printU(f"Embed structure #{EI} contains no ULDM. Removing from list.")
            
            Shell = [Mollusk[0],Mollusk[1],Mollusk[2],Mollusk[4]]
            embeds.remove(Mollusk)
            

        EI += 1
        
    
    return particles, solitons, embeds
        

    


def NBodyEnergy(MassListSI,TMDataSI,EndNum,a=0,length_units = ''): # kg, m, m/s, Int, code unit -1
    
    if a == 0:
        printU('Using Standard Newtonian Potential.', 'NBoE')
    
    else:
        
        rP =  RecPlummer(a,length_units)

        ADim = 1/rP
    
        printU(f'The Plummer Radius is {rP:.4f}({length_units})', 'NBoE')
    
    NBo = len(MassListSI)
    
    printU(f'Reconstructing Potential and Kinetic Energies for {NBo} stored objects.','NBoE')

    KS = np.zeros(int(EndNum))
    PS = np.zeros(int(EndNum))

    for i in range(int(EndNum)):

        Data = TMDataSI[i]

        for Mass1 in range(NBo):

            Index1 = int(Mass1*6)
            Position1 = Data[Index1:Index1+2]
            m1 = MassListSI[Mass1]
            if m1 == 0:
                continue

            else:
                Vx = Data[int(Index1+3)]
                Vy = Data[int(Index1+4)]
                Vz = Data[int(Index1+5)]
            
                KS[i] += 1/2*MassListSI[Mass1]*(Vx**2+Vy**2+Vz**2) # J

            for Mass2 in range (Mass1+1,NBo,1):
                Index2 = int(Mass2*6)
                Position2 = Data[Index2:Index2+2]
                m2 = MassListSI[Mass2]
                if m2 == 0:
                    continue

                r = Position1 - Position2

                rN = np.linalg.norm(r)

                if a == 0:
                    PS[i] += - 1*G*m1*m2/rN
                else:
                    PS[i] += - 1*G*m1*m2*ADim/np.sqrt(1+ADim**2*rN**2)

            
    return NBo, KS, PS

# Point Mass Kepler
def FOSR(m,r):
    return np.sqrt(m/r)

# NFW Kepler
def FOSU(m,r):
    return np.sqrt(m/r)

def PopulateWithStars(NStars, MaxMass, embeds, particles, resol, length, length_units, s_mass_unit, m_position_unit, m_velocity_unit, rIn = 0.4, rOut = 1.2, Sequential = False, CircDiv = 8):

    for Halo in embeds:
              
        GPos = np.array(Halo[1])
        GVel = np.array(Halo[2])
        GMass = Halo[0]
        GRatio = Halo[3]
        
        SolMass = GMass * (1-GRatio)
        
        SolSizeC = SolEst(SolMass,length,resol,mass_unit = s_mass_unit,length_units = length_units)
        
        SolSizeL = convert_between(SolSizeC,'',length_units,'l')
        
        for i in range(NStars):

            
            if Sequential:
                r = ((i+1)/NStars*(rOut-rIn) + rIn) * SolSizeL
                
                Mass = MaxMass
                
                
                for j in range(CircDiv):
                
                    theta = j/CircDiv * 2 * np.pi



                    m_temp, v = DefaultSolitonOrbit(resol,length, length_units, SolMass, s_mass_unit, r, m_position_unit, m_velocity_unit)

                    Position = np.array([r*np.cos(theta),r*np.sin(theta),0]) + GPos



                    Velocity = np.array([v*np.sin(theta),-v*np.cos(theta),0])  + GVel



                    particles.append([Mass,Position.tolist(),Velocity.tolist()])
                
                
            else:
                r = (np.random.random()*(rOut-rIn) + rIn) * SolSizeL
            
                Mass = MaxMass*np.random.random()

                theta = 2*np.pi*np.random.random()
            
                m_temp, v = DefaultSolitonOrbit(resol,length, length_units, SolMass, s_mass_unit, r, m_position_unit, m_velocity_unit)

                Position = np.array([r*np.cos(theta),r*np.sin(theta),0]) + GPos



                Velocity = np.array([v*np.sin(theta),-v*np.cos(theta),0])  + GVel



                particles.append([Mass,Position.tolist(),Velocity.tolist()])

    return particles


def PopulateBHWithStars(particles,rIn = 0.4, rOut = 1.2,InBias = 0, NStars = 10, MassMax = 1e-5):

    IterParticles = particles.copy()
    
    for BH in IterParticles:
       
              
        GPos = np.array(BH[1])
        GVel = np.array(BH[2])
        GMass = BH[0]
        
        if GMass == 0:
            continue
        else:

            for i in range(NStars):

                r = (np.random.random()*(rOut-rIn) + rIn)

                theta = 2*np.pi*np.random.random()

                v = FOSR(GMass,r)

                Position = np.array([r*np.cos(theta),r*np.sin(theta),0]) + GPos
                Velocity = np.array([v*np.sin(theta),-v*np.cos(theta),0]) + GVel

                Mass = MassMax*np.random.random()

                particles.append([Mass,Position.tolist(),Velocity.tolist()])

    return particles


def SolEst(mass,length,resol,mass_unit = '',length_units = '', Plot = False, Density = 0, density_unit = ''): 
    # Only deals with default solitons!
    import matplotlib.pyplot as plt
    
    code_mass = convert(mass,mass_unit,'m')
    code_length = convert(length,length_units,'l')
    
    f = LoadDefaultSoliton()
    alpha = (code_mass / 3.883) ** 2
    
    CutOff = 5.6
    
    delta_x = 0.00001
    
    rarray = np.linspace(0,code_length/2,resol//2)

    funct = 0*rarray
    for index in range(resol//2):

        if (np.sqrt(alpha) * rarray[index] <= CutOff):
            funct[index] = alpha * f[int(np.sqrt(alpha) * (rarray[index] / delta_x + 1))]
        
        else:
            funct[index] = np.nan
            
    funct = funct**2
    
    if Plot:
        plt.plot(rarray,funct,'--')
        plt.xlim([0,code_length/2])
        plt.ylim([0,funct[0]*1.1])
        plt.xlabel('Code Radial Coordinate')
        plt.ylabel('Code Density')
        
        if Density != 0:
            DensityC = convert(Density,density_unit,'d')
            plt.plot(rarray,0*rarray+DensityC,'--')
    
    try:
        RHWHM = np.where(funct <= funct[0]/2)
    
        codeHWHM = rarray[RHWHM[0][0]]
    
        return codeHWHM
    
    except IndexError:
        print('Soliton too wide or too narrow!')
        print(f'Central value: {funct[0]}',f'Minimum value: {np.min(funct)}')
        return 0

SolitonSizeEstimate = SolEst   

def SolitonSizeEstimateR(mass,length,resol,mass_unit = '',length_units = '', Plot = False, Density = 0, density_unit = ''): 
    # Only deals with default solitons!
    import matplotlib.pyplot as plt
    
    code_mass = convert(mass,mass_unit,'m')
    code_length = convert(length,length_units,'l')
    
    f = LoadDefaultSoliton()
    alpha = (code_mass / 3.883) ** 2
    
    CutOff = 5.6
    
    delta_x = 0.00001
    
    rarray = np.linspace(0,code_length/2,resol//2)

    funct = 0*rarray
    for index in range(resol//2):

        if (np.sqrt(alpha) * rarray[index] <= CutOff):
            funct[index] = alpha * f[int(np.sqrt(alpha) * (rarray[index] / delta_x + 1))]
        
        else:
            funct[index] = np.nan
            
    funct = funct**2
    
    if Plot:
        plt.plot(rarray,funct,'--')
        plt.xlim([0,code_length/2])
        plt.ylim([0,funct[0]*1.1])
        plt.xlabel('Code Radial Coordinate')
        plt.ylabel('Code Density')
        
        if Density != 0:
            DensityC = convert(Density,density_unit,'d')
            plt.plot(rarray,0*rarray+DensityC,'--')
    
    try:
        RHWHM = np.where(funct <= funct[0]/2)
    
        codeHWHM = rarray[RHWHM[0][0]]
    
        return codeHWHM, rarray, funct
    
    except IndexError:
        print('Soliton too wide or too narrow!')
        print(f'Central value: {funct[0]}',f'Minimum value: {np.min(funct)}')
        return 0, rarray, funct


def DefaultDBL(v = 10,vUnit = 'm/s'):
    
    v = convert_between(v,vUnit,'m/s','v')
    
    return 2*np.pi*hbar/(axion_mass*v)


# Relevant for Paper 1

def ParameterScanGenerator(path_to_config,ScanParams,ValuePool,save_path,
                           SaveSpace = False, KeepResol = True, KeepSmooth = False,
                          AdaptiveTime = False):
        
    if len(ScanParams) != len(ValuePool):
        raise ValueError ('You did not specify the correct number of variable pools to scan over!')
        
    else:

        printU(f'Automated scan will be performed over {len(ScanParams)} parameters.','ParamScan')
        
    Product = 1
    
    for Pool in ValuePool:
        
        Product *= len(Pool)
        
    print(f'There will be {Product} separate simulations. They are:')
    
    print(list(zip(ScanParams,ValuePool)))
  
    print('(Units are defined in the donor config file)')
        
    # Load background parameters from donor config file.
    
    NS, length, length_units, resol, duration, duration_units, step_factor, save_number, SO, save_format, s_mass_unit, s_position_unit, s_velocity_unit, solitons,start_time, m_mass_unit, m_position_unit, m_velocity_unit, particles, embeds, Uniform,Density, density_unit,a, UVel = LoadConfig(path_to_config)
    
    if SaveSpace:
        save_options = 'Minimum'
    else:
        save_options = SaveOptionsCompile(SO)
    
    if KeepSmooth:
        resol_old = resol
        a_old = a
    
    if KeepResol:
        RPL = resol/length
    
    if AdaptiveTime:
        resol_old = resol
        Density_old = Density
        duration_old = duration


    Units = []
    
    for ScanP in ScanParams:
    
        if ScanP == 'Density':
                
            Units.append(density_unit)

        elif ScanP == 'Resolution':

            Units.append('Grids')

        elif ScanP == 'TM_M':

            Units.append(m_mass_unit)

        elif ScanP == 'TM_v':

            Units.append(m_velocity_unit)

        elif ScanP == 'U_v':

            Units.append(s_velocity_unit)

        elif ScanP == 'Step_Factor':

            Units.append('')
            
        elif ScanP == 'Scaling':
            
            Units.append('')
            
            lengthOrig = length
            
        elif ScanP == 'Plummer_Radius':
                       
            KeepSmooth = False
            Units.append('xLength')

        else:
            raise ValueError('Unrecognized parameter type used.')
    
    for i in range(Product):

        Str = 'PScan'

        PreDim = Product

        for j in range(len(ScanParams)):
            
            Pool = ValuePool[j]
            
            # String Manipulation
            NPar = len(Pool)

            PreDim = PreDim // NPar

            iDisp = i

            iDiv = iDisp // PreDim % NPar 

            # Value Lookup
            
            if ScanParams[j] == 'Density':
                
                Density = Pool[iDiv]  
                PString = 'D'
                

            elif ScanParams[j] == 'Resolution':
                
                resol = Pool[iDiv] 
                
                if KeepSmooth:
                    a = a_old * (resol)/(resol_old)
                
                PString = 'R'
            
            elif ScanParams[j] == 'TM_M':
                
                particles[0][0] = Pool[iDiv]   
                PString = 'M'
            
            elif ScanParams[j] == 'TM_v':
                
                particles[0][2][1] = Pool[iDiv]
                PString = 'V'
                
            elif ScanParams[j] == 'U_v':
                
                UVel[1] = -1*Pool[iDiv]  
                PString = 'U'
                
            elif ScanParams[j] == 'Step_Factor':
                
                step_factor = Pool[iDiv]
                PString = 'F'
                
            elif ScanParams[j] == 'Scaling':
                
                length = lengthOrig * Pool[iDiv]
                PString = 'S'
                if KeepResol:
                    resol =  int(RPL * length)
            
            elif ScanParams[j] == 'Plummer_Radius':
                
                rP = Pool[iDiv] * length
                a = GenPlummer(rP,length_units)
                PString = 'P'
            
            else:
                raise ValueError('Unrecognized parameter type used.')
            
            
            Str += f'_{PString}{iDiv+1:02d}'
        
        # GenerateConfig Is Done Per i Loop
    
        GenerateConfig(NS, length, length_units, resol, duration, duration_units, step_factor, save_number, save_options, save_path, save_format, s_mass_unit, s_position_unit, s_velocity_unit, solitons,start_time, m_mass_unit, m_position_unit, m_velocity_unit, particles,embeds, Uniform,Density,density_unit,a,UVel,True,Str)
        
        print('Generated config file for', Str)
        
    file = open('{}{}'.format(save_path, '/LookUp.uldm'), "w+")
    
    file.write(f'PyUL {S_version} Parameter Scan Settings Lookup \n')
    
    for line in list(zip(ScanParams,Units,ValuePool)):
    
        file.write((str(line)+'\n'))
    file.close()
        
    return Product, save_options

def RhoEst(length,length_units,mass,mass_unit):
    lengthC = convert(length,length_units,'l')
    massC = convert(mass,mass_unit,'m')
    return massC/lengthC**3
    
DensityEstimator = RhoEst

def VizInit2D(length,length_units,resol,embeds,
              solitons,s_position_unit, s_mass_unit,
              particles,m_position_unit, Uniform, Density, UVel, rP, VScale = 1):
    
    particles, solitons, embeds = EmbedParticle(particles,embeds,solitons)
    
    
    import matplotlib.pyplot as plt
    
    PR = np.linspace(-length/2,length/2,resol,endpoint = False)
    
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    
    for i,soliton in enumerate(solitons):

        mass = soliton[0]
        
        HWHM = SolEst(mass,length,resol,s_mass_unit,length_units)

        HWHM = convert_back(HWHM,length_units,'l')
        
        position = convert_between(np.array(soliton[1]),s_position_unit,length_units,'l')
        
        circ = plt.Circle((position[1],position[0]),HWHM,fill = False)
        
        ax.add_patch(circ)
        
        velocity = np.array(soliton[2])
        
        if np.linalg.norm(velocity) != 0:
            ax.quiver(position[1],position[0],velocity[1],velocity[0],scale = VScale)
    
    for i,particle in enumerate(particles):

        
        position = convert_between(np.array(particle[1]),m_position_unit,length_units,'l')
        
        velocity = np.array(particle[2])
        
        circ = plt.Circle((position[1],position[0]),rP,fill = False)
        ax.add_patch(circ)
        
        ax.scatter(position[1],position[0])
        if np.linalg.norm(velocity) != 0:
            ax.quiver(position[1],position[0],velocity[1],velocity[0], scale = VScale)
        
    for i,embed in enumerate(embeds):
        #print(f"Visualizing Embedded Soliton #{i} (Approximate)")
        mass = embed[0] * embed[3]
        
        HWHM = SolEst(mass,length,resol,s_mass_unit,length_units)
        HWHM = convert_back(HWHM,length_units,'l')
        
        position = convert_between(np.array(embed[1]),s_position_unit,length_units,'l')
        
        circ = plt.Circle((position[1],position[0]),HWHM,fill = False)
        
        ax.add_patch(circ)
        ax.scatter(position[1],position[0])
        
    ax.set_ylim(PR[0],PR[-1])
    ax.set_xlim(PR[0],PR[-1])
    
    ax.set_xlabel(f'$x$')
    ax.set_ylabel(f'$y$')
    ax.grid(color = 'k',alpha = 0.3)
    
    if Uniform:
        if UVel[1]**2 + UVel[0]**2 != 0:
            ax.quiver(0.75,0.75,UVel[1],UVel[0])
        ax.text(0.75,0.75,f'Density: {Density}')
    
    ax.set_xticks(PR)
    ax.set_yticks(PR)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    return fig, ax


# More Pragmatic Version, Just initiates the psi, a part of current evolve()
# solitons, embeds, particles

def Init3D(length,length_units,resol,embeds,
              solitons,s_position_unit, s_mass_unit,
              particles,m_position_unit, Uniform, Density, UVel, rP):
    
    particles, solitons, embeds = EmbedParticle(particles,embeds,solitons)
    
    masslist = []
    TMState = []

##########################################################################################
    #CONVERT INITIAL CONDITIONS TO CODE UNITS

    lengthC = convert(length, length_units, 'l')

    Density = convert(Density,density_unit,'d')

    Vcell = (lengthC / float(resol)) ** 3

 ##########################################################################################
    # SET UP THE REAL SPACE COORDINATES OF THE GRID - FW Revisit

    gridvec = np.linspace(-lengthC / 2.0, lengthC / 2.0, resol, endpoint = False) # careful!

    xarray, yarray, zarray = np.meshgrid(
        gridvec, gridvec, gridvec,
        sparse=True, indexing='ij')

    ##########################################################################################
    delta_x = 0.00001 # Needs to match resolution of soliton profile array file. Default = 0.00001

    warn = 0 
    funct = pyfftw.zeros_aligned((resol, resol, resol), dtype='complex128')   

    psi = pyfftw.zeros_aligned((resol, resol, resol), dtype='complex128')

    MassCom = Density*lengthC**3

    UVelocity = convert(np.array(UVel),s_velocity_unit, 'v')

    VTot = np.linalg.norm(UVelocity)

    DensityCom = MassCom / resol**3

    psi = ne.evaluate("0*psi + sqrt(Density)")

    velx = UVelocity[0]
    vely = UVelocity[1]
    velz = UVelocity[2]
    psi = ne.evaluate("exp(1j*(velx*xarray + vely*yarray + velz*zarray))*psi")

    for EI,emb in enumerate(embeds):
        Ratio = emb[3]
        RatioBU = float(Ratio/(1-Ratio))
        try:
            delta_xL, prealphaL, betaL,CutOff = LoadSolitonConfig(RatioBU)
        except RuntimeError:
            print('==============================================================================')
            printU(f'Note that this process will not be required for repeated runs. To remove custom profiles, go to the folder /Soliton Profile Files/Custom.', 'Profiler', ToFile= GenerateLog, FilePath= LogLocation)
            printU(f'Generating profile for Soliton with MBH/MSoliton = {RatioBU:.4f}, Part 1', ToFile= GenerateLog, FilePath= LogLocation)
            GMin,GMax = BHGuess(RatioBU)
            s, BHMass = BHRatioTester(RatioBU,30,1e-6,GMin,GMax,a)
            printU(f'Generating profile for Soliton with MBH/MSoliton = {RatioBU:.4f}, Part 2', ToFile= GenerateLog, FilePath= LogLocation)
            SolitonProfile(BHMass,s,a,not Draft)
        print('==============================================================================')

        delta_xL, prealphaL, betaL,CutOff = LoadSolitonConfig(RatioBU)

        # L stands for Local, as in it's only used once.
        fL = LoadSoliton(RatioBU)

        printU(f"Loaded embedded soliton {EI} with BH-Soliton mass ratio {RatioBU:.4f}.", 'Init', ToFile= GenerateLog, FilePath= LogLocation)

        mass = convert(emb[0], s_mass_unit, 'm')*(1-Ratio)
        position = convert(np.array(emb[1]), s_position_unit, 'l')
        velocity = convert(np.array(emb[2]), s_velocity_unit, 'v')

        # Note that alpha and beta parameters are computed when the initial_f.npy soliton profile file is generated.

        alphaL = (mass / prealphaL) ** 2
        phase = emb[4]
        funct = initsoliton_jit(funct, xarray, yarray, zarray, position, alphaL, fL, delta_xL)

        if(np.isnan(funct).any()):
            print('Something is seriously wrong!')
            raise RuntimeError('Duh')
        ####### Impart velocity to solitons in Galilean invariant way
        velx = velocity[0]
        vely = velocity[1]
        velz = velocity[2]
        funct = ne.evaluate("exp(1j*(alphaL*betaL*t0 + velx*xarray + vely*yarray + velz*zarray -0.5*(velx*velx+vely*vely+velz*velz)*t0  + phase))*funct")
        psi = ne.evaluate("psi + funct")
        EI += 1 # For displaying.

    if solitons != []:
        printU(f"Loaded standard soliton radial profile.",'Init', ToFile= GenerateLog, FilePath= LogLocation)
        f = LoadDefaultSoliton()

    for s in solitons:
        mass = convert(s[0], s_mass_unit, 'm')
        position = convert(np.array(s[1]), s_position_unit, 'l')
        velocity = convert(np.array(s[2]), s_velocity_unit, 'v')
        # Note that alpha and beta parameters are computed when the initial_f.npy soliton profile file is generated.
        alpha = (mass / 3.8827652755822006) ** 2 #3.883
        #alpha = (mass / 3.883) ** 2 #3.883
        beta = 2.4538872760773143 #2.454
        #beta = 2.454 #2.454
        phase = s[3]

        funct = InitSolitonF(gridvec, position, resol, alpha, DR = DR)
        # funct = initsoliton_jit(funct, xarray, yarray, zarray, position, alpha, f, delta_x, DR = DR)

        printU(f"Using scale {DR}","Scaler")
        ####### Impart velocity to solitons in Galilean invariant way
        velx = velocity[0]
        vely = velocity[1]
        velz = velocity[2]
        funct = ne.evaluate("exp(1j*(alpha*beta*t0 + velx*xarray + vely*yarray + velz*zarray -0.5*(velx*velx+vely*vely+velz*velz)*t0  + phase))*funct")
        psi = ne.evaluate("psi + funct")


    COM = Find3BoxCOM(rho,xarray, yarray, zarray)

    for MI, particle in enumerate(particles):

        if MassChange:
            mT = particle[0] * MassFunc(0)
        else:
            mT = convert(particle[0], m_mass_unit, 'm')


        masslist.append(mT)

        position = convert(np.array(particle[1]), m_position_unit, 'l')

        if mT == 0:
            printU(f"Particle #{MI} loaded as observer.",'NBody', ToFile= GenerateLog, FilePath= LogLocation)
        else:

            DensityInit = QuickInterpolate(rho,lengthC,resol,position)
            printU(f"Particle #{MI} mass {mT:.5f} and local density {DensityInit:.5f} (code units).",'NBody', ToFile= GenerateLog, FilePath= LogLocation)
            NBDensity(loc,DensityInit)


        if Shift:
            position = GridShift(position, lengthC, resol)

        velocity = convert(np.array(particle[2]), m_velocity_unit, 'v')

        IND = int(6*MI)

        TMx = position[0]
        TMy = position[1]
        TMz = position[2]

        Vx = velocity[0]
        Vy = velocity[1]
        Vz = velocity[2]

        TMState.append([TMx,TMy,TMz,Vx,Vy,Vz])

        if mT != 0:

            distarrayTM = ne.evaluate("((xarray-TMx)**2+(yarray-TMy)**2+(zarray-TMz)**2)**0.5") # Radial coordinates
            if a == 0:
                phiTM = ne.evaluate("phiTM-mT/(distarrayTM)")
            else:
                phiTM = ne.evaluate("phiTM-a*mT/sqrt(1+a**2*distarrayTM**2)")


        if (save_options[3]):
            EGPCM += mT*QuickInterpolate(phiSP,lengthC,resol,np.array([TMx,TMy,TMz]))
        MI = int(MI + 1)

        if AutoStop and Uniform and len(particles) == 1:
            ThresholdVelocity += Vy

        if AutoStop and len(particles) == 1:
            E0 = 1/2 * mT * np.linalg.norm(velocity + UVelocity)**2  
            if E0 == 0:
                E0 = 1

    masslist = np.array(masslist)
    TMState = np.array(TMState)
    TMState = TMState.flatten(order='C')

    return TMState, psi, COM

# New addition that fixes mesh compatibility
def XYSwap(Vec):
    return [Vec[1],Vec[0],Vec[2]]


# Check for Energy Evaluation Consistency
def SmoothingScan(a,resol,length,length_units,Density,density_unit,M = 1,m_mass_unit = 'M_solar_masses', N = 20, Grids = 1, curve = 'xyz', AxVer = 2, Shift = False):
    
    lengthC = convert(length,length_units,'l')
    DensityC = convert(Density,density_unit,'d')
    massC = convert(M,m_mass_unit,'m')
    
    if AxVer == 1:
        gridvec = np.linspace(-lengthC / 2.0 + lengthC / float(2 * resol),
                  lengthC / 2.0 - lengthC / float(2 * resol), resol, endpoint = True)
    if AxVer == 4:
        gridvec = np.linspace(-lengthC / 2.0 + lengthC / float(2 * resol),
                  lengthC / 2.0 - lengthC / float(2 * resol), resol, endpoint = False)
        
    elif AxVer == 3:
        gridvec = np.linspace(-lengthC / 2.0, lengthC / 2.0, resol, endpoint= True)
        
    elif AxVer == 2:
        gridvec = np.linspace(-lengthC / 2.0, lengthC / 2.0, resol, endpoint= False)
    
    xarray, yarray, zarray = np.meshgrid(
        gridvec, gridvec, gridvec,
        sparse=True, indexing='ij')
    
    GridSize = lengthC/resol # Grid Size
    
    VCell = GridSize ** 3
    
    
    GridStep = GridSize/N # Scanning Size
    
    if curve == 'x':
        StepVec = np.array([1,0,0])

    elif curve == 'xy':
        StepVec = np.array([1,1,0])

    elif curve == 'xyz':
        StepVec = np.array([1,1,1])
     
    Origin = - (Grids)/2 * StepVec

    # Initial Potential Energy
    
    mT = massC
    
    EScan = []
    x = []
    
    for NI in range(Grids*N+1):
        
        phiTM = np.zeros([resol,resol,resol])
        
        position = Origin * GridSize + NI * StepVec* GridStep

        if Shift:      
            position = GridShift(position, lengthC, resol)
            
        
        TMx = position[0]
        TMy = position[1]
        TMz = position[2]

        distarrayTM = ne.evaluate("((xarray-TMx)**2+(yarray-TMy)**2+(zarray-TMz)**2)**0.5") # Radial coordinates

        if a == 0:
            phiTM = ne.evaluate("phiTM-mT/(distarrayTM)")
        else:
            phiTM = ne.evaluate("phiTM-a*mT/sqrt(1+a**2*distarrayTM**2)")
            
        E0 = np.sum(phiTM*DensityC*VCell)
        
        EScan.append(E0)
        x.append(position[0])
        
    return np.array(x), np.array(EScan), gridvec
        
    
    
def GridShift(position, lengthC, resol, direction = '-'):
    
    position = np.array(position)
    
    return (position - lengthC/resol/2.0).tolist()
    
    
def GetRel(array):
    return array - array[0]


def DefaultSolitonOrbit(resol,length, length_units, s_mass, s_mass_unit, m_radius, m_position_unit, m_velocity_unit = '', Silent = True, Detail = 10000, IndexCorrect = True):
    
    lengthC = convert(length,length_units,'l')
    s_massC = convert(s_mass,s_mass_unit,'m')
    m_radiC = convert(m_radius,m_position_unit,'l')
    
    
    if m_radiC >= lengthC/2:
        raise ValueError("Supplied orbital radius too large!")
    
    linearray = np.linspace(0,lengthC/2,Detail,endpoint = False)
    
    lineh = linearray[1] - linearray[0]

    f = LoadDefaultSoliton()

    delta_x = 0.00001

    alpha = (s_massC / 3.883) ** 2

    funct = np.abs(initsolitonRadial(linearray, alpha, f, delta_x,Cutoff = 5.6, IndexCorrect = IndexCorrect))**2
    
    if not Silent:
        import matplotlib.pyplot as plt

        plt.plot(linearray,funct)
        plt.xlabel('Code Length')
        plt.ylabel('Code Density')
        
        plt.vlines(m_radiC,np.min(funct),np.max(funct))
    
    try:
        CutOff = np.where(linearray >= m_radiC)[0][0]
    except:
        CutOff = len(linearray)
    
    Integrand = linearray[0:CutOff]**2 * funct[0:CutOff]
    from scipy import integrate as SINT
    MInt = 4*np.pi*SINT.simps(Integrand, x = linearray[0:CutOff])
    
    VC = np.sqrt(MInt/m_radiC)
    
    return convert_back(MInt, s_mass_unit,'m'), convert_back(VC,m_velocity_unit,'v')


def InterpolateCurve(x,y, Resol = 1800, ResolGain = 10):
    import scipy.interpolate as SInt
    t = np.arange(len(x))

    BSplineX = SInt.make_interp_spline(t,x)
    BSplineY = SInt.make_interp_spline(t,y)
    
    if ResolGain > 0:
        Resol = int(len(x) * ResolGain)
              
    T = np.linspace(t[0],t[-1],Resol)
    
    Fitx = BSplineX(T)
    Fity = BSplineY(T)
    
    return Fitx, Fity
              
def InterpolateCurve3(x,y,z, Resol = 1800, ResolGain = 10):
    import scipy.interpolate as SInt
    t = np.arange(len(x))

    BSplineX = SInt.make_interp_spline(t,x)
    BSplineY = SInt.make_interp_spline(t,y)
    BSplineZ = SInt.make_interp_spline(t,y)
    
    if ResolGain > 0:
        Resol = int(len(x) * ResolGain)
    
    T = np.linspace(t[0],t[-1],Resol)
    
    Fitx = BSplineX(T)
    Fity = BSplineY(T)
    Fitz = BSplineZ(T)
    
    return Fitx, Fity, Fitz



def FindBoxCOM(Darray,xG, yG, zG, Silent = True):
    
    Mass = np.sum(Darray)

    COM = np.array([np.sum(xG * Darray),np.sum(yG * Darray),np.sum(zG * Darray)])/Mass
    
    if not Silent:
        print(COM)
    
    return COM


def FindBoxCOMSpeed(COMLog, hC = 1):
    
    CMVLog = []
    
    # First Step

    CMVLog.append((COMLog[1]-COMLog[0])/hC)

    # Central Steps
    for i in range(1,len(COMLog)-1):  
        CMVLog.append((COMLog[i+1]-COMLog[i-1])/(2*hC))


    CMVLog.append((COMLog[-1]-COMLog[-2])/hC)

    return CMVLog


#### not very useful
def GalShift(psi,v):
    velx = v[0]
    vely = v[1]
    velz = v[2]
    return ne.evaluate("exp(1j*(velx*xG + vely*yG + velz*zG - 0.5*(velx*velx+vely*vely+velz*velz)*hC*i))*psi")
    

    
# Basis Decomposition Tools

##########################################################

def HighBasisInit_O(n,resol,mS, xarray, yarray, zarray):
    
    ## Assemble the Order to Load
    
    # Principal Quantum Number = n+1
    
    Name = f'./Soliton Profile Files/HFK/f_HFK_{n:02d}'
    
    DataName = Name + '.npy'
    MDataName = Name + '_info.uldm'
    
    # Load data
    f = np.load(DataName)
    
    # Load Metadata
    config = json.load(open(MDataName))
    
    delta_x = config["Resolution"]
    alpha = config["Alpha"]
    beta = config["Beta"]
    CutOff = config["Radial Cutoff"]
    
    alphaL = (mS / alpha) ** 2 # Scale!
    
    position = [0,0,0]
    
    funct = np.zeros([resol,resol,resol],dtype = 'complex128')
    
    return initsoliton_jit(funct, xarray, yarray, zarray, [0,0,0], alphaL, f, delta_x,CutOff)

    
    
def NLMCompile(resol, mSC, xarray,yarray,zarray,max_n = 9, Scratch = True, OnlySym = False):
    
    time0 = time.time()
    
    printU("Init. spherical grid ...",'SpH')
    LonArr, ColArr = SphBasic(resol)
    printU("Init. spherical grid ... Done!",'SpH')
    
    bases = {}
    printU('Prep. basis ...','SpH')
        
    print('-'*20)
    for n in range(max_n):
        
        RadialFN = HighBasisInit_O(n,resol,mSC,xarray,yarray,zarray) 
        
        
       
        for l in range(n+1):
            
            if OnlySym:
                mrange = 1
                
            else:
                mrange = l+1
            
            for m in range(mrange):
                
                nlmString = f"{n+1}_{l}_{m}"
                
                Ylm = SPH(m,l,LonArr,ColArr)
                
                bases[nlmString] = np.conj(RadialFN*Ylm/np.abs(Ylm))
                
                print(nlmString,end = ',')
                
        print('')     
        print('-'*20)
                
      
    printU(f"Prep. basis ... Done! Time taken: {time.time()-time0:.4g} s",'SpH')
            
    return bases
                
                
# New stuff 

 

def SphBasic(resol):
    
    GSpace = np.linspace(-1,1,resol,endpoint=False)

    xG,yG,zG = np.meshgrid(GSpace,GSpace,GSpace,indexing = 'ij')
    
    RArr = np.sqrt(xG**2+yG**2+zG**2) # R (Not Useful)
    
    LonArr =  np.arctan2(yG,xG) # THETA # Longitude
    ColArr = np.arccos(zG/RArr)  # Colatitude

    ColArr[np.isnan(ColArr)] = 0 # Filter off invalid values
    
    return LonArr, ColArr
    

    
def ExpansionCoefficientPrep(psi,Basis,loc,ix,its_per_save):
    
    save_num = int((ix + 1) / its_per_save)
        
    Result = {}
    
    for key, value in Basis.items():
        
        Result[key] = (np.sum(psi*value))
        
        
    with open(f"{loc}/Outputs/SPH_#{save_num:03d}.uldm", "w+") as outfile:
        json.dump(Result, outfile,indent=4)
    
    


def ExpansionCoefficient(psi,Basis):

    Result = {}

    
    for key, value in Basis.items():
        
        InnerProd = ne.evaluate('value*psi')
        
        Result[key] = np.sum(InnerProd)
        
    return Result


def Find3BoxCOM(rho,xGrid, yGrid, zGrid):
    COM = np.array([np.sum(xGrid * rho),np.sum(yGrid * rho),np.sum(zGrid * rho)])/np.sum(rho)
    return COM
    
