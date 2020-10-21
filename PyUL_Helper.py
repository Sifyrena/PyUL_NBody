#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 13:56:50 2020

@author: ywan598

"""

import os
import matplotlib.pyplot as plt
import numpy as np


def SSEst(save_options, save_number, resol):
    
    
    save_rho = save_options[0]
    
    save_psi = save_options[1]
    
    save_phi = save_options[6]
    
    save_phi_plane = save_options[7]
    
    save_plane = save_options[2]
    
    save_gradients = save_options[8]
    
    save_testmass = save_options[5]
    

    
    PreMult = 0
    
    if save_rho:
        print('PyUL NBody: Saving Mass Density Data (3D)')
        PreMult = PreMult + resol**3
        
    if save_psi:
        print('PyUL NBody: Saving Complex Field Data (3D)')
        PreMult = PreMult + resol**3*2
        
    if save_phi:
        print('PyUL NBody: Saving Gravitational Field Data (3D)')
        PreMult = PreMult + resol**3
        
    if save_plane:
        print('PyUL NBody: Saving Mass Density Data (2D)')
        PreMult = PreMult + resol**2
        
    if save_phi_plane:
        print('PyUL NBody: Saving Gravitational Field Data (2D)')
        PreMult = PreMult + resol**2
    
    if save_gradients:
        print('PyUL NBody: Saving NBody Gradient Data')
    
    if save_testmass:
        print('PyUL NBody: Saving NBody Position Data')
    
    return (save_number+1)*(PreMult)*8/(1024**3)
    
    

def DSManagement(save_path):
    
    print(save_path,": The current size of the folder is", round(get_size(save_path)/1024**2,3), 'Mib')

    if get_size(save_path) == 0:
        cleardir = 'N'
    else:
        print(save_path,": Do You Wish to Delete All Files Currently Stored In This Folder? [Y] \n")
        cleardir = str(input())
    
    if cleardir == 'Y':
        import shutil 
        shutil.rmtree(save_path)
        print("Folder Cleaned! \n")
    
    try:
        os.mkdir(save_path)
        print(save_path,": Save Folder Created.")
    except FileExistsError:
        if cleardir != 'Y':
            print("")
        else:
            print(save_path,": File Already Exists!")
            

            
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
    
    
    with open('{}{}'.format(save_path, '/timestamp.txt'), 'r') as timestamp:
        ts = timestamp.read()
        print('PyUL NBody: Loading Folder',ts)
        
        return ts
    

    
def Load_Data(save_path,ts,save_options,save_number):
    
    data = []
    TMdata = []
    phidata = []
    graddata = []
    
    
    save_rho = save_options[0]
    
    save_psi = save_options[1]
    
    save_phi = save_options[6]
    
    save_phi_plane = save_options[7]
    
    save_plane = save_options[2]
    
    save_gradients = save_options[8]
    
    save_testmass = save_options[5]
    
    
    loc = save_path + '/' + ts

    import time   
    import warnings 
    warnings.filterwarnings("ignore")


    EndNum = 0
    
    
    if save_plane:
        print('PyUL NBody: Loaded Planar Mass Density Data \n')
    if save_testmass:
        print('PyUL NBody: Loaded Test Mass State Data \n')
    if save_phi_plane:
        print('PyUL NBody: Loaded Planar Gravitational Field Data \n')
    if save_gradients:
        print('PyUL NBody: Loaded Test Mass Gradient Data \n')

    
    for x in np.arange(0,save_number+1,1):
    #for x in np.arange(0,550,1):    
    
        try:
            if save_plane:
                data.append(np.load('{}{}{}{}'.format(loc, '/plane_#', x, '.npy')))
            if save_testmass:

                TMdata.append(np.load('{}{}{}{}'.format(loc, '/TM_#', x, '.npy')))
            if save_phi_plane:

                phidata.append(np.load('{}{}{}{}'.format(loc, '/Field2D_#', x, '.npy')))
                
            if save_gradients:
                graddata.append(np.load('{}{}{}{}'.format(loc, '/Gradients_#', x, '.npy')))
            
            EndNum += 1
        
        except FileNotFoundError:

            print("WARNING: Run incomplete or the storage is corrupt!")

            break
        
    print("PyUL NBody: Loaded", EndNum, "Data Entries")
    return EndNum, data,  TMdata, phidata,    graddata


def SmoothingReport(a,resol,clength):
    

    GridLenFS = clength/(resol)


    
    fig_grav = plt.figure(figsize=(10, 10))

    # Diagnostics For Field Smoothing
    
    
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    
    
    FS = np.linspace(1,50,50)
    
    rR = GridLenFS*FS
    
    GOrig = -1/rR
    
    GMod = -a*1/(a*rR+np.exp(-a*rR))
    
    GDiff = - GOrig + GMod
    
    # Two little quantifiers
    BoundaryEn = next(x for x, val in enumerate(GDiff) if val < 1e-2)
    rS = FS[BoundaryEn]
    
    BoundaryEx = next(x for x, val in enumerate(GMod) if val > -a/2)
    rQ = FS[BoundaryEx]
    
    ax1.plot(FS,GOrig,'k--',label = 'Point Potential')
    
    ax1.plot([rS,rS],[2,-100],'r.-')
    ax1.plot([rQ,rQ],[2,-100],'b.-')
    
    
    ax1.plot(FS,GMod,'go',label = 'Modified Potential')
    
    ax1.set_ylim([-1.5*a,0])
    
    ax1.legend()
    
    ax1.set_ylabel('$\propto$Energy')
    ax1.grid()
    
    ax2.semilogy(FS,(GDiff),'g.')
    ax2.set_xlabel('Radial distance from origin (Grids)')
    ax2.set_ylabel('Difference')
    ax2.semilogy([rS,rS],[1e2,1e-16],'r.-')
    ax2.semilogy([rQ,rQ],[1e2,1e-16],'b.-')
    ax2.grid()
    
    plt.show()
    
    
    print('Generating Field Smoothing Report:')
    print('  The simulation runs on a %.0f^3 grid with total side length %.1f'%(resol,clength))
    print('  The simulation grid size is %.4f Code Units,\n  ' % ( GridLenFS))
    
    print('\n==========Grid Counts of Important Features=========\n')
    print("  Radius outside which the fields are practically indistinguishable (Grids): %.0f" % rS)
    print("  Modified Potential HWHM (Grids): %.0f" % rQ)
    
    

def Init(Settings,CustomParticles,CustomSolitons,resol,clength):
    
    print('The simulation box edge length is %.03f, at a resolution of %d \n' % (clength,resol))
    
    if Settings[0]:
        print("Loading 2-Body Parabola Demo \n")
        

        print("Initial x Location of Particle 1 (First Quadrant Only) \n")

        x0 = float(input("x0 ") or "1")
        
        print("Initial y Location of Particle 1 (First Quadrant Only) \n")
        y0 = float(input("y0 ") or "1")
        
        if x0<0 or y0<0:
            raise ValueError('Particle 1 must start in the first quadrant.')
        
        
        print("The masses of the particles (each) \n")

        m = float(input("m") or "8")
    
        # Focal Point Is Origin
        # y^2 = 4cx + 4c^2
    
        c0 = 1/2*(-x0+np.sqrt(x0**2+y0**2))
    
        v0 = np.sqrt(m/(2*np.sqrt(x0**2+y0**2))) #Correct
    
        xDot0 = 1
        yDot0 = 2*c0*xDot0/(y0)
    
        vNorm = np.linalg.norm([xDot0,yDot0])
    
        xDot0 = xDot0/vNorm*v0
        yDot0 = yDot0/vNorm*v0
    
        BH1 = [m,[x0,y0,0],[-xDot0,-yDot0,0]]
        BH2 = [m,[-x0,-y0,0],[xDot0,yDot0,0]]

        particles = [BH1,BH2]
        
        print("Note: This 2Body Model Does Not Come with Solitons. Turning \'Uniform\' on is recommended \n")
        
        #Soliton parameters are mass, position, velocity and phase (radians)

        solitons = []
        # solitons = []
    
        
        
    elif Settings[1]:
        
        print("Loading 2-Body Circular Orbit Demo")
        
        
        print("Mass of Particle 1 \n")

        m1 = float(input("m1 = ") or "15")
        
        print("Mass of Particle 2 \n")
        
        m2 = float(input("m2 = ") or "6")
        
        print("Initial radial coordinate of Particle 1 \n")
        
        x1 = float(input("x1 = ") or "1")

        
        x2 = x1/m2*m1 # Ensures Com Position
        
        xC = x1+x2
        
        yDot1 = np.sqrt(m2*x1/xC**2)
        yDot2 = np.sqrt(m1*x2/xC**2)
               
        BH1 = [m1,[x1,0,0],[0,yDot1,0]]
        BH2 = [m2,[-x2,0,0],[0,-yDot2,0]]
        
        
        particles = [BH2]

    
        solitons = []
        
        print("Note: This 2Body Model Does Not Come with Solitons. Turning \'Uniform\' on is recommended \n")
        #solitons = [solitonC]
        
    
        
    elif Settings[2]:
        
        print("Loading Tangyuan Demo")
        
        print("Radius of orbit relative to box size? \n")
        
        rL = float(input("r_rel = ") or "0.2")
        
        r = rL*clength
        
        print('The orbit radius is %.4f \n' % (r))
        
        print("Mass of Each Consituent Particle? \n")
        
        m = float(input("m = ") or "15")
         
        m1 = m
        mS = m
        
        
        v = np.sqrt(m/r*(1+2*np.sqrt(2)))/2
        
        print("The speed required for them to go in circle has been evaluated. You can scale it with a factor. \n")
        
        vM = float(input("Speed Factor = ") or "1")
        
        
        

        v = vM * v
        print("The initial tangential speed is %.4f \n" % (v))
        
        
        BH1 = [m1,[r,0,0],[0,v,0]]

        
        particles = [BH1]
        #Soliton parameters are mass, position, velocity and phase (radians)
        
        
        solitonB = [mS, [-r,0,0],[0,-v,0], 0]
        
        solitonC = [mS, [0,-r,0],[v,0,0], 0]
        
        solitonD = [mS, [0,r,0],[-v,0,0], 0]
        
        
        solitons = [solitonB,solitonC,solitonD]

    
    elif Settings[3]:
        # Dynamics happen along the y axis.
        
        print("Loading Collision / Scattering Demo")
        
        print("(Approx) Impact Parameter? \n")
        
        b = float(input("b = ") or "0.5")

        print("Initial Relative Speed (along axial-direction)? \n")
        
        vRel0 = float(input("vRel0 = ") or "1.5")
        
        print("Initial Separation? \n")
        
        Separation = float(input("d = ") or "5")



        Origin = [0,0,0]
        
        
        print("Mass of Particle \n")

        m1 = float(input("m = ") or "12")
        
        print("Mass of Soliton\n")
        
        mS = float(input("m2 = ") or "12")

        
        yS = Separation*m1/(m1+mS)
        xS = b*m1/(m1+mS)
        ydotS = vRel0*m1/(m1+mS)
        
        y1 = Separation*mS/(m1+mS)
        x1 = b*mS/(m1+mS)
        ydot1 = vRel0*mS/(m1+mS)
        
        BH1 = [m1,[x1,y1,0],[0,-ydot1,0]]
        particles = [BH1]
        
        soliton1 = [mS,[-xS,-yS,0],[0,ydotS,0],0]
        solitons = [soliton1]
        
        
    else:
        
        print("Loading Custom Settings")
        particles = CustomParticles
        solitons = CustomSolitons
        
        
    return particles, solitons
    

def PInit(Avg,AvgD,CW):

    if Avg:
        # Fixed Probe Locations. Unit Vector = 1HWHM
        NCV = np.array([[0,0,0],
                        [1,0,0],
                        [-1,0,0],
                        [0,1,0],
                        [0,-1,0],
                        [0,0,1],
                        [0,0,-1]])
    
        # Probe Weights. Normalized automatically. Has to be nonzero.
        NCW = np.array([CW,1,1,1,1,1,1]) # The relative weights of all of them.
        
        NCV = AvgD * NCV
        
    else:
        
        NCV = np.array([[0,0,0]])
        NCW = np.array([1]) # The relative weights of all of them.

    return NCV, NCW


def D_version():
    return 'Helper Version 1. 21 Oct 2020'
    