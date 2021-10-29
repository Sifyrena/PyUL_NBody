#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 16:14:05 2021
Yourong F Wang (FWPhys.github.io)
"""


print("Welcome to PyUltraLight 2 Performance Benchmark")

print("Please key in the letter for a specific resolution. ENTER to confirm")

print("[Potato] 64³. [A] 256³. [B] 384³. [C] 512³. [D] 1024³")

userinput = input()

if userinput == 'Potato':
    resol = 64
elif userinput == 'A' or userinput == 'a':
    resol = 256
elif userinput == 'B' or userinput == 'b':
    resol = 384
elif userinput == 'D' or userinput == 'd':
    resol = 1024
else:
    resol = 512

      
print("Note: This script will produce outputs in the path [PyUL]/Bench/.")

print("\n")

import os
import numpy as np

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    print("Please press ENTER again to proceed.")
    import PyUltraLight2 as PyUL
except ModuleNotFoundError:
    
    print("WARNING: Your current Python Executable failed to locate some required packages. [Y] to automatically attempt to resolve the issue. [N] if you with to install the packages manually. ")
    
    userinput = input()
    
    if userinput == 'Y':
      
        install('pyfftw')
        install('numba')
        install('numexpr')
        install('multiprocessing')
        print("Please press ENTER again to proceed.")
        import PyUltraLight2 as PyUL
        
    else:
        raise RuntimeError("Benchmark Aborted!")



###### Do not touch
MinVersion = 23

if (PyUL.S_version < MinVersion):
    raise RuntimeError("You need the latest PyUL!")


np.set_printoptions(suppress=True)

import math

import psutil
import platform
from datetime import datetime

import numba
import numexpr as ne
import time
import pyfftw
import os
import sys
import multiprocessing
import numpy

from numpy import sqrt, exp, log, log10, sin, cos, tan, pi

# Useful Aux Functions
ToCode = PyUL.convert
ToPhys = PyUL.convert_back
CB = PyUL.convert_between
printU = PyUL.printU

# Simulation Parameters

save_path = 'Bench'

if not os.path.exists(save_path):
    os.makedirs(save_path)
    

length, length_units = 16, ''  #
duration, duration_units = 0.5, ''  #

start_time = 0.  # For solitons only: Pre-evolve the wavefunction phase.

NS = 36

save_format = 'npy'  # npy, npz, hdf5

step_factor = 1

save_number = -1

PyUL.DispN(duration, duration_units, length, length_units, resol, step_factor,
           save_number)

Save_Options = 'Energy NBody DF'

save_options = PyUL.SaveOptionsDigest(Save_Options)

s_mass_unit = ''
s_position_unit = ''
s_velocity_unit = ''

m_mass_unit = ''
m_position_unit = ''
m_velocity_unit = ''

NSols = 8
MassMin = 9
MassMax = 15

GenRange = length/3

particles = []
embeds = []
solitons = []

VMax = 1

Momentum = np.zeros(3)
COM = np.zeros(3)

np.random.seed(seed = 100)

for i in range(NSols):
    
    Mass = np.random.random()*(MassMax-MassMin) + MassMin
    
    Position = np.random.random(3)*GenRange - GenRange/2
    
    Velocity = np.random.random(3)*VMax - VMax / 2
    
    Momentum += Mass * Velocity
    
    COM += Mass * Position
    
    solitons.append([Mass,Position.tolist(),Velocity.tolist(),np.random.random()*2*np.pi])
 
MF = 15

CorrP = -COM/MF
CorrV = -Momentum/MF

solitons.append([Mass,CorrP.tolist(),CorrV.tolist(),0])

COM += MF*CorrP

Momentum += MF*CorrV

gridspace = PyUL.MeshSpacing(resol,length,length_units, silent = True)
# PLUMMER RADIUS (IN LENGTH UNITS)
rP = gridspace / 2

a = PyUL.GenPlummer(rP,length_units)

NBo = 12
for i in range(NBo):
    
    Mass = (np.random.random()*(MassMax-MassMin) + MassMin)*0.05
    
    Position = np.random.random(3)*GenRange - GenRange/2
    
    Velocity = np.random.random(3)*VMax - VMax / 2
    
    particles.append([Mass,Position.tolist(),Velocity.tolist()])
 

printU(f"Initial condition is set-up using {NSols+1} solitons and {NBo} particles.",'Init')


Uniform = False
UVel = [0,0,0]
Density, density_unit = 0, ''

Name = ''

run_folder = PyUL.GenerateConfig(NS, length, length_units,
                                 resol, duration, duration_units, step_factor,
                                 save_number, Save_Options, save_path,
                                 save_format, s_mass_unit, s_position_unit,
                                 s_velocity_unit, solitons,start_time,
                                 m_mass_unit, m_position_unit, m_velocity_unit,
                                 particles,embeds, Uniform,Density,density_unit,
                                 a,UVel,True,Name)


Start_Time = time.time()

PyUL.evolve(save_path,
            run_folder, 
            EdgeClear = False,      # Reflexive Boundary Condition (Half Baked)
            DumpInit = False,       # Dump Initial Wavefunction
            DumpFinal = False,      # Dump Final Wavefunction
            UseInit = False,        # Use Initial Wavefunction (Specify in InitPath)
            IsoP = False,           # Zero Padded Potential
            UseDispSponge = False,  # Dispersive Boundary Condition
            SelfGravity = True,
            NBodyInterp = True,
            NBodyGravity = True,
            Shift = False,          # Shift NBody Location Between Grids (For QW)
            Simpson = False,        # Not Used.
            Silent = True,         # Quiet Mode
            AutoStop = False,       # Stop with BH
            AutoStop2 = False,      # Stop with Gravitational Field Strength
            WellThreshold = 100,    # Stop Condition
            InitPath = '',          # Everything except for the .npy file extension.
            InitWeight = 1,
            Stream = False,          # Write RK Step Results To File
            StreamChar = [])     # Locations in the vectorized TMState to Stream. (x-y flipped).

Duration = time.time()-Start_Time

sim_number = PyUL.ULDStepEst(duration,duration_units,
                                          length,length_units,
                                          resol,step_factor, 
                                          save_number = -1)

DPS = Duration/sim_number

printU('If desired, please copy and send the following text to the developer (as a GitHub ticket or via Email)','Bench')

print("="*40+'PyUL Benchmark Result')

uname = platform.uname()

print(f"System: {uname.system}")
print(f"Release: {uname.release}")
print(f"Version: {uname.version}")
print(f"Machine: {uname.machine}")
print(f"Processor: {uname.processor}")

print(f"\nAvailable CPU Threads: {multiprocessing.cpu_count()}")
print(f"Resolution: {resol}")
print(f"Avg Step (s): {DPS:.3g}")
print("="*80)

