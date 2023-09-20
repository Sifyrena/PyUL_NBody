Version   = str('PyUL') # Handle used in console.
D_version = str('Build 2023 March 07 Public') # Detailed Version
S_version = 26.32 # Short Version

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

from scipy.interpolate import CubicSpline as CS 
from scipy.special import sph_harm as SPH 

print("Importing h5py. If this functionality is not required, please comment out line #22 in the main program.")
import h5py

# Jupyter
from IPython.core.display import clear_output

num_threads = multiprocessing.cpu_count()

pi = np.pi

eV = 1.78266191e-36 # kg*c^2

# ULDM:

axion_E = float(input('Axion Mass (eV). Blank for 1e-22 eV') or 1e-22)


################################## DIALOG BOX #################################

SSLength = 4

def printU(Message,SubSys = 'Sys',ToScreen = True, ToFile = False, FilePath = ''):

    if ToScreen:
        print(f"{Version}.{SubSys.rjust(SSLength)}: {Message}")
        
    if ToFile and FilePath != "":
        with open(FilePath, "a+") as o:
            o.write(f"{GenFromTime()} {Version}.{SubSys.rjust(SSLength)}: {Message}\n")
    
printU(f"Axion Mass: {axion_E:.2g} eV.", 'Universe')

################################## CONSTANTS ##################################

axion_mass = axion_E * eV

hbar = 1.0545718e-34  # m^2 kg/s

parsec = 3.0857e16  # m

light_year = 9.4607e15  # m

solar_mass = 1.989e30  # kg

G = 6.67e-11  # kg

omega_m0 = 0.31

H_0 = 67.7 / (parsec * 1e3)  # s^-1

CritDens = 3*H_0**2/(8*pi*G)
# IMPORTANT

time_unit = (3 * H_0 ** 2 * omega_m0 / (8 * pi)) ** -0.5

length_unit = (8 * pi * hbar ** 2 / (3 * axion_mass ** 2 * H_0 ** 2 * omega_m0)) ** 0.25

mass_unit = (3 * H_0 ** 2 * omega_m0 / (8 * pi)) ** 0.25 * hbar ** 1.5 / (axion_mass ** 1.5 * G)

energy_unit = mass_unit * length_unit ** 2 / (time_unit**2)
    
#####################################################################################
### Internal Flags used for IO

SFS = '3Density 3Wfn 2Density Energy 1Density NBody 3Grav 2Grav DF 2Phase Entropy 1Grav 3GravF 2GravF 1GravF 3UMmt 2UMmt 1UMmt Momentum AngMomentum 3DensityRS 3WfnRS 2EnergyTot 2EnergyKQ 2EnergySI ULDCOM'

# .    0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25
SNM = 'R3D P3D R2D EGY R1D NTM G3D G2D DYF A2D ENT G1D F3D F2D F1D V3D V2D V1D PMT MVR R3R P3R E2T E2K E2G UCM'

SaveFlags = SFS.split()
SaveNames = SNM.split()
    
def IOName(Type):
    return SaveNames[SaveFlags.index(Type)]

def IOSave(loc,Type,save_num,save_format = 'npy',data = [], Old = False):
    
    file_name = f"{loc}/Outputs/{Type}/{IOName(Type)}_#{save_num:03d}.{save_format}"
    if Old:
        file_name = f"{loc}/Outputs/{IOName(Type)}_#{save_num:03d}.{save_format}"

    if save_format == 'npy':
        np.save(file_name,data)
    elif save_format == 'npz':
        np.savez(file_name,data)
    elif save_format == 'hdf5':
        
        f = h5py.File(file_name, 'w')
        dset = f.create_dataset("init", data=data)
        f.close()
    else:
        raise RuntimeError('Invalid output format specified!')
        
def IOLoad_npy(loc,Type,save_num):
    return np.load(f"{loc}/Outputs/{Type}/{IOName(Type)}_#{save_num:03d}.npy")

def IOLoad_h5(loc,Type,save_num):
    flname = f"{loc}/Outputs/{Type}/{IOName(Type)}_#{save_num:03d}.hdf5"
    f = h5py.File(flname,'r')
    b = f['init'][:]
    f.close()
    return np.array(b)
    
def IOLoad_npy_O(loc,Type,save_num):
    return np.load(f"{loc}/Outputs/{IOName(Type)}_#{save_num:03d}.npy")

def IOLoad_h5_O(loc,Type,save_num):
    flname = f"{loc}/Outputs/{IOName(Type)}_#{save_num:03d}.hdf5"
    f = h5py.File(flname,'r')
    b = f['init'][:]
    f.close()
    return np.array(b)

def CreateStream(loc, NS = 32, Target = 'Undefined', StreamChar = [0]):
    file = open(f'{loc}/NBStream.uldm', "w+")
    file.write(f'{Version}: NBody State Stream File.\nRK4 N body Steps Per ULDM Step: {NS//4:.0f}\nTarget v: {Target}\nVectorised TMState loci printed: {StreamChar}')
    file.close()
    
def NBStream(loc,Message):
    file = open(f'{loc}/NBStream.uldm', "a")
    file.write("\n")
    
    if type(Message) == np.ndarray:
        MesList = Message.tolist()
        
        for Mes in MesList:
            file.write(f'{Mes:.16f}, ')
    else:
        file.write(f'{Message}')
    file.close()
    
def NBDensity(loc,Density):
    file = open(f'{loc}/LocalDensities.uldm', "w+")
    file.write(f'{Density:.8f}')
    file.close()
    
def ReadLocalDensity(loc):
    with open(f'{loc}/LocalDensities.uldm', "r") as file:
        return float(file.read())

### AUX. FUNCTION TO GENERATE TIME STAMP

def GenFromTime():
    from datetime import datetime
    now = datetime.now() # current date and time
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    
    return timestamp

####################### AUX. FUNCTION TO GENERATE PROGRESS BAR
def prog_bar(iteration_number = 100, progress = 1, tinterval = 0 ,status = '',adtl = ''):
    size = 10
    
    if tinterval != 0:
        ETAStamp = time.time() + (iteration_number - progress)*tinterval

        ETA = datetime.fromtimestamp(ETAStamp).strftime("%d/%m/%Y, %H:%M:%S")
    
    else:
        ETA = '-'
        
    PROG = float(progress) / float(iteration_number)
    
    if PROG >= 1.:
        PROG, status = 1, ""
    
    block = int(round((size) * PROG))
    
    status = f'{status.ljust(SSLength)}'
  
    if block == 0:
        CM = '◎'
        PM = ''
        PL = 0
    else:
        CM = '○'
        PM = '◎' 
        PL = 1
    
    if block == size:
        PM = CM
        
    CL = 2*block - 2*PL + 1
    
    LL = size - block
    RL = size - block

    BarText = "●" * LL + PM * PL + CM * CL + PM * PL + "●" * RL
    
    shift = int(-1 * progress % (2*size+1))

    BarText = BarText[shift:-1] + BarText[0:shift]
    
    text = "\r[{}] {:.0f}% {}{}{} ({}{:.2f}s) {}".format(BarText
        , round(PROG * 100, 0),
        status, 'Exp. Time: ',ETA,'Prev.: ',tinterval,adtl)
   
    print(f'{text}', end="",flush='true')
    
    
def prog_bar_NG(iteration_number = 100, progress = 1, tinterval = 0 ,status = '',adtl = ''):
        
    if tinterval != 0:
        ETAStamp = time.time() + (iteration_number - progress)*tinterval

        ETA = datetime.fromtimestamp(ETAStamp).strftime("%d/%m/%Y, %H:%M:%S")
    
    else:
        ETA = '-'
        
    PROG = float(progress) / float(iteration_number)
    
    if PROG >= 1.:
        PROG, status = 1, ""
    
    text = f"{round(PROG * 100, 0)},{status}, Exp. Time: {ETA},'Prev.: ',{tinterval},{adtl}"
   
    printU(text,"")



####################### Credits Information
def PyULCredits(IsoP = False,UseDispSponge = False,embeds = []):
    print(f"==============================================================================")
    print(f"{Version}.{S_version}: (c) 2020 - 2021 Wang., Y. and collaborators. \nAuckland Cosmology Group\n") 
    print("Original PyUltraLight Team:\nEdwards, F., Kendall, E., Hotchkiss, S. & Easther, R.\n\
arxiv.org/abs/1807.04037")
    
    if IsoP or UseDispSponge or (embeds != []):
        print(f"\n**External Module In Use**")
    
    if IsoP:
        printU(f"Isolated ULDM Potential Implementation \nAdapted from J. L. Zagorac et al. Yale Cosmology",'External')
        
    if UseDispSponge:
        printU(f"Dispersive Sponge Condition \nAdapted from J. L. Zagorac et al. Yale Cosmology",'External')
        
    if embeds != []:
        printU(f"Embedded Soliton Profiles \nAdapted from N. Guo et al. Auckland Cosmology Group",'External')

    print(f"==============================================================================")

    

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
####################### GREEN FUNCTION DEFINITIONS FROM J. Luna Zagorac (Yale Cosmology)
# With FW Changes to Ensure Compatibility


####################### PADDED POTENTIAL FUNCTIONS

#safeInverse

def safeInverse3D(x):
    # Safe inverse for 3D arrays
    
    y = x*0
    for ind in np.ndindex(x.shape):
        if x[ind] > 0:
                    #x[i, j] = 1.0 / float(x[i, j])
                    y[ind] = np.reciprocal(x[ind])
    return y

# greenArray 

def greenArray(xyz):
    # make grid for Green's function
    x2, y2, z2 = np.meshgrid(
        xyz, xyz, xyz, 
        sparse=True, indexing='ij',
    )
    gplane = ne.evaluate("x2 + y2 + z2")
    return(gplane)

# makeDCTGreen

def makeDCTGreen(n):
    # make Green's function using a DCT
    x2 = np.arange(0, n + 1)**2
    arr = greenArray(x2)
    arr = np.sqrt(arr)
    arr = safeInverse3D(arr)
    arr[0,0,0] = 3*np.log(np.sqrt(3) + 2) - np.pi/2 #calculated in https://arxiv.org/pdf/chem-ph/9508002.pdf
    # no normalization below matches un-normalization in Mathematica
    import scipy.fft
    with scipy.fft.set_workers(num_threads):
        green =  scipy.fft.dctn(arr, type = 1, norm = None)
    return(green)
    
#makeEvenArray

def makeEvenArray(arr):
    # defined for a square 2D array
    # doubles the size of the array to make it even
    # this allows us to use the DCT
    nRow, nCol = np.shape(arr)
    # first do rows 
    arow = np.flip(arr[1:nRow-1, :], axis = 0)
    arr = np.concatenate((arr, arow))
    #now do columns
    acol = np.flip(arr[:, 1:nCol-1], axis = 1)
    arr = np.concatenate((arr, acol), axis = 1)
    #return
    return(arr)

#Plane Convolve2

def planeConvolve(green, plane, n, fft_plane, ifft_plane):
    # Convolve Green's function for each plane
    bigplane = np.pad(plane, ((0,n), (0,n)), 'constant', constant_values=0)
    green1 = makeEvenArray(green) # defined above
    temp = -1 * green1 * fft_plane(bigplane) 
    bigplane = ifft_plane(temp)
    bigplane = bigplane[:-n, :-n]
    return(bigplane)

#Isolated Potential2

def isolatedPotential(rho, green, l, fft_X, ifft_X, fft_plane, ifft_plane):
    # the main function
    # pads x-direction
    # then takes each plane, pads it, and convolves it with the appropriate Green's function plane
    n,n,n = np.shape(rho)
    rhopad = np.pad(rho, ((0,n), (0,0), (0,0)), 'constant', constant_values=0)
    #rhopad = scipy.fftpack.fftn(rhopad, axes = (0,)) #transform x-axis
    rhopad = fft_X(rhopad)
    # ndx is the index for green (so we don't have to waste memory storing EvenGreen)
    ndx = np.concatenate((np.arange(n + 1), np.flip(np.arange(1,n)) ))
    # the for loop
    for i in range(0,2*n):
        plane = rhopad[i, :, :]
        rhopad[i, :, :] = PC_jit(green[ndx[i], :, :], plane, n, fft_plane, ifft_plane)
    # - - - - - - - - - - - - - - - - - - - - - -    
    #rhopad = (1/n**2) * scipy.fftpack.ifftn(rhopad, axes = (0,)) #inverse transform x-axis
    rhopad = (1/n**2) * ifft_X(rhopad) # normalization
    rho = (l**2) * rhopad[:-n, :, :] #potential has units of length^2 in this system
    return(rho.real)



####################### CONVERTING PARAMS TO CODE UNITS

def convert(value, unit, type):
    converted = 0
    if (type == 'l'):
        if (unit == ''):
            converted = value
        elif (unit == 'm') or (unit == 'SI'):
            converted = value / length_unit
        elif (unit == 'km'):
            converted = value * 1e3 / length_unit
        elif (unit == 'pc'):
            converted = value * parsec / length_unit
        elif (unit == 'kpc'):
            converted = value * 1e3 * parsec / length_unit
        elif (unit == 'Mpc'):
            converted = value * 1e6 * parsec / length_unit
        elif (unit == 'ly'):
            converted = value * light_year / length_unit
        else:
            raise NameError('Unsupported LENGTH unit used')

    elif (type == 'm'):
        if (unit == ''):
            converted = value
        elif (unit == 'kg') or (unit == 'SI'):
            converted = value / mass_unit
        elif (unit == 'solar_masses'):
            converted = value * solar_mass / mass_unit
        elif (unit == 'M_solar_masses'):
            converted = value * solar_mass * 1e6 / mass_unit
        else:
            raise NameError('Unsupported MASS unit used')

    elif (type == 't'):
        if (unit == ''):
            converted = value
        elif (unit == 's') or (unit == 'SI'):
            converted = value / time_unit
        elif (unit == 'yr'):
            converted = value * 60 * 60 * 24 * 365 / time_unit
        elif (unit == 'kyr'):
            converted = value * 60 * 60 * 24 * 365 * 1e3 / time_unit
        elif (unit == 'Myr'):
            converted = value * 60 * 60 * 24 * 365 * 1e6 / time_unit
        elif (unit == 'Gyr'):
            converted = value * 60 * 60 * 24 * 365 * 1e9 / time_unit
        else:
            raise NameError('Unsupported TIME unit used')

    elif (type == 'v'):
        if (unit == ''):
            converted = value
        elif (unit == 'm/s') or (unit == 'SI'):
            converted = value * time_unit / length_unit
        elif (unit == 'km/s'):
            converted = value * 1e3 * time_unit / length_unit
        elif (unit == 'km/h'):
            converted = value * 1e3 / (60 * 60) * time_unit / length_unit
        elif (unit == 'c'):
            converted = value * time_unit / length_unit * 299792458
        else:
            raise NameError('Unsupported SPEED unit used')
            
            
    elif (type == 'd'):
        if (unit == ''):
            converted = value
        elif (unit == 'Crit'):
            converted = value / omega_m0 
        elif (unit == 'MSol/pc3'):
            converted = value * solar_mass / mass_unit * length_unit**3 / parsec**3     
        elif (unit == 'MMSol/kpc3'):
            converted = value * solar_mass / mass_unit * length_unit**3 / parsec**3  / 1000  
        elif (unit == 'kg/m3') or (unit == 'SI'):
            converted = value / mass_unit * length_unit**3
        else:
            raise NameError('Unsupported DENSITY unit used')
            
            
    elif (type == 'a'):
        if (unit == ''):
            converted = value
        elif (unit == 'm/s2') or (unit == 'SI'):
            converted = value / length_unit * time_unit**2   
        else:
            raise NameError('Unsupported ACCELERATION unit used')

    else:
        raise TypeError('Unsupported conversion type')

    return converted



####################### FUNCTION TO CONVERT FROM DIMENSIONLESS UNITS TO DESIRED UNITS

def convert_back(value, unit, type):
    converted = 0
    if (type == 'l'):
        if (unit == ''):
            converted = value
        elif (unit == 'm') or (unit == 'SI'):
            converted = value * length_unit
        elif (unit == 'km'):
            converted = value / 1e3 * length_unit
        elif (unit == 'pc'):
            converted = value / parsec * length_unit
        elif (unit == 'kpc'):
            converted = value / (1e3 * parsec) * length_unit
        elif (unit == 'Mpc'):
            converted = value / (1e6 * parsec) * length_unit
        elif (unit == 'ly'):
            converted = value / light_year * length_unit
        else:
            raise NameError('Unsupported LENGTH unit used')

    elif (type == 'm'):
        if (unit == ''):
            converted = value
        elif (unit == 'kg') or (unit == 'SI'):
            converted = value * mass_unit
        elif (unit == 'solar_masses'):
            converted = value / solar_mass * mass_unit
        elif (unit == 'M_solar_masses'):
            converted = value / (solar_mass * 1e6) * mass_unit
        else:
            raise NameError('Unsupported MASS unit used')

    elif (type == 't'):
        if (unit == ''):
            converted = value
        elif (unit == 's') or (unit == 'SI'):
            converted = value * time_unit
        elif (unit == 'yr'):
            converted = value / (60 * 60 * 24 * 365) * time_unit
        elif (unit == 'kyr'):
            converted = value / (60 * 60 * 24 * 365 * 1e3) * time_unit
        elif (unit == 'Myr'):
            converted = value / (60 * 60 * 24 * 365 * 1e6) * time_unit
        elif (unit == 'Gyr'):
            converted = value / (60 * 60 * 24 * 365 * 1e9) * time_unit
        else:
            raise NameError('Unsupported TIME unit used')

    elif (type == 'v'):
        if (unit == ''):
            converted = value
        elif (unit == 'm/s') or (unit == 'SI'):
            converted = value / time_unit * length_unit
        elif (unit == 'km/s'):
            converted = value / (1e3) / time_unit * length_unit
        elif (unit == 'km/h'):
            converted = value / (1e3) * (60 * 60) / time_unit * length_unit
        elif (unit == 'c'):
            converted = value * time_unit / length_unit / 299792458
        else:
            raise NameError('Unsupported SPEED unit used')
            
            
    elif (type == 'd'):
        if (unit == ''):
            converted = value
        elif (unit == 'Crit'):
            converted = value * omega_m0 
        elif (unit == 'MSol/pc3'):
            converted = value / solar_mass * mass_unit / length_unit**3 * parsec**3
        elif (unit == 'MMSol/kpc3'):
            converted = value / solar_mass * mass_unit / length_unit**3 * parsec**3 * 1000
        elif (unit == 'kg/m3') or (unit == 'SI'):
            converted = value * mass_unit / length_unit**3
        else:
            raise NameError('Unsupported DENSITY unit used')

    elif (type == 'a'):
        if (unit == ''):
            converted = value
        elif (unit == 'm/s2') or (unit == 'SI') :
            converted = value * length_unit / time_unit**2   
        else:
            raise NameError('Unsupported ACCELERATION unit used')        
            
    else:
        raise TypeError('Unsupported conversion type')

    return converted



def convert_between(value, oldunit,newunit, type):
    
    return convert_back(convert(value,oldunit,type),newunit,type)

########################FUNCTION TO CHECK FOR SOLITON OVERLAP

def overlap_check(candidate, soliton):
    for i in range(len(soliton)):
        m = max(candidate[0], soliton[i][0])
        d_sol = 5.35854 / m
        c_pos = np.array(candidate[1])
        s_pos = np.array(soliton[i][1])
        displacement = c_pos - s_pos
        distance = np.sqrt(displacement[0] ** 2 + displacement[1] ** 2 + displacement[2] ** 2)
        if (distance < 2 * d_sol):
            return False
    return True


############################FUNCTION TO PUT SPHERICAL SOLITON DENSITY PROFILE INTO 3D BOX (Uses pre-computed array)


def InitSolitonF(gridVec, position, resol, alpha, delta_x=0.00001, DR = 1):

    xAr, yAr, zAr = np.meshgrid(gridVec - position[0],
                                gridVec - position[1],
                                gridVec - position[2],
                                sparse=True,
                                indexing="ij")

    gridSize = gridVec[1] - gridVec[0]
    DistArr = ne.evaluate("sqrt(xAr**2+yAr**2+zAr**2 * DR)")

    f = alpha * LoadDefaultSoliton()
    fR = np.arange(len(f)) * delta_x / np.sqrt(alpha)

    fInterp = CS(fR, f, bc_type=("clamped", "not-a-knot"))

    DistArrPts = DistArr.reshape(resol**3)

    Eval = fInterp(DistArrPts)
    
    Eval[DistArrPts > fR[-1]] = 0 # Fix Cubic Spline Behaviour

    return Eval.reshape(resol,resol,resol)


def initsoliton(funct, xarray, yarray, zarray, position, alpha, f, delta_x,Cutoff = 9, DR = 1):
    
    funct *= 0
    
    for index in np.ndindex(funct.shape):
        
        
        # Note also that this distfromcentre is here to calculate the distance of every gridpoint from the centre of the soliton, not to calculate the distance of the soliton from the centre of the grid
        distfromcentre = (
            (xarray[index[0], 0, 0] - position[0]) ** 2 +
            (yarray[0, index[1], 0] - position[1]) ** 2 +
            (zarray[0, 0, index[2]] - position[2]) ** 2 ) ** 0.5 * DR
        # Utilises soliton profile array out to dimensionless radius.
        if (np.sqrt(alpha) * distfromcentre <= Cutoff):
         
            funct[index] = alpha * f[int(np.sqrt(alpha) * (distfromcentre / delta_x + 1))]

    return funct


############################FUNCTION TO PUT SPHERICAL SOLITON DENSITY PROFILE INTO 3D BOX (Uses pre-computed array)

def initsolitonRadial(line, alpha, f, delta_x,Cutoff = 9):
    funct = 0*line
    
    for index in np.ndindex(funct.shape):
        
        
        # Note also that this distfromcentre is here to calculate the distance of every gridpoint from the centre of the soliton, not to calculate the distance of the soliton from the centre of the grid
        distfromcentre = (
            (line[index[0]]) ** 2) ** 0.5
        # Utilises soliton profile array out to dimensionless radius 5.6.
        if (np.sqrt(alpha) * distfromcentre <= Cutoff):
         
            funct[index] = alpha * f[int(np.sqrt(alpha) * (distfromcentre / delta_x + 1))]

    return funct

"""
Save various properties of the various grids in various formats
"""

def save_grid(
        rho, psi, resol, 
        TMState, phiSP, phi, GradientLog,
        save_options,
        save_format,
        loc, ix, its_per_save
        ):

        save_num = int((ix + 1) / its_per_save)
        
        if (save_options[0]): # 3Density
            
            IOSave(loc,'3Density',save_num,save_format,rho)
                       
        if (save_options[1]): # 3Wfn
            
            IOSave(loc,'3Wfn',save_num,save_format,psi)       
            
        if (save_options[2]): # 2Density
            
            plane = rho[:, :, resol // 2]
            IOSave(loc,'2Density',save_num,save_format,plane)

        if (save_options[4]): # 1Density Now Saving to Simulated y (Paper x axis)!
            
            line = rho[resol // 2, :, resol // 2]  
            IOSave(loc,'1Density',save_num,save_format,line)
            
        if (save_options[5]): # TM
            
            IOSave(loc,'NBody',save_num,save_format,TMState)
            
        if (save_options[6]): # 3Grav
            
            IOSave(loc,'3Grav',save_num,save_format,phiSP)           
                
        if (save_options[7]): # 2Grav
            
            phiSP_slice = phiSP[:,:,int(resol/2)] # z = 0
            
            IOSave(loc,'2Grav',save_num,save_format,phiSP_slice)             
                
        if (save_options[8]): # DF
           
            IOSave(loc,'DF',save_num,save_format,GradientLog)
            
        if (save_options[9]): # 2Phase

            psislice = psi[:, :, resol // 2]
            
            argplane = np.angle(psislice)
            
            IOSave(loc,'2Phase',save_num,save_format,argplane)         
            
        if (save_options[11]): # 1Grav

            phiSP_line = phiSP[int(resol/2),:,int(resol/2)] # z = 0
            
            IOSave(loc,'1Grav',save_num,save_format,phiSP_line)
            
        if (save_options[12]): # 3GravF
            IOSave(loc,'3GravF',save_num,save_format,phi)
                
        if (save_options[13]): # 2GravF
            phi_slice = phi[:,:,int(resol/2)]
            
            IOSave(loc,'2GravF',save_num,save_format,phi_slice)
                
        if (save_options[14]): # 1GravF
            phi_line = phi[int(resol/2),:,int(resol/2)]
            
            IOSave(loc,'1GravF',save_num,save_format,phi_line)


        
# CALCULATE_ENERGIES, NEW VERSION IN 2.16
            
def calculate_energies(rho, Vcell, phiSP,phiTM, psi, karray2, fft_psi, ifft_funct, Density,Uniform, egpcmlist, egpsilist, ekandqlist, egylist, mtotlist, resol, save_grid_E = False):

    rho = rho.real
    
    #if Uniform:
    #    BoxAvg = np.mean(rho) # SHOULD BE TEMPORARILY DISABLED!
    #else:
    
    BoxAvg = 0

    # Gravitational potential energy density associated with the point masses potential

    ETM = ne.evaluate('phiTM*(rho-BoxAvg)') # Interaction in particle potential!
    ETMtot = Vcell * np.sum(ETM)
    egpcmlist.append(ETMtot) # TM Saved.

    # Gravitational potential energy density of self-interaction of the condensate
    ESI = ne.evaluate('0.5*(phiSP)*(rho-BoxAvg)') # New!
    ESItot = Vcell * np.sum(ESI)
    egpsilist.append(ESItot)
    
    Etot = ETMtot + ESItot # Begin gathering!

    # TODO: Does this reuse the memory of funct?  That is the
    # intention, but likely isn't what is happening
    funct = fft_psi(psi)
    funct = ne.evaluate('-karray2*funct')
    funct = ifft_funct(funct)
    EKQ = ne.evaluate('real(-0.5*conj(psi)*funct)')
    EKQtot = Vcell * np.sum(EKQ)
    
    ekandqlist.append(EKQtot)
    Etot += EKQtot

    egylist.append(Etot)

    # Total mass compared to background.
    Mtot = np.sum(rho)*Vcell
    mtotlist.append(Mtot)

    if save_grid_E:
        EGrid = ne.evaluate("ETM + ESI + EKQ")[:,:,resol//2]
        return EGrid, EKQ[:,:,resol//2], ESI[:,:,resol//2]
    else:
        return [], [], []
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

def InterpolateLocal(RRem,Input):
        
    while len(RRem) > 1:
                    
        Input = Input[1,:]*RRem[0] + Input[0,:]*(1-RRem[0])
        RRem = RRem[1:]
        InterpolateLocal(RRem,Input)
        
    else:
        return Input[1]*RRem + Input[0]*(1-RRem)

def FWNBody(t,TMState,masslist,phiSP,a,lengthC,resol):

    GridDist = lengthC/resol
    
    dTMdt = 0*TMState
    GradientLog = np.zeros(len(masslist)*3)
    
    for i in range(len(masslist)):
        
        #0,6,12,18,...
        Ind = int(6*i)
        IndD = int(3*i)
           
        #X,Y,Z
        poslocal = TMState[Ind:Ind+3]

        RNum = (poslocal*1/lengthC+1/2)*resol

        RPt = np.floor(RNum)
        RRem = RNum - RPt
        
        # Need special treatment if any of these is zero or close to resol!

        
        RPtX = int(RPt[0])
        RPtY = int(RPt[1])
        RPtZ = int(RPt[2])

        if (RPtX <= 0) or (RPtY <= 0) or (RPtZ <= 0):
            #raise RuntimeError (f'Particle #{i} reached boundary on the -ve side. Halting.')
            TAr = np.zeros([4,4,4])

            GradientX = 0
            GradientY = 0
            GradientZ = 0        

        elif (RPtX >= resol-4) or (RPtY >= resol-4) or (RPtZ >= resol-4):
            #raise RuntimeError (f'Particle #{i} reached boundary on the +ve side. Halting.')
            TAr = np.zeros([4,4,4])
            
            GradientX = 0
            GradientY = 0
            GradientZ = 0

        else:   
            
            TAr = phiSP[RPtX-1:RPtX+3,RPtY-1:RPtY+3,RPtZ-1:RPtZ+3] # 64 Local Grids

            GArX = (TAr[2:4,1:3,1:3] - TAr[0:2,1:3,1:3])/(2*GridDist) # 8

            GArY = (TAr[1:3,2:4,1:3] - TAr[1:3,0:2,1:3])/(2*GridDist) # 8

            GArZ = (TAr[1:3,1:3,2:4] - TAr[1:3,1:3,0:2])/(2*GridDist) # 8

            GradientX = InterpolateLocal(RRem,GArX)

            GradientY = InterpolateLocal(RRem,GArY)

            GradientZ = InterpolateLocal(RRem,GArZ)

        #XDOT
        dTMdt[Ind]   =  TMState[Ind+3]
        #YDOT
        dTMdt[Ind+1] =  TMState[Ind+4]
        #ZDOT
        dTMdt[Ind+2] =  TMState[Ind+5]
        
        #x,y,z
        
        GradientLocal = -1*np.array([[GradientX],[GradientY],[GradientZ]])

        #Initialized Against ULDM Field
        #XDDOT
        dTMdt[Ind+3] =  GradientLocal[0]
        #YDDOT
        dTMdt[Ind+4] =  GradientLocal[1]
        #ZDDOT
        dTMdt[Ind+5] =  GradientLocal[2]
    
        for ii in range(len(masslist)):
            
            if (ii != i) and (masslist[ii] != 0):
                
                IndX = int(6*ii)
                
                # print(ii)
                
                poslocalX = np.array([TMState[IndX],TMState[IndX+1],TMState[IndX+2]])
                
                rV = poslocalX - poslocal
                
                rVL = np.linalg.norm(rV) # Positive
                
                
                if a == 0:
                    F = 1/(rVL)**3
                else:                    
                    F = -(a**3)/(a**2*rVL**2+1)**(1.5) # The First Plummer
                
                # Differentiated within Note 000.0F
                
                #XDDOT with Gravity
                dTMdt[Ind+3] = dTMdt[Ind+3] - masslist[ii]*F*rV[0]
                #YDDOT
                dTMdt[Ind+4] = dTMdt[Ind+4] - masslist[ii]*F*rV[1]
                #ZDDOT
                dTMdt[Ind+5] = dTMdt[Ind+5] - masslist[ii]*F*rV[2]
        
        GradientLog[IndD  ] = GradientLocal[0]
        GradientLog[IndD+1] = GradientLocal[1]
        GradientLog[IndD+2] = GradientLocal[2]

    return dTMdt, GradientLog


def FWNBody_NI(t,TMState,masslist,phiSP,a,lengthC,resol):

    GridDist = lengthC/resol
    
    dTMdt = 0*TMState
    GradientLog = np.zeros(len(masslist)*3)
    
    for i in range(len(masslist)):
        
        #0,6,12,18,...
        Ind = int(6*i)
        IndD = int(3*i)
           
        #X,Y,Z
        poslocal = TMState[Ind:Ind+3]

        RNum = (poslocal*1/lengthC+1/2)*resol

        RPt = np.floor(RNum)
        RRem = RNum - RPt
        
        # Need special treatment if any of these is zero or close to resol!

        
        RPtX = int(RPt[0])
        RPtY = int(RPt[1])
        RPtZ = int(RPt[2])

        if (RPtX <= 0) or (RPtY <= 0) or (RPtZ <= 0):
            #raise RuntimeError (f'Particle #{i} reached boundary on the -ve side. Halting.')
            TAr = np.zeros([4,4,4])

            GradientX = 0
            GradientY = 0
            GradientZ = 0        

        elif (RPtX >= resol-4) or (RPtY >= resol-4) or (RPtZ >= resol-4):
            #raise RuntimeError (f'Particle #{i} reached boundary on the +ve side. Halting.')
            TAr = np.zeros([4,4,4])
            
            GradientX = 0
            GradientY = 0
            GradientZ = 0

        else:   
            
            TAr = phiSP[RPtX-1:RPtX+3,RPtY-1:RPtY+3,RPtZ-1:RPtZ+3] # 64 Local Grids

            GArX = (TAr[2:4,1:3,1:3] - TAr[0:2,1:3,1:3])/(2*GridDist) # 8

            GArY = (TAr[1:3,2:4,1:3] - TAr[1:3,0:2,1:3])/(2*GridDist) # 8

            GArZ = (TAr[1:3,1:3,2:4] - TAr[1:3,1:3,0:2])/(2*GridDist) # 8

            GradientX = InterpolateLocal(RRem,GArX)

            GradientY = InterpolateLocal(RRem,GArY)

            GradientZ = InterpolateLocal(RRem,GArZ)

        #XDOT
        dTMdt[Ind]   =  TMState[Ind+3]
        #YDOT
        dTMdt[Ind+1] =  TMState[Ind+4]
        #ZDOT
        dTMdt[Ind+2] =  TMState[Ind+5]
        
        #x,y,z
        
        GradientLocal = -1*np.array([[GradientX],[GradientY],[GradientZ]])

        #Initialized Against THE VOID
        #XDDOT
        dTMdt[Ind+3] =  0
        #YDDOT
        dTMdt[Ind+4] =  0
        #ZDDOT
        dTMdt[Ind+5] =  0
    
        for ii in range(len(masslist)):
            
            if (ii != i) and (masslist[ii] != 0):
                
                IndX = int(6*ii)
                
                # print(ii)
                
                poslocalX = np.array([TMState[IndX],TMState[IndX+1],TMState[IndX+2]])
                
                rV = poslocalX - poslocal
                
                rVL = np.linalg.norm(rV) # Positive

                if a == 0:
                    F = 1/(rVL)**3
                else:                    
                    F = -(a**3)/(a**2*rVL**2+1)**(1.5) # The First Plummer
                
                # Differentiated within Note 000.0F
                
                #XDDOT
                dTMdt[Ind+3] = dTMdt[Ind+3] - masslist[ii]*F*rV[0]
                #YDDOT
                dTMdt[Ind+4] = dTMdt[Ind+4] - masslist[ii]*F*rV[1]
                #ZDDOT
                dTMdt[Ind+5] = dTMdt[Ind+5] - masslist[ii]*F*rV[2]
        
        GradientLog[IndD  ] = GradientLocal[0]
        GradientLog[IndD+1] = GradientLocal[1]
        GradientLog[IndD+2] = GradientLocal[2]

    return dTMdt, GradientLog


FWNBody3 = FWNBody
FWNBody3_NI = FWNBody_NI

def NBodyAdvance(TMState,h,masslist,phiSP,a,lengthC,resol,NS,loc = '',Stream = False, StreamChar = ''):
        #
        
        
        if NS == 0: # NBody Dynamics Off
            StateLen = len(TMState)
            
            GradientLog = np.zeros_like(TMState)
            
            for i in range(StateLen//6):
                
                for j in range(3):    
                    TMState[6*i+j] += h * TMState[6*i+j+3]
            
            
            return TMState, GradientLog
        
        if NS == 1:
 
            Step, GradientLog = FWNBody3(0,TMState,masslist,phiSP,a,lengthC,resol)
            TMStateOut = TMState + Step*h
            
            return TMStateOut, GradientLog
        
        elif NS%4 == 0:
            
            NRK = int(NS/4)
            
            H = h/NRK
            
            for RKI in range(NRK):
                TMK1, Trash = FWNBody3(0,TMState,masslist,phiSP,a,lengthC,resol)
                TMK2, Trash = FWNBody3(0,TMState + H/2*TMK1,masslist,phiSP,a,lengthC,resol)
                TMK3, Trash = FWNBody3(0,TMState + H/2*TMK2,masslist,phiSP,a,lengthC,resol)
                TMK4, GradientLog = FWNBody3(0,TMState + H*TMK3,masslist,phiSP,a,lengthC,resol)
                TMState = TMState + H/6*(TMK1+2*TMK2+2*TMK3+TMK4)
                
                if Stream:
                    NBStream(loc,TMState[StreamChar])
                
            TMStateOut = TMState

            return TMStateOut, GradientLog


def NBodyAdvance_NI(TMState,h,masslist,phiSP,a,lengthC,resol,NS):
        #
        if NS == 0: # NBody Dynamics Off
            
            StateLen = len(TMState)
            
            GradientLog = np.zeros_like(TMState)
            
            for i in range(StateLen//6):
                
                for j in range(3):    
                    TMState[6*i+j] += h * TMState[6*i+j+3]
            
            
            return TMState, GradientLog
        
        
        if NS == 1:
 
            Step, GradientLog = FWNBody3_NI(0,TMState,masslist,phiSP,a,lengthC,resol)
            TMStateOut = TMState + Step*h
            
            return TMStateOut, GradientLog
        
        elif NS%4 == 0:
            
            NRK = int(NS/4)
            
            H = h/NRK
            
            for RKI in range(NRK):
                TMK1, Trash = FWNBody3_NI(0,TMState,masslist,phiSP,a,lengthC,resol)
                TMK2, Trash = FWNBody3_NI(0,TMState + H/2*TMK1,masslist,phiSP,a,lengthC,resol)
                TMK3, Trash = FWNBody3_NI(0,TMState + H/2*TMK2,masslist,phiSP,a,lengthC,resol)
                TMK4, GradientLog = FWNBody3_NI(0,TMState + H*TMK3,masslist,phiSP,a,lengthC,resol)
                TMState = TMState + H/6*(TMK1+2*TMK2+2*TMK3+TMK4)
            
            TMStateOut = TMState

            return TMStateOut, GradientLog

######################### Early June Addition (Momentum)


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



def LpEvalFast(psi,rho, funct,resol,Uarray,Kx,Ky,Kz,ifft_funct):
 
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
        max_radius = 9
        tolerance = 1e-9 # how close f(r) must be to 0 at r_max; can be changed
        
    else:
        dr = .001
        max_radius = 9 
        tolerance = 1e-7 # how close f(r) must be to 0 at r_max; can be changed
        
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

IP_jit = numba.jit(isolatedPotential)

PC_jit = numba.jit(planeConvolve)
 
Lp_jit = LpEvalFast
    
######################### New Version With Built-in I/O Management
######################### Central Function
######################### Hello There!
######################### With Code Adapted from Yale Cosmology. Full information please see LICENSE.


# New IO Functions

def ULDump(loc,psi,TMState,Status):

    np.save(f'./{loc}/{Status}_psi.npy',psi)
    np.save(f'./{loc}/{Status}_TM.npy',TMState)
    return 1
    
def ULRead(InitPath):
    
    psi = np.load(f'{InitPath}_psi.npy')
    
    return psi

def evolve(save_path,run_folder, EdgeClear = False, DumpInit = False, DumpFinal = False, UseInit = False, IsoP = False, UseDispSponge = False, SelfGravity = True, NBodyInterp = True, NBodyGravity = True, Shift = False, Simpson = False, Silent = False, AutoStop = False, AutoStop2 = False,AutoStop3 = False, KEThreshold = 0.9, WellThreshold = 100, InitPath = '', InitWeight = 1, Message = '', Stream = False, StreamChar = [0], GenerateLog = True, CenterCalc = False, NLM = False, Length_Ratio = 0.5, resolR = 64, PrintEK = True, DR = 1):
    
    if run_folder == "":
        printU('Nothing done!','Evolve')
        return
    
    # clear_output()

    Draft = True

    Method = 3 # Backward Compatibility
    
    loc = './' + save_path + '/' + run_folder
        
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

    file = open('{}{}{}'.format('./', save_path, '/latest.uldm'), "w+")
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
    
    LogLocation = f"./{save_path}/{run_folder}/evolve_{GenFromTime()}.log"
    
    PyULConfig["Misc. Settings"] = ({"Custom Soliton Draft Quallity": Draft})
    
    with open(f'./{save_path}/{run_folder}/reproducibility.uldm', "w+") as outfile:
        json.dump(PyULConfig, outfile,indent=4)

    
    NS, length, length_units, resol, duration, duration_units, step_factor, save_number, save_options, save_format, s_mass_unit, s_position_unit, s_velocity_unit, solitons,start_time, m_mass_unit, m_position_unit, m_velocity_unit, particles, embeds, Uniform,Density, density_unit,a, UVel = LoadConfig(loc)

    for SaveName in SaveOptionsCompile(save_options).split():
        os.mkdir(str(loc + '/Outputs/'+SaveName))        
    num_threads = multiprocessing.cpu_count()
    
    if resol <= 100:
        num_threads = np.min([num_threads//2,8])
        
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
        printU(f"Added a pre-generated wavefunction with pseudorandom phase.",'Init', ToFile= GenerateLog, FilePath= LogLocation)
        
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
            print(f'Generating profile for Soliton with MBH/MSoliton = {RatioBU:.4f}, Part 1', ToFile= GenerateLog, FilePath= LogLocation)
            GMin,GMax = BHGuess(RatioBU)
            s, BHMass = BHRatioTester(RatioBU,30,1e-6,GMin,GMax,a)
            print(f'Generating profile for Soliton with MBH/MSoliton = {RatioBU:.4f}, Part 2', ToFile= GenerateLog, FilePath= LogLocation)
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
            Resample3Box(psi, LocCOM, gridvec, loc, 0, save_format, Length_Ratio, resolR, Save_Rho = save_options[20], Save_Psi = save_options[21])


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
        
        Cutoff = (resol//8)
        
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
        phiSP = IP_jit(rho, green, lengthC, fft_X, ifft_X, fft_plane, ifft_plane)
        
    else:
        printU(f"Poisson Equation Solveed Using FFT.",'SP', ToFile= GenerateLog, FilePath= LogLocation)
        phiSP = irfft_phi(phik)
        
    ##########################################################################################
       
    # NBody
    EGPCM = 0
    for MI, particle in enumerate(particles):
               
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
    
    tBegin = time.time()
    
    tBeginDisp = datetime.fromtimestamp(tBegin).strftime("%d/%m/%Y, %H:%M:%S")
    

    ########################################################################################## 
    # From Yale Cosmology.
  
    if UseDispSponge:
        
        SpongeRatio = 1/2
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
        PreMult[distarray<=rp] = 0
        
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
    
    if warn == 1:
        print("WARNING: Detected significant overlap between solitons in I.V.")
    print('\n')
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


        if UseDispSponge:
            prog_bar(actual_num_steps, ix + 1, tint,'DS',PBEDisp)
            psi *= np.exp(-PreMult*h)
        
        rho = ne.evaluate("abs(abs(psi)**2)").real
        
        if CenterCalc:
            LocCOM = Find3BoxCOM(rho,xarray, yarray, zarray)
            printU(LocCOM, "COM", ToScreen = False, ToFile= GenerateLog, FilePath= LogLocation)
            
            
        phik = rfft_rho(rho)  # not actually phik but phik is defined in next line
            
        phik = ne.evaluate("-4*pi*phik/rkarray2")

        phik[0, 0, 0] = 0        

        prog_bar(actual_num_steps, ix + 1, tint,'SP',PBEDisp)
        # New Green Function Methods
        if not IsoP:
            phiSP = irfft_phi(phik)
        else:
            phiSP = IP_jit(rho, green, lengthC, fft_X, ifft_X, fft_plane, ifft_plane)

        # FW STEP MAGIC HAPPENS HERE
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
            prog_bar(actual_num_steps, ix + 1, tint,'IO',PBEDisp)
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
            
            if CenterCalc:
                Resample3Box(psi, LocCOM, gridvec, loc, int((ix + 1) / its_per_save), save_format, Length_Ratio, resolR, Save_Rho = save_options[20], Save_Psi = save_options[21])
            
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
    
    
    with open('{}{}{}'.format('./',save_path, '/latest.uldm'), 'r') as timestamp:
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
        

    
def SaveOptionsCompile(save_options):
    result = ''

    for i in zip(save_options, SaveFlags):
        if i[0]:
            result += (i[1]+' ')
            
    return result

def SaveOptionsDigest(OptionsText):
    
    save_options = np.zeros(len(SaveFlags), dtype = bool)
    
    if OptionsText == 'Minimum':
        OptionsText = 'Energy 1Density NBody DF Entropy'

    OList = OptionsText.split()
    
    for Word in OList:
        save_options[SaveFlags.index(Word)] = True
    
    return save_options.tolist()
 
def GenerateConfig(NS, length, length_units, resol, duration, duration_units, step_factor, save_number, save_options, save_path, save_format, s_mass_unit, s_position_unit, s_velocity_unit, solitons,start_time, m_mass_unit, m_position_unit, m_velocity_unit, particles,embeds, Uniform,Density,density_unit,a,UVel,NoInteraction = False,Name = ''
           ):

    tm = GenFromTime()
    if NoInteraction:

        if Name != '':
            timestamp = Name
            
        else:
            timestamp = tm + f'@{resol}'
   
    else:
        print("What is the name of the run? Blank to use generated timestamp. ABORT to cancel compilation.")
        InputName = input()
        
        if InputName == "ABORT":
            return ""
        
        elif InputName != "":
            timestamp = InputName
        else:
            timestamp = tm + f'@{resol}'

    Dir = f"./{save_path}/{timestamp}"
      
        
    try:
        os.makedirs(Dir)
    except(FileExistsError):
        import shutil
        if NoInteraction:
            shutil.rmtree(Dir)
            os.mkdir(Dir)
            
        else: 
            print(f"{Version} IO: Folder Not Empty. Are you sure you want to proceed [Y/n]? \nTo reuse the pre-existing config file, type [C].")
              
            Protect = str(input())
        
            if Protect == 'n':
                return

            elif Protect == 'Y':
                print('Pre-existing files removed.')

                shutil.rmtree(Dir)
                os.mkdir(Dir)
                
            elif Protect == 'C':
                try:
                    shutil.rmtree(str(Dir+'/Outputs'))
                except FileNotFoundError:
                    print('Using pre-existing config in folder.')
                
                return timestamp

    Conf_data = {}

    Conf_data['PyUL Version'] = S_version
    Conf_data['PyUL Version Descriptor'] = D_version
    if NoInteraction:
        Conf_data['Description'] = 'Batch Configuration'
    else:
        Conf_data['Description'] = 'Single Run Configuration'
        
    Conf_data['Config Generation Time'] = tm
    
    Conf_data['Save Options'] = ({
            'Flags': save_options,
            'All Supported Flags': SFS,
            'Number': save_number,
            'Format': save_format
            })

    Conf_data['Spatial Resolution'] = resol

    Conf_data['Temporal Step Factor'] = step_factor

    Conf_data['RK Steps'] = NS

    Conf_data['Duration'] = ({
            'Time Duration': duration,
            'Start Time': start_time,
            'Time Units': duration_units
            })


    Conf_data['Simulation Box'] = ({
    'Box Length': length,
    'Length Units': length_units,
    })

    particles, solitons, embeds = EmbedParticle(particles,embeds,solitons)
    
    Conf_data['ULDM Solitons'] = ({
    'Condition': solitons,
    'Embedded': embeds,
    'Mass Units': s_mass_unit,
    'Position Units': s_position_unit,
    'Velocity Units': s_velocity_unit
    })

    rP = RecPlummer(np.real(a),length_units)
    
    Conf_data['Matter Particles'] = ({
    'Plummer Radius': rP,
    'Condition': particles,
    'Mass Units': m_mass_unit,
    'Position Units': m_position_unit,
    'Velocity Units': m_velocity_unit
    })

    Conf_data['Central Mass'] = 0
    

    #Conf_data['Field Smoothing'] = np.real(a)

    Conf_data['Uniform Field Override'] = ({

            'Flag': Uniform,
            'Density Unit': density_unit,
            'Density Value': Density,
            'Uniform Velocity': UVel

            })


    with open('{}{}{}{}{}'.format('./', save_path, '/', timestamp, '/config.uldm'), "w+") as outfile:
        json.dump(Conf_data, outfile,indent=4)

        
    if not NoInteraction:
        printU(('Compiled Config in Folder', timestamp))

    return timestamp
    
def Runs(save_path, Automatic = True):
    runs = os.listdir(save_path)
    runs.sort()
    if Automatic:
        Latest = Load_Latest(save_path)
    else:
        Latest = 'default'
    FLog = 0
    Log = [FLog]
    for i in range(len(runs)):
        
        if os.path.isdir(os.path.join(save_path, runs[i])):
            
            FLog += 1
            Log.append(i)
            if runs[i] == Latest:
                print("[",FLog,"]: *", runs[i],sep = '' )
            else:
                print("[",FLog,"]: ", runs[i],sep = ''  )

    if FLog == 0:
        return 'EMPTY'
    
    if FLog == 1 and Automatic:
        return Latest
    
    else:
        print("Which folder do you want to analyse? Blank to load the latest one. 'X[Number]' to Delete")
        
        Ind = (input() or int(-1))

        if Ind == -1 and Automatic:
            printU(f"Loading {Latest}")
            return Latest
        
        elif Ind.startswith('X'):
            Ind = Ind[1:]
            IndTD = Log[int(Ind)]
            TDName = os.path.join(save_path, runs[IndTD])
            
            import shutil
                       
            shutil.rmtree(TDName)
            clear_output()
            return Runs(save_path, Automatic = Automatic)
        
        else:
            Ind = int(Ind)
            Ind = Log[Ind]
            printU(f"Loading {runs[Ind]}")
            return runs[Ind]

def LoadConfig(loc):
        
        configfile = loc + '/config.uldm'
        
        with open(configfile) as json_file:
            config = json.load(json_file)
               
                
        if config["PyUL Version"] > S_version:
            raise RuntimeError("Configuration file generated by a newer version of PyUL.")
        
        ### Simulation Stuff
        try:
            save_options = SaveOptionsDigest(config["Save Options"]["Flags"])
            
        except KeyError:
            save_options = config["Save Options"]["flags"]

        try: 
            save_format = config["Save Options"]["Format"]
            
        except KeyError:
            save_format = 'npy'
        
        save_number = config["Save Options"]["Number"]
        
        ### Time Stuff
        
        duration = config["Duration"]['Time Duration']
        
        start_time = config["Duration"]['Start Time']
        
        duration_units = config["Duration"]['Time Units']
        
        NS = int(config["RK Steps"])
        
        step_factor = float(config["Temporal Step Factor"])
        
        ### Space Stuff
        
        try:
            resol = int(config["Spatial Resolution"])
        except KeyError:
            resol = int(config["Spacial Resolution"])
        
        length = config["Simulation Box"]["Box Length"]
        
        length_units = config["Simulation Box"]["Length Units"]
  
        ### Black Hole Stuff
        
        particles = config["Matter Particles"]['Condition']
        
        m_mass_unit = config["Matter Particles"]['Mass Units']
        
        m_position_unit = config["Matter Particles"]['Position Units']
        
        m_velocity_unit = config["Matter Particles"]['Velocity Units']
      
        try:
            rP = config["Matter Particles"]["Plummer Radius"]
            
            a = GenPlummer(rP,length_units)
            
        except KeyError:
            
            a = config["Field Smoothing"]
            
        ### ULDM Stuff
        
        solitons = config["ULDM Solitons"]['Condition']
        
        embeds = config["ULDM Solitons"]['Embedded']
        
        s_mass_unit = config["ULDM Solitons"]['Mass Units']
        
        s_position_unit = config["ULDM Solitons"]['Position Units']
        
        s_velocity_unit = config["ULDM Solitons"]['Velocity Units']
   
        ### ULDM Modifier
        
        Uniform = config["Uniform Field Override"]["Flag"]
        density_unit = config["Uniform Field Override"]["Density Unit"]
        Density = config["Uniform Field Override"]["Density Value"]
        UVel = config["Uniform Field Override"]['Uniform Velocity']
      
         
        return  NS, length, length_units, resol, duration, duration_units, step_factor, save_number, save_options, save_format, s_mass_unit, s_position_unit, s_velocity_unit, solitons,start_time, m_mass_unit, m_position_unit, m_velocity_unit, particles, embeds, Uniform,Density, density_unit,a, UVel



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
        
    file = open('{}{}{}'.format('./', save_path, '/LookUp.uldm'), "w+")
    
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


def DefaultSolitonOrbit(resol,length, length_units, s_mass, s_mass_unit, m_radius, m_position_unit, m_velocity_unit = '', Silent = True, Detail = 10000):
    
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

    funct = np.abs(initsolitonRadial(linearray, alpha, f, delta_x,Cutoff = 5.6))**2
    
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
    
def Resample3Box(psi, COM, XAR, loc, save_num, save_format, Length_Ratio = 0.5, resolR = 192, Save_Rho = False, Save_Psi = False, Compatibility = True):
    
    from scipy.interpolate import RegularGridInterpolator as RPI
    # Note that phase correction is not performed in this version.
    
    lengthCR = -2 * XAR[0] * Length_Ratio

    if Compatibility:
        GVR = np.linspace(-lengthCR / 2.0 + lengthCR / float(2 * resolR), lengthCR / 2.0 - lengthCR / float(2 * resolR), int(resolR), endpoint = True)
    else:
        GVR = np.linspace(-lengthCR/2, lengthCR/2, int(resolR), endpoint = False)
        
    Real_I = RPI((XAR,XAR,XAR),np.real(psi),method='linear',bounds_error = False, fill_value = 0)
    Imag_I = RPI((XAR,XAR,XAR),np.imag(psi),method='linear',bounds_error = False, fill_value = 0)
    
    NewGrid = np.meshgrid(
        GVR + COM[0], GVR + COM[1], GVR + COM[2],
        sparse=False, indexing='ij')

    NewGrid_List = np.reshape(NewGrid, (3, -1), order='C').T

    IReal = Real_I(NewGrid_List)
    IImag = Imag_I(NewGrid_List)

    IPsi = ne.evaluate("IReal + 1j*IImag")

    PsiNew = np.reshape(IPsi,(resolR,resolR,resolR))
    
    if Save_Psi:
        IOSave(loc,'3WfnRS',save_num,save_format,PsiNew)
    
    if Save_Rho:
        RhoNew = ne.evaluate("conj(PsiNew)*PsiNew").real
        
        IOSave(loc,'3DensityRS',save_num,save_format,RhoNew)


def Wfn_to_PyUL1(psi):

    from scipy.interpolate import RegularGridInterpolator as RPI
    # Note that phase correction is not performed in this version.
    
    lengthCR = 1 
    resolR = psi.shape[0]
    
    GVR = np.linspace(-lengthCR / 2.0 + lengthCR / float(2 * resolR), lengthCR / 2.0 - lengthCR / float(2 * resolR), int(resolR), endpoint = True)
    XAR = np.linspace(-lengthCR/2, lengthCR/2, int(resolR), endpoint = False)
        
    Real_I = RPI((XAR,XAR,XAR),np.real(psi),method='linear',bounds_error = False, fill_value = 0)
    Imag_I = RPI((XAR,XAR,XAR),np.imag(psi),method='linear',bounds_error = False, fill_value = 0)
    
    NewGrid = np.meshgrid(
        GVR , GVR , GVR ,
        sparse=False, indexing='ij')
    
    NewGrid_List = np.reshape(NewGrid, (3, -1), order='C').T
    
    IReal = Real_I(NewGrid_List)
    IImag = Imag_I(NewGrid_List)
    
    IPsi = ne.evaluate("IReal + 1j*IImag")
    
    return np.reshape(IPsi,(resolR,resolR,resolR))
