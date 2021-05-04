Version   = str('PyUL2') # Handle used in console.
D_version = str('Build 2021 May 03') # Detailed Version
S_version = 20.3 # Short Version

import time
from datetime import datetime
import sys
import numpy as np
import numexpr as ne
import numba

import pyfftw
import h5py
import os
import scipy.fft
import multiprocessing
import json

import scipy.integrate as si

from scipy.interpolate import RegularGridInterpolator as RPI

# For Jupyter
from IPython.core.display import clear_output

try:
    import scipy.special.lambertw as LW
except ModuleNotFoundError:
    def LW(a):
        return -0.15859433956303937
    
num_threads = multiprocessing.cpu_count()

pi = np.pi

eV = 1.783e-36 # kg*c^2

# ULDM:
axion_E = 1e-22

# CDM Clumps:
#axion_E = 2e-5

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
    

### Internal Flags used for IO
SFS = '3Density 3Wfn 2Density Energy 1Density NBody 3Grav 2Grav DF 2Phase Entropy 1Grav 3GravF 2GravF 1GravF'

SNM = 'R3D P3D R2D EGY R1D NTM G3D G2D DYF A2D ENT G1D F3D F2D F1D'

SaveFlags = SFS.split()
SaveNames = SNM.split()
    
def IOName(Type):
    return SaveNames[SaveFlags.index(Type)]

def IOSave(loc,Type,save_num,save_format = 'npy',data = []):
    
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
    return np.load(f"{loc}/Outputs/{IOName(Type)}_#{save_num:03d}.npy")


### AUX. FUNCTION TO GENERATE TIME STAMP

def GenFromTime():
    from datetime import datetime
    now = datetime.now() # current date and time
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    
    return timestamp

SSLength = 5

def printU(Message,SubSys = 'Sys'):
    print(f"{Version}.{SubSys.rjust(SSLength)}: {Message}")
####################### AUX. FUNCTION TO GENERATE PROGRESS BAR
def prog_bar(iteration_number, progress, tinterval,status = '',adtl = ''):
    size = 20
    ETAStamp = time.time() + (iteration_number - progress)*tinterval
    
    ETA = datetime.fromtimestamp(ETAStamp).strftime("%d/%m/%Y, %H:%M:%S")
    
    progress = float(progress) / float(iteration_number)
    
    if progress >= 1.:
        progress, status = 1, ""
    
    block = int(round(size * progress))
    current = 1
    if block == size:
        current = 0
    
    status = f'{status.ljust(SSLength)}'
    
    text = "\r[{}] {:.0f}% {}{}{} ({}{:.2f}s) {}".format(
        "●" * (block) + "◐" * (current) + "○" * (size - block - current), round(progress * 100, 0),
        status, 'Exp. Time: ',ETA,'Prev.: ',tinterval,adtl)
   
    print(f'{text}', end="",flush='true')


####################### Credits Information
def PyULCredits(IsoP = False,UseDispSponge = False,embeds = []):
    print(f"==============================================================================")
    print(f"{Version}.{S_version}: (c) 2020 - 2021 Wang., Y. and collaborators. \nAuckland Cosmology Group\n") 
    print("Original PyUltraLight Team:\nEdwards, F., Kendall, E., Hotchkiss, S. & Easther, R.\n\
arxiv.org/abs/1807.04037")
    
    if IsoP or UseDispSponge or (embeds != []):
        print(f"\n================== External Module In Use ================== \n")
    
    if IsoP:
        printU(f"Isolated ULDM Potential Implementation \nAdapted from J. L. Zagorac et al. Yale Cosmology \n",'External')
        
    if UseDispSponge:
        printU(f"Dispersive Sponge Condition \nAdapted from J. L. Zagorac et al. Yale Cosmology \n",'External')
        
    if embeds != []:
        printU(f"Embedded Soliton Profiles \nAdapted from N. Guo et al. Auckland Cosmology Group \n",'External')

    print(f"==============================================================================")

    

def ULDStepEst(duration,duration_units,length,length_units,resol,step_factor, save_number = -1):
    
    gridlength = convert(length, length_units, 'l')
 
    t = convert(duration, duration_units, 't')
    
    delta_t = (gridlength/float(resol))**2/np.pi

    min_num_steps = np.ceil(t / delta_t)
    MinUS = int(min_num_steps//step_factor)

    print(f'The required number of ULDM steps is {MinUS}')
    
    if save_number > 0:
        
        if save_number >= MinUS:
            MinUS = int(save_number)
        
        else:
            MinUS = int(save_number * (MinUS // (save_number) + 1))
            
    print(f'The actual ULDM steps is {MinUS}')
    
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
        rhopad[i, :, :] = PC_jit(green[ndx[i], :, :], plane, n, fft_plane, ifft_plane) #make sure this indexing works 
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
        elif (unit == 'm'):
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
        elif (unit == 'kg'):
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
        elif (unit == 's'):
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
        elif (unit == 'm/s'):
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
        elif (unit == 'kg/m3'):
            converted = value / mass_unit * length_unit**3
        else:
            raise NameError('Unsupported DENSITY unit used')

    else:
        raise TypeError('Unsupported conversion type')

    return converted



####################### FUNCTION TO CONVERT FROM DIMENSIONLESS UNITS TO DESIRED UNITS

def convert_back(value, unit, type):
    converted = 0
    if (type == 'l'):
        if (unit == ''):
            converted = value
        elif (unit == 'm'):
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
        elif (unit == 'kg'):
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
        elif (unit == 's'):
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
        elif (unit == 'm/s'):
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
        elif (unit == 'kg/m3'):
            converted = value * mass_unit / length_unit**3
        else:
            raise NameError('Unsupported DENSITY unit used')

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

def initsoliton(funct, xarray, yarray, zarray, position, alpha, f, delta_x,Cutoff = 5.6):
    funct*= 0
    
    for index in np.ndindex(funct.shape):
        
        
        # Note also that this distfromcentre is here to calculate the distance of every gridpoint from the centre of the soliton, not to calculate the distance of the soliton from the centre of the grid
        distfromcentre = (
            (xarray[index[0], 0, 0] - position[0]) ** 2 +
            (yarray[0, index[1], 0] - position[1]) ** 2 +
            (zarray[0, 0, index[2]] - position[2]) ** 2
            ) ** 0.5
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
            
def calculate_energies(rho, Vcell, phiSP,phiTM, psi, karray2, fft_psi, ifft_funct, Density,Uniform, egpcmlist, egpsilist, ekandqlist, egylist, mtotlist):


    BoxAvg = np.mean(rho) # Mean Value in Box

    # Gravitational potential energy density associated with the point masses potential

    ETM = ne.evaluate('real(phiTM*(rho-BoxAvg))') # Interaction in particle potential!.
    ETMtot = Vcell * np.sum(ETM)
    egpcmlist.append(ETMtot) # TM Saved.

    # Gravitational potential energy density of self-interaction of the condensate
    ESI = ne.evaluate('real(0.5*(phiSP)*rho)') # New!
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


### Toroid or something
        
def Wrap(TMx, TMy, TMz, gridlength):
    
    if TMx > gridlength/2:
        TMx = TMx - gridlength
        
    if TMx < -gridlength/2:
        TMx = TMx + gridlength
        
        
    if TMy > gridlength/2:
        TMy = TMy - gridlength
    
    if TMy < -gridlength/2:
        TMy = TMy + gridlength
        
        
    if TMz > gridlength/2:
        TMz = TMz - gridlength
    
    if TMz < -gridlength/2:
        TMz = TMz + gridlength
        
    return TMx,TMy,TMz

FWrap = Wrap

### For Immediate Interpolation of Field Energy

def QuickInterpolate(Field,gridlength,resol,position):
        #Code Position
                
        RNum = (position*1/gridlength+1/2)*resol

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

def FWNBody(t,TMState,masslist,phiSP,a,gridlength,resol):

    GridDist = gridlength/resol
    
    dTMdt = 0*TMState
    GradientLog = np.zeros(len(masslist)*3)
    
    for i in range(len(masslist)):
        
        #0,6,12,18,...
        Ind = int(6*i)
        IndD = int(3*i)
           
        #X,Y,Z
        poslocal = TMState[Ind:Ind+3]

        RNum = (poslocal*1/gridlength+1/2)*resol

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

            GArZ = (TAr[1:3,1:3,2:4] - TAr[1:3,1:3,2:4])/(2*GridDist) # 8

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


def FWNBody_NI(t,TMState,masslist,phiSP,a,gridlength,resol):

    GridDist = gridlength/resol
    
    dTMdt = 0*TMState
    GradientLog = np.zeros(len(masslist)*3)
    
    for i in range(len(masslist)):
        
        #0,6,12,18,...
        Ind = int(6*i)
        IndD = int(3*i)
           
        #X,Y,Z
        poslocal = TMState[Ind:Ind+3]

        RNum = (poslocal*1/gridlength+1/2)*resol

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

            GArZ = (TAr[1:3,1:3,2:4] - TAr[1:3,1:3,2:4])/(2*GridDist) # 8

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

def NBodyAdvance(TMState,h,masslist,phiSP,a,gridlength,resol,NS):
        #
        if NS == 0: # NBody Dynamics Off
            
            Step, GradientLog = FWNBody3(0,TMState,masslist,phiSP,a,gridlength,resol)
            
            return TMState, GradientLog
        
        if NS == 1:
 
            Step, GradientLog = FWNBody3(0,TMState,masslist,phiSP,a,gridlength,resol)
            TMStateOut = TMState + Step*h
            
            return TMStateOut, GradientLog
        
        elif NS%4 == 0:
            
            NRK = int(NS/4)
            
            H = h/NRK
            
            for RKI in range(NRK):
                TMK1, Trash = FWNBody3(0,TMState,masslist,phiSP,a,gridlength,resol)
                TMK2, Trash = FWNBody3(0,TMState + H/2*TMK1,masslist,phiSP,a,gridlength,resol)
                TMK3, Trash = FWNBody3(0,TMState + H/2*TMK2,masslist,phiSP,a,gridlength,resol)
                TMK4, GradientLog = FWNBody3(0,TMState + H*TMK3,masslist,phiSP,a,gridlength,resol)
                TMState = TMState + H/6*(TMK1+2*TMK2+2*TMK3+TMK4)
                
            TMStateOut = TMState

            return TMStateOut, GradientLog


def NBodyAdvance_NI(TMState,h,masslist,phiSP,a,gridlength,resol,NS):
        #
        if NS == 0: # NBody Dynamics Off
            
            Step, GradientLog = FWNBody3_NI(0,TMState,masslist,phiSP,a,gridlength,resol)
            
            return TMState, GradientLog
        
        if NS == 1:
 
            Step, GradientLog = FWNBody3_NI(0,TMState,masslist,phiSP,a,gridlength,resol)
            TMStateOut = TMState + Step*h
            
            return TMStateOut, GradientLog
        
        elif NS%4 == 0:
            
            NRK = int(NS/4)
            
            H = h/NRK
            
            for RKI in range(NRK):
                TMK1, Trash = FWNBody3_NI(0,TMState,masslist,phiSP,a,gridlength,resol)
                TMK2, Trash = FWNBody3_NI(0,TMState + H/2*TMK1,masslist,phiSP,a,gridlength,resol)
                TMK3, Trash = FWNBody3_NI(0,TMState + H/2*TMK2,masslist,phiSP,a,gridlength,resol)
                TMK4, GradientLog = FWNBody3_NI(0,TMState + H*TMK3,masslist,phiSP,a,gridlength,resol)
                TMState = TMState + H/6*(TMK1+2*TMK2+2*TMK3+TMK4)
            
            TMStateOut = TMState

            return TMStateOut, GradientLog



######################### Soliton Init Factory Setting!

def LoadDefaultSoliton():
    
    f = np.load('./Soliton Profile Files/initial_f.npy')
    
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
            
            Response = str(input())
            
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
        
        with open(Save_Folder + RATIOName + '_info.txt', "w+") as outfile:
            json.dump(Profile_Config, outfile,indent=4)

    print ('Successfully Initiated Soliton Profile.')
    
def LoadSolitonConfig(Ratio): 
    
    Ratio = float(Ratio)
    
    RatioN = f"{Ratio:.4f}"
    
    FileName = './Soliton Profile Files/Custom/f_'+RatioN+'_Pro_info.txt'
    
    if os.path.isfile(FileName):
        with open(configfile) as json_file:
            config = json.load(json_file)
            
    else:
        FileName = './Soliton Profile Files/Custom/f_'+RatioN+'_Dra_info.txt'

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

def evolve(save_path,run_folder, EdgeClear = False, DumpInit = False, DumpFinal = False, UseInit = False, IsoP = False, UseDispSponge = False, SelfGravity = True, NBodyInterp = True, NBodyGravity = True, Silent = False, AutoStop = False, AutoStop2 = False, WellThreshold = 100, InitPath = '', InitWeight = 1, Message = ''):
    
    clear_output()

    Draft = True

    Method = 3 # Backward Compatibility
    
    loc = './' + save_path + '/' + run_folder
        
    if DumpInit and (InitPath == ''):
        raise RuntimeError("Must supply initial wavefunction!")
        
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

    
    file = open('{}{}{}'.format('./', save_path, '/latest.txt'), "w+")
    file.write(run_folder)
    file.close()

    
    NS, central_mass, length, length_units, resol, duration, duration_units, step_factor, save_number, save_options, save_format, s_mass_unit, s_position_unit, s_velocity_unit, solitons,start_time, m_mass_unit, m_position_unit, m_velocity_unit, particles, embeds, Uniform,Density, density_unit,a, B, UVel = LoadConfig(loc)

        
    num_threads = multiprocessing.cpu_count()
    
    if resol <= 100:
        num_threads = np.min([num_threads//2,8])
        
    printU(f"Using {num_threads} CPU Threads for FFT.",'FFT')
    # External Credits Print
    PyULCredits(IsoP,UseDispSponge,embeds)
    print(f"{Version}\n")
    # Embedded particles are Pre-compiled into the list.
    
    if not Uniform:
        Density = 0
        UVel = [0,0,0]
    
    if a>=1e8:
        printU(f"Smoothing has been turned off!",'NBody')
        a = 0
    
    printU(f"Loaded Parameters from {loc}",'IO')

    NumSol = len(solitons)
    NumTM = len(particles)
            
    if (Method == 3): # 1 = Real Space Interpolation (Orange), 2 = Fourier Sum (White)
        printU(f"Using Linear Interpolation for gravity.",'NBody')
    
    printU(f"Simulation grid resolution is {resol}^3.",'FFT')
    
    if a == 0:
        printU(f"Using 1/r Point Mass Potential.",'NBody')
    

    if EdgeClear:
        print("WARNING: The Wavefunction on the boundary planes will be Auto-Zeroed at every iteration.")

    print('--------------------------Additional Settings---------------------------------')

    if NBodyGravity == False:
        print(f"Particle gravity OFF.")    
    else:
        print(f"Particle gravity  ON.")     
        
    if SelfGravity == False:
        print(f"ULDM self-gravity OFF.")    
    else:
        print(f"ULDM self-gravity  ON.")
        
    if NBodyInterp == False:
        print(f"NBody response to ULDM OFF.")
    else:
        print(f"NBody response to ULDM  ON.")
    
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

    gridlength = convert(length, length_units, 'l')
    
    b = gridlength * B/2

    t = convert(duration, duration_units, 't')

    t0 = convert(start_time, duration_units, 't')
    
    cmass = convert(central_mass, m_mass_unit, 'm')

    Density = convert(Density,density_unit,'d')
    
    Vcell = (gridlength / float(resol)) ** 3
    
    ne.set_num_threads(num_threads)

    ##########################################################################################
    # Backwards Compatibility
    
    NCV = np.array([[0,0,0]])
    NCW = np.array([1])

    save_path = os.path.expanduser(save_path)

    ##########################################################################################
    # SET UP THE REAL SPACE COORDINATES OF THE GRID - FW Revisit

    gridvec = np.linspace(-gridlength / 2.0, gridlength / 2.0, resol, endpoint=False)
    
    xarray, yarray, zarray = np.meshgrid(
        gridvec, gridvec, gridvec,
        sparse=True, indexing='ij')
    
    WN = 2*np.pi*np.fft.fftfreq(resol, gridlength/(resol)) # 2pi Pre-multiplied
    
    Kx,Ky,Kz = np.meshgrid(WN,WN,WN,sparse=True, indexing='ij',)
    
##########################################################################################
    # SET UP K-SPACE COORDINATES FOR COMPLEX DFT

    kvec = 2 * np.pi * np.fft.fftfreq(resol, gridlength / float(resol))
    
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
        printU(f"IO: Loaded initial wavefunction from {InitPath}",'IO')
        
        MassCom = Density*gridlength**3

        UVelocity = convert(np.array(UVel),s_velocity_unit, 'v')

        DensityCom = MassCom / resol**3

        print('========================Dispersive Background====================================')
        printU(f"Solitons overridden with a pre-generated wavefunction with pseudorandom phase.",'Init')
        
    # INITIALISE SOLITONS WITH SPECIFIED MASS, POSITION, VELOCITY, PHASE

    psi = pyfftw.zeros_aligned((resol, resol, resol), dtype='complex128')

    MassCom = Density*gridlength**3

    UVelocity = convert(np.array(UVel),s_velocity_unit, 'v')

    if AutoStop and NumTM == 1:
        ThresholdVelocity = -1*UVelocity[1]
    else:
        ThresholdVelocity = 0

    DensityCom = MassCom / resol**3
    
    if Uniform:
        print('========================Uniform Background====================================')
        printU(f"Solitons overridden with a uniform wavefunction with no phase.",'Init')
        printU(f"Background ULDM mass in domain is {MassCom:.4f}, at {Density:.4f} per grid.",'Init')
        printU(f"Background Global velocity is (x,y,z): {UVel[1]},{UVel[0]},{UVel[2]}.",'Init')
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
            printU(f'Note that this process will not be required for repeated runs. To remove custom profiles, go to the folder /Soliton Profile Files/Custom.', 'Profiler')
            print(f'Generating profile for Soliton with MBH/MSoliton = {RatioBU:.4f}, Part 1')
            GMin,GMax = BHGuess(RatioBU)
            s, BHMass = BHRatioTester(RatioBU,30,1e-6,GMin,GMax,a)
            print(f'Generating profile for Soliton with MBH/MSoliton = {RatioBU:.4f}, Part 2')
            SolitonProfile(BHMass,s,a,not Draft)
        print('==============================================================================')

        delta_xL, prealphaL, betaL,CutOff = LoadSolitonConfig(RatioBU)

        # L stands for Local, as in it's only used once.
        fL = LoadSoliton(RatioBU)

        printU(f"Loaded embedded soliton {EI} with BH-Soliton mass ratio {RatioBU:.4f}.", 'Init')

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
        printU(f"Loaded unperturbed soliton.",'Init')
        f = LoadDefaultSoliton()

    for s in solitons:
        mass = convert(s[0], s_mass_unit, 'm')
        position = convert(np.array(s[1]), s_position_unit, 'l')
        velocity = convert(np.array(s[2]), s_velocity_unit, 'v')
        # Note that alpha and beta parameters are computed when the initial_f.npy soliton profile file is generated.
        alpha = (mass / 3.883) ** 2
        beta = 2.454
        phase = s[3]
        funct = initsoliton_jit(funct, xarray, yarray, zarray, position, alpha, f, delta_x)
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
    
    ifft_funct = pyfftw.builders.ifftn(funct, axes=(0, 1, 2), threads=num_threads)       
    
    # Experimental Zero Ridder!
    #if not Uniform:
    #    print(f"{Version} SP: Peforming an Additional FFT Step to Get rid of zeroes.")
    #    psi = ifft_funct(fft_psi(psi)) # New FFT Step 

    rho = ne.evaluate("real(abs(psi)**2)")
     ##########################################################################################
    # COMPUTE SIZE OF TIMESTEP (CAN BE INCREASED WITH step_factor)

    delta_t = (gridlength/float(resol))**2/np.pi

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
    ##########################################################################################
    # SETUP PADDED POTENTIAL HERE (From JLZ)

    rhopad = pyfftw.zeros_aligned((2*resol, resol, resol), dtype='float64')
    bigplane = pyfftw.zeros_aligned((2*resol, 2*resol), dtype='float64')

    fft_X = pyfftw.builders.fftn(rhopad, axes=(0, ), threads=num_threads)
    ifft_X = pyfftw.builders.ifftn(rhopad, axes=(0, ), threads=num_threads)

    fft_plane = pyfftw.builders.fftn(bigplane, axes=(0, 1), threads=num_threads)
    ifft_plane = pyfftw.builders.ifftn(bigplane, axes=(0, 1), threads=num_threads)
    
    phiSP = pyfftw.zeros_aligned((resol, resol, resol), dtype='float64')
    phiTM = pyfftw.zeros_aligned((resol, resol, resol), dtype='float64') # New, separate treatment.

    fft_phi = pyfftw.builders.fftn(phiSP, axes=(0, 1, 2), threads=num_threads)
    ##########################################################################################
    # SETUP K-SPACE FOR RHO (REAL)

    rkvec = 2 * np.pi * np.fft.fftfreq(resol, gridlength / float(resol))
    
    krealvec = 2 * np.pi * np.fft.rfftfreq(resol, gridlength / float(resol))
    
    rkxarray, rkyarray, rkzarray = np.meshgrid(
        rkvec, rkvec, krealvec,
        sparse=True, indexing='ij'
    )

    rkarray2 = ne.evaluate("rkxarray**2+rkyarray**2+rkzarray**2")

    rfft_rho = pyfftw.builders.rfftn(rho, axes=(0, 1, 2), threads=num_threads)
    
    phik = rfft_rho(rho)  # not actually phik but phik is defined in next line
    
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
            printU(f"Using pre-computed Green function for simulation region.",'SP')
        except FileNotFoundError:
            if not os.path.exists('./Green Functions/'):
                os.mkdir('./Green Functions/')
            green = makeDCTGreen(resol) #make Green's function ONCE
            printU(f"Generating Green function for simulation region.",'SP')
            np.save(f'./Green Functions/G{resol}.npy',green)
            
        #green = makeEvenArray(green)
        phiSP = IP_jit(rho, green, gridlength, fft_X, ifft_X, fft_plane, ifft_plane)
        
    else:
        printU(f"Poisson Equation Solveed Using FFT.",'SP')
        phiSP = irfft_phi(phik)
        
    ##########################################################################################
       
    # FW NBody Vanilla
    EGPCM = 0
    for MI, particle in enumerate(particles):
               
        mT = convert(particle[0], m_mass_unit, 'm')
        
        if mT == 0:
            printU(f"Particle #{MI} loaded as observer.",'NBody')
        else:
            printU(f"Particle #{MI} loaded, with (code) mass {mT:.3f}",'NBody')
        
        masslist.append(mT)

        position = convert(np.array(particle[1]), m_position_unit, 'l')  
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
            EGPCM += mT*QuickInterpolate(phiSP,gridlength,resol,np.array([TMx,TMy,TMz]))
        MI = int(MI + 1)
        
        if AutoStop and Uniform and len(particles) == 1:
            ThresholdVelocity += Vy
            
        if AutoStop and len(particles) == 1:
            E0 = 1/2 * mT * np.linalg.norm(velocity + UVelocity)**2  
            if E0 == 0:
                E0 = 1
            # Always shooting towards the right.

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
    
    TMStateDisp = TMState.reshape((-1,6))
    
    printU(f"The test mass initial state (vectorised) is:", 'NBody')
    print(TMStateDisp)
        
    MI = 0
    
    GridMass = [Vcell*np.sum(rho)] # Mass of ULDM in Grid
    
    if (save_options[3]):
        egylist = []
        egpcmlist = []
        egpsilist = []
        ekandqlist = []
        mtotlist = []
        egpcmMlist = [EGPCM]

        calculate_energies(rho, Vcell, phiSP,phiTM, psi, karray2, fft_psi, ifft_funct, Density,Uniform, egpcmlist, egpsilist, ekandqlist, egylist, mtotlist)
    
    GradientLog = np.zeros(NumTM*3)

    if save_options[10]:
        
        EntropyLog = [-1*np.sum(ne.evaluate('rho*log(rho)'))]
        np.save(os.path.join(os.path.expanduser(loc), "Outputs/Entro.npy"), EntropyLog)

    #######################################
    
    if np.isnan(rho).any() or np.isnan(psi).any():
        raise RuntimeError("Something is seriously wrong.")
    
    if DumpInit:
        printU(f'Successfully initiated Wavefunction and NBody Initial Conditions. Dumping to file.','IO')
    
        ULDump(loc,psi,TMState,'Init')
        
        
    else:
        printU(f'Successfully initiated Wavefunction and NBody Initial Conditions.', 'Init')

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
    # Chapel Code From Yale Cosmology.
  
    if UseDispSponge:
        
        SpongeRatio = 1/2
        # This works in grid units in Chapel. We make it work in Code Units.
        rn = 1/2*gridlength
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
        
        printU(f'Dispersive Sponge Condition Pre Multiplier Ready.','SP')
    ##########################################################################################
    # LOOP NOW BEGINS
    if Silent:
        clear_output()
    print(f"{D_version}\nExternal Message: {Message}")
    printU(f"Simulation name is {loc}",'Runtime')
    printU(f"{resol} Resolution for {duration:.4g}{duration_units}",'Runtime')
    printU(f"Simulation Started at {tBeginDisp}.",'Runtime')
            
    HaSt = 1  # 1 for a half step 0 for a full step

    tenth = float(save_number/10) #This parameter is used if energy outputs are saved while code is running.
    if actual_num_steps == save_number:
        printU(f"Taking {int(actual_num_steps)} ULDM steps", 'Runtime')
    else:
        printU(f"Taking {int(actual_num_steps)} ULDM steps @ {save_number} snapshots", 'Runtime')
    
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
        funct = ne.evaluate("funct*exp(-1j*0.5*h*karray2)")
        
        psi = ifft_funct(funct)


        if UseDispSponge:
            prog_bar(actual_num_steps, ix + 1, tint,'DS',PBEDisp)
            psi *= np.exp(-PreMult*h)
                    
        rho = ne.evaluate("real(abs(psi)**2)")
        
        phik = rfft_rho(rho)
        
        phik = ne.evaluate("-4*pi*(phik)/rkarray2")
        
        phik[0, 0, 0] = 0 # Kill the base frequency.

        prog_bar(actual_num_steps, ix + 1, tint,'SP',PBEDisp)
        # New Green Function Methods
        if not IsoP:
            phiSP = irfft_phi(phik)
        else:
            phiSP = IP_jit(rho, green, gridlength, fft_X, ifft_X, fft_plane, ifft_plane)

        # FW STEP MAGIC HAPPENS HERE
        prog_bar(actual_num_steps, ix + 1, tint,'RK4',PBEDisp)
        
        if NBodyInterp:

            TMState, GradientLog = NBodyAdvance(TMState,h,masslist,phiSP,a,gridlength,resol,NS)
            
        else:

            TMState, GradientLog = NBodyAdvance_NI(TMState,h,masslist,phiSP,a,gridlength,resol,NS)
 
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

                    if b > 0:
                        phiTM = phiTM * 1/2*(np.tanh((b-distarrayTM)*Steep)+1)

                    if (save_options[3]) and ((ix + 1) % its_per_save) == 0:
                        EGPCM += mT*QuickInterpolate(phiSP,gridlength,resol,np.array([TMx,TMy,TMz]))
            
        if AutoStop and len(particles) == 1:
            velocity = TMState[3:6]
            EKDisp = (1/2 * mT * np.linalg.norm(velocity + UVelocity)**2 ) / E0
            PBEDisp = f'KE~{100*EKDisp:.2f}%'
            
        if SelfGravity:
            phi = ne.evaluate('phiSP + phiTM')
        else: 
            phi = phiTM
            
            

        prog_bar(actual_num_steps, ix + 1, tint,'FT',PBEDisp)
        #Next if statement ensures that an extra half step is performed at each save point
        if (((ix + 1) % its_per_save) == 0) and HaSt == 0:
            psi = ne.evaluate("exp(-1j*0.5*h*phi)*psi")
            rho = ne.evaluate("real(abs(psi)**2)")
            HaSt = 1

            prog_bar(actual_num_steps, ix + 1, tint,'IO',PBEDisp)
            #Next block calculates the energies at each save, not at each timestep.
            if (save_options[3]):
                calculate_energies(rho, Vcell, phiSP,phiTM, psi, karray2, fft_psi, ifft_funct, Density,Uniform, egpcmlist, egpsilist, ekandqlist, egylist, mtotlist)
           
################################################################################
# SAVE DESIRED OUTPUTS
        if ((ix + 1) % its_per_save) == 0:
        
            egpcmMlist.append(EGPCM)
            EGPCM = 0
            
            save_grid(
                rho, psi, resol, 
                TMState, phiSP, phi, GradientLog,
                save_options,
                save_format,
                loc, ix, its_per_save
                )
            
            GridMass.append(Vcell*np.sum(rho))
            np.save(os.path.join(os.path.expanduser(loc), "Outputs/ULDMass.npy"), GridMass)
            
            if (save_options[3]):  
                np.save(os.path.join(os.path.expanduser(loc), "Outputs/egylist.npy"), egylist)
                np.save(os.path.join(os.path.expanduser(loc), "Outputs/egpcmlist.npy"), egpcmlist)
                np.save(os.path.join(os.path.expanduser(loc), "Outputs/egpcmMlist.npy"), egpcmMlist)
                np.save(os.path.join(os.path.expanduser(loc), "Outputs/egpsilist.npy"), egpsilist)
                np.save(os.path.join(os.path.expanduser(loc), "Outputs/ekandqlist.npy"), ekandqlist)
                np.save(os.path.join(os.path.expanduser(loc), "Outputs/masseslist.npy"), mtotlist)
         
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

                    file = open(f'{loc}/StoppingTime.txt', "w+")
                    file.write(f"{TIntegrate}")
                    file.close()
                    TimeWritten = True
                    
        if AutoStop2:
            if np.min(phi) < phiRef:
                print('\n')
                printU('Gravitational field runaway threshold reached!','Consistency')
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
    printU(f"Run Complete. Time Elapsed (d:h:m:s): {day:.0f}:{hour:.0f}:{minutes:.0f}:{seconds:.2f}",'Runtime')
    if DumpFinal:
        printU(f'Dumped final state to file.','IO')
        ULDump(loc,psi,TMState,'Final')
        
    if AutoStop and not TimeWritten:
        file = open(f'{loc}/StoppingTime.txt', "w+")
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
    
    

def DSManagement(save_path):
    
    print('[',save_path,']',": The current size of the folder is", round(get_size(save_path)/1024**2,3), 'Mib')

    if get_size(save_path) == 0:
        cleardir = 'N'
    else:
        print('[',save_path,']',": Do You Wish to Delete All Files Currently Stored In This Folder? [Y] \n")
        cleardir = str(input())
    
    if cleardir == 'Y':
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
    
    
    with open('{}{}{}'.format('./',save_path, '/latest.txt'), 'r') as timestamp:
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
        print('ULHelper: Loaded Planar Mass Density Data \n')
    if save_testmass:
        print('ULHelper: Loaded Test Mass State Data \n')
    if save_phi_plane:
        print('ULHelper: Loaded Planar Gravitational Field Data \n')
    if save_gradients:
        print('ULHelper: Loaded Test Mass Gradient Data \n')
    if save_phase_plane:
        print('ULHelper: Loaded Planar ULD Phase Data \n')
        

    
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
        
    print("ULHelper: Loaded", EndNum, "Data Entries")
    return EndNum, data,  TMdata, phidata,    graddata, phasedata


def Load_npys(loc,save_options, LowMem = False):
    

    
    if save_options[0] or save_options[1] or save_options[6] or save_options[12]: 
        printU('3D saves are not automatically loaded. Please load them manually.','IO')
        save_options[0] = False
        save_options[1] = False
        save_options[6] = False
        save_options[12] = False
    
    if LowMem:
        printU('Skipping 2D data. Please load them manually.','IO')
        save_options[2] = False
        save_options[7] = False
        save_options[13] = False
    
    
    SaveWordList = SaveOptionsCompile(save_options).split()
    
    Out = {}
    
    Out['Directory'] = loc
    
    
    
    for Word in SaveWordList:
        if (Word != 'Energy') and (Word != 'Entropy'): 
            Out[Word] = []
        
    import time   
    import warnings 
    warnings.filterwarnings("ignore")

    EndNum = 0
    
    x = 0
    success = True

    while success:
        
        try:
            for Word in SaveWordList:
                if (Word != 'Energy') and (Word != 'Entropy'): 
                    Out[Word].append(IOLoad_npy(loc,Word,x))        
            x += 1
        
        except FileNotFoundError:
            success = False

    printU(f"Loaded {x} Data Entries",'LoadNPY')
    return x, Out


def SmoothingReport(a,resol,clength,B = 0):
    import matplotlib.pyplot as plt

    GridLenFS = clength/(resol)
    
    COMin = resol/2
    COMid = resol/2*np.sqrt(2)
    COMax = resol/2*np.sqrt(3)
   
    fig_grav = plt.figure(figsize=(12, 12))

    # Diagnostics For Field Smoothing
    
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    
    
    FS = np.arange(resol)+1
    
    rR = GridLenFS*FS
    
    GOrig = -1/rR
    
    GMod = -a*1/np.sqrt(1+a**2*rR**2)
    
    GDiff = - GOrig + GMod
    
    # Two little quantifiers
    BoundaryEn = next(x for x, val in enumerate(GDiff) if val < 1e-2)
    rS = FS[BoundaryEn]
    
    BoundaryEx = next(x for x, val in enumerate(GMod) if val > -a/2)
    rQ = FS[BoundaryEx]
    
    ax1.plot(FS,GOrig,'k--',label = 'Point Potential')
       
    
    ax1.plot(FS,GMod,'g.',label = 'Modified Potential')
        
    ax1.set_ylim([-1.5*a,0])
    
    
    ax1.vlines(rS,-1.5*a,0,color = 'red')
    ax1.vlines(rQ,-1.5*a,0,color = 'blue')
    
    ax1.vlines([COMin,COMid,COMax],-1.5*a,0,color = 'black')
    
    ax1.legend()
    
    ax1.set_ylabel('$\propto$Energy')
    ax1.grid()
    
    ax2.semilogy(FS,(GDiff),'g.')
    ax2.set_xlabel('Radial distance from origin (Grids)')
    ax2.set_ylabel('Difference')
    ax2.vlines(rS,0,1e5,color = 'red')
    ax2.vlines(rQ,0,1e5,color = 'blue')
    
    ax2.set_ylim([np.min(GDiff),np.max(GDiff)])
    ax2.grid()
    
    plt.show()
    
    
    print('Generating Field Smoothing Report:')
    print('  The simulation runs on a %.0f^3 grid with total side length %.1f'%(resol,clength))
    print('  The simulation grid size is %.4f Code Units,\n  ' % ( GridLenFS))
    
    print('\n==========Grid Counts of Important Features=========\n')
    print("  Radius outside which the fields are practically indistinguishable (Grids): %.0f" % rS)
    print("  Modified Potential HWHM (Grids): %.0f" % rQ)
    

def CutoffReport(B,resol,clength,Steep = 100):
    import matplotlib.pyplot as plt
    fig_b = plt.figure(figsize=(7, 5))

    b = B*clength/2
    
    # Diagnostics For Field Cutoff
    ax1 = plt.subplot(111)
    
    xAr = np.linspace(-clength/2,clength/2,resol,endpoint = False)
    
    RAr = ne.evaluate('sqrt(xAr*xAr)')
                
    Premult = 1/2*(np.tanh((b-RAr)*Steep)+1)
    
    ax1.plot(xAr,Premult)
    ax1.set_ylim([0,1.1])
    
        
def EmbedParticle(particles,embeds,solitons):

    EI = 0
    
    embedsIter = embeds.copy()
    
    for Mollusk in embedsIter:
 
        Mass = Mollusk[0]*Mollusk[3]
        

        print(f"ULHelper: Calculating and loading the mass of embedded particle #{EI}.")
        Pearl = [Mass,Mollusk[1],Mollusk[2]]

        if not (Pearl in particles):
            particles.append(Pearl)
        
        if Mollusk[3] == 0:
            print(f"ULHelper: Embed structure #{EI} contains no black hole. Moving to regular solitons list.")
            
            Shell = [Mollusk[0],Mollusk[1],Mollusk[2],Mollusk[4]]
            solitons.append(Shell)
            embeds.remove(Mollusk)
            
        if Mollusk[3] == 1:
            print(f"ULHelper: Embed structure #{EI} contains no ULDM. Removing from list.")
            
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
 
def GenerateConfig(NS, central_mass, length, length_units, resol, duration, duration_units, step_factor, save_number, save_options, save_path, save_format, s_mass_unit, s_position_unit, s_velocity_unit, solitons,start_time, m_mass_unit, m_position_unit, m_velocity_unit, particles,embeds, Uniform,Density,density_unit,a,B,UVel,NoInteraction = False,Name = ''
           ):

    tm = GenFromTime()
    if NoInteraction:

        if Name != '':
            timestamp = Name
            
        else:
            timestamp = tm + f'@{resol}'
   
    else:
        print("What is the name of the run? Leave blank to use automatically generated timestamp.")
        InputName = input()

        if InputName != "":
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
            print(f"{Version} IO: Folder Not Empty. Are you sure you want to proceed [Y/n]? \nTo continue using the pre-existing config file, type [C].")
              
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

    Conf_data['Matter Particles'] = ({
    'Condition': particles,
    'Mass Units': m_mass_unit,
    'Position Units': m_position_unit,
    'Velocity Units': m_velocity_unit
    })

    Conf_data['Central Mass'] = 0

    Conf_data['Field Smoothing'] = np.real(a)

    Conf_data['NBody Cutoff Factor'] = np.real(0)


    Conf_data['Uniform Field Override'] = ({

            'Flag': Uniform,
            'Density Unit': density_unit,
            'Density Value': Density,
            'Uniform Velocity': UVel

            })


    with open('{}{}{}{}{}'.format('./', save_path, '/', timestamp, '/config.txt'), "w+") as outfile:
        json.dump(Conf_data, outfile,indent=4)

        
    if not NoInteraction:
        print('ULHelper: Compiled Config in Folder', timestamp)

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
        print("Which folder do you want to analyse? Blank to load the latest one.")
        Ind = int(input() or -1)

        if Ind == -1 and Automatic:
            print(f"ULHelper: Loading {Latest}")
            return Latest
        else:
            Ind = Log[Ind]
            print(f"ULHelper: Loading {runs[Ind]}")
            return runs[Ind]

def LoadConfig(loc):
        
        configfile = loc + '/config.txt'
        
        with open(configfile) as json_file:
            config = json.load(json_file)
        
        central_mass = 0
        
                
        if config["PyUL Version"] > S_version:
            raise RuntimeError("Configuration file generated by a newer version of PyUL.")
        
        ### Simulation Stuff
        try:
            save_options = SaveOptionsDigest(config["Save Options"]["Flags"])
            
        except KeyError: # Backwards Compatibility
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
        
        a = config["Field Smoothing"]
        
        B = config['NBody Cutoff Factor']
        
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
      
         
        return  NS, central_mass, length, length_units, resol, duration, duration_units, step_factor, save_number, save_options, save_format, s_mass_unit, s_position_unit, s_velocity_unit, solitons,start_time, m_mass_unit, m_position_unit, m_velocity_unit, particles, embeds, Uniform,Density, density_unit,a, B, UVel



def NBodyEnergy(MassListSI,TMDataSI,EndNum,a=0,length_units = ''): # kg, m, m/s, Int, code unit -1
    
    if a == 0:
        print('Warning: Using Standard Newtonian Potential.')
    
    else:
        ADim = convert(a,length_units,'l')
        
    NBo = len(MassListSI)
    print(f'Reconstructing Potential and Kinetic Energies for {NBo} stored objects.')

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
                    PS[i] += - 1*G*m1*m2*ADim/np.sqrt(1+ADim*2*rN**2)

            
    return NBo, KS, PS

# Point Mass Kepler
def FOSR(m,r):
    return np.sqrt(m/r)

# NFW Kepler
def FOSU(m,r):
    return np.sqrt(m/r)

def PopulateWithStars(embeds,particles,rIn = 0.4,rOut = 1.2,InBias = 0, NStars = 10, MassMax = 1e-5):

    for Halo in embeds:
              
        GPos = np.array(Halo[1])
        GVel = np.array(Halo[2])
        GMass = Halo[0]

        
        for i in range(NStars):

            r = (np.random.random()*(rOut-rIn) + rIn)

            theta = 2*np.pi*np.random.random()
            
            v = FOSR(GMass,r)

            Position = np.array([r*np.cos(theta),r*np.sin(theta),0]) + GPos
            Velocity = np.array([v*np.sin(theta),-v*np.cos(theta),0]) + GVel

            Mass = MassMax*np.random.random()

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


def SolEst(mass,length,resol,mass_unit = '',length_units = '', Plot = False, Density = 0, density_unit = ''): # Only deals with default solitons!
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
        print('Soliton too wide!')
        return 0

SolitonSizeEstimate = SolEst   

def DefaultDBL(v = 10,vUnit = 'm/s'):
    
    v = convert_between(v,vUnit,'m/s','v')
    
    return 2*np.pi*hbar/(axion_mass*v)


# Relevant for Paper 1
def SliceFinderC(TMState,resol,length,verbose = False):

    TMState = TMState[0:3]
    RNum = (np.array(TMState)*1/length+1/2)*resol
    RPt = np.floor(RNum)
    if verbose:
        print(f'The test mass particle #0 is closest to {RPt[0]}x,{RPt[1]}y,{RPt[2]}z in the data.')
    return RPt

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
    
    NS, central_mass, length, length_units, resol, duration, duration_units, step_factor, save_number, SO, save_format, s_mass_unit, s_position_unit, s_velocity_unit, solitons,start_time, m_mass_unit, m_position_unit, m_velocity_unit, particles, embeds, Uniform,Density, density_unit,a, B, UVel = LoadConfig(path_to_config)
    
    if SaveSpace:
        save_options = 'Minimal'
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
    
    if SaveSpace:
        save_options = [False,False,False,True,True,True,False,False,True,False]

    Units = []
    
    for ScanP in ScanParams:
    
        if ScanP == 'Density':
                
            Units.append(density_unit)

        elif ScanP == 'Resolution':

            Units.append('Grids')

        elif ScanP == 'TM_Mass':

            Units.append(m_mass_unit)

        elif ScanP == 'TM_vY':

            Units.append(m_velocity_unit)

        elif ScanP == 'UVelY':

            Units.append(s_velocity_unit)

        elif ScanP == 'Step_Factor':

            Units.append('')
            
        elif ScanP == 'Scaling':
            
            Units.append('')
            
            lengthOrig = length
            
        elif ScanP == 'Smoothing':
            KeepSmooth = False
            Units.append('')

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
            
            elif ScanParams[j] == 'TM_Mass':
                
                particles[0][0] = Pool[iDiv]   
                PString = 'M'
            
            elif ScanParams[j] == 'TM_vY':
                
                particles[0][2][1] = Pool[iDiv]
                PString = 'V'
                
            elif ScanParams[j] == 'UVelY':
                
                UVel[1] = Pool[iDiv]  
                PString = 'U'
                
            elif ScanParams[j] == 'Step_Factor':
                
                step_factor = Pool[iDiv]
                PString = 'F'
                
            elif ScanParams[j] == 'Scaling':
                
                length = lengthOrig * Pool[iDiv]
                PString = 'S'
                
                resol =  int(RPL * length)
            
            elif ScanParams[j] == 'Smoothing':
                a = Pool[iDiv]
                PString = 'A'
            
            else:
                raise ValueError('Unrecognized parameter type used.')
            
            
            Str += f'_{PString}{iDiv+1:02d}'
        
        # GenerateConfig Is Done Per i Loop
    
        GenerateConfig(NS, central_mass, length, length_units, resol, duration, duration_units, step_factor, save_number, save_options, save_path, save_format, s_mass_unit, s_position_unit, s_velocity_unit, solitons,start_time, m_mass_unit, m_position_unit, m_velocity_unit, particles,embeds, Uniform,Density,density_unit,a,B,UVel,True,Str)
        
        print('Generated config file for', Str)
        
    file = open('{}{}{}'.format('./', save_path, '/LookUp.txt'), "w+")
    
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
              particles,m_position_unit, Uniform, Density, UVel):
    
    print(length_units)
    particles, solitons, embeds = EmbedParticle(particles,embeds,solitons)
    import matplotlib.pyplot as plt
    PR = np.linspace(-length/2,length/2,resol,endpoint = False)
    
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    
    for i,soliton in enumerate(solitons):
        #print(f"Visualizing Soliton #{i}")
        mass = soliton[0]
        
        HWHM = SolEst(mass,length,resol,s_mass_unit,length_units)
        #print(f"FW WAS HERE {HWHM}")
        
        HWHM = convert_back(HWHM,length_units,'l')
        
        position = convert_between(np.array(soliton[1]),s_position_unit,length_units,'l')
        
        circ = plt.Circle((position[1],position[0]),HWHM,fill = False)
        
        ax.add_patch(circ)
    
    for i,particle in enumerate(particles):
        #print(f"Visualizing TM #{i}")
        position = convert_between(np.array(particle[1]),m_position_unit,length_units,'l')
        
        ax.scatter(position[1],position[0])
        
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
    
    ax.set_xlabel(f'x / {length_units}')
    ax.set_ylabel(f'y / {length_units}')
    ax.grid()
    
    if Uniform:
        if UVel[1]**2 + UVel[0]**2 != 0:
            ax.quiver(0.75,0.75,UVel[1],UVel[0])
        ax.text(0.75,0.75,f'Density: {Density}')
    
    return fig, ax

# New addition that fixes mesh compatibility
def XYSwap(Vec):
    return [Vec[1],Vec[0],Vec[2]]


class Simulation():
    
    def __init__(self,ConfigFile = ''):
        
        if ConfigFile == '':
            self.resol = 128