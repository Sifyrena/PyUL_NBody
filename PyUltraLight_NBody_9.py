Version   = str('PyUL9') # Handle used in console.
D_version = str('Integrator Build 2020 12 02') # Detailed Version
S_version = 9 # Short Version

import time
from datetime import datetime
import sys
import numpy as np
import numexpr as ne
import numba
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

import pyfftw
import h5py
import os
import scipy.fftpack
import multiprocessing
import json

import scipy.integrate as si


from scipy.interpolate import RegularGridInterpolator as RPI

# For Jupyter
from IPython.core.display import clear_output

try:
    import scipy.special.lambertw as LW
except ModuleNotFoundError:
    print('WARNING: scipy.special not installed. Using Pre-Computed value for W(-e^-2)')
    def LW(a):
        return -0.15859433956303937

pi = np.pi

hbar = 1.0545718e-34  # m^2 kg/s
parsec = 3.0857e16  # m
light_year = 9.4607e15  # m
solar_mass = 1.989e30  # kg
axion_mass = 1e-22 * 1.783e-36  # kg
G = 6.67e-11  # N m^2 kg^-2
omega_m0 = 0.31
H_0 = 67.7 * 1e3 / (parsec * 1e6)  # s^-1

# IMPORTANT

length_unit = (8 * np.pi * hbar ** 2 / (3 * axion_mass ** 2 * H_0 ** 2 * omega_m0)) ** 0.25
time_unit = (3 * H_0 ** 2 * omega_m0 / (8 * np.pi)) ** -0.5
mass_unit = (3 * H_0 ** 2 * omega_m0 / (8 * np.pi)) ** 0.25 * hbar ** 1.5 / (axion_mass ** 1.5 * G)


def LoadConfig(loc):
        
        configfile = loc + '/config.txt'
        
        with open(configfile) as json_file:
            config = json.load(json_file)
        
        central_mass = config["Central Mass"]
        
        ### Simulation Stuff
        
        save_options = config["Save Options"]["flags"]
        
        save_path = config["Save Options"]["folder"]
        
        hdf5 = config["Save Options"]["hdf5"]
        
        npy = config["Save Options"]["npy"]
        
        npz = config["Save Options"]["npz"]
        
        save_number = config["Save Options"]["number"]
        
        ### Time Stuff
        
        duration = config["Duration"]['Time Duration']
        
        start_time = config["Duration"]['Start Time']
        
        duration_units = config["Duration"]['Time Units']
        
        NS = int(config["RK Steps"])
        
        step_factor = float(config["Temporal Step Factor"])
        
        ### Space Stuff
        
        a = config["Field Smoothing"]
        
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

        ### Field Averaging
        
        NCV = np.array(config["Field-averaging Probes"]["Probe Array"])
        NCW = np.array(config["Field-averaging Probes"]["Probe Weights"])
        
        return  NS, central_mass, length, length_units, resol, duration, duration_units, step_factor, save_number, save_options, save_path, npz, npy, hdf5, s_mass_unit, s_position_unit, s_velocity_unit, solitons,start_time, m_mass_unit, m_position_unit, m_velocity_unit, particles, embeds, Uniform,Density, density_unit,a, NCV,NCW



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
    green =  scipy.fftpack.dctn(arr, type = 1, norm = None)
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

####################### AUX. FUNCTION TO GENERATE PROGRESS BAR

def prog_bar(iteration_number, progress, tinterval):
    size = 20
    status = ""
    
    ETAStamp = time.time() + (iteration_number - progress)*tinterval
    
    ETA = datetime.fromtimestamp(ETAStamp).strftime("%d/%m/%Y, %H:%M:%S")
    
    progress = float(progress) / float(iteration_number)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(size * progress))
    text = "\r[{}] {:.0f}% {}{}{} ({}{:.2f}s)".format(
        "龘" * block + "口" * (size - block), round(progress * 100, 0),
        status, 'Expected Finish Time: ',ETA,'Prev. Step: ',tinterval)
    
    sys.stdout.write(text)
    sys.stdout.flush()

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
            raise NameError('Unsupported length unit used')

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
            raise NameError('Unsupported mass unit used')

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
        else:
            raise NameError('Unsupported mass unit used')

    elif (type == 'v'):
        if (unit == ''):
            converted = value
        elif (unit == 'm/s'):
            converted = value * time_unit / length_unit
        elif (unit == 'km/s'):
            converted = value * 1e3 * time_unit / length_unit
        elif (unit == 'km/h'):
            converted = value * 1e3 / (60 * 60) * time_unit / length_unit
        else:
            raise NameError('Unsupported speed unit used')
            
            
    elif (type == 'd'):
        if (unit == ''):
            converted = value
        elif (unit == 'MSol/pc3'):
            converted = value * solar_mass / mass_unit * length_unit**3 / parsec**3
        elif (unit == 'kg/m3'):
            converted = value / mass_unit * length_unit**3
        else:
            raise NameError('Unsupported speed unit used')

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
            raise NameError('Unsupported length unit used')

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
            raise NameError('Unsupported mass unit used')

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
        else:
            raise NameError('Unsupported time unit used')

    elif (type == 'v'):
        if (unit == ''):
            converted = value
        elif (unit == 'm/s'):
            converted = value / time_unit * length_unit
        elif (unit == 'km/s'):
            converted = value / (1e3) / time_unit * length_unit
        elif (unit == 'km/h'):
            converted = value / (1e3) * (60 * 60) / time_unit * length_unit
        else:
            raise NameError('Unsupported speed unit used')
            
            
    elif (type == 'd'):
        if (unit == ''):
            converted = value
        elif (unit == 'MSol/pc3'):
            converted = value / solar_mass * mass_unit / length_unit**3 * parsec**3
        elif (unit == 'kg/m3'):
            converted = value * mass_unit / length_unit**3
        else:
            raise NameError('Unsupported speed unit used')

    else:
        raise TypeError('Unsupported conversion type')

    return converted

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

def initsoliton(funct, xarray, yarray, zarray, position, alpha, f, delta_x):
    for index in np.ndindex(funct.shape):
        # Note also that this distfromcentre is here to calculate the distance of every gridpoint from the centre of the soliton, not to calculate the distance of the soliton from the centre of the grid
        distfromcentre = (
            (xarray[index[0], 0, 0] - position[0]) ** 2 +
            (yarray[0, index[1], 0] - position[1]) ** 2 +
            (zarray[0, 0, index[2]] - position[2]) ** 2
        ) ** 0.5
        # Utilises soliton profile array out to dimensionless radius 5.6.
        if (np.sqrt(alpha) * distfromcentre <= 5.6):
            funct[index] = alpha * f[int(np.sqrt(alpha) * (distfromcentre / delta_x + 1))]

        else:
            funct[index] = 0
    return funct


def save_grid(
        rho, psi, resol, TMState, phisp,
        save_options,
        npy, npz, hdf5,
        loc, ix, its_per_save, GradientLog
        ):

        """
        Save various properties of the various grids in various formats
        """

        save_num = int((ix + 1) / its_per_save)

        if (save_options[0]):
            if (npy):
                file_name = "Outputs/rho_#{0}.npy".format(save_num)
                np.save(
                    os.path.join(os.path.expanduser(loc), file_name),
                    rho
                )
            if (npz):
                file_name = "Outputs/rho_#{0}.npz".format(save_num)
                np.savez(
                    os.path.join(os.path.expanduser(loc), file_name),
                    rho
                )
            if (hdf5):
                file_name = "Outputs/rho_#{0}.hdf5".format(save_num)
                file_name = os.path.join(os.path.expanduser(loc), file_name)
                f = h5py.File(file_name, 'w')
                dset = f.create_dataset("init", data=rho)
                f.close()
                
                
        if (save_options[1]): # Temporarily Appropriated!
            
            if (npy):
                file_name = "Outputs/psi_#{0}.npy".format(save_num)
                np.save(
                    os.path.join(os.path.expanduser(loc), file_name),
                    psi
                )
            if (npz):
                file_name = "Outputs/psi_#{0}.npz".format(save_num)
                np.savez(
                    os.path.join(os.path.expanduser(loc), file_name),
                    psi
                )
            if (hdf5):
                file_name = "Outputs/psi_#{0}.hdf5".format(save_num)
                file_name = os.path.join(os.path.expanduser(loc), file_name)
                f = h5py.File(file_name, 'w')
                dset = f.create_dataset("init", data=psi)
                f.close()
                
        if (save_options[2]):
            plane = rho[:, :, int(resol / 2)]
            if (npy):
                file_name = "Outputs/plane_#{0}.npy".format(save_num)
                np.save(
                    os.path.join(os.path.expanduser(loc), file_name),
                    plane
                )
            if (npz):
                file_name = "Outputs/plane_#{0}.npz".format(save_num)
                np.savez(
                    os.path.join(os.path.expanduser(loc), file_name),
                    plane
                )
            if (hdf5):
                file_name = "Outputs/plane_#{0}.hdf5".format(save_num)
                file_name = os.path.join(os.path.expanduser(loc), file_name)
                f = h5py.File(file_name, 'w')
                dset = f.create_dataset("init", data=plane)
                f.close()
                
                
        
                
        if (save_options[4]):
            line = rho[:, int(resol / 2), int(resol / 2)]
            file_name2 = "Outputs/line_#{0}.npy".format(save_num)
            np.save(
                os.path.join(os.path.expanduser(loc), file_name2),
                line
            )

        if (save_options[5]):
            if (npy):
                file_name = "Outputs/TM_#{0}.npy".format(save_num)
                np.save(
                    os.path.join(os.path.expanduser(loc), file_name),
                    TMState
                )
            if (npz):
                file_name = "Outputs/TM_#{0}.npz".format(save_num)
                np.savez(
                    os.path.join(os.path.expanduser(loc), file_name),
                    TMState
                )
            if (hdf5):
                file_name = "Outputs/TM_#{0}.hdf5".format(save_num)
                file_name = os.path.join(os.path.expanduser(loc), file_name)
                f = h5py.File(file_name, 'w')
                dset = f.create_dataset("init", data=TMState)
                f.close()
                
                
        if (save_options[6]):
            if (npy):
                file_name = "Outputs/Field3D_#{0}.npy".format(save_num)
                np.save(
                    os.path.join(os.path.expanduser(loc), file_name),
                    phisp
                )
            if (npz):
                file_name = "Outputs/Field3D_#{0}.npz".format(save_num)
                np.savez(
                    os.path.join(os.path.expanduser(loc), file_name),
                    phisp
                )
            if (hdf5):
                file_name = "Outputs/Field3D_#{0}.hdf5".format(save_num)
                file_name = os.path.join(os.path.expanduser(loc), file_name)
                f = h5py.File(file_name, 'w')
                dset = f.create_dataset("init", data=phisp)
                f.close()
                
                
                
        if (save_options[7]):
            
            phisp_slice = phisp[:,:,int(resol/2)] # z = 0
            
            if (npy):
                file_name = "Outputs/Field2D_#{0}.npy".format(save_num)
                np.save(
                    os.path.join(os.path.expanduser(loc), file_name),
                    phisp_slice
                )
            if (npz):
                file_name = "Outputs/Field2D_#{0}.npz".format(save_num)
                np.savez(
                    os.path.join(os.path.expanduser(loc), file_name),
                    phisp_slice
                )
            if (hdf5):
                file_name = "Outputs/Field2D_#{0}.hdf5".format(save_num)
                file_name = os.path.join(os.path.expanduser(loc), file_name)
                f = h5py.File(file_name, 'w')
                dset = f.create_dataset("init", data=phisp_slice)
                f.close()
                
                
        if (save_options[8]):
            
            if (npy):
                file_name = "Outputs/Gradients_#{0}.npy".format(save_num)
                np.save(
                    os.path.join(os.path.expanduser(loc), file_name),
                    GradientLog
                )
            if (npz):
                file_name = "Outputs/Gradients_#{0}.npz".format(save_num)
                np.savez(
                    os.path.join(os.path.expanduser(loc), file_name),
                    GradientLog
                )
            if (hdf5):
                file_name = "Outputs/Gradients_#{0}.hdf5".format(save_num)
                file_name = os.path.join(os.path.expanduser(loc), file_name)
                f = h5py.File(file_name, 'w')
                dset = f.create_dataset("init", data=GradientLog)
                f.close()
                
                
        if (save_options[9]): # Temporarily Appropriated!
            
            psislice = psi[:, :, int(resol / 2)]
            
            psiarg = np.angle(psislice)
            
            if (npy):
                file_name = "Outputs/Arg2D_#{0}.npy".format(save_num)
                np.save(
                    os.path.join(os.path.expanduser(loc), file_name),
                    psiarg
                )
            if (npz):
                file_name = "Outputs/Arg2D_#{0}.npz".format(save_num)
                np.savez(
                    os.path.join(os.path.expanduser(loc), file_name),
                    psiarg
                )
            if (hdf5):
                file_name = "Outputs/Arg2D_#{0}.hdf5".format(save_num)
                file_name = os.path.join(os.path.expanduser(loc), file_name)
                f = h5py.File(file_name, 'w')
                dset = f.create_dataset("init", data=psiarg)
                f.close()


# CALCULATE_ENERGIES, FW

def calculate_energiesF(save_options, resol,
        psi, cmass, TMState, masslist, Vcell, phisp, karray2, funct,
        fft_psi, ifft_funct,
        egpcmlist, egpsilist, ekandqlist, egylist, mtotlist,xarray, yarray, zarray, a):
    
    if (save_options[3]):
        
            egyarr = pyfftw.zeros_aligned((resol, resol, resol), dtype='float64')
            
            mPhi = pyfftw.zeros_aligned((resol, resol, resol), dtype='float64')

            # Gravitational potential energy density associated with the point masses potential
            
            egyarr = ne.evaluate('real((abs(psi))**2)')
            
            for MI in range(len(masslist)):
        
                State = TMState[int(MI*6):int(MI*6+5)]
            
                TMx = State[0]
                TMy = State[1]
                TMz = State[2]
                
                mT = masslist[MI]
            
                distarrayTM = ne.evaluate("((xarray-TMx)**2+(yarray-TMy)**2+(zarray-TMz)**2)**0.5") # Radial coordinates
                                
                mPhi = ne.evaluate("mPhi-a*mT/(a*distarrayTM+exp(-a*distarrayTM))") # Potential Due to Test Mass
                
                phisp = ne.evaluate("phisp+a*mT/(a*distarrayTM+exp(-a*distarrayTM))") # Restoring Self-Interaction
                
            
            egyarr = ne.evaluate('real(mPhi*egyarr)')
            egpcmlist.append(Vcell * np.sum(egyarr))
            tot = Vcell * np.sum(egyarr)

            # Gravitational potential energy density of self-interaction of the condensate
            egyarr = ne.evaluate('real(0.5*(phisp)*real((abs(psi))**2))') # Restored
            egpsilist.append(Vcell * np.sum(egyarr))
            tot = tot + Vcell * np.sum(egyarr)

            # TODO: Does this reuse the memory of funct?  That is the
            # intention, but likely isn't what is happening
            funct = fft_psi(psi)
            funct = ne.evaluate('-karray2*funct')
            funct = ifft_funct(funct)
            egyarr = ne.evaluate('real(-0.5*conj(psi)*funct)')
            ekandqlist.append(Vcell * np.sum(egyarr))
            tot = tot + Vcell * np.sum(egyarr)

            egylist.append(tot)

            egyarr = ne.evaluate('real((abs(psi))**2)')
            mtotlist.append(Vcell * np.sum(egyarr))



### Toroid or something
            
def FWrap(TMx, TMy, TMz, gridlength):
    
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
            
            
######################### NO LONGER RELEVANT
## FieldProcess Doesn't Use That Much Time

def FieldProcess(phik,gridlength,resol):
    # Parts of FieldGradient that's shared for all masses.
    
    length = gridlength

    FieldFT = np.zeros([resol,resol,resol],dtype = 'complex128')

    # Input
    
    FieldFTD = phik

    # Padding Into Cube - Converting rfftn to fftn #
    FieldFT[:,:,0:int(resol/2+1)] = FieldFTD

    for i in range(int(resol/2+1),resol):
        FieldFT[:,:,i] = np.conj(FieldFTD[:,:,int(resol-i)]) # The smile of Hermite
        
    return FieldFT
    
    # Using Fourier Series

def FieldGradient(gridlength,Kx,Ky,Kz,FieldFT,position,resol):
    
    xT = position[0]
    yT = position[1]
    zT = position[2]

    # Store Sine Waves as Fourier Grid
    TMxGrid = ne.evaluate("1j*Kx*exp(1j*(Kx*(xT+gridlength/2)+Ky*(yT+gridlength/2)+Kz*(zT+gridlength/2)))")
    TMyGrid = ne.evaluate("1j*Ky*exp(1j*(Kx*(xT+gridlength/2)+Ky*(yT+gridlength/2)+Kz*(zT+gridlength/2)))")
    TMzGrid = ne.evaluate("1j*Kz*exp(1j*(Kx*(xT+gridlength/2)+Ky*(yT+gridlength/2)+Kz*(zT+gridlength/2)))")
    
    # Returned Values (Real Part Kept)
    GradX = np.real(np.sum(ne.evaluate("FieldFT*TMxGrid"))/resol**3)
    GradY = np.real(np.sum(ne.evaluate("FieldFT*TMyGrid"))/resol**3)
    GradZ = np.real(np.sum(ne.evaluate("FieldFT*TMzGrid"))/resol**3)
    
    #print('\n',GradX,'\n',GradY,'\n',GradZ)
    
    #print(max(np.abs(GradX),np.abs(GradY),np.abs(GradZ)))
    return np.array([GradX, GradY, GradZ])

    

def FWNBodyR(t,TMState,masslist,FieldFT,gridlength,resol,Kx,Ky,Kz,a,HWHM,NCV,NCW):
   
    dTMdt = 0*TMState
    GradientLog = np.zeros(len(masslist)*3)
    
    for i in range(len(masslist)):
        
        #0,6,12,18,...
        Ind = int(6*i)
        IndD = int(3*i)
        
        #XDOT
        dTMdt[Ind]   =  TMState[Ind+3]
        #YDOT
        dTMdt[Ind+1] =  TMState[Ind+4]
        #ZDOT
        dTMdt[Ind+2] =  TMState[Ind+5]
        
        #x,y,z
        
        GradientLocal = np.zeros(3)
        poslocal = np.array([TMState[Ind],TMState[Ind+1],TMState[Ind+2]])
        
        for NC in range(len(NCW)): # This is now an average
            if NCW[NC] != 0:
                poslocalNC = poslocal + HWHM*NCV[NC]
                GradientLocal = GradientLocal -1*FieldGradient(gridlength,Kx,Ky,Kz,FieldFT,poslocalNC,resol)*NCW[NC]
            
        #Initialized Against ULDM Field
        #XDDOT
        dTMdt[Ind+3] =  GradientLocal[0]
        #YDDOT
        dTMdt[Ind+4] =  GradientLocal[1]
        #ZDDOT
        dTMdt[Ind+5] =  GradientLocal[2]
    
        for ii in range(len(masslist)):
            
            if ii != i:
                
                IndX = int(6*ii)
                
                # print(ii)
                
                poslocalX = np.array([TMState[IndX],TMState[IndX+1],TMState[IndX+2]])
                
                rV = poslocalX - poslocal
                
                rVL = np.linalg.norm(rV)
                
                rSmooth = (-a**2+a**2*np.exp(-a*rVL))/(rVL*(a*rVL+np.exp(-a*rVL))**2)
                
                # Differentiated within Note 000.0F
                
                #XDDOT with Gravity
                dTMdt[Ind+3] = dTMdt[Ind+3] - masslist[ii]*rSmooth*rV[0]
                #YDDOT
                dTMdt[Ind+4] = dTMdt[Ind+4] - masslist[ii]*rSmooth*rV[1]
                #ZDDOT
                dTMdt[Ind+5] = dTMdt[Ind+5] - masslist[ii]*rSmooth*rV[2]
        
        GradientLog[IndD] = GradientLocal[0]
        GradientLog[IndD+1] = GradientLocal[1]
        GradientLog[IndD+2] = GradientLocal[2]
                
    
    return dTMdt, GradientLog


def FloaterAdvanceR(TMState,h,NS,FieldFT,masslist,gridlength,resol,Kx,Ky,Kz,a, HWHM,NCV,NCW):
        # We pass on phik into the Gradient Function Above. This saves one inverse FFT call.
        # The N-Body Simulation is written from scratch

        # 
        if NS == 0:
 
            Step, GradientLog = FWNBodyR(0,TMState,masslist,FieldFT,gridlength,resol,Kx,Ky,Kz,a,HWHM,NCV,NCW)
            TMStateOut = TMState + Step*h
        

        elif NS%4 == 0:
            
            NRK = int(NS/4)
            
            H = h/NRK
            
            for RKI in range(NRK):
                TMK1, Trash = FWNBodyR(0,TMState,masslist,FieldFT,gridlength,resol,Kx,Ky,Kz,a,HWHM,NCV,NCW)
                TMK2, Trash = FWNBodyR(0,TMState + H/2*TMK1,masslist,FieldFT,gridlength,resol,Kx,Ky,Kz,a,HWHM,NCV,NCW)
                TMK3, Trash = FWNBodyR(0,TMState + H/2*TMK2,masslist,FieldFT,gridlength,resol,Kx,Ky,Kz,a,HWHM,NCV,NCW)
                TMK4, GradientLog = FWNBodyR(0,TMState + H*TMK3,masslist,FieldFT,gridlength,resol,Kx,Ky,Kz,a,HWHM,NCV,NCW)
                TMState = TMState + H/6*(TMK1+2*TMK2+2*TMK3+TMK4)
                
            TMStateOut = TMState

        return TMStateOut, GradientLog
    


### Actual Rigid NBody Integrator
# Rewritten Interpolation 

def FieldGradientI(position,GxI,GyI,GzI):
            
    # Returned Values (Real Part Kept)
    GradX = GxI(position)
    GradY = GyI(position)
    GradZ = GzI(position)
        #print(GradX)
    
    #print(max(np.abs(GradX),np.abs(GradY),np.abs(GradZ))) 
    
    return np.array([GradX, GradY, GradZ])


def FWNBodyI(t,TMState,masslist,GxI,GyI,GzI,a,HWHM,NCV,NCW):
   
    dTMdt = 0*TMState
    GradientLog = np.zeros(len(masslist)*3)
    
    for i in range(len(masslist)):
        
        #0,6,12,18,...
        Ind = int(6*i)
        IndD = int(3*i)
        
        #XDOT
        dTMdt[Ind]   =  TMState[Ind+3]
        #YDOT
        dTMdt[Ind+1] =  TMState[Ind+4]
        #ZDOT
        dTMdt[Ind+2] =  TMState[Ind+5]
        
        #x,y,z
        
        GradientLocal = np.array([[0],[0],[0]])

        poslocal = np.array([TMState[Ind],TMState[Ind+1],TMState[Ind+2]])
        
        for NC in range(len(NCW)): # This is now an average
            if NCW[NC] != 0:
                poslocalNC = poslocal + HWHM*NCV[NC]
                GradientAdd = -1*FieldGradientI(poslocalNC,GxI,GyI,GzI)*NCW[NC]
                GradientLocal = GradientLocal + GradientAdd
        

        #Initialized Against ULDM Field
        #XDDOT
        dTMdt[Ind+3] =  GradientLocal[0]
        #YDDOT
        dTMdt[Ind+4] =  GradientLocal[1]
        #ZDDOT
        dTMdt[Ind+5] =  GradientLocal[2]
    
        for ii in range(len(masslist)):
            
            if ii != i:
                
                IndX = int(6*ii)
                
                # print(ii)
                
                poslocalX = np.array([TMState[IndX],TMState[IndX+1],TMState[IndX+2]])
                
                rV = poslocalX - poslocal
                
                rVL = np.linalg.norm(rV)
                
                rSmooth = (-a**2+a**2*np.exp(-a*rVL))/(rVL*(a*rVL+np.exp(-a*rVL))**2)
                
                # Differentiated within Note 000.0F
                
                #XDDOT with Gravity
                dTMdt[Ind+3] = dTMdt[Ind+3] - masslist[ii]*rSmooth*rV[0]
                #YDDOT
                dTMdt[Ind+4] = dTMdt[Ind+4] - masslist[ii]*rSmooth*rV[1]
                #ZDDOT
                dTMdt[Ind+5] = dTMdt[Ind+5] - masslist[ii]*rSmooth*rV[2]
        
        GradientLog[IndD] = GradientLocal[0]
        GradientLog[IndD+1] = GradientLocal[1]
        GradientLog[IndD+2] = GradientLocal[2]
                
    
    return dTMdt, GradientLog

def FloaterAdvanceI(TMState,h,masslist,NS,GxI,GyI,GzI,a,HWHM,NCV,NCW):
        # We pass on phik into the Gradient Function Above. This saves one inverse FFT call.
        # The N-Body Simulation is written from scratch

        # 
        if NS == 0:
 
            Step, GradientLog = FWNBodyI_jit(0,TMState,masslist,GxI,GyI,GzI,a,HWHM,NCV,NCW)
            
            TMStateOut = TMState + Step*h
        

        elif NS%4 == 0:
            
            NRK = int(NS/4)
            
            H = h/NRK
            
            for RKI in range(NRK):
                TMK1, Trash = FWNBodyI_jit(0,TMState,masslist,GxI,GyI,GzI,a,HWHM,NCV,NCW)
                TMK2, Trash = FWNBodyI_jit(0,TMState + H/2*TMK1,masslist,GxI,GyI,GzI,a,HWHM,NCV,NCW)
                TMK3, Trash = FWNBodyI_jit(0,TMState + H/2*TMK2,masslist,GxI,GyI,GzI,a,HWHM,NCV,NCW)
                TMK4, GradientLog = FWNBodyI_jit(0,TMState + H*TMK3,masslist,GxI,GyI,GzI,a,HWHM,NCV,NCW)
                TMState = TMState + H/6*(TMK1+2*TMK2+2*TMK3+TMK4)
                
            TMStateOut = TMState

        return TMStateOut, GradientLog


    
######################### Soliton Init Factory Setting!

def LoadDefaultSoliton(Profile):
    
    f = np.load(Profile)
    
    print(f"\n{Version} Load Soliton: Loaded original PyUL soliton profiles.")
    
    return f

######################### Soliton Init

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
            return -(2/r)*c+2*b*a - 2*(Smoo*BHmass / (Smoo*r+np.exp(-Smoo*r)))*a + 2 * Lambda * a **3

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
            
            raise ValueError("First loop of shooting algorithm failed to converge to given ratio.")
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

    def g1(r, a, b, c, BHmass = BHmass):
        return -(2/r)*c+2*b*a - 2*(Smoo*BHmass / (Smoo*r+np.exp(-Smoo*r)))*a + 2 * Lambda * a **3

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
        
        return delta_x, alpha, beta
    
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

print(f'{Version} Runtime: JIT optimisations under development. Please ignore Numba warnings for now.')

FloaterAdvanceI_jit = numba.jit(FloaterAdvanceI)

initsoliton_jit = numba.jit(initsoliton)

FWNBodyI_jit = numba.jit(FWNBodyI)

IP_jit = numba.jit(isolatedPotential)

PC_jit = numba.jit(planeConvolve)
    
    
######################### New Version With Built-in I/O Management
def evolve(save_path,run_folder,Method,Draft):
    
    clear_output()
    print(f"=========={Version}: {D_version}==========")
    
    timestamp = run_folder
    
    configfile = './' + save_path + '/' + run_folder
    
    file = open('{}{}{}'.format('./', save_path, '/latest.txt'), "w+")
    file.write(run_folder)
    
    loc = configfile
            
    num_threads = multiprocessing.cpu_count()
    
    NS, central_mass, length, length_units, resol, duration, duration_units, step_factor, save_number, save_options, save_path, npz, npy, hdf5, s_mass_unit, s_position_unit, s_velocity_unit, solitons,start_time, m_mass_unit, m_position_unit, m_velocity_unit, particles,embeds, Uniform,Density, density_unit,a, NCV,NCW = LoadConfig(configfile)

    # Embedded particles are Pre-compiled into the list.
    
    print(f"{Version} IO   : Loaded Parameters from {configfile}")

    NumSol = len(solitons)
    NumTM = len(particles)

    if Method == 2:
        print(f"{Version} NBody: Using DCT Sum interpolation For gravity.")
            
    if Method == 1: # 1 = Real Space Interpolation (Orange), 2 = Fourier Sum (White)
        print(f"{Version} NBody: Using Linear Interpolation for gravity.")
    
    print(f"{Version} NBody: The point mass field smoothing factor is {a:.5f}.")
    
    print(f"{Version} SP: Simulation grid resolution is {resol}^3.")
    
    HWHM = (LW(-np.exp(-2))+2)/a # inversely proportional to a.
    
    HWHM = np.real(HWHM)
    
    if HWHM > length / 4:
        print("WARNING: Field Smoothing factor too small, and the perfomance will be compromised.")
    
    if HWHM <= length / (4*resol):
        print("WARNING: The field peak is narrower than half a grid. There may be artifitial energy fluctuations.")
    
    if np.sum(NCW) != 0:
        NCW = NCW/np.sum(NCW) # Normalized to 1
    else:
        NCW[0] = 1
    
    
    masslist = []
    
    TMState = []

    ##########################################################################################
    #SET INITIAL CONDITIONS

    gridlength = convert(length, length_units, 'l')

    t = convert(duration, duration_units, 't')

    t0 = convert(start_time, duration_units, 't')
    
    cmass = convert(central_mass, m_mass_unit, 'm')

    Density = convert(Density,density_unit,'d')
    
    Vcell = (gridlength / float(resol)) ** 3
    
    ne.set_num_threads(num_threads)
        
    ##########################################################################################
    # CREATE THE TIMESTAMPED SAVE DIRECTORY AND CONFIG.TXT FILE

    save_path = os.path.expanduser(save_path)
       
    ##########################################################################################
    # SET UP THE REAL SPACE COORDINATES OF THE GRID - FW Revisit

    gridvec = np.linspace(-gridlength / 2.0, gridlength / 2.0, resol, endpoint=False)
    
    xarray, yarray, zarray = np.meshgrid(
        gridvec, gridvec, gridvec,
        sparse=True, indexing='ij')
    
    distarray = ne.evaluate("(xarray**2+yarray**2+zarray**2)**0.5") # Radial coordinates
    
    WN = 2*np.pi*np.fft.fftfreq(resol, gridlength/(resol)) # 2pi Pre-multiplied
    
    Kx,Ky,Kz = np.meshgrid(WN,WN,WN,sparse=True, indexing='ij',)
    
    ##########################################################################################
    # SET UP K-SPACE COORDINATES FOR COMPLEX DFT (NOT RHO DFT)

    kvec = 2 * np.pi * np.fft.fftfreq(resol, gridlength / float(resol))
    
    kxarray, kyarray, kzarray = np.meshgrid(
        kvec, kvec, kvec,
        sparse=True, indexing='ij',
    )
    
    karray2 = ne.evaluate("kxarray**2+kyarray**2+kzarray**2")
    ##########################################################################################
    
    delta_x = 0.00001 # Needs to match resolution of soliton profile array file. Default = 0.00001

    warn = 0

    psi = pyfftw.zeros_aligned((resol, resol, resol), dtype='complex128')
    
    funct = pyfftw.zeros_aligned((resol, resol, resol), dtype='complex128')    
    
    # INITIALISE SOLITONS WITH SPECIFIED MASS, POSITION, VELOCITY, PHASE

    if Uniform:
        
        DensityCom = Density * (gridlength**3) / resol**3
        print('==============================================================================')
        print(f"{Version} Init: Solitons overridden with a uniform wavefunction with no phase.")
        print(f"{Version} Init: Initial density is {DensityCom:.5f} per mesh grid.")
        print('==============================================================================')
        psi = ne.evaluate("0*psi + sqrt(DensityCom)")
        
        
    else:
        # Load Embedded Objects first
        
        EI = 0
        for emb in embeds:

            # 0.     1.     2.          3.                   4.
            # [mass,[x,y,z],[vx,vy,vz], BH-Total Mass Ratio, Phase]
            Ratio = emb[3]
            RatioBU = float(Ratio/(1-Ratio))
            try:
                delta_xL, prealphaL, betaL = LoadSolitonConfig(RatioBU)
            except RuntimeError:
                print('==============================================================================')
                print(f'{Version} IO: Note that this process will not be required for repeated runs. To remove custom profiles, go to the folder /Soliton Profile Files/Custom.')
                print(f'Generating profile for Soliton with MBH/MSoliton = {RatioBU:.4f}, Part 1')
                GMin,GMax = BHGuess(RatioBU)
                s, BHMass = BHRatioTester(RatioBU,30,1e-6,GMin,GMax,a)
                print(f'Generating profile for Soliton with MBH/MSoliton = {RatioBU:.4f}, Part 2')
                SolitonProfile(BHMass,s,a,False)
                print('==============================================================================')
                delta_xL, prealphaL, betaL = LoadSolitonConfig(RatioBU)

            # L stands for Local, as in it's only used once.
            fL = LoadSoliton(RatioBU)

            print(f"{Version} Init: Loaded embedded soliton {EI} with BH-Soliton mass ratio {RatioBU:.4f}.")

            mass = convert(emb[0], s_mass_unit, 'm')*(1-Ratio)

            position = convert(np.array(emb[1]), s_position_unit, 'l')
            velocity = convert(np.array(emb[2]), s_velocity_unit, 'v')

            # Note that alpha and beta parameters are computed when the initial_f.npy soliton profile file is generated.

            alphaL = (mass / prealphaL) ** 2

            phase = emb[4]

            funct = initsoliton_jit(funct, xarray, yarray, zarray, position, alphaL, fL, delta_xL)
            ####### Impart velocity to solitons in Galilean invariant way
            velx = velocity[0]
            vely = velocity[1]
            velz = velocity[2]
            funct = ne.evaluate("exp(1j*(alphaL*betaL*t0 + velx*xarray + vely*yarray + velz*zarray -0.5*(velx*velx+vely*vely+velz*velz)*t0  + phase))*funct")
            psi = ne.evaluate("psi + funct")

            EI += 1


        if solitons != []:
            
            print(f"{Version} Init: Loaded unperturbed soliton.")
            f = LoadDefaultSoliton('./Soliton Profile Files/initial_f.npy')

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
       

        
    rho = ne.evaluate("real(abs(psi)**2)")

    fft_psi = pyfftw.builders.fftn(psi, axes=(0, 1, 2), threads=num_threads)
    
    ifft_funct = pyfftw.builders.ifftn(funct, axes=(0, 1, 2), threads=num_threads)

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

    h = t / float(actual_num_steps)

    ##########################################################################################
    # SETUP PADDED POTENTIAL HERE (From JLZ)

    rhopad = pyfftw.zeros_aligned((2*resol, resol, resol), dtype='float64')
    bigplane = pyfftw.zeros_aligned((2*resol, 2*resol), dtype='float64')

    fft_X = pyfftw.builders.fftn(rhopad, axes=(0, ), threads=num_threads)
    ifft_X = pyfftw.builders.ifftn(rhopad, axes=(0, ), threads=num_threads)

    fft_plane = pyfftw.builders.fftn(bigplane, axes=(0, 1), threads=num_threads)
    ifft_plane = pyfftw.builders.ifftn(bigplane, axes=(0, 1), threads=num_threads)
    
    phisp = pyfftw.zeros_aligned((resol, resol, resol), dtype='float64')

    fft_phi = pyfftw.builders.fftn(phisp, axes=(0, 1, 2), threads=num_threads)

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

    ##########################################################################################
    # COMPUTE INTIAL VALUE OF POTENTIAL

    if Uniform:
        print(f"{Version} SP: Poisson Equation Solveed Using FFT.")
        phisp = irfft_phi(phik)
    else:
        print(f"{Version} SP: Initiating Green functions for simulation region.")   
        green = makeDCTGreen(resol) #make Green's function ONCE
        #green = makeEvenArray(green)
        phisp = IP_jit(rho, green, gridlength, fft_X, ifft_X, fft_plane, ifft_plane)
        
    ##########################################################################################
    # FW NBody
    MI = 0
    
    for particle in particles:
               
        TMmass = convert(particle[0], m_mass_unit, 'm')
        
        if TMmass == 0:
            print(f"{Version} NBody: Particle #{MI} loaded as observer.")
        else:
            print(f"{Version} NBody: Particle #{MI} loaded, with (code) mass {TMmass:.3f}")
        
        masslist.append(TMmass)

        position = convert(np.array(particle[1]), m_position_unit, 'l')  
        velocity = convert(np.array(particle[2]), m_velocity_unit, 'v')
        
        IND = int(6*MI)
        
        TMx = position[0]
        TMy = position[1]
        TMz = position[2]
        
        Vx = velocity[0]
        Vy = velocity[1]
        Vz = velocity[2]
                       
        distarrayTM = ne.evaluate("((xarray-TMx)**2+(yarray-TMy)**2+(zarray-TMz)**2)**0.5") # Radial coordinates
        
        TMState.append([TMx,TMy,TMz,Vx,Vy,Vz])
        
        phisp = ne.evaluate("phisp-a*TMmass/(a*distarrayTM+exp(-a*distarrayTM))")
        
        MI = int(MI + 1)
        
        # FW
    
    masslist = np.array(masslist)
    TMState = np.array(TMState)
    TMState = TMState.flatten(order='C')
    
    TMStateDisp = TMState.reshape((-1,6))
    
    print(f"{Version} NBody: The initial state vector is:")
    print(TMStateDisp)
        
    MI = 0
    
    if (save_options[3]):
        egylist = []
        egpcmlist = []
        egpsilist = []
        ekandqlist = []
        mtotlist = []

        calculate_energiesF(save_options, resol,
        psi, cmass, TMState, masslist, Vcell, phisp, karray2, funct,
        fft_psi, ifft_funct,
        egpcmlist, egpsilist, ekandqlist, egylist, mtotlist,xarray, yarray, zarray, a )
    
    GradientLog = np.zeros(NumTM*3)
    
    os.mkdir(str(loc + '/Outputs'))
    ##########################################################################################
    # PRE-LOOP SAVE I.E. INITIAL CONFIG
    save_grid(
            rho, psi, resol, TMState, phisp,
            save_options,
            npy, npz, hdf5,
            loc, -1, 1, GradientLog,
    )
    
    
    print(f'{Version} IO: Successfully initiated Wavefunction, (Green Function), and NBody Initial Conditions. Dumping to file.')
    
    np.save(f'{loc}/Initial_psi.npy',psi)
    np.save(f'{loc}/Initial_NBody.npy',psi)
    
    if not Uniform:
        np.save(f'{loc}/Initial_green.npy',green) 

    tBegin = time.time()
    
    tBeginDisp = datetime.fromtimestamp(tBegin).strftime("%d/%m/%Y, %H:%M:%S")
    
    print("======================================================")
    
    print(f"{Version} Runtime: Simulation Started at {tBeginDisp}.")
    ##########################################################################################
    # PRE-LOOP ENERGY CALCULATION

    ##########################################################################################
    # LOOP NOW BEGINS

    HaSt = 1  # 1 for a half step 0 for a full step

    tenth = float(save_number/10) #This parameter is used if energy outputs are saved while code is running.

    print(f"{Version} Runtime: The total number of ULDM simulation steps is {int(actual_num_steps)}")
    
    if warn == 1:
        print("WARNING: Detected significant overlap between solitons in I.V.")
    print('\n')
    tinit = time.time()
  
    for ix in range(actual_num_steps):
        
        if HaSt == 1:
            psi = ne.evaluate("exp(-1j*0.5*h*phisp)*psi")
            HaSt = 0

        else:
            psi = ne.evaluate("exp(-1j*h*phisp)*psi")
               
        funct = fft_psi(psi)
        funct = ne.evaluate("funct*exp(-1j*0.5*h*karray2)")
        
        psi = ifft_funct(funct)
        rho = ne.evaluate("real(abs(psi)**2)")
        
        phik = rfft_rho(rho)
        
        phik = ne.evaluate("-4*pi*(phik)/rkarray2")
        
        phik[0, 0, 0] = 0


        # New Green Function Methods
        if Uniform:
            phisp = irfft_phi(phik)
        else:
            phisp = IP_jit(rho, green, gridlength, fft_X, ifft_X, fft_plane, ifft_plane)
        
            
        # FW TEST STEP MAGIC HAPPENS HERE

        if Method == 2:
            TMState, GradientLog = FloaterAdvanceR(TMState,h,masslist,NS,FieldFT,masslist,gridlength,resol,Kx,Ky,Kz,a, HWHM,NCV,NCW)
        
        elif Method == 1:
            
            phiGrad = np.gradient(phisp, gridvec,gridvec,gridvec, edge_order = 2)
            
            phiGradX = phiGrad[0]
            
            GxI = RPI([gridvec,gridvec,gridvec],phiGradX,method = 'linear',bounds_error = False, fill_value = 0)
            
            phiGradY = phiGrad[1]
            
            GyI = RPI([gridvec,gridvec,gridvec],phiGradY,method = 'linear',bounds_error = False, fill_value = 0)
            
            phiGradZ = phiGrad[2]
            
            GzI = RPI([gridvec,gridvec,gridvec],phiGradZ,method = 'linear',bounds_error = False, fill_value = 0)
            
            TMState, GradientLog = FloaterAdvanceI_jit(TMState,h,masslist,NS,GxI,GyI,GzI,a,HWHM,NCV,NCW)
    
        for MI in range(NumTM):
        
            State = TMState[int(MI*6):int(MI*6+5)]
            
            TMx = State[0]
            TMy = State[1]
            TMz = State[2]
            
            if np.max(np.abs([TMx,TMy,TMz])) > gridlength/2:
                
                TMx, TMy, TMz = FWrap(TMx, TMy, TMz, gridlength)
                
                print(f"{Version} Runtime: Particle {MI} out of bounds, wrapping back.")
                print(TMx,TMy,TMz,gridlength/2)
                TMState[int(MI*6)]  = TMx
                TMState[int(MI*6+1)]= TMy
                TMState[int(MI*6+2)]= TMz
                
            mT = masslist[MI]
            
            distarrayTM = ne.evaluate("((xarray-TMx)**2+(yarray-TMy)**2+(zarray-TMz)**2)**0.5") # Radial coordinates
        
            phisp = ne.evaluate("phisp-a*mT/(a*distarrayTM+exp(-a*distarrayTM))")
            
        #phisp = ne.evaluate("phisp-a*cmass/(a*distarray+exp(-a*distarray))")


        #Next if statement ensures that an extra half step is performed at each save point
        if (((ix + 1) % its_per_save) == 0) and HaSt == 0:
            psi = ne.evaluate("exp(-1j*0.5*h*phisp)*psi")
            rho = ne.evaluate("real(abs(psi)**2)")
            HaSt = 1

            #Next block calculates the energies at each save, not at each timestep.
            if (save_options[3]):
                calculate_energiesF(save_options, resol, psi, 
                                    cmass, TMState, masslist, Vcell, phisp, karray2, funct,
                                    fft_psi, ifft_funct, egpcmlist, egpsilist, ekandqlist,
                                    egylist, mtotlist,xarray, yarray, zarray, a)

           
        ################################################################################
        # SAVE DESIRED OUTPUTS
        if ((ix + 1) % its_per_save) == 0:
                        
            save_grid(
                    rho, psi, resol, TMState, phisp,
                    save_options,
                    npy, npz, hdf5,
                    loc, ix, its_per_save, GradientLog,
            )
            
            
            
            if (save_options[3]):  
                np.save(os.path.join(os.path.expanduser(loc), "Outputs/egylist.npy"), egylist)
                np.save(os.path.join(os.path.expanduser(loc), "Outputs/egpcmlist.npy"), egpcmlist)
                np.save(os.path.join(os.path.expanduser(loc), "Outputs/egpsilist.npy"), egpsilist)
                np.save(os.path.join(os.path.expanduser(loc), "Outputs/ekandqlist.npy"), ekandqlist)
                np.save(os.path.join(os.path.expanduser(loc), "Outputs/masseslist.npy"), mtotlist)


        ################################################################################
        # UPDATE INFORMATION FOR PROGRESS BAR

        tint = time.time() - tinit
        tinit = time.time()
        prog_bar(actual_num_steps, ix + 1, tint)

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
    print('')
    print('')
    print(f"{Version} Runtime: Run Complete. Time Elapsed (d:h:m:s): {day:.0f}:{hour:.0f}:{minutes:.0f}:{seconds:.2f}")

    
      