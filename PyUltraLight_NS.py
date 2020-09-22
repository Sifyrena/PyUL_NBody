# PyUL_Floater_Advanced

#######################THIS IS AN FWPHYS PHD SUB PROJECT
# Codename 000.01 - 000.11
# Build K
# Created: 10am, 2020/Aug/05
# This Archive: 10am, 2020/Sept/15
# Publication Type: Internal Uses Only

#######################PURPOSE
# PyUL Test Mass Simulation Scheme. -Grad(Phi) = F = ma.
# This program simulates all the familiar soliton dynamics (untouched), 
# while also allowing the user to add a non-interacting test mass of arbitrary inertial mass
# whose trajectory is integrated with an RK4 solver at every simulation step. 

# This version fixes energy computation, taking bits from the Notebook.

#######################USAGE UPGRADES AND DESIGN NOTES
# A new type of save files: Test Mass Coords and Speed, a Numpy array with 6 floats for each data point.
# The mass enters the computation in code units ().
# The mass must not come close to the periodic boundary, unless some sophisticated maths can be implemented.

# Anatomy of the potential used for computation: 
    #V_ULDM: Given by Fourier (See 000.06 Notes)
    #V_Central: Local Eval. Infinity Corrected if mass lives on grid point.

import time
import sys
import numpy as np
import numexpr as ne
import numba
import pyfftw
import h5py
import os

from scipy.integrate import solve_ivp

from scipy.interpolate import RegularGridInterpolator

from IPython.core.display import clear_output


def D_version():
    return str('2020 09 21, Equinox Special Bugfix')

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

####################### FUNCTION TO GENERATE PROGRESS BAR

def prog_bar(iteration_number, progress, tinterval,TMState):
    size = 25
    status = ""
    progress = float(progress) / float(iteration_number)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(size * progress))
    text = "\r[{}] {:.0f}% {}{}{}{}".format(
        "囧" * block + "口" * (size - block), round(progress * 100, 0),
        status, ' The previous soliton simulation step took ', round(tinterval,3), ' seconds.')
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
        loc, ix, its_per_save
        ):

        """
        Save various properties of the various grids in various formats
        """

        save_num = int((ix + 1) / its_per_save)

        if (save_options[0]):
            if (npy):
                file_name = "rho_#{0}.npy".format(save_num)
                np.save(
                    os.path.join(os.path.expanduser(loc), file_name),
                    rho
                )
            if (npz):
                file_name = "rho_#{0}.npz".format(save_num)
                np.savez(
                    os.path.join(os.path.expanduser(loc), file_name),
                    rho
                )
            if (hdf5):
                file_name = "rho_#{0}.hdf5".format(save_num)
                file_name = os.path.join(os.path.expanduser(loc), file_name)
                f = h5py.File(file_name, 'w')
                dset = f.create_dataset("init", data=rho)
                f.close()
        if (save_options[2]):
            plane = rho[:, :, int(resol / 2)]
            if (npy):
                file_name = "plane_#{0}.npy".format(save_num)
                np.save(
                    os.path.join(os.path.expanduser(loc), file_name),
                    plane
                )
            if (npz):
                file_name = "plane_#{0}.npz".format(save_num)
                np.savez(
                    os.path.join(os.path.expanduser(loc), file_name),
                    plane
                )
            if (hdf5):
                file_name = "plane_#{0}.hdf5".format(save_num)
                file_name = os.path.join(os.path.expanduser(loc), file_name)
                f = h5py.File(file_name, 'w')
                dset = f.create_dataset("init", data=plane)
                f.close()
        if (save_options[1]):
            if (npy):
                file_name = "psi_#{0}.npy".format(save_num)
                np.save(
                    os.path.join(os.path.expanduser(loc), file_name),
                    psi
                )
            if (npz):
                file_name = "psi_#{0}.npz".format(save_num)
                np.savez(
                    os.path.join(os.path.expanduser(loc), file_name),
                    psi
                )
            if (hdf5):
                file_name = "psi_#{0}.hdf5".format(save_num)
                file_name = os.path.join(os.path.expanduser(loc), file_name)
                f = h5py.File(file_name, 'w')
                dset = f.create_dataset("init", data=psi)
                f.close()
        if (save_options[4]):
            line = rho[:, int(resol / 2), int(resol / 2)]
            file_name2 = "line_#{0}.npy".format(save_num)
            np.save(
                os.path.join(os.path.expanduser(loc), file_name2),
                line
            )

        if (save_options[5]):
            if (npy):
                file_name = "TM_#{0}.npy".format(save_num)
                np.save(
                    os.path.join(os.path.expanduser(loc), file_name),
                    TMState
                )
            if (npz):
                file_name = "TM_#{0}.npz".format(save_num)
                np.savez(
                    os.path.join(os.path.expanduser(loc), file_name),
                    TMState
                )
            if (hdf5):
                file_name = "TM_#{0}.hdf5".format(save_num)
                file_name = os.path.join(os.path.expanduser(loc), file_name)
                f = h5py.File(file_name, 'w')
                dset = f.create_dataset("init", data=TMState)
                f.close()
                
                
        if (save_options[6]):
            if (npy):
                file_name = "Field3D_#{0}.npy".format(save_num)
                np.save(
                    os.path.join(os.path.expanduser(loc), file_name),
                    phisp
                )
            if (npz):
                file_name = "Field3D_#{0}.npz".format(save_num)
                np.savez(
                    os.path.join(os.path.expanduser(loc), file_name),
                    phisp
                )
            if (hdf5):
                file_name = "Field3D_#{0}.hdf5".format(save_num)
                file_name = os.path.join(os.path.expanduser(loc), file_name)
                f = h5py.File(file_name, 'w')
                dset = f.create_dataset("init", data=phisp)
                f.close()
                
                
                
        if (save_options[7]):
            
            phisp_slice = phisp[:,:,int(resol/2)] # z = 0
            
            if (npy):
                file_name = "Field2D_#{0}.npy".format(save_num)
                np.save(
                    os.path.join(os.path.expanduser(loc), file_name),
                    phisp_slice
                )
            if (npz):
                file_name = "Field2D_#{0}.npz".format(save_num)
                np.savez(
                    os.path.join(os.path.expanduser(loc), file_name),
                    phisp_slice
                )
            if (hdf5):
                file_name = "Field2D_#{0}.hdf5".format(save_num)
                file_name = os.path.join(os.path.expanduser(loc), file_name)
                f = h5py.File(file_name, 'w')
                dset = f.create_dataset("init", data=phisp_slice)
                f.close()
            
            
            phisp

# CALCULATE_ENERGIES IS NOW OBSOLETE. FW 000.01D

def calculate_energies(save_options, resol,
        psi, cmass, distarray, Vcell, phisp, karray2, funct,
        fft_psi, ifft_funct,
        egpcmlist, egpsilist, ekandqlist, egylist, mtotlist,
        ):
    if (save_options[3]):

            egyarr = pyfftw.zeros_aligned((resol, resol, resol), dtype='float64')

            # Gravitational potential energy density associated with the central potential
            egyarr = ne.evaluate('real((abs(psi))**2)')
            egyarr = ne.evaluate('real((-cmass/distarray)*egyarr)')
            egpcmlist.append(Vcell * np.sum(egyarr))
            tot = Vcell * np.sum(egyarr)

            # Gravitational potential energy density of self-interaction of the condensate
            egyarr = ne.evaluate('real(0.5*(phisp+(cmass)/distarray)*real((abs(psi))**2))')
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
            
# THIS IS THE NEW CALCULATE_ENERGIES. FW 000.01J

def calculate_energiesF(save_options, resol,
        psi, cmass, TMState, masslist, Vcell, phisp, karray2, funct,
        fft_psi, ifft_funct,
        egpcmlist, egpsilist, ekandqlist, egylist, mtotlist,xarray, yarray, zarray, a
        ):
    
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

######################### FUNCTION TO EVALUATE GRADIENT USING FOURIER SERIES
## FieldProcess Doesn't Use That Much Time

def FieldProcess(phik,gridlength,resol,NumSol):
    # Parts of FieldGradient that's shared for all masses.
    # We don't actually do extra FFT calls at all!
    
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

def FieldGradient(gridlength,Kx,Ky,Kz,FieldFT,position,resol,NumSol):
    
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

    
    #print(max(np.abs(GradX),np.abs(GradY),np.abs(GradZ)))
    return np.array([GradX, GradY, GradZ])

######################### FUNCTION FOR N-BODY WITH VARIABLE MASS

def FWNBody(t,TMState,masslist,FieldFT,gridlength,resol,Kx,Ky,Kz,epsilon,NumSol):
    
    dTMdt = 0*TMState
    
    for i in range(len(masslist)):
        
        #0,6,12,18,...
        Ind = int(6*i)
        
        #XDOT
        dTMdt[Ind]   =  TMState[Ind+3]
        #YDOT
        dTMdt[Ind+1] =  TMState[Ind+4]
        #ZDOT
        dTMdt[Ind+2] =  TMState[Ind+5]
        
        #x,y,z
        poslocal = np.array([TMState[Ind],TMState[Ind+1],TMState[Ind+2]])
        
        # a = -Grad(Phi)
        GradientLocal = -1*FieldGradient(gridlength,Kx,Ky,Kz,FieldFT,poslocal,resol,NumSol)
            
        #Initialized Against ULDM Field
        #XDDOT
        dTMdt[Ind+3] =  GradientLocal[0]
        #YDDOT
        dTMdt[Ind+4] =  GradientLocal[1]
        #ZDDOT
        dTMdt[Ind+5] =  GradientLocal[2]               
    
    return dTMdt
    
def FloaterAdvance(TMState,h,NS,FieldFT,masslist,gridlength,resol,Kx,Ky,Kz,epsilon,NumSol):
        # We pass on phik into the Gradient Function Above. This saves one inverse FFT call.
        # The N-Body Simulation is written from scratch

        # 
        if NS == 0:

            TMStateOut = TMState + FWNBody(0,TMState,masslist,FieldFT,gridlength,resol,Kx,Ky,Kz,epsilon,NumSol)*h
        
        elif NS%4 == 0:
            
            NRK = int(NS/4)
            
            H = h/NRK
            
            for RKI in range(NRK):
                TMK1 = FWNBody(0,TMState,masslist,FieldFT,gridlength,resol,Kx,Ky,Kz,epsilon,NumSol)   
                TMK2 = FWNBody(0,TMState + H/2*TMK1,masslist,FieldFT,gridlength,resol,Kx,Ky,Kz,epsilon,NumSol)
                TMK3 = FWNBody(0,TMState + H/2*TMK2,masslist,FieldFT,gridlength,resol,Kx,Ky,Kz,epsilon,NumSol)
                TMK4 = FWNBody(0,TMState + H*TMK3,masslist,FieldFT,gridlength,resol,Kx,Ky,Kz,epsilon,NumSol)
                TMState = TMState + H/6*(TMK1+2*TMK2+2*TMK3+TMK4)
                
            TMStateOut = TMState
            
        else:
            t_step = h/NS
            TMStateInteg = solve_ivp(
                fun = lambda t, y: FWNBody(t,y,masslist,FieldFT,gridlength,resol,Kx,Ky,Kz,epsilon,NumSol),
                t_span = [0,h], 
                y0 = TMState, 
                method = 'RK23',
                max_step = t_step)
        
            TMStateOut = TMStateInteg.y
            TMStateOut = TMStateOut[:,-1]
        
        #print(TMStateOut[3:5])
        
        return TMStateOut

######################### FUNCTION TO INITIALIZE SOLITONS AND EVOLVE

def evolve(central_mass, num_threads, length, length_units, 
            resol, duration, duration_units, step_factor, 
            save_number, save_options, save_path, npz, npy, hdf5, 
            s_mass_unit, s_position_unit, s_velocity_unit, solitons,
            start_time, m_mass_unit, m_position_unit, m_velocity_unit, particles,
            Uniform,Density,a):
    
    print ('PyUL_NBody: Build ',D_version(),'\n')
    
    print ('PyUL_NBody: Initializing Simulation. \n')
    
    NumSol = len(solitons)
    NumTM = len(particles)
    
    # For Compatibility Safety
    NS = 0 # Euler Integration ONLY for now
    Method = 1 # No other methods for field gradient from this point on
    Infini = False
    
    print ('PyUL_NBody: This run contains', NumSol ,'ULDM solitons and', NumTM, 'point Mass partcles.\n', 'Each Soliton step has the Test masses advanced by', NS, 'steps \n')
    
    masslist = []
    
    TMState = []
    
    
    ##########################################################################################
    #SET INITIAL CONDITIONS

    if (length_units == ''):
        gridlength = length
    else:
        gridlength = convert(length, length_units, 'l')
    if (duration_units == ''):
        t = duration
    else:
        t = convert(duration, duration_units, 't')
    if (duration_units == ''):
        t0 = start_time
    else:
        t0 = convert(start_time, duration_units, 't')
    if (s_mass_unit == ''):
        cmass = central_mass
    else:
        cmass = convert(central_mass, s_mass_unit, 'm')

    Vcell = (gridlength / float(resol)) ** 3
    
    if Infini:
        epsilon = gridlength/(4*float(resol))
    else:
        epsilon = 0

    ne.set_num_threads(num_threads)

    initsoliton_jit = numba.jit(initsoliton)

    ##########################################################################################
    # CREATE THE TIMESTAMPED SAVE DIRECTORY AND CONFIG.TXT FILE

    save_path = os.path.expanduser(save_path)
    tm = time.localtime()
    tBegin = time.time()

    talt = ['0', '0', '0']
    for i in range(3, 6):
        if tm[i] in range(0, 10):
            talt[i - 3] = '{}{}'.format('0', tm[i])
        else:
            talt[i - 3] = tm[i]
    timestamp = '{}{}{}{}{}{}{}{}{}{}{}{}{}'.format(tm[0], '.', tm[1], '.', tm[2], '_', talt[0], '_', talt[1], '_', talt[2], '_', resol)
    file = open('{}{}{}'.format('./', save_path, '/timestamp.txt'), "w+")
    file.write(timestamp)
    os.makedirs('{}{}{}{}'.format('./', save_path, '/', timestamp))
    file = open('{}{}{}{}{}'.format('./', save_path, '/', timestamp, '/config.txt'), "w+")
    file.write('=======================PyUL_NBody=============\n')
    file.write('\n')
    file.write(('{}{}'.format('resol = ', resol)))
    file.write('\n')
    file.write(('{}{}'.format('axion_mass (kg) = ', axion_mass)))
    file.write('\n')
    file.write(('{}{}'.format('length (code units) = ', gridlength)))
    file.write('\n')
    file.write(('{}{}'.format('duration (code units) = ', t)))
    file.write('\n')
    file.write(('{}{}'.format('start_time (code units) = ', t0)))
    file.write('\n')
    file.write(('{}{}'.format('step_factor  = ', step_factor)))
    file.write('\n')
    file.write(('{}{}'.format('central_mass (code units) = ', cmass)))
    file.write('\n\n')
    
    if not Uniform:
        file.write(('{}'.format('solitons ([mass, [x, y, z], [vx, vy, vz], phase]): \n')))

        for s in range(NumSol):
            file.write(('{}{}{}{}{}'.format('soliton#', s, ' = ', solitons[s], '\n')))

        file.write(('{}{}{}{}{}{}'.format('\ns_mass_unit = ', s_mass_unit, ', s_position_unit = ', s_position_unit, ', s_velocity_unit = ', s_velocity_unit)))
        file.write('\n\nNote: If the above units are blank, this means that the soliton parameters were specified in code units')
        
    else:
        file.write(('Using Uniform Initial ULDM Distribution.\n'))
        file.write(('{}{}\n'.format('Density =',Density)))
    
    file.write(('{}'.format('\n masses ([mass, [x, y, z], [vx, vy, vz]]): \n')))
    for TM in range(NumTM):
        file.write(('{}{}{}{}{}'.format('particle#', TM, ' = ', particles[TM], '\n')))
            
    file.write(('{}{}{}{}{}{}'.format('\nm_mass_unit = ', m_mass_unit, ', m_position_unit = ', m_position_unit, ', m_velocity_unit = ', m_velocity_unit)))
    file.write('\n\nNote: If the above units are blank, this means that the soliton parameters were specified in code units')
    
    file.close()

    loc = save_path + '/' + timestamp


    ##########################################################################################
    # SET UP THE REAL SPACE COORDINATES OF THE GRID - FW Revisit

    gridvec = np.linspace(-gridlength / 2.0, gridlength / 2.0, resol, endpoint=False)
    
    xarray, yarray, zarray = np.meshgrid(
        gridvec, gridvec, gridvec,
        sparse=True, indexing='ij',
    )
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

    f = np.load('./Soliton Profile Files/initial_f.npy')


    for k in range(NumSol):
        if (k != 0):
            if (not overlap_check(solitons[k], solitons[:k])):
                warn = 1
            else:
                warn = 0

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
            
    if Uniform and NumSol != 0:
        print("PyUL_NBody: Replacing Initial Solitons with a Uniform Wavefunction.")
        psi = ne.evaluate("0*psi + sqrt(Density)")
        
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
    
    phik = ne.evaluate("-4*3.141593*phik/rkarray2")

    phik[0, 0, 0] = 0
    
    irfft_phi = pyfftw.builders.irfftn(phik, axes=(0, 1, 2), threads=num_threads)

    ##########################################################################################
    # COMPUTE INTIAL VALUE OF POTENTIAL

    phisp = pyfftw.zeros_aligned((resol, resol, resol), dtype='float64')
    
    fft_phi = pyfftw.builders.fftn(phisp, axes=(0, 1, 2), threads=num_threads)
    
    phisp = irfft_phi(phik)
    
    # Debug Rollback
    phisp = ne.evaluate("phisp-a*cmass/(a*distarray+exp(-a*distarray))")
    
    
    # FW
    MI = 0
    
    for particle in particles:
        print('PyUL_NBody: Initialized Particle #', MI,'\n')
              
        TMmass = convert(particle[0], m_mass_unit, 'm')
        
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
    
    print('PyUL_NBody: The Initial TM State List Is\n', TMState)
    
    if cmass != 0:
        print('\n PyUL_NBody: Testing with Central Mass, ',central_mass)
    
    
    MI = 0
    ##########################################################################################
    # PRE-LOOP ENERGY CALCULATION

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
        
    
    ##########################################################################################
    # PRE-LOOP SAVE I.E. INITIAL CONFIG
    save_grid(
            rho, psi, resol, TMState, phisp,
            save_options,
            npy, npz, hdf5,
            loc, -1, 1,
    )

    
    print ('\nPyUL_NBody: Initialization Complete. Beginning Simulation in 3 sec \n')
    time.sleep(3)
    
    ##########################################################################################
    # LOOP NOW BEGINS

    halfstepornot = 1  # 1 for a half step 0 for a full step

    tenth = float(save_number/10) #This parameter is used if energy outputs are saved while code is running.

    clear_output()
    print("PyUL_NBody:",D_version(),'\n')
    print("PyUL_NBody: The total number of ULDM simulation steps is %.0f" % actual_num_steps)
    print('\n')
    if NS == 0:
        print("PyUL_NBody: Advancing NBody Simulation Using Explicit Euler.\n")
    
    if warn == 1:
        print("WARNING: Detected significant overlap between solitons in I.V.")
    print('\n')
    tinit = time.time()
    
    for ix in range(actual_num_steps):
        
        if halfstepornot == 1:
            psi = ne.evaluate("exp(-1j*0.5*h*phisp)*psi")
            halfstepornot = 0
        
        else:
            psi = ne.evaluate("exp(-1j*h*phisp)*psi")
        funct = fft_psi(psi)
        funct = ne.evaluate("funct*exp(-1j*0.5*h*karray2)")
        psi = ifft_funct(funct)
        rho = ne.evaluate("real(abs(psi)**2)")
        
        phik = rfft_rho(rho)  # not actually phik but phik is defined on next line
        
        phik = ne.evaluate("-4*3.141593*(phik)/rkarray2")
        
        phik[0, 0, 0] = 0
        
        phisp = irfft_phi(phik)
    
        # The maths is now reordered: NBody Potential Happens Here!
        for MI in range(NumTM):
        
            State = TMState[int(MI*6):int(MI*6+5)]
            
            TMx = State[0]
            TMy = State[1]
            TMz = State[2]
            
            mT = masslist[MI]
            
            distarrayTM = ne.evaluate("((xarray-TMx)**2+(yarray-TMy)**2+(zarray-TMz)**2)**0.5") # Radial coordinates
        
            phisp = ne.evaluate("phisp-a*mT/(a*distarrayTM+exp(-a*distarrayTM))")
            
        phisp = ne.evaluate("phisp-a*cmass/(a*distarray+exp(-a*distarray))")

        FieldFT = fft_phi(phisp)

        FieldFT[0, 0, 0] = 0
                
        if np.any(np.isnan(FieldFT)):
            print('ERROR, NaN Occurred. Something is seriously wrong!')
            print(phik)
            time.sleep(200)
            break
            
        # FW TEST STEP MAGIC HAPPENS HERE
        
        # This Converts the RFFTn to FFTn
        
            
        TMState = FloaterAdvance(TMState,h,NS,FieldFT,masslist,gridlength,resol,Kx,Ky,Kz,epsilon,NumSol)
        
        #Next if statement ensures that an extra half step is performed at each save point
        if (((ix + 1) % its_per_save) == 0) and halfstepornot == 0:
            psi = ne.evaluate("exp(-1j*0.5*h*phisp)*psi")
            rho = ne.evaluate("real(abs(psi)**2)")
            halfstepornot = 1

            #Next block calculates the energies at each save, not at each timestep.
            if (save_options[3]):
                calculate_energiesF(save_options, resol, psi, 
                                    cmass, TMState, masslist, Vcell, phisp, karray2, funct,
                                    fft_psi, ifft_funct, egpcmlist, egpsilist, ekandqlist,
                                    egylist, mtotlist,xarray, yarray, zarray, a)

            #Uncomment next section if partially complete energy lists desired as simulation runs.
            #In this way, some energy data will be saved even if the simulation is terminated early.

            if (save_options[3]):
                if (ix+1) % tenth == 0:
                    label = (ix+1)/tenth
                    file_name = "{}{}".format(label,'egy_cumulative.npy')
                    np.save(os.path.join(os.path.expanduser(loc), file_name), egylist)
                    file_name = "{}{}".format(label,'egpcm_cumulative.npy')
                    np.save(os.path.join(os.path.expanduser(loc), file_name), egpcmlist)
                    file_name = "{}{}".format(label,'egpsi_cumulative.npy')
                    np.save(os.path.join(os.path.expanduser(loc), file_name), egpsilist)
                    file_name = "{}{}".format(label,'ekandq_cumulative.npy')
                    np.save(os.path.join(os.path.expanduser(loc), file_name), ekandqlist)


        ################################################################################
        # SAVE DESIRED OUTPUTS
        if ((ix + 1) % its_per_save) == 0:
                        
            save_grid(
                    rho, psi, resol, TMState, phisp,
                    save_options,
                    npy, npz, hdf5,
                    loc, ix, its_per_save,
            )

        ################################################################################
        # UPDATE INFORMATION FOR PROGRESS BAR

        tint = time.time() - tinit
        tinit = time.time()
        prog_bar(actual_num_steps, ix + 1, tint,TMState)

    ################################################################################
    # LOOP ENDS

    clear_output()
    print ('\n')
    tFinal = time.time()
    
    Time = tFinal - tBegin - 3
    
    day = Time // (24 * 3600)
    Time = Time % (24 * 3600)
    hour = Time // 3600
    Time %= 3600
    minutes = Time // 60
    Time %= 60
    seconds = Time
    
    print("PyUL_NBody: Run Complete. Time Elapsed (d:h:m:s): %d:%d:%d:%d" % (day, hour, minutes, seconds))

    if warn == 1:
        print("WARNING: Significant overlap between solitons in initial conditions")

    if (save_options[3]):
        file_name = "egylist.npy"
        np.save(os.path.join(os.path.expanduser(loc), file_name), egylist)
        file_name = "egpcmlist.npy"
        np.save(os.path.join(os.path.expanduser(loc), file_name), egpcmlist)
        file_name = "egpsilist.npy"
        np.save(os.path.join(os.path.expanduser(loc), file_name), egpsilist)
        file_name = "ekandqlist.npy"
        np.save(os.path.join(os.path.expanduser(loc), file_name), ekandqlist)
        file_name = "masseslist.npy"
        np.save(os.path.join(os.path.expanduser(loc), file_name), mtotlist)
        
    return timestamp





