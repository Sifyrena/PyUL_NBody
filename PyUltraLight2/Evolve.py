# I WISH TO GET THE FUNCTION WORKING FIRST! FW 
# Thanks to Nyx / Axionyx for inspiring this new workflow.

import numpy as np
import multiprocessing
import os
import time
from datetime import datetime

import pyfftw
import numexpr as ne

from PyUltraLight2.Version import *
from PyUltraLight2.Credits import *

from PyUltraLight2.Solitons.Profile import *

from PyUltraLight2.Utils.printU import printU
from PyUltraLight2.Utils.Find3BoxCOM import Find3BoxCOM
from PyUltraLight2.Utils.prog_bar import prog_bar
from PyUltraLight2.Init.Config import Config

from PyUltraLight2.Integration.NBodyAdvance import NBodyAdvance, NBodyAdvance_NI
from PyUltraLight2.Interpolation.Resample import *

from PyUltraLight2.DerivedQuantities.Energy import *

from PyUltraLight2.Utils.IO import GenFromTime, SaveOptionsCompile, SaveOptionsDigest, ULRead, IOSave, save_grid
from PyUltraLight2.Universe.Universe import *

from PyUltraLight2.Init.Solitons import InitSolitonF


# Numerical Tools
# from PyUltraLight2.Interpolation import Lin3D, Jax3D
#from PyUltraLight2.Integration import RK4Explicit, FT

def __call__():
    return Evolve()

def Evolve(config_or_path, Silent = False, Message = '', **kwargs):
    # IDEAL USE CASE: like the Nyx-family, from command line on its own.
    """
    - config_or_path: Config object or str
        If Config, uses it directly.
        If str, loads the Config from the file path.
    """

    if isinstance(config_or_path, Config):
        config = config_or_path
    elif isinstance(config_or_path, str):
        config = Config()
        config.FromFile(config_or_path)
    else:
        raise TypeError("config_or_path must be a Config object or a path to a config file")
    
    for key, value in kwargs.items():
        if key == "Saving_Loc":
            printU(f"Updating Run Location to {value}", "Init")
            config.Saving["Loc"] = value

        if key == "DSponge":
            printU(f"Setting Dispersive Sponge to {value}.", "Init")
            config.BC["DSponge"] = value

        if key == "IsoP":
            printU(f"Setting Isolated Potential to {value}.", "Init")
            config.BC["IsoP"] = value

        if key == "CenterCalc":
            printU(f"Setting COM Evaluation to {value}.", "Init")
            config.ADVANCED["CenterCalc"] = value



        if hasattr(config, key):
            printU(f"Overriding config: {key} = {value} (was {getattr(config, key)})","Init")
            setattr(config, key, value)
        else:
            printU(f"Adding new config attribute: {key} = {value}","Init")
            setattr(config, key, value)

    if config.Saving["Loc"] != "./":
        loc = config.Saving["Loc"]
    else:
        loc = f"./Simulations/{GenFromTime()}"

    # IO

    try:
        os.mkdir(str(loc))
        os.mkdir(str(loc + '/Outputs'))
        
    except(FileExistsError):
        
        if Silent:
            Protect = 'Y'
        else:
            printU(f"{Version}: Folder Contains Outputs. Remove current files and Proceed [Y/n]?", 'IO')

            Protect = str(input())
        
        if Protect == 'n':
            return loc
        
        elif Protect == 'Y':
            import shutil
            
            print('Pre-existing Output files removed.')
            
            shutil.rmtree(str(loc + '/Outputs'))
            os.mkdir(str(loc + '/Outputs'))
            
        else:
            return loc
    GenerateLog = config.RUNTIME["GenerateLog"]
    LogLocation = f"{loc}/evolve.log"

    printU(f"Starting RUN at {loc}!", ToFile= GenerateLog, FilePath= LogLocation)

    # Load Universe Settings from config

    m22 = config.uldm["m22"]
    Universe = ULDMUniverse(m22)
        
    axion_E = Universe.axion_E
    length_unit = Universe.length_unit
    mass_unit = Universe.mass_unit
    energy_unit = Universe.energy_unit
    
    convert = Universe.convert
    convert_back = Universe.convert_back
    convert_between = Universe.convert_between

    # Load Run Params

    save_options = SaveOptionsDigest(config.Saving["Flags"])
    save_number = config.Saving["Number"]
    save_format = config.Saving["Format"]

    # Extracting the necessary variables from the config instance
    duration = config.Time["TimeDuration"]
    start_time = config.Time["StartTime"]
    duration_units = config.Time["TimeUnits"]
    NS = config.Time["RKSteps"]
    step_factor = config.Time["StepFactor"]

    t = convert(duration, duration_units, 't')

    t0 = convert(start_time, duration_units, 't')

    resol = config.Space["Resolution"]
    length = config.Space["Box"]["BoxLength"]
    length_units = config.Space["Box"]["LengthUnits"]

    lengthC = convert(length, length_units, 'l')
    
    # Black Hole Stuff
    particles = config.BlackHole["MatterParticles"]["Condition"]
    m_mass_unit = config.BlackHole["MatterParticles"]["MassUnits"]
    m_position_unit = config.BlackHole["MatterParticles"]["PositionUnits"]
    m_velocity_unit = config.BlackHole["MatterParticles"]["VelocityUnits"]

    rP = config.BlackHole["MatterParticles"]["PlummerRadius"]
    smoothing = config.BlackHole["FieldSmoothing"]

    if smoothing == "Auto":
        smoothing = 2 * resol / lengthC # Old trick to return the usual rP value.
        config.BlackHole["FieldSmoothing"] = smoothing

    # ULDM Stuff
    solitons = config.uldm["Solitons"]["Condition"]
    embeds = config.uldm["Solitons"]["Embedded"]

    s_mass_unit = config.uldm["Solitons"]["MassUnits"]
    s_position_unit = config.uldm["Solitons"]["PositionUnits"]
    s_velocity_unit = config.uldm["Solitons"]["VelocityUnits"]

    # ULDM Modifier
    Uniform = config.uldm["Modifier"]["UniformFieldAddOn"]["Flag"]
    density_unit = config.uldm["Modifier"]["UniformFieldAddOn"]["DensityUnit"]
    density_value = config.uldm["Modifier"]["UniformFieldAddOn"]["DensityValue"]
    uniform_velocity = config.uldm["Modifier"]["UniformFieldAddOn"]["UniformVelocity"]

    # ULDM Self Interaction
    SelfInt = config.uldm["SI"]
    lambda_hat = config.uldm["LHat"]

    # Initialization Flags
    DumpInit = config.INIT["DumpInit"]
    DumpFinal = config.INIT["DumpFinal"]
    UseInit = config.INIT["UseInit"]
    InitPath = config.INIT["InitPath"]
    InitWeight = config.INIT["InitWeight"]

    # Boundary Conditions
    EdgeClear = config.BC["EdgeClear"]
    IsoP = config.BC["IsoP"]
    UseDispSponge = config.BC["DSponge"]

    # SIM
    SelfGravity = config.SIM["SelfGravity"]
    NBodyInterp = config.SIM["NBodyInterp"]
    NBodyGravity = config.SIM["NBodyGravity"]

    # LEGACY
    Shift = config.LEGACY["Shift"]
    NLM = config.LEGACY["NLM"]
    DR = config.LEGACY["DR"]

    # IO Override
    Stream = config.IO["Stream"]
    StreamChar = config.IO["StreamChar"]

    # Advanced
    CenterCalc = config.ADVANCED["CenterCalc"]
    ComputeQuad = config.ADVANCED["ComputeQuad"]
    ExtPhi = config.ADVANCED["ExtPhi"] # The current way is ugly, maybe treat this as a path, or only take floating point numbers which by definition make no difference ... 

    # Subsampled Grid
    Length_Ratio = config.SAMPLING["Length_Ratio"]
    resolR = config.SAMPLING["resolR"]

    # Extras
    PrintEK = config.EXTRAS["PrintEK"]
    MassChange = config.EXTRAS["MassChange"]
    MassFunc = config.EXTRAS["MassFunc"]
    CorrectDrift = config.EXTRAS["CorrectDrift"]
    CorrectFreq = config.EXTRAS["CorrectFreq"]

    # Special Halting Conditions
    AutoStop = config.SPECIAL_HALTING_CONDITIONS["AutoStop"]
    AutoStop2 = config.SPECIAL_HALTING_CONDITIONS["AutoStop2"]
    AutoStop3 = config.SPECIAL_HALTING_CONDITIONS["AutoStop3"]
    KEThreshold = config.SPECIAL_HALTING_CONDITIONS["KEThreshold"]
    WellThreshold = config.SPECIAL_HALTING_CONDITIONS["WellThreshold"]

    Credits(IsoP, UseDispSponge, embeds, SelfInt)

    num_threads = multiprocessing.cpu_count()
    if resol < 128:
        num_threads = np.min([num_threads,4])

    printU(f"Using {num_threads} CPU Threads for FFT.",'FFT', ToFile= GenerateLog, FilePath= LogLocation)

    for SaveName in SaveOptionsCompile(save_options).split():
        os.mkdir(str(loc + '/Outputs/'+SaveName))

    config.VERSION = S_version

    config.Saving["Flags"] = SaveOptionsCompile(save_options)

    config.ToFile(f"{loc}/config.uldm")

    if not Uniform:
        Density = 0
        UVel = [0,0,0]
    
    if smoothing>=1e8:
        printU(f"Smoothing has been turned off!",'NBody')
        smoothing = 0

    printU(f"Copied Current Config to {loc}",'IO', ToFile= GenerateLog, FilePath= LogLocation)
    printU(f"Data to save:\n{SaveOptionsCompile(save_options)}",'IO', ToFile= GenerateLog, FilePath= LogLocation)

    NumSol = len(solitons)
    NumTM = len(particles)

    UseJax = False
    if not (UseJax): # 1 = Real Space Interpolation (Orange), 2 = Fourier Sum (White)
        printU(f"Using Linear Interpolation for gravity.",'NBody', ToFile= GenerateLog, FilePath= LogLocation)
    
    printU(f"Simulation grid resolution is {resol}^3.",'FFT', ToFile= GenerateLog, FilePath= LogLocation)
    
    if smoothing == 0:
        printU(f"Using 1/r Point Mass Potential.",'NBody', ToFile= GenerateLog, FilePath= LogLocation)
    
    if EdgeClear:
        print("WARNING: The Wavefunction on the boundary planes will be Auto-Zeroed at every iteration.")

    print('==========================Consistency=================================')

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
    

    print('===============================================================================')

    TIntegrate = 0
    
    TimeWritten = False
    
    masslist = []
    
    TMState = []

    ##########################################################################################
    #CONVERT INITIAL CONDITIONS TO CODE UNITS



    Density = convert(Density,density_unit,'d')
    
    Vcell = (lengthC / float(resol)) ** 3
    
    ne.set_num_threads(num_threads)

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
        printU("Created an additional N body stream file at root folder for variables {StreamChar}.",'NBody', ToFile= GenerateLog, FilePath= LogLocation)

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
        
        # Mixing with previous wavefunction
        psi = ne.evaluate("psi + funct")


    if solitons != []:
        printU(f"Loaded standard soliton radial profile.",'Init', ToFile= GenerateLog, FilePath= LogLocation)
        f = LoadDefaultSoliton()

    for s in solitons:
        mass = convert(s[0], s_mass_unit, 'm')
        position = convert(np.array(s[1]), s_position_unit, 'l')
        velocity = convert(np.array(s[2]), s_velocity_unit, 'v')
        # Note that alpha and beta parameters are computed when the initial_f.npy soliton profile file is generated.
        alpha = (mass / prealpha) ** 2 #3.883
        phase = s[3]
        
        funct = InitSolitonF(gridvec, position, resol, alpha, DR = DR)
        # funct = initsoliton_jit(funct, xarray, yarray, zarray, position, alpha, f, delta_x, DR = DR)
        if DR != 1:
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

        else:
            raise ValueError("Mixing Fraction can only be in [0,1] or -1 for phase mixing only.")

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
        
        Cutoff = (resol//16) # Default "MOAT" is 1/8 of the simulation lengthwise.
        
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
        printU(f'Successfully initiated Wavefunction and NBody Initial Conditions. Dumping to file and Quitting.','IO', ToFile= GenerateLog, FilePath= LogLocation)
    
        ULDump(loc,psi,TMState,'Init')
        
        return loc
        
    
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
    
    printU("Initialised data saved!",'Init', ToFile= GenerateLog, FilePath= LogLocation)
    
    if ComputeQuad:
        #r_sq = ne.evaluate("xarray**2+yarray**2+zarray**2")
        LocCOM = Find3BoxCOM(rho,xarray, yarray, zarray)
        Qij = QuadrupoleSecond(rho, gridvec, LocCOM)
        IOSave(loc,'Quadrupole',0,save_format,data = Qij)
        
    tBegin = time.time()
    
    tBeginDisp = datetime.fromtimestamp(tBegin).strftime("%d/%m/%Y, %H:%M:%S")
    
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

                IOSave(loc,'2Momentum',momentum_I,save_format,data = np.array([pXAr[:,:,resol//2],   pYAr[:,:,resol//2], pZAr[:,:,resol//2]]))
                
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

            TMState, GradientLog = NBodyAdvance(TMState,h,masslist,phiSP,smoothing,lengthC,resol,NS, loc, Stream, StreamChar)
            
        else:

            TMState, GradientLog = NBodyAdvance_NI(TMState,h,masslist,phiSP,smoothing,lengthC,resol,NS)
 
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
    return loc