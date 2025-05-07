import numpy as np
import h5py

# Function to generate default folder names
def GenFromTime():
    from datetime import datetime
    now = datetime.now() # current date and time
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    
    return timestamp

# We still have this thing that is poetic but redundant.

SFS = '3Density 3Wfn 2Density Energy 1Density NBody 3Grav 2Grav DF 2Phase Entropy 1Grav 3GravF 2GravF 1GravF 3UMmt 2UMmt 1UMmt Momentum AngMomentum 3DensityRS 3WfnRS 2EnergyTot 2EnergyKQ 2EnergySI ULDCOM 2DensityRS Quadrupole 2Momentum'

# .    0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29
SNM = 'R3D P3D R2D EGY R1D NTM G3D G2D DYF A2D ENT G1D F3D F2D F1D V3D V2D V1D PMT MVR R3R P3R E2T E2K E2G UCM R2R QIJ P2D'

SaveFlags = SFS.split()
SaveNames = SNM.split()

def IOName(Type):
    return SaveNames[SaveFlags.index(Type)]

# Saving Files

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
        dset = f.create_dataset(Type, data=data)
        f.close()
    else:
        raise RuntimeError('Invalid output format specified!')
        

# Loading Files

def IOLoad_npy(loc,Type,save_num):
    return np.load(f"{loc}/Outputs/{Type}/{IOName(Type)}_#{save_num:03d}.npy")

def IOLoad_h5(loc,Type,save_num):
    flname = f"{loc}/Outputs/{Type}/{IOName(Type)}_#{save_num:03d}.hdf5"
    f = h5py.File(flname,'r')
    b = f[Type][:]
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

# NBody Streaming Useful for Accessing Intermediate Parts of a Simulation

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
        
        
def SaveOptionsCompile(save_options): # True False to String
    result = ''

    for i in zip(save_options, SaveFlags):
        if i[0]:
            result += (i[1]+' ')
            
    return result

def SaveOptionsDigest(OptionsText): # String to True False
    
    save_options = np.zeros(len(SaveFlags), dtype = bool)
    
    if OptionsText == 'Minimum':
        OptionsText = 'Energy 1Density NBody DF Entropy'

    OList = OptionsText.split()
    
    for Word in OList:
        save_options[SaveFlags.index(Word)] = True
    
    return save_options.tolist()
 
def Runs(save_path, Automatic = False):

    import os
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
            print(f"Loading {Latest}")
            return Latest
        
        elif Ind.startswith('X'):
            Ind = Ind[1:]
            IndTD = Log[int(Ind)]
            TDName = os.path.join(save_path, runs[IndTD])
            
            import shutil
                       
            shutil.rmtree(TDName)
            return Runs(save_path, Automatic = Automatic)
        
        else:
            Ind = int(Ind)
            Ind = Log[Ind]
            print(f"Loading {runs[Ind]}")
            return runs[Ind]

# OLD VERSION IS NOT FUN

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


def ULDump(loc,psi,TMState,Status):
    np.save(f'{loc}/{Status}_psi.npy',psi)
    np.save(f'{loc}/{Status}_TM.npy',TMState)
    return 1
    
def ULRead(InitPath):
    psi = np.load(f'{InitPath}_psi.npy')
    return psi

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