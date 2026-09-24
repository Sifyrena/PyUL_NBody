import numpy as np
import h5py
import os


def get_pyul2_root():
    """Dynamically compute the root address of the PyUL2 library."""
    # This finds the directory containing this script file, i.e., PyUL2's root.
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Function to generate default folder names
def GenFromTime():
    from datetime import datetime
    now = datetime.now() # current date and time
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    
    return timestamp

# We still have this thing that is poetic but redundant.
# Utils/IO.py

from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class SaveDefinition:
    """A single saveable quantity.
    
    flag     : user-facing name, used in config strings ('3Density', 'Energy', ...)
    dirname  : short folder/filename prefix ('R3D', 'EGY', ...) — kept for
               compatibility with existing Data loaders and downstream scripts
    description : human-readable description for docs and --help output
    """
    flag: str
    dirname: str
    description: str


SAVE_CATALOG = (
    SaveDefinition('3Density',   'R3D', 'Full 3D density field |ψ|²'),
    SaveDefinition('3Wfn',       'P3D', 'Full 3D complex wavefunction ψ'),
    SaveDefinition('2Density',   'R2D', 'Density, xy mid-plane slice'),
    SaveDefinition('Energy',     'EGY', 'Global energy components (KQ, SI, TM). Should always be computed.'),
    SaveDefinition('1Density',   'R1D', 'Density, y-axis line through centre'),
    SaveDefinition('NBody',      'NTM', 'N-body state (position, velocity). Should always be computed.'),
    SaveDefinition('3Grav',      'G3D', 'Full 3D self-gravity potential Φ_SP'),
    SaveDefinition('2Grav',      'G2D', 'Self-gravity, xy mid-plane slice'),
    SaveDefinition('DF',         'DYF', 'Dynamical friction gradient log'),
    SaveDefinition('2Phase',     'A2D', 'Phase arg(ψ), xy mid-plane slice'),
    SaveDefinition('Entropy',    'ENT', 'Global -∫ρ ln ρ dV'),
    SaveDefinition('1Grav',      'G1D', 'Self-gravity, y-axis line'),
    SaveDefinition('3GravF',     'F3D', 'Full 3D total potential Φ_SP + Φ_TM'),
    SaveDefinition('2GravF',     'F2D', 'Total potential, xy mid-plane slice'),
    SaveDefinition('1GravF',     'F1D', 'Total potential, y-axis line'),
    SaveDefinition('3DensityRS', 'R3R', 'Subsampled 3D density around COM'),
    SaveDefinition('3WfnRS',     'P3R', 'Subsampled 3D wavefunction around COM'),
    SaveDefinition('2EnergyTot', 'E2T', 'Energy density, xy mid-plane slice'),
    SaveDefinition('2EnergyKQ',  'E2K', 'Kinetic-quantum energy density slice'),
    SaveDefinition('2EnergySI',  'E2G', 'Self-interaction energy density slice'),
    SaveDefinition('ULDCOM',     'UCM', 'ULDM centre-of-mass position'),
    SaveDefinition('2DensityRS', 'R2R', 'Subsampled density, 2D slice'),
)


# Derived lookup tables. Built once at import, O(1) access.
SaveFlags = [s.flag for s in SAVE_CATALOG]
SaveNames = [s.dirname for s in SAVE_CATALOG]
_FLAG_TO_DIRNAME = {s.flag: s.dirname for s in SAVE_CATALOG}
_FLAG_TO_INDEX   = {s.flag: i for i, s in enumerate(SAVE_CATALOG)}


def IOName(flag: str) -> str:
    """Short directory/filename prefix for a given save flag."""
    try:
        return _FLAG_TO_DIRNAME[flag]
    except KeyError:
        raise ValueError(
            f"Unknown save flag '{flag}'. "
            f"Valid flags: {', '.join(SaveFlags)}"
        )


def save_flag_index(flag: str) -> int:
    """Integer index for a save flag — used by boolean save_options arrays."""
    return _FLAG_TO_INDEX[flag]

# Save Options Sanity

def SaveOptionsCompile(save_options) -> str:
    """Convert a boolean vector to space-separated flag names."""
    return ' '.join(
        s.flag for s, flag_on in zip(SAVE_CATALOG, save_options) if flag_on
    )


def SaveOptionsDigest(options_text: str) -> list[bool]:
    """Convert space-separated flag names to a boolean vector.
    
    The shorthand 'Minimum' expands to a sensible default set.
    """
    if options_text == 'Minimum':
        options_text = 'Energy 1Density NBody 1GravF'

    result = [False] * len(SAVE_CATALOG)
    for flag in options_text.split():
        try:
            result[_FLAG_TO_INDEX[flag]] = True
        except KeyError:
            raise ValueError(
                f"Unknown save flag '{flag}' in options string. "
                f"Valid flags: {', '.join(SaveFlags)}"
            )
    return result

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
         
def ListRuns(save_path, Automatic = False):

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

Runs = ListRuns
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


def Load_npys(loc,save_options, Extension = "npy", Old = False):
    
    if Extension == "hdf5":
        Loader = IOLoad_h5
    else: 
        Loader = IOLoad_npy
    
    if Old:
        if Extension == "hdf5":
            Loader = IOLoad_h5_O
        else: 
            Loader = IOLoad_npy_O

    print('3D saves are not automatically loaded. Please load them manually!')
    save_options[0] = False
    save_options[1] = False
    save_options[6] = False
    save_options[12] = False
    save_options[20] = False
    save_options[21] = False
        
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

    print(f"Loaded {x} Data Entries from {loc}!")
    
    return x, Out


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
    
class Data:
        
    def __init__(self, save_path):
        from PyUltraLight2.Universe.Universe import ULDMUniverse
        from PyUltraLight2.Init.Config import Config

        self.save_path = save_path
        self.loc = save_path
 
        config = Config()
        config.FromFile(f"{self.loc}/config.uldm") 

        
        self.config = config
        # Initialize the ULDM Universe based on config
        m22 = self.config.uldm["m22"]
        self.Universe = ULDMUniverse(m22)
        self.axion_E = self.Universe.axion_E
        self.length_unit = self.Universe.length_unit
        self.mass_unit = self.Universe.mass_unit
        self.energy_unit = self.Universe.energy_unit
        self.convert = self.Universe.convert
        self.convert_back = self.Universe.convert_back
        self.convert_between = self.Universe.convert_between

        # Load Config-dependent values
        self._load_config_values()

        # Load arrays (you can customize this based on 2D/1D flags)
        self._load_npy_data()

    def _load_config_values(self):
        c = self.config
        space = c.Space
        time = c.Time
        uldm = c.uldm
        bh = c.BlackHole

        self.resolution = space["Resolution"]
        self.length = space["Box"]["BoxLength"]
        self.dx = self.length / self.resolution
        self.resol = self.resolution # just a shorthand we are used to.
        self.length_units = space["Box"]["LengthUnits"]
        self.duration = time["TimeDuration"]
        self.duration_units = time["TimeUnits"]
        self.start_time = time["StartTime"]
        self.step_factor = time["StepFactor"]
        
        # 
        
        if c.LEGACY["Shift"]:
            self.xAr = np.linspace(-1*(self.length-dx)/2,(self.length-dx)/2, self.resol, endpoint = True)
        else:
            self.xAr = np.linspace(-self.length/2, self.length/2, self.resol, endpoint = False)
        
        self.save_number = c.Saving["Number"] # if early termination may not bre reached
        
        if self.save_number == -1:
            self.save_number = c.ULDStepEst()
            
        
        self.save_format = c.Saving["Format"]
        self.save_flags = SaveOptionsDigest(c.Saving["Flags"])

        self.particles = bh["MatterParticles"]["Condition"]
        self.solitons = uldm["Solitons"]["Condition"]
        self.embeds = uldm["Solitons"]["Embedded"]
        self.density = uldm["Modifier"]["UniformFieldAddOn"]["DensityValue"]
        self.density_unit = uldm["Modifier"]["UniformFieldAddOn"]["DensityUnit"]
        self.uniform_velocity = uldm["Modifier"]["UniformFieldAddOn"]["UniformVelocity"]




    def _load_npy_data(self):
        
        # Always load energy.
        self.ETotal = np.load((self.loc + '/Outputs/egylist.npy'), allow_pickle=True) 
        self.EGP_NB = np.load((self.loc + '/Outputs/egpcmlist.npy'), allow_pickle=True) 
        self.EGP_NB2 = np.load((self.loc + '/Outputs/egpcmMlist.npy'), allow_pickle=True) 
        self.EGP_UL = np.load((self.loc + '/Outputs/egpsilist.npy'), allow_pickle=True) 
        self.EKQ = np.load((self.loc + '/Outputs/ekandqlist.npy'), allow_pickle=True)
        self.mTotal = np.load((self.loc + '/Outputs/masseslist.npy'), allow_pickle=True) 
        
        self.EndNum, self.Loaded = Load_npys(self.loc, self.save_flags, Extension=self.save_format)

        self.Tp = np.arange(self.EndNum) / self.save_number * self.duration #

        if self.EndNum < self.save_number:
            print("Did the run end early?")
            
        self.nbody = np.array(self.Loaded["NBody"])
        #self.grad = np.array(self.Loaded["DF"])
        #self.center_of_mass = np.array(self.Loaded["ULDCOM"])

        self._apply_unit_conversions()

    def _apply_unit_conversions(self):
        CB = self.convert_between
        ToPhys = self.convert_back

        self.duration_Myr = CB(self.duration, self.duration_units, "Myr", "t")
        self.length_kpc = CB(self.length, self.length_units, "kpc", "l")
        self.length_code = self.convert(self.length, self.length_units, "l")

        # Pre-multipliers
        self.time_array = np.arange(self.EndNum) * self.duration_Myr / (self.save_number + 1)

        self.XPre = ToPhys(1, 'kpc', 'l')
        self.VPre = ToPhys(1, 'km/s', 'v')
        self.XPreSI = ToPhys(1, 'm', 'l')
        self.VPreSI = ToPhys(1, 'm/s', 'v')
        
        self.EPre = self.Universe.energy_unit
        if len(self.nbody[0]) != 0:
            IArray = np.arange(len(self.nbody[0]))
            self.nbody_S = self.nbody.copy()
            self.nbody_S[:, IArray % 6 <= 2] *= self.XPre
            self.nbody_S[:, IArray % 6 >= 3] *= self.VPre

            self.nbody_SI = self.nbody.copy()
            self.nbody_SI[:, IArray % 6 <= 2] *= self.XPreSI
            self.nbody_SI[:, IArray % 6 >= 3] *= self.VPreSI

    def get_mass_list(self, unit="M_solar_masses"):
        CB = self.convert_between
        unit_type = 'm'
        return [CB(m[0], self.config.BlackHole["MatterParticles"]["MassUnits"], unit, unit_type) for m in self.particles]