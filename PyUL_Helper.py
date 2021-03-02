Version   = str('ULHelper') # Handle used in console.
D_version = str('Helper Build 2021 03 02') # Detailed Version
S_version = 9 # Short Version

print(D_version)

import os
import matplotlib.pyplot as plt
import numpy as np
import time
import json

def SSEst(save_options, save_number, resol):
    
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
        print('ULHelper: Saving Mass Density Data (3D)')
        PreMult = PreMult + resol**3
        
    if save_psi:
        print('ULHelper: Saving Complex Field Data (3D)')
        PreMult = PreMult + resol**3*2
        
    if save_phi:
        print('ULHelper: Saving Gravitational Field Data (3D)')
        PreMult = PreMult + resol**3
        
    if save_plane:
        print('ULHelper: Saving Mass Density Data (2D)')
        PreMult = PreMult + resol**2
        
    if save_phase_plane:
        print('ULHelper: Saving ULD Argument Data (2D)')
        PreMult = PreMult + resol**2
        
    if save_phi_plane:
        print('ULHelper: Saving Gravitational Field Data (2D)')
        PreMult = PreMult + resol**2
    
    if save_gradients:
        print('ULHelper: Saving NBody Gradient Data')
    
    if save_testmass:
        print('ULHelper: Saving NBody Position Data')
    
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
    
    
    with open('{}{}{}'.format('./',save_path, '/latest.txt'), 'r') as timestamp:
        ts = timestamp.read()
        print('ULHelper: Loading Folder',ts)
        
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
                data.append(np.load('{}{}{}{}'.format(loc, '/Outputs/plane_#', x, '.npy')))
            if save_testmass:

                TMdata.append(np.load('{}{}{}{}'.format(loc, '/Outputs/TM_#', x, '.npy')))
            if save_phi_plane:

                phidata.append(np.load('{}{}{}{}'.format(loc, '/Outputs/Field2D_#', x, '.npy')))
                
            if save_gradients:
                graddata.append(np.load('{}{}{}{}'.format(loc, '/Outputs/Gradients_#', x, '.npy')))
                
            if save_phase_plane:
                phasedata.append(np.load('{}{}{}{}'.format(loc, '/Outputs/Arg2D_#', x, '.npy')))
            
            EndNum += 1
        
        except FileNotFoundError:

            print("WARNING: Run incomplete or the storage is corrupt!")

            break
        
    print("ULHelper: Loaded", EndNum, "Data Entries")
    return EndNum, data,  TMdata, phidata,    graddata, phasedata


def SmoothingReport(a,resol,clength):
    

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
    
    GMod = -a*1/(a*rR+np.exp(-a*rR))
    
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
        
        print("ULHelper: Loading Custom Settings")
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


def Load_Config(configpath):
    
    with open(configpath, 'r') as configfile:
        config = configfile.read()
        print(config)

        
def EmbedParticle(particles,embeds):
    
    EI = 0
    
    for Mollusk in embeds:
        
        print(f"ULHelper: Calculating and loading the mass of embedded particle #{EI}.")
        
        Mass = Mollusk[0]*Mollusk[3]
        
        Pearl = [Mass,Mollusk[1],Mollusk[2]]
        
        if not (Pearl in particles):
            particles.append(Pearl)
            
        EI += 1
        
    
    return particles
        
       
def GenFromTime():
    tm = time.localtime()
    talt = ['0', '0', '0']
    for i in range(3, 6):
        if tm[i] in range(0, 10):
            talt[i - 3] = '{}{}'.format('0', tm[i])
        else:
            talt[i - 3] = tm[i]
    timestamp = '{}{}{}{}{}{}{}{}{}'.format(tm[0], tm[1], tm[2], '_', talt[0], '', talt[1], '', talt[2])
    
    return timestamp


def GenerateConfig(NS, central_mass, length, length_units, resol, duration, duration_units, step_factor, save_number, save_options, save_path, npz, npy, hdf5, s_mass_unit, s_position_unit, s_velocity_unit, solitons,start_time, m_mass_unit, m_position_unit, m_velocity_unit, particles,embeds, Uniform,Density,density_unit,a,B,UVel,NoInteraction = False,Name = ''
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

           

    os.makedirs('{}{}{}{}'.format('./', save_path, '/', timestamp))


    Conf_data = {}

    Conf_data['Helper Version'] = D_version

    Conf_data['Config Generation Time'] = tm

    Conf_data['Save Options'] = ({
            'flags': save_options,
            'folder': save_path,
            'number': save_number,
            'npz': npz,
            'npy': npy,
            'hdf5': hdf5
            })


    Conf_data['Spacial Resolution'] = resol

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


    Conf_data['ULDM Solitons'] = ({
    'Condition': solitons,
    'Embedded': embeds,
    'Mass Units': s_mass_unit,
    'Position Units': s_position_unit,
    'Velocity Units': s_velocity_unit
    })


    particles = EmbedParticle(particles,embeds)

    Conf_data['Matter Particles'] = ({
    'Condition': particles,
    'Mass Units': m_mass_unit,
    'Position Units': m_position_unit,
    'Velocity Units': m_velocity_unit
    })

    Conf_data['Field-averaging Probes'] = ({

            'Probe Array': 'Defunct',
            'Probe Weights': 'Defunct' 

            })

    Conf_data['Central Mass'] = central_mass

    Conf_data['Field Smoothing'] = a

    Conf_data['NBody Cutoff Factor'] = B


    Conf_data['Uniform Field Override'] = ({

            'Flag': Uniform,
            'Density Unit': density_unit,
            'Density Value': Density,
            'Uniform Velocity': UVel

            })


    with open('{}{}{}{}{}'.format('./', save_path, '/', timestamp, '/config.txt'), "w+") as outfile:
        json.dump(Conf_data, outfile,indent=4)

    print('ULHelper: Compiled Config in Folder', timestamp)

    return timestamp
    
def Runs(save_path):
    runs = os.listdir(save_path)
    runs.sort()
    Latest = Load_Latest(save_path)
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
    
    if FLog == 1:
        return Latest
    
    else:
        print("Which folder do you want to analyse? Blank to load the latest one.")
        Ind = int(input() or -1)

        if Ind == -1:
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
        
        B = config['NBody Cutoff Factor']
        
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
      
         
        return  NS, central_mass, length, length_units, resol, duration, duration_units, step_factor, save_number, save_options, save_path, npz, npy, hdf5, s_mass_unit, s_position_unit, s_velocity_unit, solitons,start_time, m_mass_unit, m_position_unit, m_velocity_unit, particles, embeds, Uniform,Density, density_unit,a, B, UVel



def NBodyEnergy(particles,EndNum,TMdata,m_mass_unit,a):
    NBo = len(particles)
    ML = []
    for i in range(NBo):
        particle = particles[i]

        mass = particle[0]
        
        ML.append(mass)

        print(ML)
    

    KS = np.zeros(int(EndNum))
    PS = np.zeros(int(EndNum))

    print(len(particles))
    for i in range(int(EndNum)):

        Data = TMdata[i]


        if len(particles)>=2:


            for Mass1 in range(NBo):

                Index1 = int(Mass1*6)
                Position1 = Data[Index1:Index1+2]
                m1 = ML[Mass1]

                for Mass2 in range (Mass1+1,NBo,1):
                    Index2 = int(Mass2*6)
                    Position2 = Data[Index2:Index2+2]
                    m2 = ML[Mass2]

                    r = Position1 - Position2

                    rN = np.linalg.norm(r)

                    PS[i] = PS[i] - 1*m1*m2*a/(a*rN+np.exp(-1*a*rN))



        for particleID in range(NBo):
            Vx = Data[int(6*particleID+3)]
            Vy = Data[int(6*particleID+4)]
            Vz = Data[int(6*particleID+5)]

            KS[i] = KS[i] + 1/2*ML[particleID]*(Vx**2+Vy**2+Vz**2) 
            
            
    return NBo, KS, PS