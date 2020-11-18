#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 13:56:50 2020

@author: ywan598

"""

D_version = 'Helper Build 4. 18 Nov 2020'

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
        print('ULHelper: Loaded Planar Mass Density Data \n')
    if save_testmass:
        print('ULHelper: Loaded Test Mass State Data \n')
    if save_phi_plane:
        print('ULHelper: Loaded Planar Gravitational Field Data \n')
    if save_gradients:
        print('ULHelper: Loaded Test Mass Gradient Data \n')

    
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
        
    print("ULHelper: Loaded", EndNum, "Data Entries")
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
    
       
def GenerateConfig(central_mass, length, length_units, resol, duration, duration_units, step_factor, save_number, save_options, save_path, npz, npy, hdf5, s_mass_unit, s_position_unit, s_velocity_unit, solitons,start_time, m_mass_unit, m_position_unit, m_velocity_unit, particles, Uniform,Density,a, NCV,NCW
           ):
    
        tm = time.localtime()


        print("What is the name of the run? Leave blank to use automatically generated timestamp.")
        InputName = input()
        
        if InputName != "":
            
            timestamp = InputName
            
        else:
            
            talt = ['0', '0', '0']
            for i in range(3, 6):
                if tm[i] in range(0, 10):
                    talt[i - 3] = '{}{}'.format('0', tm[i])
                else:
                    talt[i - 3] = tm[i]
            timestamp = '{}{}{}{}{}{}{}{}{}{}{}{}{}'.format(tm[0], '.', tm[1], '.', tm[2], '_', talt[0], '_', talt[1], '_', talt[2], '_', resol)
        
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
        
        Conf_data['Field-averaging Probes'] = ({
                
                'Probe Array': NCV.tolist(),
                'Probe Weights': NCW.tolist()
                
                })
        
        Conf_data['Central Mass'] = central_mass
        
        Conf_data['Field Smoothing'] = a
        
        
        Conf_data['Uniform Field Override'] = ({
                
                'Flag': Uniform,
                'Density Value': Density
                
                })
        
        
        with open('{}{}{}{}{}'.format('./', save_path, '/', timestamp, '/config.txt'), "w+") as outfile:
            json.dump(Conf_data, outfile,indent=4)
            
        return timestamp
    

def Runs(save_path):
    runs = os.listdir(save_path)
    runs.sort()

    
    for i in range(len(runs)):
        
        if os.path.isdir(os.path.join(save_path, runs[i])):
            
    
        
            print("[",i,"]: ", runs[i] )
            
            LS = i
        
    print("Which Folder Do You Want to Run? Leave blank to run the last one.")
    
    Ind = int(input() or LS)
   
    
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
        
        s_mass_unit = config["ULDM Solitons"]['Mass Units']
        
        s_position_unit = config["ULDM Solitons"]['Position Units']
        
        s_velocity_unit = config["ULDM Solitons"]['Velocity Units']
        
        
        ### ULDM Modifier
        
        Uniform = config["Uniform Field Override"]["Flag"]
        Density = config["Uniform Field Override"]["Density Value"]
        
        
        ### Field Averaging
        
        NCV = np.array(config["Field-averaging Probes"]["Probe Array"])
        NCW = np.array(config["Field-averaging Probes"]["Probe Weights"])
        
         
        return  central_mass, length, length_units, resol, duration, duration_units, step_factor, save_number, save_options, save_path, npz, npy, hdf5, s_mass_unit, s_position_unit, s_velocity_unit, solitons,start_time, m_mass_unit, m_position_unit, m_velocity_unit, particles, Uniform,Density,a, NCV,NCW


def AnimSummary(TimeStamp,save_path, VX,VY,FPS,Loga,Skip,ToFile):
    import matplotlib.animation

    BarWidth = 1  # the width of the bars

    MovieX = VX

    MovieY = VY

    loc = save_path + '/' + TimeStamp
    
    central_mass, length, length_units, resol, duration, duration_units, step_factor, save_number, save_options, save_path, npz, npy, hdf5, s_mass_unit, s_position_unit, s_velocity_unit, solitons,start_time, m_mass_unit, m_position_unit, m_velocity_unit, particles, Uniform,Density,a, NCV,NCW = LoadConfig(loc)
    
    EndNum, data, TMdata, phidata, graddata = Load_Data(save_path,TimeStamp, save_options,save_number)
    
    
    ML = []
    for i in range(len(particles)):
        particle = particles[i]
        
        mass = particle[0]
                
        ML.append(mass)
    
        print(ML)

    KS = np.zeros(int(EndNum))
    PS = np.zeros(int(EndNum))

    for i in range(int(EndNum)):
        
        Data = TMdata[i]
        
        if len(particles)==2:
    
            r = Data[0:2] - Data[6:8]
    
            rN = np.linalg.norm(r)
            m1 = ML[0]
            m2 = ML[1]
        
            PS[i] = -1*m1*m2*a/(a*rN+np.exp(-1*a*rN))
       
        for particleID in range(len(particles)):
            Vx = Data[int(6*particleID+3)]
            Vy = Data[int(6*particleID+4)]
            Vz = Data[int(6*particleID+5)]
            
            KS[i] = KS[i] + 1/2*ML[particleID]*(Vx**2+Vy**2+Vz**2) 

    egylist = np.load('{}{}'.format(loc, '/egylist.npy'))
  
    egylistD = egylist - egylist[1]
    
    PSD = PS - PS[1]
    
    KSD = KS - KS[1]
    
    TotalED = PSD+KSD+egylistD

    try:
        VTimeStamp = TimeStamp
    except NameError:
        VTimeStamp = str('Debug')
    
    AnimName = '{}{}{}{}{}'.format("./",save_path,"/AnimSummary_",VTimeStamp,'.mp4')
    
    if ToFile:
        print("Saving ",AnimName)
    
    
    # Defining Grid System and Plotting Variables
    
    NumSol = len(solitons)
    
    figAS = plt.figure(figsize=(MovieX, MovieY))
    gs = figAS.add_gridspec(4, 4)
    
    AS_GradGraph = figAS.add_subplot(gs[0, :])
    AS_GradGraph.set_title('Acceleration of Particle #1')
    
    AS_FieldPlane = figAS.add_subplot(gs[1:3,0:2])
    AS_FieldPlane.set_title('2D Gravitational Field')
    
    AS_RhoPlane = figAS.add_subplot(gs[1:3, 2:4])
    AS_RhoPlane.set_title('2D Mass Density')
    
    
    AS_EDelta = figAS.add_subplot(gs[3, :])
    AS_EDelta.set_title('Energy Change Snapshot #1')
    
    
    AS_FieldPlane.set_aspect('equal')
    
    if Loga:
        
        if Uniform:
            data0 = np.log(np.array(data)/Density)
            print("Initial Field is Uniform. Evaluating Change Ratio.")
        
        else:
            data0 = np.log(data)
            
        planemax = np.max(data0)
        planemin = -50
        
        print("Using Log Plot, the Contour Level Limits Are")
        print(planemax,planemin)
        
    else:
        data0 = (data)
        planemax = np.max(data0)
        planemin = np.min(data0)
        
            
    levels = np.linspace(planemin, planemax, int(resol/2))
    
    PlotRange = np.linspace(-length/2, length/2,resol,endpoint = False)
    
    BarLabels = ['Total Energy', 'Mass Kinetic Energy', 'ULDM Total Energy']
    
    BarX = np.arange(len(BarLabels))  # the label locations
    
    BarMax = np.max(np.abs(TotalED))
        
    graddataP = (graddata)
    
    
    DTEMaxChange = 0.
    DTEMinChange = 0.
    
    phimax = np.max(phidata)
    phimin = np.min(phidata)
    
    def animateAS(i,DTEMaxChange,DTEMinChange):
        
        if Skip != 1 and i == 0:
            print("We are skipping some frames.")
        i = int(Skip*i-1)
    
        # Acceleration Graph
    
        AS_GradGraph.plot(i,graddataP[i][0],'r.',label = '$x$')
        AS_GradGraph.plot(i,graddataP[i][1],'g.',label = '$y$')
        AS_GradGraph.plot(i,graddataP[i][2],'b.',label = '$z$')
       
        # Field Graph
        
        sliced = phidata[i]
    
        AS_FieldPlane.imshow(sliced,origin='lower',vmin = phimin, vmax = phimax)
    
        AS_FieldPlane.set_xticks([])
        AS_FieldPlane.set_yticks([])
        
        # Density Graph
        
        AS_RhoPlane.cla()
        AS_RhoPlane.set_aspect('equal')
        
        AS_RhoPlane.set_xticks([])
        AS_RhoPlane.set_yticks([])
        
        AS_RhoPlane.set_xlim([-length/2,length/2])
        AS_RhoPlane.set_ylim([-length/2,length/2])
        AS_RhoPlane.get_xaxis().set_ticks([])
        AS_RhoPlane.get_yaxis().set_ticks([])
        
        AS_RhoPlane.contour(PlotRange,PlotRange,data0[i], levels=levels, vmin=planemin, vmax=planemax,cmap = 'coolwarm')
    
    
        TMStateLoc = TMdata[i]
        for particleID in range(len(particles)):
            Vx = TMStateLoc[int(6*particleID+3)]
            Vy = TMStateLoc[int(6*particleID+4)]
            Vz = TMStateLoc[int(6*particleID+5)]
            
    
            TMx = TMStateLoc[int(6*particleID)]
            TMy = TMStateLoc[int(6*particleID+1)]
            TMz = TMStateLoc[int(6*particleID+2)]
            
    
            AS_RhoPlane.plot([TMy],[TMx],'ko')
            AS_RhoPlane.quiver([TMy],[TMx],[Vy],[Vx])
            
            
        # Bar Graph
    
        AS_EDelta.cla()
        
        DTE = TotalED[i]
        DKE = KSD[i]
        DUE = egylistD[i]
        
        DTEMaxChange = np.max([DTEMaxChange,DTE])
        DTEMinChange = np.min([DTEMinChange,DTE])
        
        EnergyEntry = [DTE, DKE, DUE]
        
        rects1 = AS_EDelta.bar(BarX, EnergyEntry, BarWidth)
    
        zeroLine = AS_EDelta.plot((-0.5,2.5),(0,0),'k--')
        
        maxLine = AS_EDelta.plot((-0.5,2.5),(DTEMaxChange,DTEMaxChange),'r-.')
        
        minLine = AS_EDelta.plot((-0.5,2.5),(DTEMinChange,DTEMinChange),'b-.')
    
        # Add some text for labels, title and custom x-axis tick labels, etc.
        AS_EDelta.set_ylabel('Absolute Change Since Onset ($\mathcal{E}$)')
        AS_EDelta.set_ylim(-2*BarMax,2*BarMax)
        AS_EDelta.set_title('Energy Change for Snapshot %.0f'%i)
        AS_EDelta.set_xticks(BarX)
        AS_EDelta.set_xticklabels(BarLabels)
    

        
        if i%FPS == 0 and i!= 0:
            print('Animated %.0f seconds out of %.2f seconds of data.' % (i/FPS, EndNum/FPS))
        
        if i == EndNum-1:
            clear_output()
            print('Animation Complete:', AnimName)
    
            
    interval = 0.00001 #in seconds
    aniAS = matplotlib.animation.FuncAnimation(figAS,animateAS,int(EndNum/Skip),fargs=(DTEMaxChange,DTEMinChange),interval=interval*1e+3,blit=False)
    
    Writer = matplotlib.animation.writers['ffmpeg']
    
    writer = Writer(fps=FPS, metadata=dict(artist='PyUltraLightF'))
    
    if ToFile:
        aniAS.save(AnimName, writer=writer)
    else:
        from IPython.display import HTML
        animated_plot0 = HTML(aniAS.to_jshtml())
        figAS.clear()
        display(animated_plot0) 