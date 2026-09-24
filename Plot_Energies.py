#!/usr/bin/env python3

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PyUltraLight2.Utils.printU import printU
from PyUltraLight2.Utils.prog_bar import prog_bar
from PyUltraLight2.Init.Config import Config
# Get Universe Settings


def RecPlummer(a,length_units):   
    rP = 1/convert(a,length_units,'l')
    return rP

def NBodyEnergy(MassListSI,TMDataSI,EndNum,a=0,length_units = ''): 
    
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

def main(loc, relative=False):
        
    config = Config()
    config.FromFile(loc)
        
    m22 = config.uldm["m22"]
    Universe = ULDMUniverse(m22)
        
    axion_E = Universe.axion_E
    length_unit = Universe.length_unit
    mass_unit = Universe.mass_unit
    energy_unit = Universe.energy_unit

    convert = Universe.convert
    convert_back = Universe.convert_back
    convert_between = Universe.convert_between
    
    CB = convert_between 
    
    # Constants
    EPre = 1.0  # Replace with your energy prefactor if not 1
    UVel = [0, 0, 0]  # Replace with real values if needed
    
    MassListSI = ...
    TMDataSI = ...
    EndNum = ...
    a = ...
    s_velocity_unit = ...
    length_units = ...
    TimeStamp = ...
    resol = ...
    lengthKpc = ...
    durationMyr = ...
    Tp = ...
    
    ETStyle, EUColor, ETColor, ENColor = '-', 'orange', 'black', 'grey'  # Replace with your styling
    
    particles = ...  # Should be defined as a list

    # Figure Sizes
    EFigSize = (10, 6.18)

    # File Names
    def energy_path(name): return os.path.join(loc, name)
    EnergyName = energy_path("Energy_Total.jpg")

    # Load Data
    egylist = np.load(energy_path('Outputs/egylist.npy'), allow_pickle=True) * EPre
    egpcmlist = np.load(energy_path('Outputs/egpcmMlist.npy'), allow_pickle=True) * EPre
    egpcmlist2 = np.load(energy_path('Outputs/egpcmlist.npy'), allow_pickle=True) * EPre
    egpsilist = np.load(energy_path('Outputs/egpsilist.npy'), allow_pickle=True) * EPre
    ekandqlist = np.load(energy_path('Outputs/ekandqlist.npy'), allow_pickle=True) * EPre
    mtotlist = np.load(energy_path('Outputs/masseslist.npy'), allow_pickle=True) * EPre

    # Reconstruct NBody Energy
    NBo, KS, PS = NBodyEnergy(MassListSI, TMDataSI, EndNum, a, length_units)

    if relative:
        K0 = KS[0]
        EUnit = '$E_k(0)$'
        if UVel != [0, 0, 0]:
            VRelSI = CB(np.linalg.norm(UVel), s_velocity_unit, 'm/s', 'v')
            printU(f'Initial Relative Speed is {VRelSI:.3f} m/s', 'QW')
            K0 = 0.5 * MassListSI[0] * VRelSI**2
    else:
        K0 = 1
        EUnit = 'J'

    MES = PS + KS
    MESD = GetRel(MES) / K0
    EKQD = GetRel(ekandqlist) / K0
    EGPD = GetRel(egpsilist) / K0
    ECMD = GetRel(egpcmlist) / K0
    ECOD = GetRel(egpcmlist2) / K0
    KSD = GetRel(KS) / K0
    PSD = GetRel(PS) / K0

    EUOld = egylist
    EUOldD = GetRel(EUOld) / K0
    EUNew = egpsilist + ekandqlist + egpcmlist2
    EUNewD = GetRel(EUNew) / K0
    ETOld = EUOld + MES
    ETOldD = GetRel(ETOld) / K0
    ETNew = EUNew + MES
    ETNewD = GetRel(ETNew) / K0
    EROld = ETOld / ETOld[0]
    ERNew = ETNew / ETNew[0]

    # Plotting
    fig = plt.figure(figsize=EFigSize)
    ax = fig.add_subplot(111)
    ax.plot(Tp, EUNewD, ETStyle, color=EUColor, label='Total ULDM Energy')
    ax.plot(Tp, ETNewD, ETStyle, color=ETColor, label='Total Energy of System', lw=5)

    if len(particles) >= 2:
        ax.plot(Tp, MESD, ETStyle, color=ENColor, label='Total Mechanical Energy of Particles')

    ax.set_ylabel(f'$ΔE / $ {EUnit}')
    ax.legend(ncol=3, bbox_to_anchor=(0.5, -0.4), loc='lower center')
    plt.xlabel('Time / Myr')
    plt.title('Energy Change of System')
    plt.savefig(EnergyName, format='jpg', dpi=72)
    plt.show()

    # Output info
    sim_info = '\n'.join((
        TimeStamp,
        f'Resolution: {resol:.0f}^3',
        f'Box Length: {lengthKpc:.3f} kpc',
        f'Simulation Time Length: {durationMyr:.3f} Myr',
    ))
    print(sim_info)

    nbody_info = '\n'.join((
        TimeStamp,
        f'Number of Bodies: {NBo:.0f}',
    ))
    print(nbody_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot ULDM energy data from simulation output.')
    parser.add_argument('savepath', help='Path to simulation output folder')
    parser.add_argument('--relative', action='store_true', help='Use relative energy normalization')

    args = parser.parse_args()
    main(args.savepath, args.relative)
