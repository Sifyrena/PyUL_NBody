##### The Original PyUL NBody

import numpy as np

from PyUltraLight2.Interpolation.Lin3D import InterpolateLocal

def _local_phi_patch(phiSP, RPtX, RPtY, RPtZ, resol, Periodic):
    """4x4x4 patch of phiSP straddling grid point (RPtX,RPtY,RPtZ), i.e.
    indices RPt-1..RPt+2 in each axis (see InterpolateLocal for the
    2x2x2 sub-cube convention used on top of this).

    Periodic=True wraps the patch around the box (correct for the
    FFT-periodic Poisson solve, IsoP=False) instead of the old
    hard-zero-gradient fallback near the edges, which was a numerical
    safety net rather than physics: a particle within ~2 cells of any
    face would silently feel zero force from the periodic field it is
    actually still immersed in.

    Periodic=False (IsoP=True, isolated/vacuum potential) keeps that
    fallback, since there's no periodicity to wrap into and phiSP is
    expected to fall off near the domain edge anyway.
    """
    if Periodic:
        ix = np.arange(RPtX - 1, RPtX + 3) % resol
        iy = np.arange(RPtY - 1, RPtY + 3) % resol
        iz = np.arange(RPtZ - 1, RPtZ + 3) % resol
        return phiSP[np.ix_(ix, iy, iz)]

    if (RPtX <= 0) or (RPtY <= 0) or (RPtZ <= 0) or \
       (RPtX >= resol - 4) or (RPtY >= resol - 4) or (RPtZ >= resol - 4):
        return None

    return phiSP[RPtX-1:RPtX+3, RPtY-1:RPtY+3, RPtZ-1:RPtZ+3]


def FWNBody(t,TMState,masslist,phiSP,smoothing,lengthC,resol,Periodic=True):

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

        RPtX = int(RPt[0])
        RPtY = int(RPt[1])
        RPtZ = int(RPt[2])

        TAr = _local_phi_patch(phiSP, RPtX, RPtY, RPtZ, resol, Periodic)

        if TAr is None:
            GradientX = 0
            GradientY = 0
            GradientZ = 0

        else:

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
                
                
                if smoothing == 0:
                    F = 1/(rVL)**3
                else:                    
                    F = -(smoothing**3)/(smoothing**2*rVL**2+1)**(1.5) # The First Plummer
                
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


def FWNBody_NI(t,TMState,masslist,phiSP,smoothing,lengthC,resol,Periodic=True):

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

        RPtX = int(RPt[0])
        RPtY = int(RPt[1])
        RPtZ = int(RPt[2])

        TAr = _local_phi_patch(phiSP, RPtX, RPtY, RPtZ, resol, Periodic)

        if TAr is None:
            GradientX = 0
            GradientY = 0
            GradientZ = 0

        else:

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

                if smoothing == 0:
                    F = 1/(rVL)**3
                else:                    
                    F = -(smoothing**3)/(smoothing**2*rVL**2+1)**(1.5) # The First Plummer
                
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


def NBodyAdvance(TMState,h,masslist,phiSP,smoothing,lengthC,resol,NS,loc = '',Stream = False, StreamChar = '',Periodic = True):
        #


        if NS == 0: # NBody Dynamics Off
            StateLen = len(TMState)

            GradientLog = np.zeros_like(TMState)

            for i in range(StateLen//6):

                for j in range(3):
                    TMState[6*i+j] += h * TMState[6*i+j+3]


            return TMState, GradientLog

        if NS == 1:

            Step, GradientLog = FWNBody3(0,TMState,masslist,phiSP,smoothing,lengthC,resol,Periodic)
            TMStateOut = TMState + Step*h

            return TMStateOut, GradientLog

        elif NS%4 == 0:

            NRK = int(NS/4)

            H = h/NRK

            for RKI in range(NRK):
                TMK1, _ = FWNBody3(0,TMState,masslist,phiSP,smoothing,lengthC,resol,Periodic)
                TMK2, _ = FWNBody3(0,TMState + H/2*TMK1,masslist,phiSP,smoothing,lengthC,resol,Periodic)
                TMK3, _ = FWNBody3(0,TMState + H/2*TMK2,masslist,phiSP,smoothing,lengthC,resol,Periodic)
                TMK4, GradientLog = FWNBody3(0,TMState + H*TMK3,masslist,phiSP,smoothing,lengthC,resol,Periodic)
                TMState = TMState + H/6*(TMK1+2*TMK2+2*TMK3+TMK4)

                if Stream:
                    NBStream(loc,TMState[StreamChar])

            TMStateOut = TMState

            return TMStateOut, GradientLog


def NBodyAdvance_NI(TMState,h,masslist,phiSP,smoothing,lengthC,resol,NS,Periodic = True):
        #
        if NS == 0: # NBody Dynamics Off

            StateLen = len(TMState)

            GradientLog = np.zeros_like(TMState)

            for i in range(StateLen//6):

                for j in range(3):
                    TMState[6*i+j] += h * TMState[6*i+j+3]


            return TMState, GradientLog


        if NS == 1:

            Step, GradientLog = FWNBody3_NI(0,TMState,masslist,phiSP,smoothing,lengthC,resol,Periodic)
            TMStateOut = TMState + Step*h

            return TMStateOut, GradientLog

        elif NS%4 == 0:

            NRK = int(NS/4)

            H = h/NRK

            for RKI in range(NRK):
                TMK1, _ = FWNBody3_NI(0,TMState,masslist,phiSP,smoothing,lengthC,resol,Periodic)
                TMK2, _ = FWNBody3_NI(0,TMState + H/2*TMK1,masslist,phiSP,smoothing,lengthC,resol,Periodic)
                TMK3, _ = FWNBody3_NI(0,TMState + H/2*TMK2,masslist,phiSP,smoothing,lengthC,resol,Periodic)
                TMK4, GradientLog = FWNBody3_NI(0,TMState + H*TMK3,masslist,phiSP,smoothing,lengthC,resol,Periodic)
                TMState = TMState + H/6*(TMK1+2*TMK2+2*TMK3+TMK4)

            TMStateOut = TMState

            return TMStateOut, GradientLog
        


def FWNBodySimple(TMState,masslist,smoothing = 0):
    
    dTMdt = 0*TMState
    
    for i in range(len(masslist)):
        
        #0,6,12,18,...
        Ind = int(6*i)
           
        #XDOT = vx
        dTMdt[Ind]   =  TMState[Ind+3]
        #YDOT = vy
        dTMdt[Ind+1] =  TMState[Ind+4]
        #ZDOT = vz
        dTMdt[Ind+2] =  TMState[Ind+5]
        
        #x,y,z

        # Now, change velocities
        
        #Initialized Against ULDM Field
        #XDDOT
        dTMdt[Ind+3] =  0
        #YDDOT
        dTMdt[Ind+4] =  0
        #ZDDOT
        dTMdt[Ind+5] =  0
 
        poslocal = TMState[Ind:Ind+3]
    
        for ii in range(len(masslist)):
            
            if (ii != i) and (masslist[ii] != 0):
                
                IndX = int(6*ii)
                
                # print(ii)
                
                poslocalX = np.array([TMState[IndX],TMState[IndX+1],TMState[IndX+2]])
                
                rV = poslocalX - poslocal
                
                rVL = np.linalg.norm(rV) # Positive
                
                if smoothing == 0:
                    F = 1/(rVL)**3
                else:                    
                    F = -(smoothing**3)/(smoothing**2*rVL**2+1)**(1.5) # The First Plummer
                
                # Differentiated within Note 000.0F
                
                #XDDOT with Gravity
                dTMdt[Ind+3] -= masslist[ii]*F*rV[0]
                #YDDOT
                dTMdt[Ind+4] -= masslist[ii]*F*rV[1]
                #ZDDOT
                dTMdt[Ind+5] -= masslist[ii]*F*rV[2]

    return dTMdt
