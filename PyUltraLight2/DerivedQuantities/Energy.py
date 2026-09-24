import numpy as np
import numexpr as ne

def calculate_energies(rho, Vcell, phiSP,phiTM, psi, karray2, fft_psi, ifft_funct, Density,Uniform, egpcmlist, egpsilist, ekandqlist, egylist, mtotlist, resol, save_grid_E = False, SelfInt = False, lambda_hat = 0):

    rho = rho.real
    
    #if Uniform:
    #    BoxAvg = np.mean(rho) # SHOULD BE TEMPORARILY DISABLED!
    #else:
    
    BoxAvg = 0

    # Gravitational potential energy density associated with the point masses potential

    ETM = ne.evaluate('phiTM*(rho-BoxAvg)') # Interaction in particle potential!
    ETMtot = Vcell * np.sum(ETM)
    egpcmlist.append(ETMtot) # TM Saved.

    # Gravitational potential energy density of self-interaction of the condensate
    ESI = ne.evaluate('0.5*(phiSP)*(rho-BoxAvg)') # New!
    if SelfInt:
        ESI += ne.evaluate('0.5*lambda_hat*(rho-BoxAvg)**2')
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

    if save_grid_E:
        EGrid = ne.evaluate("ETM + ESI + EKQ")[:,:,resol//2]
        return EGrid, EKQ[:,:,resol//2], ESI[:,:,resol//2]
    else:
        return [], [], []