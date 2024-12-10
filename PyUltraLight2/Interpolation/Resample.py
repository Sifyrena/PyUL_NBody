import numpy as np
import numexpr as ne

def Resample3Box(psi, COM, XAR, loc, save_num, save_format, Length_Ratio = 0.5, resolR = 128, Save_Rho = False, Save_Psi = False, Compatibility = False, Save_Rho2 = False):
    
    from scipy.interpolate import RegularGridInterpolator as RPI
    # Note that phase correction is not performed in this version.
    
    lengthCR = -2 * XAR[0] * Length_Ratio

    resol = psi.shape[0]

    if resolR <= 4:
        resolR =int(resol * Length_Ratio * resolR) # This is a ratio!

    if Compatibility:
        GVR = np.linspace(-lengthCR / 2.0 + lengthCR / float(2 * resolR), lengthCR / 2.0 - lengthCR / float(2 * resolR), int(resolR), endpoint = True)
    else:
        GVR = np.linspace(-lengthCR/2, lengthCR/2, int(resolR), endpoint = False)
        
    Real_I = RPI((XAR,XAR,XAR),np.real(psi),method='linear',bounds_error = False, fill_value = 0)
    Imag_I = RPI((XAR,XAR,XAR),np.imag(psi),method='linear',bounds_error = False, fill_value = 0)
    
    NewGrid = np.meshgrid(
        GVR + COM[0], GVR + COM[1], GVR + COM[2],
        sparse=False, indexing='ij')

    NewGrid_List = np.reshape(NewGrid, (3, -1), order='C').T

    IReal = Real_I(NewGrid_List)
    IImag = Imag_I(NewGrid_List)

    IPsi = ne.evaluate("IReal + 1j*IImag")

    PsiNew = np.reshape(IPsi,(resolR,resolR,resolR))
    
    if Save_Psi:
        IOSave(loc,'3WfnRS',save_num,save_format,PsiNew)
    
    if Save_Rho:
        RhoNew = ne.evaluate("conj(PsiNew)*PsiNew").real
        
        IOSave(loc,'3DensityRS',save_num,save_format,RhoNew)

    if Save_Rho2:
        RhoNew = ne.evaluate("conj(PsiNew)*PsiNew").real
        IOSave(loc,'2DensityRS',save_num,save_format,RhoNew[:,:,resolR//2])


def Wfn_to_PyUL1(psi):

    from scipy.interpolate import RegularGridInterpolator as RPI
    # Note that phase correction is not performed in this version.
    
    lengthCR = 1 
    resolR = psi.shape[0]
    
    GVR = np.linspace(-lengthCR / 2.0 + lengthCR / float(2 * resolR), lengthCR / 2.0 - lengthCR / float(2 * resolR), int(resolR), endpoint = True)
    XAR = np.linspace(-lengthCR/2, lengthCR/2, int(resolR), endpoint = False)
        
    Real_I = RPI((XAR,XAR,XAR),np.real(psi),method='linear',bounds_error = False, fill_value = 0)
    Imag_I = RPI((XAR,XAR,XAR),np.imag(psi),method='linear',bounds_error = False, fill_value = 0)
    
    NewGrid = np.meshgrid(
        GVR , GVR , GVR ,
        sparse=False, indexing='ij')
    
    NewGrid_List = np.reshape(NewGrid, (3, -1), order='C').T
    
    IReal = Real_I(NewGrid_List)
    IImag = Imag_I(NewGrid_List)
    
    IPsi = ne.evaluate("IReal + 1j*IImag")
    
    return np.reshape(IPsi,(resolR,resolR,resolR))

