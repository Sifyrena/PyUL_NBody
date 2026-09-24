import numpy as np
import numexpr as ne
from scipy.interpolate import CubicSpline as CS 

from PyUltraLight2.Solitons.Profile import LoadDefaultSoliton
############################FUNCTION TO PUT SPHERICAL SOLITON DENSITY PROFILE INTO 3D BOX (Uses pre-computed array)

def InitSolitonF(gridVec, position, resol, alpha, delta_x=0.00001, DR = 1.0):

    xAr, yAr, zAr = np.meshgrid(gridVec - position[0],
                                gridVec - position[1],
                                gridVec - position[2],
                                sparse=True,
                                indexing="ij")

    gridSize = gridVec[1] - gridVec[0]
    DistArr = ne.evaluate("sqrt(xAr**2+yAr**2+zAr**2 * DR)")

    f = alpha * LoadDefaultSoliton()
    fR = np.arange(len(f)) * delta_x / np.sqrt(alpha)

    fInterp = CS(fR, f, bc_type=("clamped", "not-a-knot"))

    DistArrPts = DistArr.reshape(resol**3)

    Eval = fInterp(DistArrPts)
    
    Eval[DistArrPts > fR[-1]] = 0 # Fix Cubic Spline Behaviour

    return Eval.reshape(resol,resol,resol)

############################FUNCTION TO PUT SPHERICAL SOLITON DENSITY PROFILE INTO 3D BOX (Uses pre-computed array)

def initsolitonRadial(line, alpha, f, delta_x,Cutoff = 9, IndexCorrect = False):
    funct = 0*line
    
    for index in np.ndindex(funct.shape):
        
        
        # Note also that this distfromcentre is here to calculate the distance of every gridpoint from the centre of the soliton, not to calculate the distance of the soliton from the centre of the grid
        distfromcentre = (
            (line[index[0]]) ** 2) ** 0.5
        # Utilises soliton profile array out to dimensionless radius 5.6.
        if (np.sqrt(alpha) * distfromcentre <= Cutoff):
            if IndexCorrect:
                funct[index] = alpha * f[int(np.sqrt(alpha) * (distfromcentre / delta_x))]
            else:
                funct[index] = alpha * f[int(np.sqrt(alpha) * (distfromcentre / delta_x + 1))]
    return funct