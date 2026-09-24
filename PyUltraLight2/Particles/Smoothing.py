##################### THIS CODE IS NOT USED CURRENTLY.

# Utility Functions to Initialise the Plummer Spheres

def MeshSpacing(resol,length,length_units, convert, convert_back):
    clength = convert(length,length_units,'l')
    lengthpc = convert_back(clength,'pc','l')

    return length/resol

# Particles/Smoothing.py  (cleaned, syntax-correct version)

def plummer_smoothing_from_radius(rP, length_units, convert_back):
    """Plummer smoothing parameter `a` in inverse code length,
    given a smoothing radius rP in user units."""
    return convert_back(1 / rP, length_units, 'l')

def plummer_smoothing_auto(resol, length, length_units, convert, grid_cells=0.5):
    """Default Plummer smoothing when the user selects 'Auto'.
    
    Returns `a` in inverse code length, set so the smoothing scale is
    `grid_cells` times the grid spacing. Default 0.5 matches the historical
    behaviour of `a = 2 * resol / lengthC` (smoothing = half a cell).
    """
    clength = convert(length, length_units, 'l')
    return 1.0 / (grid_cells * clength / resol)