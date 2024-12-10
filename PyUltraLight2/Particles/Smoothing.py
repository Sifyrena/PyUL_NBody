# Utility Functions to Initialise the Plummer Spheres

def MeshSpacing(resol,length,length_units, convert, convert_back):
    clength = convert(length,length_units,'l')
    lengthpc = convert_back(clength,'pc','l')

    return length/resol
    
def GenPlummer(rP,length_units, silent = True, resol = 0,length = 0, convert, convert_back):
    a = convert_back(1/rP,length_units,'l')
    return a # IN CODE UNITS (LENGTH^-1)

def GenPlummerAuto(resol, length, length_units,convert, co):
    clength = convert(length,length_units,'l')
    return convert_back(resol/clength, length_units, "l"))