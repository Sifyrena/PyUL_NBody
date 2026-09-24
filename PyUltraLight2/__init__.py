from PyUltraLight2.Credits import *
from PyUltraLight2.Version import *

from .Init.Config import Config

__version__ = S_version
__author__ = "Frank Wang and Auckland Cosmology"

from .Utils.printU import printU
from .Universe.Universe import *

axion_E = 1
length_unit = None
mass_unit = None
energy_unit = None

def AxionMass(m22 = 1):

    Universe = ULDMUniverse(m22)
    
    global axion_E, length_unit, mass_unit, energy_unit
    global convert, convert_back, convert_between
    
    axion_E = Universe.axion_E
    length_unit = Universe.length_unit
    mass_unit = Universe.mass_unit
    energy_unit = Universe.energy_unit
    
    convert = Universe.convert
    convert_back = Universe.convert_back
    convert_between = Universe.convert_between

# Not Directly Exposed to User
from PyUltraLight2.Utils.prog_bar import *
from PyUltraLight2.Utils.IO import *

from PyUltraLight2.Evolve import *
from PyUltraLight2.Init.Config import *
from PyUltraLight2.Solitons.Profile import *

LDSo = LoadDefaultSoliton