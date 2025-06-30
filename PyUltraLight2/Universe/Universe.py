# Fundamental Constants and Conversion Tools. All parametrised with m22.

import numpy as np
import astropy.units as u
from astropy.constants import c, G, hbar, M_sun

pi = np.pi

# Electron volt in SI units (energy)
eV = u.eV  # Already defined in astropy.units

parsec = u.pc  # astropy handles this automatically
light_year = u.lyr

# Solar mass
solar_mass = M_sun  # from astropy.constants

# Cosmological parameters
omega_m0 = 0.31
H_0 = 67.7 * u.km / u.s / u.Mpc  # Hubble constant in standard cosmology units

# Convert H_0 to SI units (1/s)
#H_0 = H_0_value.to(1 / u.s)

# Critical density
CritDens = (3 * H_0**2 / (8 * pi * G))

# Define your custom time unit (still in SI)
time_unit = ((3 * H_0**2 * omega_m0 / (8 * pi))**-0.5)

# ULDM:

m22eV = 1e-22 * eV  # Default value

class ULDMUniverse():

    def __init__(self, mass = 1):
    
        self.m22 = mass
        self.axion_E = mass * m22eV
        self.axion_mass = self.axion_E / c **2
        
        self.time_unit = time_unit
        self.length_unit = (8 * pi * hbar ** 2 / (3 * self.axion_mass ** 2 * H_0 ** 2 * omega_m0)) ** 0.25
        
        
        self.mass_unit = (3 * H_0 ** 2 * omega_m0 / (8 * pi)) ** 0.25 * hbar ** 1.5 / (self.axion_mass ** 1.5 * G)
        
        self.energy_unit = self.mass_unit * self.length_unit ** 2 / (time_unit**2)
        
    def convert(self ,value, unit, type):
        converted = 0
        if (type == 'l'):
            if (unit == ''):
                converted = value
            elif (unit == 'm') or (unit == 'SI'):
                converted = value / self.length_unit
            elif (unit == 'km'):
                converted = value * 1e3 / self.length_unit
            elif (unit == 'pc'):
                converted = value * parsec / self.length_unit
            elif (unit == 'kpc'):
                converted = value * 1e3 * parsec / self.length_unit
            elif (unit == 'Mpc'):
                converted = value * 1e6 * parsec / self.length_unit
            elif (unit == 'ly'):
                converted = value * light_year / self.length_unit
            else:
                raise NameError('Unsupported LENGTH unit used')
    
        elif (type == 'm'):
            if (unit == ''):
                converted = value
            elif (unit == 'kg') or (unit == 'SI'):
                converted = value / self.mass_unit
            elif (unit == 'solar_masses'):
                converted = value * solar_mass / self.mass_unit
            elif (unit == 'M_solar_masses'):
                converted = value * solar_mass * 1e6 / self.mass_unit
            else:
                raise NameError('Unsupported MASS unit used')
    
        elif (type == 't'):
            if (unit == ''):
                converted = value
            elif (unit == 's') or (unit == 'SI'):
                converted = value / time_unit
            elif (unit == 'yr'):
                converted = value * 60 * 60 * 24 * 365 / time_unit
            elif (unit == 'kyr'):
                converted = value * 60 * 60 * 24 * 365 * 1e3 / time_unit
            elif (unit == 'Myr'):
                converted = value * 60 * 60 * 24 * 365 * 1e6 / time_unit
            elif (unit == 'Gyr'):
                converted = value * 60 * 60 * 24 * 365 * 1e9 / time_unit
            else:
                raise NameError('Unsupported TIME unit used')
    
        elif (type == 'v'):
            if (unit == ''):
                converted = value
            elif (unit == 'm/s') or (unit == 'SI'):
                converted = value * time_unit / self.length_unit
            elif (unit == 'km/s'):
                converted = value * 1e3 * time_unit / self.length_unit
            elif (unit == 'km/h'):
                converted = value * 1e3 / (60 * 60) * time_unit / self.length_unit
            elif (unit == 'c'):
                converted = value * time_unit / self.length_unit * c
            else:
                raise NameError('Unsupported SPEED unit used')
                
                
        elif (type == 'd'):
            if (unit == ''):
                converted = value
            elif (unit == 'Crit'):
                converted = value / omega_m0 
            elif (unit == 'MSol/pc3'):
                converted = value * solar_mass / self.mass_unit * self.length_unit**3 / parsec**3     
            elif (unit == 'MMSol/kpc3'):
                converted = value * solar_mass / self.mass_unit * self.length_unit**3 / parsec**3  / 1000  
            elif (unit == 'kg/m3') or (unit == 'SI'):
                converted = value / self.mass_unit * self.length_unit**3
            else:
                raise NameError('Unsupported DENSITY unit used')
                
                
        elif (type == 'a'):
            if (unit == ''):
                converted = value
            elif (unit == 'm/s2') or (unit == 'SI'):
                converted = value / self.length_unit * time_unit**2   
            else:
                raise NameError('Unsupported ACCELERATION unit used')
    
        
        elif (type == 'p'):
            if (unit == ''):
                converted = value
            elif (unit == 'kgm/s') or (unit == "Nr") or (unit == 'SI'):
                converted = value / self.mass_unit / self.length_unit * time_unit 
            else:
                raise NameError('Unsupported MOMENTUM unit used')
    
        else:
            raise TypeError('Unsupported conversion type')
    
        return converted

    ####################### FUNCTION TO CONVERT FROM DIMENSIONLESS UNITS TO DESIRED UNITS
    def convert_back(self, value, unit, type):
        converted = 0
        if (type == 'l'):
            if (unit == ''):
                converted = value
            elif (unit == 'm') or (unit == 'SI'):
                converted = value * self.length_unit
            elif (unit == 'km'):
                converted = value / 1e3 * self.length_unit
            elif (unit == 'pc'):
                converted = value / parsec * self.length_unit
            elif (unit == 'kpc'):
                converted = value / (1e3 * parsec) * self.length_unit
            elif (unit == 'Mpc'):
                converted = value / (1e6 * parsec) * self.length_unit
            elif (unit == 'ly'):
                converted = value / light_year * self.length_unit
            else:
                raise NameError('Unsupported LENGTH unit used')
    
        elif (type == 'm'):
            if (unit == ''):
                converted = value
            elif (unit == 'kg') or (unit == 'SI'):
                converted = value * self.mass_unit
            elif (unit == 'solar_masses'):
                converted = value / solar_mass * self.mass_unit
            elif (unit == 'M_solar_masses'):
                converted = value / (solar_mass * 1e6) * self.mass_unit
            else:
                raise NameError('Unsupported MASS unit used')
    
        elif (type == 't'):
            if (unit == ''):
                converted = value
            elif (unit == 's') or (unit == 'SI'):
                converted = value * self.time_unit
            elif (unit == 'yr'):
                converted = value / (60 * 60 * 24 * 365) * self.time_unit
            elif (unit == 'kyr'):
                converted = value / (60 * 60 * 24 * 365 * 1e3) * self.time_unit
            elif (unit == 'Myr'):
                converted = value / (60 * 60 * 24 * 365 * 1e6) * self.time_unit
            elif (unit == 'Gyr'):
                converted = value / (60 * 60 * 24 * 365 * 1e9) * self.time_unit
            else:
                raise NameError('Unsupported TIME unit used')
    
        elif (type == 'v'):
            if (unit == ''):
                converted = value
            elif (unit == 'm/s') or (unit == 'SI'):
                converted = value / time_unit * self.length_unit
            elif (unit == 'km/s'):
                converted = value / (1e3) / time_unit * self.length_unit
            elif (unit == 'km/h'):
                converted = value / (1e3) * (60 * 60) / time_unit * self.length_unit
            elif (unit == 'c'):
                converted = value * time_unit / self.length_unit / c
            else:
                raise NameError('Unsupported SPEED unit used')
                
                
        elif (type == 'd'):
            if (unit == ''):
                converted = value
            elif (unit == 'Crit'):
                converted = value * omega_m0 
            elif (unit == 'MSol/pc3'):
                converted = value / solar_mass * self.mass_unit / self.length_unit**3 * parsec**3
            elif (unit == 'MMSol/kpc3'):
                converted = value / solar_mass * self.mass_unit / self.length_unit**3 * parsec**3 * 1000
            elif (unit == 'kg/m3') or (unit == 'SI'):
                converted = value * self.mass_unit / self.length_unit**3
            else:
                raise NameError('Unsupported DENSITY unit used')
    
        elif (type == 'a'):
            if (unit == ''):
                converted = value
            elif (unit == 'm/s2') or (unit == 'SI') :
                converted = value * self.length_unit / time_unit**2   
            else:
                raise NameError('Unsupported ACCELERATION unit used')     
    
        
        elif (type == 'p'):
            if (unit == ''):
                converted = value
            elif (unit == 'kgm/s') or (unit == "Nr") or (unit == 'SI'):
                converted = value * self.mass_unit * self.length_unit / time_unit 
            else:
                raise NameError('Unsupported MOMENTUM unit used')
                
        else:
            raise TypeError('Unsupported conversion type')
    
        return converted
    
    def convert_between(self,value, oldunit,newunit, type):
        
        return self.convert_back(self.convert(value,oldunit,type),newunit,type)