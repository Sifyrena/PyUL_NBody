### THIS FILE HANDLES THE EXPORTING AND IMPORTING OF CONFIGURATION FILES
from PyUltraLight2.Utils.printU import printU
from PyUltraLight2.Utils.IO import SaveOptionsDigest, SaveOptionsCompile
from PyUltraLight2.Version import S_version

def __call__():
    return Config()

import inspect
import textwrap

def serialize_ext_phi(ext_phi):
    import numpy as np
    from pathlib import Path 
    match ext_phi:
        case int() | float():
            return ext_phi
        
        case np.ndarray():
            path = "ext_phi_array.npy"
            np.save(path, ext_phi)
            return {"type": "array_path", "value": path}
        
        case str() | Path():
            return {"type": "path", "value": str(ext_phi)}
        
        case _ if callable(ext_phi):
            try:
                source = textwrap.dedent(inspect.getsource(ext_phi))
                return {"type": "source", "value": source, "name": ext_phi.__name__}
            except OSError:
                # Function defined in interactive session (e.g. Jupyter)
                # Fall back to recording the function name only — not reconstructable
                # but at least ToFile() doesn't crash
                import warnings
                warnings.warn(
                    f"ExtPhi callable '{ext_phi.__name__}' was defined interactively "
                    f"and cannot be serialised to the config file. "
                    f"The simulation will run, but this config cannot be reloaded. "
                    f"Define ExtPhi in a .py file for full reproducibility.",
                    UserWarning
                )
                return {"type": "interactive_fn", "name": ext_phi.__name__}
        
        case None:
            return {"type": "none"}
        
        case _:
            raise NotImplementedError(f"ExtPhi type '{type(ext_phi).__name__}' is not supported.")
        
def deserialize_ext_phi(d):
    if isinstance(d, (int, float)):
        return d
    match d["type"]:

        case "array_path":
            return np.load(d["value"])
        case "path":
            return Path(d["value"])
        case "source":
            namespace = {}
            exec(compile(d["value"], "<string>", "exec"), namespace)
            return namespace[d["name"]]
        case "none":
            return None        

class Config:
    def __init__(self):
        # Save Options
        self.Saving = {
            "Loc": "./",
            "Flags": "Minimum",
            "Format": "npy",
            "Number": -1
        }
        
        # Duration
        self.Time = {
            "TimeDuration": 1,
            "StartTime": 0,
            "TimeUnits": "",
            "RKSteps": 32,
            "StepFactor": 1
        }
        
        # Space
        self.Space = {
            "Resolution": 256,
            "Box": {
                "BoxLength": 1,
                "LengthUnits": ""
            }
        }
        
        # Black Hole
        self.BlackHole = {
            "MatterParticles": {
                "Condition": [],
                "MassUnits": "",
                "PositionUnits": "",
                "VelocityUnits": "",
                "PlummerRadius": ""
            },
            "FieldSmoothing": "Auto", # Auto is the half-grid smoothing we know and love.
            "Sink": {
                "Flag": False,           # Enable moving BH absorption sink
                "ParticleIdx": 0,        # Index into MatterParticles.Condition this sink follows
                "Amplitude": 0.0,        # V0 of the imaginary potential, code units. Ignored if Dynamic=True.
                "Radius": 0.05,          # Gaussian sigma of the sink, code length units. Ignored if Dynamic=True.
                "Feedback": True,        # Add absorbed ULDM mass onto the particle's mass
                "Dynamic": False,        # Recompute Amplitude/Radius every step from the current
                                         # particle mass & speed via BondiHoyleCalibration(), instead
                                         # of using the fixed Amplitude/Radius above.
                "VFloor": 0.05,          # Minimum speed used in the Dynamic calibration (code units)
                                         # - keeps Radius finite when the particle is near-stationary
                                         # (e.g. at a bound orbit's turning point). Not a physical
                                         # soliton sound speed, just a numerical floor.
                "RadiusCap": None,       # Max Radius allowed in Dynamic mode (code length units).
                                         # The Bondi radius grows as M^2/v^3 combined with Mdot, so a
                                         # bound/oscillating particle can drive it far past the box
                                         # as it slows and grows - this caps it so the sink stays a
                                         # local drain rather than draining the whole grid. Defaults
                                         # to 0.25*BoxLength if left None.
                "ConserveMomentum": False  # Deposit the absorbed ULDM's momentum (from its local
                                         # phase gradient) onto the particle's velocity, instead of
                                         # only adding mass. Without this the sink is a pure mass
                                         # leak with no recoil - unphysical, since real accretion
                                         # exerts a drag force set by the relative velocity of the
                                         # swallowed gas. Costs 3 extra FFTs per step.
            }
        }
        
        # ULDM
        self.uldm = {
            "m22": 1,
            "SI": False,
            "LHat": 0,
            "Solitons": {
                "Condition": [[1,[0,0,0],[0,0,0],0]],
                "Embedded": [],
                "MassUnits": "",
                "PositionUnits": "",
                "VelocityUnits": ""
            },
            "Modifier": {
                "UniformFieldAddOn": {
                    "Flag": False,
                    "DensityUnit": "",
                    "DensityValue": 0,
                    "UniformVelocity": [0,0,0]
                }
            }
        }
        
        # Initialization Flags
        self.INIT = {
            "DumpInit": False,
            "DumpFinal": False,
            "UseInit": False,
            "InitPath": "",
            "InitWeight": 1
        }
        
        # Boundary Conditions
        self.BC = {
            "EdgeClear": False,
            "IsoP": False,
            "DSponge": False
        }
        
        # Simulation
        self.SIM = {
            "SelfGravity": True,
            "NBodyInterp": True,
            "NBodyGravity": True
        }
        
        # Compatibility
        self.LEGACY = {
            "Shift": False,
            "NLM": False,
            "DR": 1
        }
        
        # IO Override
        self.IO = {
            "Stream": False,
            "StreamChar": [0]
        }
        
        # IO
        self.RUNTIME = {
            "GenerateLog": True
        }
        
        # Advanced
        self.ADVANCED = {
            "CenterCalc": False,
            "ComputeQuad": False,
            "ExtPhi": 0
        }
        
        # Subsampled Grid
        self.SAMPLING = {
            "Length_Ratio": 0.5,
            "resolR": 64
        }
        
        # Extras
        self.EXTRAS = {
            "PrintEK": True,
            "MassChange": False,
            "MassFunc": "",
            "CorrectDrift": False,
            "CorrectFreq": 6
        }
        
        # Special Halting Conditions
        self.SPECIAL_HALTING_CONDITIONS = {
            "AutoStop": False,
            "AutoStop2": False,
            "AutoStop3": False,
            "KEThreshold": 0.9,
            "WellThreshold": 100
        }

        self.VERSION = S_version

    def to_dict(self):
        """Converts the class attributes into a nested dictionary."""
        return {
            key: value for key, value in self.__dict__.items()
        }

    def ToFile(self, file_path):
        """Saves the configuration to a JSON file."""
        import json
        
        ext_phi = self.ADVANCED["ExtPhi"]
        self.ADVANCED["ExtPhi"] = serialize_ext_phi(ext_phi)
        
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    def FromFile(self, file_path):
        """Loads the configuration from a JSON file."""
        import json
        with open(file_path, 'r') as f:
            data = json.load(f)
            self.__dict__.update(data)
            
        ext_phi = self.ADVANCED["ExtPhi"]
        self.ADVANCED["ExtPhi"] = deserialize_ext_phi(ext_phi)
            
    def ULDStepEst(self,save_number = -1):
        import numpy as np
        duration = self.Time["TimeDuration"]
        duration_units = self.Time["TimeUnits"]
        length = self.Space["Box"]["BoxLength"]
        length_units = self.Space["Box"]["LengthUnits"]
        resol = self.Space["Resolution"]  
        step_factor = self.Time["StepFactor"]
        
        from PyUltraLight2.Universe.Universe import ULDMUniverse
        self.Universe = ULDMUniverse(self.uldm["m22"])
        self.axion_E = self.Universe.axion_E
        self.length_unit = self.Universe.length_unit
        self.mass_unit = self.Universe.mass_unit
        self.energy_unit = self.Universe.energy_unit
        self.convert = self.Universe.convert
        self.convert_back = self.Universe.convert_back
        self.convert_between = self.Universe.convert_between
        
        
        lengthC = self.convert(length, length_units, 'l')
    
        t = self.convert(duration, duration_units, 't')
        
        delta_t = (lengthC/float(resol))**2/np.pi

        min_num_steps = np.ceil(t / delta_t)
        MinUS = int(min_num_steps//step_factor)

        #print(f'The required number of ULDM steps is {MinUS}')
        
        if save_number > 0:
            
            if save_number >= MinUS:
                MinUS = int(save_number)
            
            else:
                MinUS = int(save_number * (MinUS // (save_number) + 1))
                
        #print(f'The actual ULDM steps is {MinUS}')
        
        return MinUS
    
    def RunSignature(self):
        """
        Returns a string summarizing the run:
        axion mass (m22), box length, duration, and resolution.
        Example:
            "m22=1.0, Box=1 [LengthUnits], Duration=10 [TimeUnits] @ 256^3"
        """
        m22 = self.uldm.get("m22", "UNKNOWN")
        box = self.Space["Box"].get("BoxLength", "UNKNOWN")
        box_units = self.Space["Box"].get("LengthUnits", "")
        dur = self.Time.get("TimeDuration", "UNKNOWN")
        dur_units = self.Time.get("TimeUnits", "")
        res = self.Space.get("Resolution", "UNKNOWN")
        
        Lambda = self.uldm.get("LHat","0")
        
        if Lambda!=0:
            sig = f"M{m22}_Λ{Lambda}_L{box}{box_units}_T{dur}{dur_units}@{res}"
        else:
            sig = f"M{m22}_L{box}{box_units}_T{dur}{dur_units}@{res}"
        return sig
