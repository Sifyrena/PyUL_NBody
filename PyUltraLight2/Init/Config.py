### THIS FILE HANDLES THE EXPORTING AND IMPORTING OF CONFIGURATION FILES
from PyUltraLight2.Utils.printU import printU
from PyUltraLight2.Utils.IO import SaveOptionsDigest, SaveOptionsCompile
from PyUltraLight2.Version import S_version

def __call__():
    return Config()

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
            "FieldSmoothing": "Auto" # Auto is the half-grid smoothing we know and love.
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
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    def FromFile(self, file_path):
        """Loads the configuration from a JSON file."""
        import json
        with open(file_path, 'r') as f:
            data = json.load(f)
            self.__dict__.update(data)