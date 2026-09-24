import numpy as np
import os 
from PyUltraLight2.Utils.IO import get_pyul2_root

def LoadDefaultSoliton():
    """Load the soliton file relative to PyUL2's root directory."""
    root_dir = get_pyul2_root()  # Dynamically compute the root path
    soliton_path = os.path.join(root_dir, "./Solitons/f0.npy")
    return np.load(soliton_path)

def overlap_check(candidate, soliton):
    for i in range(len(soliton)):
        m = max(candidate[0], soliton[i][0])
        d_sol = 5.35854 / m
        c_pos = np.array(candidate[1])
        s_pos = np.array(soliton[i][1])
        displacement = c_pos - s_pos
        distance = np.sqrt(displacement[0] ** 2 + displacement[1] ** 2 + displacement[2] ** 2)
        if (distance < 2 * d_sol):
            return False
    return True

beta = 2.4538872760773143
prealpha = 3.8827652755822006
delta_x = 0.00001
