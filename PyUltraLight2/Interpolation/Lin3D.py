# Interpolation/Lin3D.py
import numpy as np

def trilinear_weights(frac):
    """Compute the 8 trilinear weights for fractional position `frac` in [0,1)^3.
    
    Returned as a flat array in (i,j,k) = (0,0,0), (0,0,1), (0,1,0), ...
    order matching c.flatten()[n] for a 2x2x2 cube c indexed as c[i,j,k].
    """
    fx, fy, fz = frac
    gx, gy, gz = 1.0 - frac
    return np.array([
        gx*gy*gz, gx*gy*fz, gx*fy*gz, gx*fy*fz,
        fx*gy*gz, fx*gy*fz, fx*fy*gz, fx*fy*fz,
    ])


def interpolate_in_cube(frac, cube):
    """Trilinear interpolation inside a pre-sliced 2x2x2 cube at fractional
    position `frac`. Caller is responsible for bounds checking.
    
    Hot-path version: used in the N-body RK4 inner loop where the cube has
    already been sliced out of a larger field.
    """
    fx, fy, fz = frac
    gx, gy, gz = 1.0 - frac
    c = cube
    return (gx*gy*gz * c[0,0,0] + gx*gy*fz * c[0,0,1]
          + gx*fy*gz * c[0,1,0] + gx*fy*fz * c[0,1,1]
          + fx*gy*gz * c[1,0,0] + fx*gy*fz * c[1,0,1]
          + fx*fy*gz * c[1,1,0] + fx*fy*fz * c[1,1,1])


def sample_field(field, lengthC, resol, position, out_of_bounds=0.0):
    """Trilinear interpolation of `field` at world-space `position`.
    
    Convenience wrapper: finds the grid cell, handles bounds, and evaluates.
    Use this for one-off samples (diagnostics, energy evaluation).
    For repeated samples at the same position (e.g. three gradient components),
    slice the cube yourself and call interpolate_in_cube directly.
    """
    grid_pos = (position / lengthC + 0.5) * resol
    idx = np.floor(grid_pos).astype(int)
    frac = grid_pos - idx
    ix, iy, iz = idx

    if np.any(idx <= 0) or np.any(idx >= resol - 1):
        return out_of_bounds

    cube = field[ix:ix+2, iy:iy+2, iz:iz+2]
    return interpolate_in_cube(frac, cube)

InterpolateLocal = interpolate_in_cube
QuickInterpolate = sample_field