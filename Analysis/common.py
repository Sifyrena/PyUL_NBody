"""Shared loading helpers for the Analysis/ scripts.

All of these operate on a run's output directory `loc` (the string
returned by Evolve(), i.e. the folder containing config.uldm and Outputs/).
"""

import os
import glob
import re

import numpy as np

from PyUltraLight2.Init.Config import Config
from PyUltraLight2.Universe.Universe import ULDMUniverse
from PyUltraLight2.Utils.IO import IOName


class Run:
    """Lazy accessor for one simulation's output directory."""

    def __init__(self, loc):
        self.loc = loc.rstrip('/')
        self.config = Config()
        self.config.FromFile(f"{self.loc}/config.uldm")

        self.resol = self.config.Space["Resolution"]
        length = self.config.Space["Box"]["BoxLength"]
        length_units = self.config.Space["Box"]["LengthUnits"]

        self.universe = ULDMUniverse(self.config.uldm["m22"])
        self.lengthC = self.universe.convert(length, length_units, 'l')

        self.particles = self.config.BlackHole["MatterParticles"]["Condition"]
        m_mass_unit = self.config.BlackHole["MatterParticles"]["MassUnits"]
        self.n_particles = len(self.particles)
        self.initial_masses = np.array([
            self.universe.convert(p[0], m_mass_unit, 'm') for p in self.particles
        ])

        self.sink = self.config.BlackHole.get("Sink", {})
        self.sink_flag = self.sink.get("Flag", False)
        self.sink_idx = self.sink.get("ParticleIdx", 0)

    def save_numbers(self, flag):
        """Sorted list of available snapshot indices for a given save flag
        (e.g. '2Density', 'NBody'), based on what's actually on disk."""
        dirname = IOName(flag)
        pattern = f"{self.loc}/Outputs/{flag}/{dirname}_#*.npy"
        nums = []
        for f in glob.glob(pattern):
            m = re.search(r'#(\d+)\.npy$', f)
            if m:
                nums.append(int(m.group(1)))
        return sorted(nums)

    def has(self, flag):
        return len(self.save_numbers(flag)) > 0

    def load(self, flag, n):
        dirname = IOName(flag)
        return np.load(f"{self.loc}/Outputs/{flag}/{dirname}_#{n:03d}.npy")

    def load_series(self, flag):
        """All available snapshots for a flag, stacked along axis 0, plus
        the matching list of save indices."""
        nums = self.save_numbers(flag)
        return nums, np.array([self.load(flag, n) for n in nums])

    def load_scalar(self, name):
        """Load one of the plain Outputs/*.npy time series (ULDMass,
        BHMass, egylist, ...). Returns None if it wasn't saved."""
        path = f"{self.loc}/Outputs/{name}.npy"
        if not os.path.exists(path):
            return None
        return np.load(path, allow_pickle=True)

    def particle_positions(self, TMState):
        """TMState (flat, len 6*n_particles) -> (n_particles, 3) positions."""
        TMState = np.asarray(TMState)
        return TMState.reshape(-1, 6)[:, 0:3]

    def particle_velocities(self, TMState):
        TMState = np.asarray(TMState)
        return TMState.reshape(-1, 6)[:, 3:6]

    def mass_history(self):
        """(n_saves, n_particles) code-unit mass of every particle at every
        NBody save point. Non-sink particles are held at their initial mass
        (MassChange is not accounted for here); the sink particle's mass
        comes from BHMass.npy when available."""
        nums, _ = self.load_series('NBody')
        n_saves = len(nums)
        masses = np.tile(self.initial_masses, (n_saves, 1))

        if self.sink_flag:
            bhmass = self.load_scalar('BHMass')
            if bhmass is not None:
                n = min(len(bhmass), n_saves)
                masses[:n, self.sink_idx] = bhmass[:n]

        return masses
