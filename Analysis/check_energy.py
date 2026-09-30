#!/usr/bin/env python3
"""Energy checker.

Plots the field-energy components (Outputs/egpcmlist.npy,
egpsilist.npy, ekandqlist.npy -> egylist.npy = their sum), reconstructs
particle kinetic energy from Outputs/NBody/ + the mass history (constant per
particle, or the sink's growing mass from BHMass.npy), and plots system total
energy (field + particle KE) with its fractional drift from the initial value
as a conservation check.

Momentum is NOT saved at runtime by this codebase (no Outputs/*momentum*
file is ever written), so it is intentionally skipped here - there's nothing
to load.

Usage:
    python Analysis/check_energy.py <run_dir> [--out FILE.png] [--show]
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Analysis.common import Run


def particle_kinetic_energy(run):
    """0.5 * sum_i m_i(t) * v_i(t)^2 at each NBody save point, in code units."""
    nums, tm = run.load_series('NBody')
    if len(nums) == 0:
        return None, None

    masses = run.mass_history()  # (n_saves, n_particles)
    n = min(len(masses), tm.shape[0])

    ke = np.zeros(n)
    for i in range(n):
        vel = run.particle_velocities(tm[i])
        speed2 = np.sum(vel**2, axis=1)
        ke[i] = 0.5 * np.sum(masses[i] * speed2)

    return np.array(nums[:n]), ke


def main(loc, out=None, show=False):
    run = Run(loc)

    egylist = run.load_scalar('egylist')
    egpcmlist = run.load_scalar('egpcmlist')
    egpsilist = run.load_scalar('egpsilist')
    ekandqlist = run.load_scalar('ekandqlist')

    if egylist is None:
        print("No Outputs/egylist.npy found ('Energy' save flag not set) - nothing to check.")
        return

    steps = np.arange(len(egylist))

    pk_steps, pke = particle_kinetic_energy(run)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=False)

    ax = axes[0]
    if egpcmlist is not None:
        ax.plot(steps, egpcmlist, label='Particle-field ($\\Phi_{TM}$)')
    if egpsilist is not None:
        ax.plot(steps, egpsilist, label='Self-gravity/SI ($\\Phi_{SP}$)')
    if ekandqlist is not None:
        ax.plot(steps, ekandqlist, label='Kinetic+quantum')
    ax.plot(steps, egylist, 'k-', lw=2, label='Total field energy')
    ax.set_ylabel('Energy (code units)')
    ax.set_xlabel('Save index')
    ax.legend(fontsize=8)
    ax.set_title(f'Field energy components: {os.path.basename(run.loc)}')

    ax2 = axes[1]
    if pke is not None:
        n = min(len(egylist), len(pke))
        total = egylist[:n] + pke[:n]
        drift = (total - total[0]) / (np.abs(total[0]) + 1e-30)

        ax2.plot(steps[:n], pke[:n], label='Particle kinetic energy')
        ax2.plot(steps[:n], egylist[:n], label='Field energy')
        ax2.plot(steps[:n], total, 'k-', lw=2, label='System total')
        ax2b = ax2.twinx()
        ax2b.plot(steps[:n], drift * 100, 'r--', lw=1, label='Fractional drift (%)')
        ax2b.set_ylabel('Drift from $E(0)$ (%)', color='r')
        ax2b.tick_params(axis='y', colors='r')

        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='best')

        print(f"Total energy drift over run: {drift[-1]*100:.4f}%")
        if run.sink_flag:
            print("NOTE: Sink.Flag is on for this run - the absorbing potential "
                  "(psi *= exp(-h*V_sink)) is explicitly non-Hamiltonian: mass "
                  "(and the energy it carries) leaves the tracked budget entirely "
                  "when absorbed. A large drift here reflects that removal, not a "
                  "conservation bug. This check is only a strict pass/fail test "
                  "for Sink.Flag=False runs; for sink runs, read it as 'how much "
                  "energy left with the accreted mass', not an error signal.")
    else:
        print("No Outputs/NBody/ snapshots found - skipping particle KE / total energy.")

    ax2.set_ylabel('Energy (code units)')
    ax2.set_xlabel('Save index')
    title2 = 'System total energy (field + particles)'
    if run.sink_flag:
        title2 += ' -- Sink ON: drift expected (energy leaves with absorbed mass)'
    ax2.set_title(title2, fontsize=9)

    plt.tight_layout()

    out = out or os.path.join(run.loc, 'energy_check.png')
    plt.savefig(out, dpi=120)
    print(f"Saved: {out}")

    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', help='Simulation output directory (as returned by Evolve())')
    parser.add_argument('--out', default=None, help='Output PNG path (default: <run_dir>/energy_check.png)')
    parser.add_argument('--show', action='store_true', help='Also display the plot interactively')
    args = parser.parse_args()
    main(args.run_dir, args.out, args.show)
