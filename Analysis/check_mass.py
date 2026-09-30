#!/usr/bin/env python3
"""Mass-conservation checker.

Plots total ULDM mass on the grid (Outputs/ULDMass.npy) and, if the sink was
enabled, the sink particle's mass (Outputs/BHMass.npy) over the run, plus the
residual (ULDM lost) - (BH mass gained), which should sit at ~0 (down to
grid/edge-loss effects) if mass bookkeeping is self-consistent.

Usage:
    python Analysis/check_mass.py <run_dir> [--out FILE.png] [--show]
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


def main(loc, out=None, show=False):
    run = Run(loc)

    uldmass = run.load_scalar('ULDMass')
    bhmass = run.load_scalar('BHMass') if run.sink_flag else None

    if uldmass is None:
        print("No Outputs/ULDMass.npy found - nothing to check.")
        return

    steps = np.arange(len(uldmass))

    n_panels = 2 if bhmass is not None else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 4 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    ax = axes[0]
    ax.plot(steps, uldmass, '-o', ms=3, color='tab:blue', label='ULDM mass on grid')
    if bhmass is not None:
        ax.plot(steps[:len(bhmass)], bhmass, '-o', ms=3, color='tab:orange',
                 label=f'Sink particle mass (idx {run.sink_idx})')
    ax.set_ylabel('Mass (code units)')
    ax.legend()
    ax.set_title(f'Mass tracking: {os.path.basename(run.loc)}')

    if bhmass is not None:
        n = min(len(uldmass), len(bhmass))
        uldm_lost = uldmass[0] - uldmass[:n]
        bh_gained = bhmass[:n] - bhmass[0]
        residual = bh_gained - uldm_lost

        ax2 = axes[1]
        ax2.plot(steps[:n], uldm_lost, '-', color='tab:blue', label='ULDM mass lost')
        ax2.plot(steps[:n], bh_gained, '--', color='tab:orange', label='Sink mass gained')
        ax2.plot(steps[:n], residual, '-', color='tab:red', label='Residual (gained - lost)')
        ax2.axhline(0, color='k', lw=0.5)
        ax2.set_ylabel('Mass (code units)')
        ax2.set_xlabel('Save index')
        ax2.legend()

        max_res = np.max(np.abs(residual))
        max_scale = max(np.max(np.abs(uldm_lost)), 1e-30)
        print(f"Max |residual| = {max_res:.3e} code units "
              f"({max_res/max_scale*100:.4f}% of max ULDM mass lost)")
    else:
        axes[0].set_xlabel('Save index')

    plt.tight_layout()

    out = out or os.path.join(run.loc, 'mass_check.png')
    plt.savefig(out, dpi=120)
    print(f"Saved: {out}")

    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', help='Simulation output directory (as returned by Evolve())')
    parser.add_argument('--out', default=None, help='Output PNG path (default: <run_dir>/mass_check.png)')
    parser.add_argument('--show', action='store_true', help='Also display the plot interactively')
    args = parser.parse_args()
    main(args.run_dir, args.out, args.show)
