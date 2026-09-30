#!/usr/bin/env python3
"""2Density + particle animator.

Animates the saved xy mid-plane density slices (Outputs/2Density/) with the
matter particle(s) overlaid as scatter markers, marker area scaled to each
particle's current mass (constant for ordinary particles; growing over time
for the sink particle, read from Outputs/BHMass.npy).

Output format is chosen by the --out extension: .mp4 (needs ffmpeg on PATH,
smaller file / better quality - preferred for presentations) or .gif
(portable, no external dependency).

Usage:
    python Analysis/animate_run.py <run_dir> [--out FILE.mp4] [--fps 12] [--log]
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Analysis.common import Run


def main(loc, out=None, fps=12, log=False, marker_scale=800.0):
    run = Run(loc)

    if not run.has('2Density'):
        print("No Outputs/2Density/ snapshots found ('2Density' save flag not set).")
        return
    if not run.has('NBody'):
        print("No Outputs/NBody/ snapshots found ('NBody' save flag not set) - "
              "animating density only, no particle overlay.")

    d_nums, density = run.load_series('2Density')

    if run.has('NBody'):
        n_nums, tm = run.load_series('NBody')
        masses = run.mass_history()
        n_frames = min(len(d_nums), len(n_nums))
    else:
        tm = None
        n_frames = len(d_nums)

    half = run.lengthC / 2.0
    extent = [-half, half, -half, half]

    fig, ax = plt.subplots(figsize=(6, 6))

    frame0 = density[0]
    plot0 = np.log10(frame0 + 1e-30) if log else frame0
    im = ax.imshow(plot0.T, origin='lower', extent=extent, cmap='inferno', animated=True)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('log10 density' if log else 'density')

    scat = ax.scatter([], [], c='cyan', edgecolors='white', linewidths=0.5, zorder=5)

    ax.set_xlabel('x (code units)')
    ax.set_ylabel('y (code units)')
    title = ax.set_title('')

    max_mass = run.initial_masses.max() if run.n_particles > 0 else 1.0
    if run.sink_flag:
        bhmass = run.load_scalar('BHMass')
        if bhmass is not None:
            max_mass = max(max_mass, bhmass.max())

    def sizes_for(i):
        if tm is None or run.n_particles == 0:
            return np.array([]), np.array([])
        pos = run.particle_positions(tm[i])
        m = masses[i]
        nonzero = m > 0
        s = marker_scale * (m[nonzero] / max_mass)
        return pos[nonzero, 0:2], s

    def update(i):
        plot_i = np.log10(density[i] + 1e-30) if log else density[i]
        im.set_data(plot_i.T)
        im.set_clim(vmin=plot_i.min(), vmax=plot_i.max())

        xy, s = sizes_for(i)
        if len(xy) > 0:
            scat.set_offsets(xy)
            scat.set_sizes(s)
        title.set_text(f'save #{d_nums[i]:03d}')
        return im, scat, title

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)

    out = out or os.path.join(run.loc, 'density_animation.mp4')
    if out.lower().endswith('.mp4'):
        writer = FFMpegWriter(fps=fps, bitrate=4000)
    else:
        writer = PillowWriter(fps=fps)
    anim.save(out, writer=writer)
    print(f"Saved: {out} ({n_frames} frames)")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', help='Simulation output directory (as returned by Evolve())')
    parser.add_argument('--out', default=None, help='Output path, .mp4 or .gif (default: <run_dir>/density_animation.mp4)')
    parser.add_argument('--fps', type=int, default=12)
    parser.add_argument('--log', action='store_true', help='Plot log10(density) instead of density')
    parser.add_argument('--marker-scale', type=float, default=800.0,
                         help='Max scatter marker area (points^2) for the most massive particle')
    args = parser.parse_args()
    main(args.run_dir, args.out, args.fps, args.log, args.marker_scale)
