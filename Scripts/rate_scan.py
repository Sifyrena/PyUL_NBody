#!/usr/bin/env python3
"""Run one point of the stone-skipping accretion-rate scan.

One call = one run, so it maps directly onto a SLURM array task.

Usage (from the repo root):
    python Scripts/rate_scan.py --res 128 --f 1e-3
    python Scripts/rate_scan.py --res 128 --f none     # fixed mass, sink off
    python Scripts/rate_scan.py --res 128 --model Unruh --f 1   # wave-regime rate, f=1 on top

f multiplies the sink amplitude (Sink.RateScale) on top of --model:
BHL -> absorption rate = f * Mdot_BHL; Unruh -> f * Mdot_Unruh. Prints the run directory on completion.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PyUltraLight2.Init.Config import Config
from PyUltraLight2.Evolve import Evolve


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--res', type=int, default=128, help='Grid resolution N (N^3)')
    p.add_argument('--f', default='1.0', help="RateScale multiplier, or 'none' to switch the sink off")
    p.add_argument('--model', default='BHL', choices=['BHL', 'Unruh'], help='Sink.Model')
    p.add_argument('--out', default=None, help='Base directory to write runs under (Saving.Loc), e.g. a project SSD path')
    p.add_argument('--name', default=None, help="Run folder name (Saving.Name); default '{auto}_<model>_f<f>' so array tasks never collide")
    p.add_argument('--config', default='StoneSkipping_Baseline_Accretion.uldm')
    args = p.parse_args()

    c = Config()
    c.FromFile(args.config)
    c.Space['Resolution'] = args.res
    if args.f.lower() == 'none':
        c.BlackHole['Sink']['Flag'] = False
    else:
        c.BlackHole['Sink']['Flag'] = True
        c.BlackHole['Sink']['RateScale'] = float(args.f)
        c.BlackHole['Sink']['Model'] = args.model

    if args.out:
        c.Saving['Loc'] = args.out
    sink_tag = 'nosink' if args.f.lower() == 'none' else f"{args.model}_f{args.f}"
    c.Saving['Name'] = args.name or f"{{auto}}_{sink_tag}"

    loc = Evolve(c, Silent=True)
    print(f"DONE res={args.res} f={args.f} -> {loc}", flush=True)


if __name__ == "__main__":
    main()
