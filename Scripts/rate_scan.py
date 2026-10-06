#!/usr/bin/env python3
"""Run one point of the stone-skipping accretion-rate scan.

One call = one run, so it maps directly onto a SLURM array task.

Usage (from the repo root):
    python Scripts/rate_scan.py --res 128 --f 1e-3
    python Scripts/rate_scan.py --res 128 --f none     # fixed mass, sink off

f multiplies the Bondi-Hoyle-calibrated sink amplitude (Sink.RateScale):
absorption rate = f * Mdot_BHL. Prints the run directory on completion.
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

    loc = Evolve(c, Silent=True)
    print(f"DONE res={args.res} f={args.f} -> {loc}", flush=True)


if __name__ == "__main__":
    main()
