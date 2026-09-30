# Analysis

Standalone post-processing scripts for `Evolve()` output directories. All
three take the run directory (the string `Evolve()` returns / prints as
`DONE: ...`) as their first argument and are safe to run from anywhere
(they resolve imports relative to the repo root).

```bash
python Analysis/check_mass.py   <run_dir>   # ULDM mass vs. sink particle mass, conservation residual
python Analysis/check_energy.py <run_dir>   # field energy components + particle KE + system total
python Analysis/animate_run.py  <run_dir>   # 2Density animation with particle markers scaled to mass
```

Each writes a PNG/GIF into `<run_dir>` by default (`--out` to override),
and each degrades gracefully (prints why, does nothing else) if the
relevant `Saving.Flags` weren't enabled for that run:

- `check_mass.py` needs `ULDMass.npy` (always written) and, for the
  residual panel, `BHMass.npy` (only written when `Sink.Flag=True`).
- `check_energy.py` needs `Energy` in `Saving.Flags`. If `NBody` is also
  present it reconstructs particle kinetic energy (using each particle's
  initial mass, or the sink particle's growing mass from `BHMass.npy`) and
  plots system total energy. **If `Sink.Flag=True`, a large drift is
  expected and printed as such** — the absorbing potential is
  non-Hamiltonian, so energy leaves with the absorbed mass and isn't
  tracked in this budget. Treat the check as strict conservation only for
  `Sink.Flag=False` runs.
- `animate_run.py` needs `2Density`; `NBody` is optional (adds the
  particle overlay). Marker area scales linearly with each particle's
  current mass relative to the most massive particle over the run.

## What's intentionally not here

**Momentum.** Nothing in `Evolve()` saves field or particle momentum at
runtime (no `Outputs/*momentum*` file is ever written), so there is no
`check_momentum.py` — there's nothing on disk to check. Verifying momentum
conservation (as was done ad hoc for the `ConserveMomentum` sink feature)
requires full wavefunction snapshots (`3Wfn`) at every save point to
integrate `j = Im(psi* grad psi)` over the grid, which is expensive and
not part of a normal run's output.

## `common.py`

Shared `Run` class used by all three scripts — loads `config.uldm`, lists
available snapshots for a given save flag, loads a snapshot or a full
time series, and builds a `(n_saves, n_particles)` mass-history array
(constant per particle except the sink particle, which uses `BHMass.npy`).
Reuse it directly for custom analysis:

```python
from Analysis.common import Run
run = Run("Simulations/M1_L1_T1@64_20260101_000000")
nums, density = run.load_series("2Density")
masses = run.mass_history()
```
