# PyUltraLight2 v2.40.2 — Refactor Audit

Audit of the attempted modularization (mid-2025, manual + GPT-4 assist). Scope: verify that the new `Evolve.py` + split submodules actually work standalone and identify what needs fixing before the BH accretion module is layered on top.

## Headline

**The refactor did not complete.** `Evolve.py` does not import cleanly, and even if the imports are patched, it calls ~10 functions that live only in the legacy `Current.py` monolith and were never moved or re-exported. The package will not run as-is. `Current.py:evolve()` is still the only working entry point.

That said, the refactor **structure** is mostly sound — Config, Solitons, isolated_potential, NBodyAdvance, Resample, Universe, IO are all well-scoped and working. The work that remains is finishing the extraction: a day or two of mechanical migration, plus a handful of real bugs.

---

## Category 1: Will not import / will crash at runtime

These are hard blockers. Fix these first.

### 1.1 Circular import: `Init/Config.py` ↔ `Utils/IO.py`

Config.py line 3 does `from PyUltraLight2.Utils.IO import SaveOptionsDigest, SaveOptionsCompile`.
IO.py line 414 (module-scope, not inside a function) does `from PyUltraLight2.Init.Config import Config`.

`import PyUltraLight2` dies with `ImportError: cannot import name 'Config' from partially initialized module`.

**Fix:** move the `from ... import Config` inside `IO.py` into the `Data.__init__` method where it's actually used. Likewise the `from ...Universe.Universe import ULDMUniverse` on the line above — both are only needed by `Data`, never at module scope.

### 1.2 `Evolve.py` uses unimported symbols

The following are referenced in `Evolve.py` but defined only in `Current.py` and never imported:

| Symbol | Used at | Source |
|---|---|---|
| `QuickInterpolate` | lines 730, 762, 1059 | `Current.py:97` |
| `GridShift` | line 736 | `Current.py:3138` |
| `QuadrupoleSecond` | lines 857, 1083 | `Current.py:868` |
| `ULDump` | lines 834, 1195 | `Current.py:837` (duplicate at `Utils/IO.py:257` — actually importable) |
| `pEval` | lines 599, 938 | `Current.py:177` |
| `Lp_jit` (alias of `LpEval`) | lines 611, 951 | `Current.py:829` |
| `CreateStream`, `NBDensity` | line 482, elsewhere | `Utils/IO.py` (present, just not imported) |
| `clear_output` | line 883 | IPython — not imported |
| `initsoliton_jit` | line 522 (commented, but see 1.4) | `Current.py:823` |

**Fix:** move these into appropriately-named submodules and import them explicitly. Proposed mapping:

- `QuickInterpolate`, `GridShift` → `Interpolation/Lin3D.py` (trilinear, grid math)
- `QuadrupoleSecond` → `DerivedQuantities/Quadrupole.py` (which currently contains a stale copy of `calculate_energies` — see 1.3)
- `pEval`, `LpEval`/`Lp_jit`, `LUQuick`/`L_jit` → `DerivedQuantities/Momentum.py` (ditto)
- `ULDump`, `ULRead`, `CreateStream`, `NBDensity` are already in `Utils/IO.py` — just add them to the star-import there, or explicit-import in Evolve.py
- `clear_output` → remove; or wrap in `try: from IPython.display import clear_output; except ImportError: def clear_output(): pass`

### 1.3 `Momentum.py` and `Quadrupole.py` contain copy-paste of `Energy.py`

Both files literally contain `def calculate_energies(...)` with the same body as `Energy.py`. They were stubs that someone (or GPT-4) filled with the wrong template and never revisited. The files must be actually filled with the momentum/quadrupole code from `Current.py` (see mapping above).

### 1.4 Undefined name `a` for Plummer smoothing

`Evolve.py` renamed the variable `a` → `smoothing` for the N-body smoothing scale, but four sites still reference the old name:

```
line 755:  if a == 0:             # → if smoothing == 0:
line 779:  if a == 0:             # → if smoothing == 0:
line 782:  phiRef = - a * ...     # → phiRef = - smoothing * ...
line 1058: if a == 0:             # → if smoothing == 0:
```

Every one of these will `NameError` the first time the relevant branch is hit.

### 1.5 `Use_EP` may be unbound

```python
if ExtPhi != 0:
    Use_EP = True
# ... then later in the loop:
if Use_EP:  # NameError if ExtPhi was 0 at entry
```

Combined with the fact that `ExtPhi != 0` raises `ValueError: truth value of an array is ambiguous` whenever `ExtPhi` is an `ndarray`, this check is broken three ways.

**Fix:** initialize `Use_EP = False` unconditionally, then set to True from the result of `resolve_ext_phi` (which is the function that actually knows whether a real external potential was provided).

### 1.6 `Init/Solitons.py`: typo `functEm`

Line 49 returns `functEm` (should be `funct`). `initsolitonRadial` is dead code right now — nothing imports it — but the file won't compile into a `.pyc` if anything tries to import `*` from here, because this is inside a function def so it's a runtime error only. Still, fix it.

### 1.7 `Particles/Smoothing.py`: syntax error

```python
def GenPlummer(rP,length_units, silent = True, resol = 0,length = 0, convert, convert_back):
```

Non-default argument after default — `SyntaxError`. The file also has a stray `)` at the end of `GenPlummerAuto`. Not imported anywhere, but it shouldn't sit in the repo like this.

---

## Category 2: Real bugs (not import-blockers, but will misbehave)

### 2.1 The ExtPhi time-dependent branch is not actually wired up

`resolve_ext_phi` correctly distinguishes `f(x,y,z)` from `f(x,y,z,t)`, and returns `(None, True, phi_fn)` for the 4-arg case. But after that:

```python
if is_time_dep:
    ext_phi = phi_fn(xarray, yarray, zarray, 0)  # Not Implemented
elif phi_static is not None:
    ExtPhi = phi_static
else:
    ExtPhi = 0.0
```

Two problems:

1. The time-dependent branch writes to `ext_phi` (lowercase, new name), while the rest of the code reads from `ExtPhi`. So even the `t=0` value is lost.
2. The loop never re-evaluates `phi_fn` at the current `t`. `EPIm = np.imag(ExtPhi)` is computed once before the loop at line 770 and is never updated.

This is a critical piece of design for a moving/time-dependent sink. Recommend redesigning the interface entirely — see **Proposal** below.

### 2.2 `Interpolation/Lin3D.py`: broken recursion

```python
def InterpolateLocal(RRem, Input):
    while len(RRem) > 1:
        Input = Input[1,:]*RRem[0] + Input[0,:]*(1-RRem[0])
        RRem = RRem[1:]
        InterpolateLocal(RRem, Input)   # recursive call, return value DISCARDED
    else:
        return Input[1]*RRem + Input[0]*(1-RRem)
```

This returns the right answer only by accident — the `while` loop does all the real work, the recursion is pointless, and `while/else` fires on `len(RRem)==1`. The code is hard to reason about and a future refactor could easily break it.

**Fix:** drop the recursion entirely.

```python
def InterpolateLocal(RRem, Input):
    for r in RRem[:-1]:
        Input = Input[1,...]*r + Input[0,...]*(1-r)
    return Input[1]*RRem[-1] + Input[0]*(1-RRem[-1])
```

### 2.3 `Evolve.py:488`: concatenation before normalization

```python
psi = ne.evaluate("psi + funct")   # at line 488, before soliton loop
```

This looks fine, but note that when `Uniform=True`, the `funct` variable still holds the initial wavefunction from `UseInit` (loaded with `psiEx = ULRead(InitPath)` then `funct = fft_psi(psi)` at the bottom). `funct` is a working buffer and its contents between the two sites are not well-defined. In `Current.py:evolve` this is managed more carefully. Worth a clean pass when touching that section.

### 2.4 Output folder protection logic

```python
try:
    os.mkdir(str(loc + '/Outputs'))
except(FileExistsError):
    if Silent:
        Protect = 'Y'
    else:
        printU(...)
        Protect = str(input())
    if Protect == 'n':
        return loc
    elif Protect == 'Y':
        shutil.rmtree(str(loc + '/Outputs'))
        os.mkdir(...)
    else:
        return loc
```

`shutil` is not imported in `Evolve.py` (only `import shutil` inside the `elif` block — that actually works in Python, but it imports in a subscope). More importantly, the default `Silent=False` branch blocks on `input()` — this will hang any batch job that starts from a directory where Outputs already exists. Consider a `force=True` kwarg path for scripted runs.

---

## Category 3: Style / structural

None of these are bugs, but they'll matter when you publish the code.

### 3.1 `Current.py` should die

106 kB, 3407 lines, contains three copies of `evolve`-like functions, embedded soliton/BH solver code, and lots of dead code. Once Category 1 is complete, `Current.py` should be deleted from the repository and the `__pycache__` entries cleared. Keep the legacy version on a tag (`v2.39-final` or similar) for reference; do not ship it with v2.40+.

### 3.2 `Credits.py` references `from .Version import *`

This works but blurs the import graph. A tiny module; import `from PyUltraLight2.Version import Version, S_version, D_version` explicitly.

### 3.3 `Evolve.py` is still a 1209-line god function

Even once all imports close, `Evolve()` itself is too long. The natural cut points are:

1. Config resolution & directory setup → `Init/Setup.py`
2. Grid construction → `Init/Grid.py`
3. Wavefunction initialization (soliton + uniform + UseInit + perturbation) → extended `Init/Solitons.py` or new `Init/Wavefunction.py`
4. N-body state setup → `Init/NBodyState.py`
5. The main time loop → keep in `Evolve.py` but make it much leaner
6. Periodic save block (lines 1075–1200) → `Utils/SaveLoop.py`

You don't have to do all of these to ship v2.40.2. But don't add the BH sink on top of the current mess.

### 3.4 `Universe.Universe.ULDMUniverse.convert`

The huge `if/elif` chain is correct but hard to maintain. A dictionary lookup `UNIT_TABLE = {('l', 'pc'): parsec, ...}` would remove 150 lines and make unit errors impossible. Not urgent.

---

## Proposal: redesigning the external-potential hook for the BH sink

The current `ExtPhi` interface — "a thing that is an array, path, or callable of `(x,y,z)` or `(x,y,z,t)`" — has two problems for what you actually need:

1. **It's trying to serve too many use cases.** Fixed-in-space external gravitational potentials, time-varying fields, and moving sinks attached to particles are three different physics cases with different update requirements and different accuracy needs. They should not share a single config key.

2. **The sink must follow the BH**, and the BH position is in `TMState`, which evolves *inside* the step. `f(x,y,z,t)` is the wrong signature; we need `f(x, y, z, TMState, masslist)` — or better, a dedicated object with state.

I suggest replacing `ADVANCED.ExtPhi` with a structured `ADVANCED.External` dict that explicitly distinguishes cases:

```python
self.ADVANCED = {
    "CenterCalc": False,
    "ComputeQuad": False,
    "External": {
        # gravitational potential that sits on the grid, purely real, for e.g. a
        # fixed tidal field or a background halo. Evaluated once at t=0 or on a
        # user-provided schedule.
        "FixedPhi": None,          # ndarray | path | callable(x,y,z) | None

        # imaginary / non-Hermitian potential — the dark→black sink.
        # Evaluated EACH step at the current BH position.
        "Sinks": [
            # one dict per BH that absorbs ULDM
            # {"ParticleIdx": 0,
            #  "Profile": "gaussian" | "plummer" | callable(r, **params),
            #  "Amplitude": V0,        # code units
            #  "Radius": r_sink,       # code units
            #  "Feedback": True,       # add absorbed mass to masslist[i]
            # }
        ],
    }
}
```

The key design points:

- **Separate static external Φ from the sink.** They have different update frequencies and different physics.
- **Sinks are per-particle and know their particle index.** `External.Sinks[i].ParticleIdx = 0` means "this sink is glued to BH #0 and follows it".
- **Profiles are parametric, not arbitrary callables, by default.** A Gaussian `V(r) = V₀·exp(-r²/2σ²)` covers 95% of physics cases and can be evaluated on a sub-cube around the BH at negligible cost. `callable(r, **params)` is the escape hatch for exotic profiles.
- **Absorbed mass feeds back into `masslist` automatically** (with a flag to turn off, for diagnostic runs where you want to see the soliton disappear without the BH changing mass).

I'll sketch the implementation in the next message. Before that, please confirm:

1. Do you have any existing simulations / notebooks that use the current `ExtPhi` interface? If so we keep it working as a deprecated path. If not, we can replace cleanly.
2. For the sink profile, do you want Gaussian, Plummer-like (`1/(r² + r_s²)`), or hard cutoff (`V₀ for r < r_s, 0 else`) as the first implementation? The Gaussian is easiest to reason about theoretically (smooth, analytic flux integral) but the sharp cutoff is closest to a physical absorbing boundary.
3. What's your working convention for the feedback: does `M_BH` increase by exactly `ΔM_ULDM` (bare rest mass accretion), or do you want the option to channel some fraction into the particle KE (radiation-like loss mechanism)?

---

## Summary — what to do, in order

1. **Resolve the circular import** (20 min). Move IO.py's imports of Config/ULDMUniverse into the Data class.
2. **Fill in `Momentum.py` and `Quadrupole.py`** with the real code from `Current.py` (2 h).
3. **Move `QuickInterpolate`, `GridShift` → `Interpolation/Lin3D.py`** (1 h).
4. **Fix `a` → `smoothing` everywhere in `Evolve.py`** (5 min, grep-replace).
5. **Fix `Use_EP` and the ExtPhi detection** (30 min).
6. **Fix `Lin3D.InterpolateLocal` recursion** (5 min).
7. **Fix `Solitons.py` `functEm` typo** (5 min).
8. **Delete or fix `Particles/Smoothing.py`** (5 min).
9. **Write a smoke-test script** that runs one short no-BH, no-sink evolution and confirms mass + energy conservation to expected precision (1 h). This is the gate before any BH accretion work.
10. **Then** start the BH accretion module with the redesigned External hook.

After step 9, we have a working Evolve that matches `Current.py:evolve()` up to minor numerical differences from the new structure. That's the right place to be before adding physics.
