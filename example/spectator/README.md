# Spectator-qubit fly-by studies

Studies of a static SLM trap holding a single Rb87 "spectator" atom while a
moving tweezer flies past at a fixed transverse distance `d`. We measure how
often the spectator gets dragged out of the SLM, captured by the moving trap,
or simply heated.

Two flavours of moving tweezer live here:

| Script                            | Moving trap         | Output dir   |
| --------------------------------- | ------------------- | ------------ |
| `spectator_qubits.py`             | analytical AOD Gaussian | `render/` |
| `spectator_qubits_ripa.py`        | gridded RIPA (tricubic, loaded from npz) | `render/` |
| `spectator_qubits_dynamics.py`    | renders representative AOD fly-by GIFs | `render/` |
| `spectator_qubits_speed_scan.py`  | fixed AOD cell, scans speed only | `render/` |

The two main scripts share their setup almost verbatim — same SLM, same
ensemble, same fly-by geometry, same `(depth × distance)` sweep, same speed
comb averaged into mean ± σ bands. The only difference is what the moving
tweezer *is*.

## Geometry

```
            spectator SLM (static, at origin)
                  ●
                  |   ← y
                  |
     ←-------- moving tweezer center -------→   (x)
                  ↑ at fixed y = d, z = 0
```

The moving tweezer translates linearly along x from `−FLYBY_HALF_LENGTH_UM`
to `+FLYBY_HALF_LENGTH_UM` over `DURATION_S`, at fixed transverse offset
`d` (the `distance` swept) and depth (the `depth` swept). Speed averaging
(`SPEED_FACTORS`, log-spaced around 1) scales the duration without changing
the path length, smearing out the resonance where the fly-by interaction
time hits an integer multiple of the SLM trap period.

## How the two flavours differ

**AOD ([spectator_qubits.py](spectator_qubits.py))** — analytical
`GaussianTrap` template + `RampSequence` → `MovingGaussianTrap`. Depth is
ramp-driven; the trap shape is the standard cylindrically-symmetric
Gaussian and is identical at every distance. `AOD_WAIST_RADIAL_UM = 1.0`,
`AOD_WAIST_AXIAL_UM = 5.0`.

**RIPA ([spectator_qubits_ripa.py](spectator_qubits_ripa.py))** —
`GriddedTrap` loaded from `example/spectator/ripa/ripa_gridded_trap.npz` (produced
by [ripa/generate_ripa_gridded_trap.py](ripa/generate_ripa_gridded_trap.py)).
The npz stores a normalized intensity grid; per-depth rescaling computes
`U = −depth × I/I_peak` and feeds a fresh `GriddedTrap` (tricubic
interpolation, 4×4×4 stencil). The ramp is reused for the spatial path;
its `depths_uK` column is ignored because `GriddedTrap` bakes the depth
into the stored grid at construction. `RIPA_WAIST_UM = 1.18` is a label
constant only — the trap shape is whatever the npz contains.

The RIPA script keeps the *normalized d/w* sample points (0.6, 0.9, 1.2,
1.5, 1.8, 2.1) identical to the AOD script's, so plot abscissas align;
physical offsets are scaled by `RIPA_WAIST_UM / AOD_WAIST_RADIAL_UM`.

## Outputs

Both main scripts cache the sweep to an `.npz` and write `.png` and `.pdf`
figures into the shared `render/` subdir (filenames are flavour-prefixed, so
the AOD and RIPA outputs never collide):

```
render/
  spectator_qubits_sweep.npz          (cached AOD sweep; delete to force re-run)
  spectator_qubits_drag_lines.{png,pdf}
  spectator_qubits_heating_lines.{png,pdf}
  spectator_qubits_dynamics_*.gif     (from spectator_qubits_dynamics.py)
  spectator_qubits_speed_scan.png     (from spectator_qubits_speed_scan.py)
  spectator_qubits_ripa_sweep.npz     (cached RIPA sweep)
  spectator_qubits_ripa_drag_lines.{png,pdf}
  spectator_qubits_ripa_heating_lines.{png,pdf}
  spectator_qubits_ripa_capture_lines.{png,pdf}
  spectator_qubits_ripa_flyby_*.gif   (when GENERATE_ANIMATION=True)
```

## Running

From the repository root:

```bash
python3 example/spectator/spectator_qubits.py            # AOD sweep
python3 example/spectator/spectator_qubits_ripa.py       # RIPA sweep (needs the npz first)
python3 example/spectator/spectator_qubits_dynamics.py   # AOD representative GIFs
python3 example/spectator/spectator_qubits_speed_scan.py # AOD speed scan
```

The RIPA script depends on `example/spectator/ripa/ripa_gridded_trap.npz`. If it's
missing, generate it once with:

```bash
python3 example/spectator/ripa/generate_ripa_gridded_trap.py
```

## Shared knobs

Both flavours read the same physical parameters from their respective
top-of-file constants (`SLM_*`, `INITIAL_TEMPERATURE_UK`, `ENSEMBLE_SIZE`,
`TIMESTEP_S`, `DURATION_S`, `LOSS_RADIUS_UM`, `RANDOM_SEED`,
`FLYBY_HALF_LENGTH_UM`, `SPEED_FACTORS`, `DEPTHS_UK`, …). They are not
hot-imported across files, so edit them in the script you care about.

`spectator_qubits_dynamics.py` and `spectator_qubits_speed_scan.py` *do*
import constants and helpers from `spectator_qubits.py` (the AOD script);
they are AOD-flavoured by design.
