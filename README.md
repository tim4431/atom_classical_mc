# atom_classical_mc

Classical Monte Carlo tools for estimating Rb87 heating and loss during
time-dependent optical tweezer transfer.

![mc atom trajectories](demo/slm_to_aod_transfer_3d.png)

![transfer animation](demo/slm_to_aod_transfer.gif)

The core library lives directly in `src/`. See `doc/plan.md` for the model
assumptions and public API.

Run the concrete SLM-to-AOD transfer example with:

```bash
python3 example/aod/slm_to_aod_transfer.py
```

The example writes separate suffixed figures into `example/aod/render/`:
`_traj` for trajectory/ramp geometry, `_energy` for heating/loss/motional
occupation distributions, and `_3d` for the atom trajectories with the AOD
center path. (`example/aod/transfer_animation.py` renders the same transfer
as a GIF.)

The example also prints harmonic radial/axial trap frequencies and motional
occupation estimates for atoms decomposed in the initial SLM trap and final AOD
trap basis.

By default, the simulator rejects and resamples atoms that are already unbound
in the initial trap, so reported loss is conditioned on successful initial
loading.

## Near-resonant light forces (MOT, molasses, probe beams)

`run_simulation` optionally couples an internal-state backend
(steady-state, rate-equation, or stochastic-jump populations of the
effective two-level cycling transition) with the momentum update
(per-photon recoil kicks plus the conservative traps). Laser beams,
magnetic fields, and species data (Rb85/Rb87 D2) are freely
configurable — a MOT is just one such configuration; see
`doc/light_matter_plan.md` for the model. `example/tweezer_probe_heating.py`
shows the mixed case: recoil heating of a tweezer-trapped atom.

Run the Rb85 MOT capture-and-cooling example with:

```bash
python3 example/mot/rb85_mot.py --save-plot
```

This writes the summary figure to `example/mot/render/rb85_mot_summary.png`
(add `--gif` for an animated cooling-cloud GIF alongside it).
