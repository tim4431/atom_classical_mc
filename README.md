# atom_classical_mc

Classical Monte Carlo tools for estimating Rb87 heating and loss during
time-dependent optical tweezer transfer.

![mc atom trajectories](demo/slm_to_aod_transfer_3d.png)

![transfer animation](demo/slm_to_aod_transfer.gif)

The core library lives directly in `src/`. See `doc/plan.md` for the model
assumptions and public API.

Run the concrete SLM-to-AOD transfer example with:

```bash
python3 example/slm_to_aod_transfer.py
```

The example writes separate suffixed figures: `_traj` for trajectory/ramp
geometry, `_energy` for heating/loss/motional occupation distributions, and
`_3d` for the atom trajectories with the AOD center path.

The example also prints harmonic radial/axial trap frequencies and motional
occupation estimates for atoms decomposed in the initial SLM trap and final AOD
trap basis.

By default, the simulator rejects and resamples atoms that are already unbound
in the initial trap, so reported loss is conditioned on successful initial
loading.

Compare AOD position ramp profiles with:

```bash
python3 example/position_ramp_compare.py
```

The default comparison includes linear, cubic smoothstep, quintic minimum-jerk,
and sinusoidal ramps with the same trap depths, timing, and random seed. It uses
a moderately shallow AOD depth so the default plot shows nonzero transfer
errors instead of a saturated all-survive case. It
plots ramp shape, `1 - p` transfer error, and temperature gain. Edit the
constants at the top of the script to change the compared profiles or output path.
