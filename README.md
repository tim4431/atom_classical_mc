# atom_classical_mc

Classical Monte Carlo tools for estimating Rb87 heating and loss during
time-dependent optical tweezer transfer.

![mc atom trajectories](demo/slm_to_aod_transfer_3d.png)

The core library lives in `src/atom_classical_mc`. See `doc/plan.md` for the
model assumptions and public API.

Run the concrete SLM-to-AOD transfer example with:

```bash
python3 example/slm_to_aod_transfer.py
```

Install the optional plotting dependency and use `--plot` or `--save-plot` for
2D diagnostics. The example writes separate suffixed figures: `_traj` for
trajectory/ramp geometry and `_energy` for heating/loss/motional occupations.
Use `--plot-3d` or `--save-3d-plot` for the standalone 3D atom trajectory and
AOD center path view.

The example also prints harmonic radial/axial trap frequencies and motional
occupation estimates for atoms decomposed in the initial SLM trap and final AOD
trap basis.

By default, the simulator rejects and resamples atoms that are already unbound
in the initial trap, so reported loss is conditioned on successful initial
loading.
