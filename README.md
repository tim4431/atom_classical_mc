# atom_classical_mc

Classical Monte Carlo tools for estimating Rb87 heating and loss during
time-dependent optical tweezer transfer.

The core library lives in `src/atom_classical_mc`. See `doc/plan.md` for the
model assumptions and public API.

Run the concrete SLM-to-AOD transfer example with:

```bash
python3 example/slm_to_aod_transfer.py
```

Install the optional plotting dependency and use `--plot` or `--save-plot` for
trajectory, heating, survival, and ramp-sequence visualization.
