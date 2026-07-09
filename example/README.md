# Examples

Runnable demonstrations of the simulator. Run every script from the repository
root (each one bootstraps `sys.path` to the repo root before importing `src`):

```bash
python3 example/aod/slm_to_aod_transfer.py
python3 example/mot/rb85_mot.py --save-plot
```

## Layout

| Path | What it covers |
| --- | --- |
| `aod/` | SLM→moving-AOD tweezer transfer (conservative pipeline) and its GIF animation |
| `mot/` | Near-resonant light-force MOTs (Rb85 MOT, tri-sector grating MOT) |
| `spectator/` | Spectator-qubit fly-by suite (AOD + gridded RIPA); see `spectator/README.md` |
| `astigmatism_check.py` | `AstigmaticAODTrap` sanity check |
| `gridded_vs_analytical.py` | `GriddedTrap` vs analytical `GaussianTrap` |
| `tweezer_probe_heating.py` | Recoil heating of a tweezer atom (mixed conservative + light) |

## Rendering convention

**Every example writes its figures, GIFs, and cached `.npz` sweeps into a
`render/` folder next to the script** (`example/aod/render/`,
`example/mot/render/`, `example/spectator/render/`, and `example/render/` for
the top-level scripts). `render/` is git-ignored — these outputs are
regenerable, so they are never committed.

When adding a new example, follow the same pattern:

```python
HERE = os.path.dirname(__file__)
RENDER_DIR = os.path.join(HERE, "render")
...
os.makedirs(RENDER_DIR, exist_ok=True)
fig.savefig(os.path.join(RENDER_DIR, "my_figure.png"), dpi=180)
```

The one exception is *input* data an example loads at runtime (e.g.
`spectator/ripa/ripa_gridded_trap.npz`, consumed by the RIPA fly-by): that
stays next to its consumer, not in `render/`.

Curated showcase images embedded in the top-level `README.md` live in the
repo-root `demo/` folder and are committed by hand; they are not produced by
the `render/` flow.
