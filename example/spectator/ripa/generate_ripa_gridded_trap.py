"""Generate a 3D trap potential from the RIPA focal pattern and cache it as a
tricubic ``GriddedTrap``.

Pipeline:

1. Build the RIPA rays (`ripa_lib.vipa_focus.vipa_rays`, vendored from the
   VIPA simulation library) from
   ``PARAMS_10_TWZ``.
2. For each z-slice in the z range, compute the focal-plane
   intensity `|E(x,y,z)|^2` with `ripa_lib.crosssections.crosssection_xy`
   and resample (bilinear) onto `(x_axis, y_axis)`.
3. Stack the slices into a `(Nx, Ny, Nz)` array, normalise so the peak
   intensity becomes the configured trap depth (red-detuned dipole trap:
   `U = -depth * I / I_peak`), and wrap in a `GriddedTrap` with
   tricubic interpolation.
4. Save the grid + axes to `.npz`. Plot the trap potential on the
   xOy plane at `z=0` and on the yOz plane at `x=0`, both by querying
   the GriddedTrap (so the figures reflect what the simulator would see).

Run from the repository root::

    python3 example/spectator/ripa/generate_ripa_gridded_trap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ATOM_ROOT = HERE.parents[2]
# Put atom_classical_mc's repo root on the path first so that `from src.* ...`
# resolves to *this* project, not the (also `src/`-shaped) vipa_focus_simulation
# library that lives under lib/. The RIPA code we need was copied into
# example/spectator/ripa/ripa_lib/ so it is importable by a non-clashing package name.
if str(ATOM_ROOT) not in sys.path:
    sys.path.insert(0, str(ATOM_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from src.trap import GriddedTrap  # noqa: E402
from src.units import joule_to_microkelvin, microkelvin_to_joule  # noqa: E402
from ripa_lib.crosssections import crosssection_xy  # noqa: E402
from ripa_lib.vipa_focus import PARAMS_10_TWZ, vipa_rays  # noqa: E402

PARAMS = PARAMS_10_TWZ
PARAMS_NAME = "10_TWZ"

# z range and grid count are not constrained by the RIPA params; set them here.
# x/y range and counts are derived from PARAMS below.
Z_RANGE_UM = 10
NZ = 151

DEPTH_UK = 500.0

# Fitted bias of the focal minimum in the raw RIPA intensity (µm). Subtract
# from the grid axes so the trap center sits at (0, 0, 0) where the
# simulator expects.
CENTER_BIAS_X_UM = 0.060
CENTER_BIAS_Y_UM = -0.148

# Zoom window (half-extent, µm) for the inset xOy view in plot_slices.
ZOOM_HALF_EXTENT_UM = 1.0

OUTPUT_NPZ = "ripa_gridded_trap.npz"
OUTPUT_PLOT = "ripa_gridded_slices.png"


def xy_axes_from_params(params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Build the focal-plane (x, y) axes from a RIPA params dict.

    Range comes from ``extent_f`` (the focal-plane half-extent the RIPA
    crosssection already crops to). Grid spacing matches the native Fourier
    spacing ``lambda * f / D`` produced by ``calc_field_after_lens`` on an
    aperture of size ``D``, so the resampling onto our axes is at the
    diffraction-limited resolution of the lens.
    """
    half_extent_m = float(params["extent_f"])
    native_dx_m = params["lambda"] * params["f"] / params["D"]
    n = int(np.round(2 * half_extent_m / native_dx_m)) + 1
    if n % 2 == 0:
        n += 1
    axis = np.linspace(-half_extent_m, half_extent_m, n)
    return axis, axis.copy()


def _bilinear_resample(
    src_y: np.ndarray, src_x: np.ndarray, src_values: np.ndarray,
    dst_y: np.ndarray, dst_x: np.ndarray,
) -> np.ndarray:
    """Bilinear resample `src_values[i,j] = f(src_y[i], src_x[j])` onto the
    target grid `(dst_y, dst_x)`. Out-of-bounds queries get 0.

    Pure NumPy; avoids dragging scipy in. Returns shape `(len(dst_y), len(dst_x))`.
    """

    dy = src_y[1] - src_y[0]
    dx = src_x[1] - src_x[0]
    ny = src_values.shape[0]
    nx = src_values.shape[1]

    Yq, Xq = np.meshgrid(dst_y, dst_x, indexing="ij")
    fy = (Yq - src_y[0]) / dy
    fx = (Xq - src_x[0]) / dx
    iy = np.floor(fy).astype(np.int64)
    ix = np.floor(fx).astype(np.int64)
    wy = fy - iy
    wx = fx - ix

    valid = (iy >= 0) & (iy <= ny - 2) & (ix >= 0) & (ix <= nx - 2)
    iy_c = np.clip(iy, 0, ny - 2)
    ix_c = np.clip(ix, 0, nx - 2)

    c00 = src_values[iy_c, ix_c]
    c01 = src_values[iy_c, ix_c + 1]
    c10 = src_values[iy_c + 1, ix_c]
    c11 = src_values[iy_c + 1, ix_c + 1]
    interp = (
        c00 * (1 - wy) * (1 - wx)
        + c01 * (1 - wy) * wx
        + c10 * wy * (1 - wx)
        + c11 * wy * wx
    )
    return np.where(valid, interp, 0.0)


def compute_intensity_grid(
    params: dict,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
) -> np.ndarray:
    """Compute `I(x, y, z)` on the user grid by RIPA per-z xy slices."""

    rays = vipa_rays(params)
    params_local = dict(params)
    # crosssection_xy crops the lens output to ±params["extent_f"]; widen so
    # it definitely covers the user-requested xy range.
    user_half_extent = max(
        float(np.max(np.abs(x_axis))),
        float(np.max(np.abs(y_axis))),
    )
    params_local["extent_f"] = max(
        float(params.get("extent_f", 0.0)),
        1.2 * user_half_extent + 1e-9,
    )

    nx, ny, nz = len(x_axis), len(y_axis), len(z_axis)
    grid = np.empty((nx, ny, nz), dtype=float)
    for k, z in enumerate(z_axis):
        xf, yf, _, intensity = crosssection_xy(
            rays,
            params_local,
            zf=float(z),
            show_focus=False,
            show_E_field=False,
            tqdm_enable=False,
        )
        # intensity has shape (len(yf), len(xf)).
        slab_yx = _bilinear_resample(yf, xf, intensity, y_axis, x_axis)
        # Transpose to (Nx, Ny) for our (x, y, z) layout.
        grid[:, :, k] = slab_yx.T
        print(
            f"  z[{k + 1:>3d}/{nz}] = {z * 1e6:+8.3f} um   "
            f"I_peak in slice = {float(np.max(intensity)):.3e}"
        )
    return grid


def build_trap_from_grid(
    potential_j: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    name: str,
) -> GriddedTrap:
    spacing = np.array(
        [
            x_axis[1] - x_axis[0],
            y_axis[1] - y_axis[0],
            z_axis[1] - z_axis[0],
        ],
        dtype=float,
    )
    origin = np.array([x_axis[0], y_axis[0], z_axis[0]], dtype=float)
    return GriddedTrap(
        grid_potential_j=potential_j,
        origin_local_m=origin,
        spacing_m=spacing,
        center_m=np.zeros(3, dtype=float),
        interpolation="tricubic",
        name=name,
    )


def plot_slices(
    trap: GriddedTrap,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    output_path: Path | None,
    title_suffix: str,
    zoom_half_extent_m: float = ZOOM_HALF_EXTENT_UM * 1e-6,
) -> None:
    import matplotlib.pyplot as plt

    plot_res = 300
    x_dense = np.linspace(x_axis[0], x_axis[-1], plot_res)
    y_dense = np.linspace(y_axis[0], y_axis[-1], plot_res)
    z_dense = np.linspace(z_axis[0], z_axis[-1], plot_res)

    # xOy at z=0 (full)
    X_xy, Y_xy = np.meshgrid(x_dense, y_dense, indexing="xy")  # (Ny, Nx)
    pts_xy = np.stack([X_xy, Y_xy, np.zeros_like(X_xy)], axis=-1)
    u_xy_uK = joule_to_microkelvin(trap.potential(pts_xy))

    # xOy at z=0 (zoom)
    zoom = float(zoom_half_extent_m)
    x_zoom = np.linspace(-zoom, zoom, plot_res)
    y_zoom = np.linspace(-zoom, zoom, plot_res)
    X_zoom, Y_zoom = np.meshgrid(x_zoom, y_zoom, indexing="xy")
    pts_zoom = np.stack([X_zoom, Y_zoom, np.zeros_like(X_zoom)], axis=-1)
    u_zoom_uK = joule_to_microkelvin(trap.potential(pts_zoom))

    # yOz at x=0
    Y_yz, Z_yz = np.meshgrid(y_dense, z_dense, indexing="xy")  # (Nz, Ny)
    pts_yz = np.stack([np.zeros_like(Y_yz), Y_yz, Z_yz], axis=-1)
    u_yz_uK = joule_to_microkelvin(trap.potential(pts_yz))

    vmin = float(min(np.min(u_xy_uK), np.min(u_yz_uK), np.min(u_zoom_uK)))
    vmax = 0.0

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)

    im0 = axes[0].pcolormesh(
        x_dense * 1e6, y_dense * 1e6, u_xy_uK,
        cmap="inferno", shading="auto", vmin=vmin, vmax=vmax,
    )
    axes[0].set_xlabel(r"$x$ (µm)")
    axes[0].set_ylabel(r"$y$ (µm)")
    axes[0].set_title(f"xOy plane at z = 0   {title_suffix}")
    axes[0].set_aspect("equal")
    # Mark the zoom window on the full-extent panel.
    z_um = zoom * 1e6
    axes[0].plot(
        [-z_um, z_um, z_um, -z_um, -z_um],
        [-z_um, -z_um, z_um, z_um, -z_um],
        color="cyan", linewidth=0.8, linestyle="--",
    )
    fig.colorbar(im0, ax=axes[0], label="U (µK)")

    im1 = axes[1].pcolormesh(
        x_zoom * 1e6, y_zoom * 1e6, u_zoom_uK,
        cmap="inferno", shading="auto", vmin=vmin, vmax=vmax,
    )
    axes[1].set_xlabel(r"$x$ (µm)")
    axes[1].set_ylabel(r"$y$ (µm)")
    axes[1].set_title(f"xOy zoom  (±{z_um:.2f} µm)")
    axes[1].set_aspect("equal")
    axes[1].axhline(0.0, color="white", linewidth=0.5, alpha=0.4)
    axes[1].axvline(0.0, color="white", linewidth=0.5, alpha=0.4)
    fig.colorbar(im1, ax=axes[1], label="U (µK)")

    im2 = axes[2].pcolormesh(
        y_dense * 1e6, z_dense * 1e6, u_yz_uK,
        cmap="inferno", shading="auto", vmin=vmin, vmax=vmax,
    )
    axes[2].set_xlabel(r"$y$ (µm)")
    axes[2].set_ylabel(r"$z$ (µm)")
    axes[2].set_title("yOz plane at x = 0")
    axes[2].set_aspect("auto")
    fig.colorbar(im2, ax=axes[2], label="U (µK)")

    if output_path is None:
        plt.show()
    else:
        fig.savefig(output_path, dpi=140)
        print(f"Saved plot to {output_path}")


def main() -> None:
    x_axis, y_axis = xy_axes_from_params(PARAMS)
    z_axis = np.linspace(-Z_RANGE_UM * 1e-6, Z_RANGE_UM * 1e-6, NZ)

    nx, ny = len(x_axis), len(y_axis)
    x_range_um = float(x_axis[-1]) * 1e6
    y_range_um = float(y_axis[-1]) * 1e6

    print(
        f"RIPA PARAMS_{PARAMS_NAME}.   "
        f"Grid ({nx},{ny},{NZ}) over "
        f"x=±{x_range_um:.2f} um, y=±{y_range_um:.2f} um, "
        f"z=±{Z_RANGE_UM:.2f} um.\n"
        f"Computing intensity slabs..."
    )

    intensity = compute_intensity_grid(PARAMS, x_axis, y_axis, z_axis)
    peak_intensity = float(np.max(intensity))
    if peak_intensity <= 0.0:
        raise RuntimeError(
            "RIPA intensity grid peak is zero. Try a smaller z range or a "
            "different parameter set so the focal volume falls inside the box."
        )

    # Re-center: the focal minimum sits at (CENTER_BIAS_X, CENTER_BIAS_Y) in the
    # RIPA frame. Shift the axes (data unchanged) so that point maps to (0, 0).
    x_axis = x_axis - CENTER_BIAS_X_UM * 1e-6
    y_axis = y_axis - CENTER_BIAS_Y_UM * 1e-6
    print(
        f"Applied center bias correction: dx = {CENTER_BIAS_X_UM:+.4f} µm, "
        f"dy = {CENTER_BIAS_Y_UM:+.4f} µm."
    )

    depth_j = float(microkelvin_to_joule(DEPTH_UK))
    potential_j = -depth_j * (intensity / peak_intensity)

    out_npz = HERE / OUTPUT_NPZ
    np.savez(
        out_npz,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
        intensity=intensity,
        potential_j=potential_j,
        depth_uK=np.array(DEPTH_UK),
        peak_intensity=np.array(peak_intensity),
        params_name=np.array(PARAMS_NAME),
    )
    print(f"Saved grid to {out_npz}")

    trap = build_trap_from_grid(
        potential_j, x_axis, y_axis, z_axis, name=f"ripa_{PARAMS_NAME}"
    )
    u_origin_uK = float(
        joule_to_microkelvin(trap.potential(np.array([0.0, 0.0, 0.0])))
    )
    min_idx = np.unravel_index(int(np.argmin(potential_j)), potential_j.shape)
    min_pos_um = (
        x_axis[min_idx[0]] * 1e6,
        y_axis[min_idx[1]] * 1e6,
        z_axis[min_idx[2]] * 1e6,
    )
    u_min_uK = float(joule_to_microkelvin(np.min(potential_j)))
    print(
        f"GriddedTrap built (tricubic). U at origin: {u_origin_uK:+.4f} µK; "
        f"grid minimum: {u_min_uK:+.4f} µK at "
        f"(x,y,z) = ({min_pos_um[0]:+.2f}, {min_pos_um[1]:+.2f}, "
        f"{min_pos_um[2]:+.2f}) µm; configured depth: {DEPTH_UK:.4f} µK."
    )

    plot_path = HERE / OUTPUT_PLOT
    plot_slices(
        trap,
        x_axis,
        y_axis,
        z_axis,
        plot_path,
        title_suffix=f"(PARAMS_{PARAMS_NAME}, depth {DEPTH_UK:.0f} µK)",
    )


if __name__ == "__main__":
    main()
