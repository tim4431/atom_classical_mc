import numpy as np
import matplotlib.pyplot as plt
from .core import *
from matplotlib.colors import LogNorm, Normalize


def crosssection_xy(
    rays: List[dict],
    params: dict,
    zf: float = 0.0,
    show_focus: bool = True,
    log_scale: bool = False,
    show_E_field: bool = False,
    **kwargs,
):
    """
    Cross-section of |E|² at focal plane zf.
    returns:
    - xf       : 1-D array (metres)
    - yf       : 1-D array (metres)
    - E_tilde  : 2-D array, shape (len(yf), len(xf))   complex electric field after lens at zf
    - intensity: 2-D array, shape (len(yf), len(xf))   |E_tilde|²
    -----------
    """
    D = params["D"]
    RESOLUTION_X = params["RESOLUTION_X"]
    N_grid = int(D / (2 * RESOLUTION_X)) * 2 + 1
    xi = np.linspace(-D / 2, D / 2, N_grid)
    yi = np.linspace(-D / 2, D / 2, N_grid)
    Xi, Yi = np.meshgrid(xi, yi, indexing="xy")
    # print(Xi.shape, Yi.shape)

    Ei = rays2elec2d(Xi, Yi, rays, params, **kwargs)
    # print("Total input power (arb. units):", np.sum(np.abs(Ei) ** 2))

    zfi = params.get("zfi", None)
    if zfi is not None:
        print(f"Propagating initial zf = {zfi*1e3:.2f} mm")
        Ei = freespace_propagation(xi, yi, Ei, params["lambda"], zfi)
    if show_E_field:
        extent = [-D / 2 * 1e6, D / 2 * 1e6, -D / 2 * 1e6, D / 2 * 1e6]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Intensity subplot
        im1 = axes[0].imshow(
            np.abs(Ei) ** 2,
            extent=extent,
            origin="lower",
            cmap="rainbow",
            aspect="equal",
            vmin=0,
        )
        axes[0].set_xlabel(r"$x$ (source plane, $\mu$m)")
        axes[0].set_ylabel(r"$y$ (source plane, $\mu$m)")
        axes[0].set_title("Source plane intensity")
        plt.colorbar(im1, ax=axes[0], label="Intensity (arb.)")

        # Phase subplot
        im2 = axes[1].imshow(
            np.angle(Ei) + np.abs(Ei) ** 2 / np.max(np.abs(Ei) ** 2),
            extent=extent,
            origin="lower",
            cmap="Reds",
            aspect="equal",
        )
        axes[1].set_xlabel(r"$x$ (source plane, $\mu$m)")
        axes[1].set_ylabel(r"$y$ (source plane, $\mu$m)")
        axes[1].set_title("Source plane phase")
        plt.colorbar(im2, ax=axes[1], label="Phase (rad)")

        plt.tight_layout()
        plt.show()
    (xf, yf), E_tilde = calc_field_after_lens(
        xi,
        yi,
        Ei,
        wl=params["lambda"],
        f=params["f"],
        zf=zf,
        **kwargs,
    )
    if params.get("pinhole"):
        print("Applying pinhole...")
        pinhole_dict = params.get("pinhole")
        pinhole_diameter = pinhole_dict.get("diameter", 5e-6)
        pinhole_x = pinhole_dict.get("x", 0.0)
        pinhole_y = pinhole_dict.get("y", 0.0)
        mask_pinhole = np.zeros_like(E_tilde, dtype=bool)
        Xf, Yf = np.meshgrid(xf, yf, indexing="xy")
        mask_pinhole[
            np.sqrt((Xf - pinhole_x) ** 2 + (Yf - pinhole_y) ** 2)
            <= pinhole_diameter / 2
        ] = True
        E_tilde *= mask_pinhole
        # propagate by zf_pinhole
        zf_pinhole = pinhole_dict.get("zf_pinhole", 0.0e-3)
        print(f"Propagating by zf_pinhole = {zf_pinhole*1e3:.2f} mm")
        E_tilde = freespace_propagation(xf, yf, E_tilde, params["lambda"], zf_pinhole)
    #
    extent_f = params["extent_f"]
    # crop data to ±extent_f
    xf_mask = np.abs(xf) < extent_f
    yf_mask = np.abs(yf) < extent_f
    xf = xf[xf_mask]
    yf = yf[yf_mask]
    E_tilde = E_tilde[np.ix_(yf_mask, xf_mask)]
    intensity = np.abs(E_tilde) ** 2

    if show_focus:
        # plot
        extent_um = extent_f * 1e6
        plt.figure(figsize=(6, 5))
        plt.imshow(
            intensity,
            extent=[-extent_um, extent_um, -extent_um, extent_um],
            norm=(
                LogNorm(vmin=np.max(intensity) * 1e-5, vmax=np.max(intensity))
                if log_scale
                else Normalize(vmin=0, vmax=np.max(intensity))
            ),
            origin="lower",
            cmap="rainbow",
            aspect="equal",
        )
        plt.colorbar(label="Intensity (arb.)")
        # ticks = np.arange(-extent_um, extent_um + 1e-12, 100)
        # plt.xticks(ticks, ticks.astype(int))
        # plt.yticks(ticks, ticks.astype(int))
        plt.xlabel(r"$x_f$ (focal plane, $\mu$m)")
        plt.ylabel(r"$y_f$ (focal plane, $\mu$m)")
        plt.title(rf"Interference pattern (zf = {zf*1e6:.1f} µm)")
        plt.tight_layout()
        plt.show()

    return xf, yf, E_tilde, intensity


def crosssection_x(
    rays: List[dict],
    params: dict,
    zf: float = 0.0,
    show_focus: bool = False,
    **kwargs,
):
    """
    Return |E|² at Y_f≈0 as a function of x_f.

    Returns
    -------
    xf       : 1-D array (metres)
    profile  : 1-D array  |E|²(x_f)   shape == len(xf)
    """

    D = params["D"]
    RESOLUTION_X = params["RESOLUTION_X"]
    N_grid = int(D / (2 * RESOLUTION_X)) * 2 + 1
    xi = np.linspace(-D / 2, D / 2, N_grid)
    yi = np.array([0])

    Xi, Yi = np.meshgrid(xi, yi, indexing="xy")
    Ei = rays2elec2d(Xi, Yi, rays, params, **kwargs)
    zfi = params.get("zfi", None)
    if zfi is not None:
        print(f"Propagating initial zf = {zfi*1e3:.2f} mm")
        Ei = freespace_propagation(xi, yi, Ei, params["lambda"], zfi)
    (xf, yf), profile = calc_field_after_lens(
        xi, yi, Ei, wl=params["lambda"], f=params["f"], zf=zf, **kwargs
    )
    profile = np.abs(profile) ** 2  # shape (1, len(yf))
    profile = profile.flatten()  # shape (len(yf),)

    extent_f = params["extent_f"]
    xf_mask = np.abs(xf) < extent_f
    xf = xf[xf_mask]
    profile = profile[xf_mask]

    if show_focus:
        plt.figure()
        plt.plot(xf * 1e6, profile)
        plt.xlabel(r"$x_f$ (µm)")
        # plt.xlim(-extent_f * 1e6, extent_f * 1e6)
        plt.ylabel("Intensity (arb.)")
        plt.title(f"Cross-section   $z_0={zf*100:.2f}$ cm")
        plt.tight_layout()
        plt.show()

    return xf, profile


def crosssection_xz(
    rays: List[dict],
    params: dict,
    extent_z: float = 20e-6,
    n_z: int = 51,
    show_focus: bool = True,
    **kwargs,
):
    """
    Compute and optionally plot |E|² at X_f≈0 as a function of y_f and z_f.

    Parameters
    ----------
    params    : dict of beam/lens parameters
    extent_z  : scan range in z (±extent_z)
    n_z       : number of z points
    alpha     : HG-mode parameter
    mmax      : highest Hermite–Gaussian index included
    show_focus : whether to display the plot

    Returns
    -------
    z_scan    : 1-D array of z_f values (meters)
    xf        : 1-D array of x_f values (meters)
    profiles  : 2-D array, shape (len(xf), len(z_scan)), |E|²(x_f, z_f)
    """
    z_scan = np.linspace(-extent_z, extent_z, n_z)
    profiles = []
    for z in tqdm(z_scan, desc="z scan"):
        xf, prof = crosssection_x(
            rays, params, zf=z, show_focus=False, tqdm_enable=False, **kwargs
        )
        profiles.append(prof)

    profiles = np.array(profiles)  # shape (n_z, n_xf)
    profiles = profiles.T  # shape (n_xf, n_z)

    if show_focus:
        extent_f = params["extent_f"]
        extent = [
            -extent_z * 1e6,
            extent_z * 1e6,
            -extent_f * 1e6,
            extent_f * 1e6,
        ]
        plt.figure(figsize=(6, 5))
        plt.imshow(
            profiles,
            extent=extent,
            origin="lower",
            aspect="auto",
            cmap="rainbow",
            vmin=0,
        )
        plt.colorbar(label="Normalised intensity")
        plt.xlabel(r"$z_0$ (µm)")
        plt.ylabel(r"$y_f$ (µm)")
        plt.title(r"Cross-section $x_f\!\approx\!0$ in the $y$–$z$ plane")
        plt.tight_layout()
        plt.show()

    return z_scan, xf, profiles
