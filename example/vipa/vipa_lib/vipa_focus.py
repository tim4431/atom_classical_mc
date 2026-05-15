import numpy as np
from typing import Tuple, List
from .core import *


def vipa_rays(
    params: dict,
) -> np.ndarray:
    """
    Complex electric field in the vipa output plane.
    """
    Nx = params["Nx"]
    Ny = params["Ny"]
    lx = params["lx"]
    ly = params["ly"]
    FSR_Ratio = params["FSR_Ratio"]
    phi = params["phi"]
    w = params["w"]
    d = params["d"]
    phase_amp_func = params.get("phase_amp_func", None)
    displacement_func = params.get("displacement_func", None)
    phase_correction = params.get("phase_correction", None)
    #

    rays = []
    for nx in range(Nx):
        for ny in range(Ny):
            ix = nx - (Nx - 1) / 2
            iy = ny - (Ny - 1) / 2
            center_x = (ix) * d
            center_y = (iy) * d
            if displacement_func is not None:
                dx, dy = displacement_func(ix, iy, nx, ny)
                center_x += dx
                center_y += dy
            intensity = np.exp(-(nx * lx + ny * ly))
            phase = -(ix * FSR_Ratio + iy) * (phi)
            if phase_correction is not None:
                # pass
                phase += phase_correction[nx, ny]
            #
            ray = {
                "x": center_x,
                "y": center_y,
                "w": w,
                "ix": ix,
                "iy": iy,
                "nx": nx,
                "ny": ny,
                "intensity": intensity,
                "phase": phase,
            }
            if phase_amp_func is not None:
                phase_amp_func_i = (
                    lambda Xi, Yi, ix=ix, iy=iy, nx=nx, ny=ny: phase_amp_func(
                        ix, iy, nx, ny, Xi, Yi
                    )
                )
                ray["phase_amp_func"] = phase_amp_func_i
            rays.append(ray)
    #
    return rays


def rays_from_file(
    filename: str,
    params: dict,
) -> np.ndarray:
    data = np.load(filename, allow_pickle=True)
    #
    FSR_Ratio = params["FSR_Ratio"]
    phi = params["phi"]
    #
    xList = data["xList"] * 1e-2  # optable unit is cm
    yList = data["yList"] * 1e-2
    # center the beams
    xList -= np.mean(xList)
    yList -= np.mean(yList)
    tXList = data["tXList"]
    tYList = data["tYList"]
    tXList -= np.mean(tXList)
    tYList -= np.mean(tYList)
    IList = data["IList"]
    print(xList.shape, yList.shape, tXList.shape, tYList.shape, IList.shape)
    #
    Nx = len(xList)
    Ny = 1
    assert (
        len(xList) == Nx * Ny
    ), f"Number of traced rays {len(xList)} does not match Nx*Ny={Nx*Ny}"
    print(f"Number of traced rays: {len(xList)} matching Nx*Ny={Nx*Ny}")
    #
    wl = params["lambda"]

    def tilt_phase(tx, ty, Xi, Yi):
        k = 2 * np.pi / wl
        return (tx * k) * Xi + (ty * k) * Yi, np.ones_like(Xi)

    #
    rays = []  # for elec2d format
    for ny in range(Ny):
        for nx in range(Nx):
            idx = ny * Nx + nx
            ix = nx - (Nx - 1) / 2
            iy = ny - (Ny - 1) / 2
            intensity = IList[idx]
            phase = -(ix * FSR_Ratio + iy) * (phi)
            phase_amp_func_i = lambda Xi, Yi, tx=tXList[idx], ty=tYList[
                idx
            ]: tilt_phase(tx, ty, Xi, Yi)
            ray = {
                "x": xList[idx],
                "y": yList[idx],
                "w": 61e-6,
                "ix": ix,
                "iy": iy,
                "nx": nx,
                "ny": ny,
                "intensity": intensity,
                "phase": phase,
                "phase_amp_func": phase_amp_func_i,
            }
            rays.append(ray)

    # field = rays2elec2d(Xi, Yi, rays, params, **kwargs)
    # return field
    return rays


DSP = -100e-6  # 20 um


def misaligned_tilt(ix, iy, nx, ny, Xi, Yi):
    tx = -np.cos(2 * np.pi * nx / 4) * (DSP / 46.7e-3)
    return (tx * (2 * np.pi / 780e-9)) * Xi, np.ones_like(Xi)


def misaligned_displacement(ix, iy, nx, ny):
    # dx = -np.sin(2 * np.pi * nx / 4) * DSP
    # dy = 0
    dx = 0
    dy = -np.sin(2 * np.pi * ny / 4) * DSP
    return dx, dy


PARAMS_100 = {
    "Nx": 100,
    "Ny": 100,
    "FSR_Ratio": 100,
    "lx": 0.01,
    "ly": 0.01,
    "w": 61e-6,  # beam waist
    "d": 360e-6,  # beam spacing
    "f": 0.2,
    "phi": 0.0,
    "lambda": 780e-9,
    "D": 40e-2,  # real space extent
    "RESOLUTION_X": 25e-6,  # real space resolution
    "extent_f": 1000e-6,  # focal plane extent, only for plotting
    # "phase_amp_func": misaligned_tilt,
    # "displacement_func": misaligned_displacement,
    # "gouy_phase": 0.0,
}
PARAMS_80 = {
    "Nx": 80,
    "Ny": 80,
    "FSR_Ratio": 80,
    "lx": 0.01,
    "ly": 0.01,
    "w": 73.82e-6,  # beam waist
    "d": 420e-6,  # beam spacing
    "f": 0.2,
    "phi": 0.0,
    "lambda": 780e-9,
    "D": 20e-2,  # real space extent
    "RESOLUTION_X": 25e-6,  # real space resolution
    "extent_x": 1e-2,
    "extent_f": 800e-6,  # focal plane extent, only for plotting
    # "phase_amp_func": misaligned_tilt,
    # "displacement_func": misaligned_displacement,
    "zfi": None,
    # "gouy_phase": 0.0,
}
PARAMS_10 = {
    "Nx": 8,
    "Ny": 9,
    "FSR_Ratio": 22.0,
    "Lrt": 2.311,
    "lx": 0.198,
    "ly": 0.052,
    "w": 108e-6,  # beam waist
    "d": 1000e-6,  # beam spacing
    "f": 0.2,
    "phi": 0.0,
    "lambda": 780e-9,
    "D": 10e-2,  # real space extent
    "RESOLUTION_X": 25e-6,  # real space resolution
    "extent_x": 1e-2,
    "extent_f": 800e-6,  # focal plane extent, only for plotting
    # "phase_amp_func": misaligned_tilt,
    # "displacement_func": misaligned_displacement,
    "zfi": None,
    # "gouy_phase": 0.0,
}

PARAMS_10_TWZ = {
    "Nx": 8,
    "Ny": 9,
    "FSR_Ratio": 22.0,
    "Lrt": 2.311,
    "lx": 0.198,
    "ly": 0.052,
    "w": 108e-6,  # beam waist
    "d": 1000e-6,  # beam spacing
    "f": 0.017,
    "phi": (8.2 / 11 + 2 * 8) * np.pi / 11,
    "lambda": 780e-9,
    "D": 20e-2,  # real space extent
    "RESOLUTION_X": 25e-6,  # real space resolution
    "extent_x": 1e-2,
    "extent_f": 10e-6,  # focal plane extent, only for plotting
    # "phase_amp_func": misaligned_tilt,
    # "displacement_func": misaligned_displacement,
    "zfi": None,
    # "gouy_phase": 0.0,
}