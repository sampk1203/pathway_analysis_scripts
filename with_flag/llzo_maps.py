"""
llzo_maps.py - shared numerical helpers for 01_density_free_energy.py and 02_compare_temps.py.

  deposit_cic       cloud-in-cell (trilinear) deposition of points on a periodic grid (no nearest-voxel aliasing)
  adaptive_smooth   adaptive-bandwidth Gaussian smoothing (narrow kernels where Li is dense, wide where sparse)
  sphere_values     grid values inside a sphere around a point (periodic)
  apply_shift       periodic roll of a map by a FIXED shift (A), rounded to whole voxels
  project           min projection along an axis (NaN-aware), optionally through a slab around the box centre
"""
import warnings

import numpy as np

import llzo_svg  # noqa: F401  (every saved .png also gets an editable .svg)
from scipy.ndimage import gaussian_filter, map_coordinates

PANELS = [(2, "x", "y", 0, 1), (1, "x", "z", 0, 2), (0, "y", "z", 1, 2)]   # (collapsed axis, xlabel, ylabel, ix, iy)


def deposit_cic(frac, n):
    """Trilinear (cloud-in-cell) deposit of points frac (M,3) in [0,1) onto a periodic grid n=(nx,ny,nz).
    Voxel i is centred at (i+0.5)/n. Total weight is conserved; result is counts per voxel."""
    n = np.asarray(n, int)
    g = frac * n - 0.5
    i0 = np.floor(g).astype(int)
    d = g - i0
    out = np.zeros(int(np.prod(n)))
    for dx in (0, 1):
        wx = d[:, 0] if dx else 1.0 - d[:, 0]
        ix = (i0[:, 0] + dx) % n[0]
        for dy in (0, 1):
            wy = d[:, 1] if dy else 1.0 - d[:, 1]
            iy = (i0[:, 1] + dy) % n[1]
            for dz in (0, 1):
                wz = d[:, 2] if dz else 1.0 - d[:, 2]
                iz = (i0[:, 2] + dz) % n[2]
                idx = (ix * n[1] + iy) * n[2] + iz
                out += np.bincount(idx, weights=wx * wy * wz, minlength=out.size)
    return out.reshape(tuple(n))


def kernel_voxels(sigma, voxel):
    """Effective number of voxels under a 3D Gaussian of width sigma (A): (2 sqrt(pi) sigma / voxel)^3."""
    return float(np.prod(2.0 * np.sqrt(np.pi) * sigma / np.asarray(voxel)))


def adaptive_smooth(frac, n, voxel, sigma_pilot, s_min, s_max, nclass=6):
    """Adaptive-bandwidth (Abramson) smoothing. A fixed-sigma pilot density sets, for every sample, a bandwidth
        h_i = sigma_pilot * sqrt( g / rho_pilot(x_i) ),   clipped to [s_min, s_max],   g = geometric mean of rho_pilot
    so samples in dense regions (Li sites) get narrow kernels (peaks are not blurred) and samples in sparse regions
    get wide ones (noise is averaged). Samples are grouped into `nclass` log-spaced bandwidths; each group is
    deposited (CIC) and smoothed once. Returns (smoothed counts/voxel, pilot counts/voxel, cic counts, h_i)."""
    counts = deposit_cic(frac, n)
    pilot = gaussian_filter(counts, sigma_pilot / voxel, mode="wrap")
    pv = map_coordinates(pilot, (frac * n - 0.5).T, order=1, mode="grid-wrap")
    pv = np.maximum(pv, 1e-12)
    gm = np.exp(np.mean(np.log(pv)))
    h = np.clip(sigma_pilot * np.sqrt(gm / pv), s_min, s_max)
    ks = np.geomspace(s_min, s_max, nclass)
    cls = np.abs(np.log(h)[:, None] - np.log(ks)[None, :]).argmin(axis=1)
    out = np.zeros(tuple(n))
    for k in range(nclass):
        m = cls == k
        if m.any():
            out += gaussian_filter(deposit_cic(frac[m], n), ks[k] / voxel, mode="wrap")
    return out, pilot, counts, h


def sphere_values(field, L, p, r):
    """Values of `field` (periodic grid over box L, origin 0) in voxels whose centre is within r (A) of point p."""
    n = np.array(field.shape)
    vox = np.asarray(L) / n
    k = np.ceil(r / vox).astype(int)
    ic = np.floor(np.asarray(p) / vox).astype(int)
    rng = [np.arange(-k[i], k[i] + 1) for i in range(3)]
    sub = field[np.ix_(*[(ic[i] + rng[i]) % n[i] for i in range(3)])]
    off = [(ic[i] + rng[i] + 0.5) * vox[i] - p[i] for i in range(3)]
    d2 = off[0][:, None, None] ** 2 + off[1][None, :, None] ** 2 + off[2][None, None, :] ** 2
    m = d2 < r * r
    return sub[m]


def apply_shift(F, L, shift):
    """Periodic roll of map F by `shift` (A), rounded to whole voxels. Returns (rolled map, shift actually applied, A).
    Use the same `shift` for every temperature of a system so maps stay aligned voxel-for-voxel."""
    n = np.array(F.shape)
    vox = np.asarray(L, float) / n
    sv = np.rint(np.asarray(shift, float) / vox).astype(int)
    return np.roll(F, tuple(sv), axis=(0, 1, 2)), sv * vox


def project(F, axis, slab=0.0, L=None):
    """NaN-aware minimum along `axis`. slab > 0: only voxels within +/- slab (A) of the box centre plane
    (meaningful after apply_shift with the dopant at the centre). All-NaN lines give NaN (drawn grey)."""
    if slab > 0:
        n = F.shape[axis]
        vox = L[axis] / n
        k = int(np.ceil(slab / vox))
        sl = [slice(None)] * 3
        sl[axis] = slice(max(n // 2 - k, 0), n // 2 + k + 1)
        F = F[tuple(sl)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmin(F, axis=axis)
