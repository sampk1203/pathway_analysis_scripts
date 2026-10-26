#!/usr/bin/env python3
"""
08_phonon_band_center.py - Li phonon band centre from the MD dump (VACF -> PDOS -> first moment).  No sites, no hops.

Li velocities come from central differences of the unwrapped positions (framework-COM drift removed, same as 05):
    v(t) = [x(t+dt) - x(t-dt)] / 2dt
VACF of all Li (sum over atoms and xyz, unbiased, Hann-tapered over --win-ps) -> cosine transform -> g(f) (Li-projected PDOS).
    band centre  f_c = int f g(f) df / int g(f) df    over  [--fmin, fmax]   (also printed without the --fmin cut)
Central-difference attenuation sinc(2 pi f dt)^2 is divided out of g.

HARD LIMIT: dump spacing dt. Nyquist = 1/(2 dt). Li modes are ~5-15 THz, so dt must be <= ~10 fs. Larger dt: aliasing,
no correction possible -> script stops (--force overrides; the numbers are then meaningless). 100 fs dumps -> Nyquist 5 THz.
Fix is a new short run: NVE (or weak thermostat), e.g.  dump 2 all custom 2 dump_phonon.lammpstrj id type xu yu zu

Classical MD: no quantum occupation weighting. Hopping adds a low-f diffusive part to g (pulls f_c down): compare systems
and temperatures with the same protocol; the --fmin cut (default 1 THz) removes most of it and is reported separately.
Checks printed: f_c at 2x VACF window (should not move) and block standard error (--blocks contiguous segments).

usage:
  python 08_phonon_band_center.py /path/to/folder
  python 08_phonon_band_center.py /path/to/folder --T 800 1200 --win-ps 3 --fmin 1
  python 08_phonon_band_center.py /path/to/run --dump-pattern 'MD-{T}/dump.{T}.lammpstrj' --elements Li La Zr Al O --fold 2 2 2
(system-layout flags --dump-dir --prefix --dump-pattern --elements --element-list --dopants --fold --name as in 01-07)
outputs (<outdir>/<system>/): phonon_band_center.csv, pdos_Li.png
"""
import argparse
import csv
import os
import sys

import numpy as np

import llzo_io as io

THZ_TO_MEV = 4.135667696


def load(cfg, T, tmin):
    traj = io.prepare(cfg, io.dump_path(cfg, T), tmin)
    drift = io.framework_drift(cfg, traj)
    lm = io.li_mask(cfg, traj)
    li = traj.pos[:, lm, :] - drift[:, None, :]
    return li, float(np.median(np.diff(traj.time_ps)))


def pdos(v, dt_ps, win):
    """v (F,N,3). Returns f (THz), g (not normalised). VACF window = win frames."""
    n = v.shape[0]
    f = np.fft.rfft(v, n=2 * n, axis=0)
    acf = np.fft.irfft(f * np.conj(f), axis=0)[:win].sum(axis=(1, 2))
    acf = acf / (n - np.arange(win))
    acf = acf / acf[0] * np.hanning(2 * win)[win:]
    sym = np.concatenate([acf, acf[-2:0:-1]])
    freq = np.fft.rfftfreq(len(sym), dt_ps)
    g = np.clip(np.fft.rfft(sym).real, 0, None)
    x = 2 * np.pi * freq * dt_ps                       # central difference over 2 dt: amplitude sin(x)/x
    ok = np.sinc(x / np.pi) > 0.3
    g = np.where(ok, g / np.where(ok, np.sinc(x / np.pi) ** 2, 1.0), 0.0)
    return freq, g


def centre(freq, g, fmin, fmax):
    m = (freq >= fmin) & (freq <= fmax)
    return float((freq[m] * g[m]).sum() / g[m].sum())


def velocities(li, dt_ps):
    return (li[2:] - li[:-2]) / (2 * dt_ps)


def analyse(li, dt_ps, win_ps, fmin, fmax, nblocks):
    v = velocities(li, dt_ps)
    win = max(8, int(round(win_ps / dt_ps)))
    if 4 * win > v.shape[0]:
        raise SystemExit(f"trajectory ({v.shape[0]} frames) too short for window {win} frames; lower --win-ps")
    f, g = pdos(v, dt_ps, win)
    f2, g2 = pdos(v, dt_ps, min(2 * win, v.shape[0] // 2))
    fc, fc_nocut = centre(f, g, fmin, fmax), centre(f, g, 0.0, fmax)
    fc2 = centre(f2, g2, fmin, fmax)
    seg = v.shape[0] // nblocks
    if seg >= 4 * win:
        b = [centre(*pdos(v[k * seg:(k + 1) * seg], dt_ps, win), fmin, fmax) for k in range(nblocks)]
        se = float(np.std(b, ddof=1) / np.sqrt(nblocks))
    else:
        se = float("nan")
    return f, g, fc, fc_nocut, fc2, se, win


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="folder with the dumps + element_list (or a settings .yaml)")
    ap.add_argument("--defaults")
    ap.add_argument("--T", type=int, nargs="*")
    ap.add_argument("--tmin", type=float)
    ap.add_argument("--outdir", default="phonon08")
    ap.add_argument("--win-ps", type=float, default=3.0, help="VACF window, ps (frequency resolution ~ 1/win)")
    ap.add_argument("--fmin", type=float, default=1.0, help="THz, lower cut of the band-centre integral (hopping tail)")
    ap.add_argument("--fmax", type=float, default=30.0, help="THz, upper limit (also capped at 0.8 x Nyquist)")
    ap.add_argument("--blocks", type=int, default=4, help="contiguous segments for the standard error")
    ap.add_argument("--max-dt-fs", type=float, default=10.0, help="largest dump spacing accepted")
    ap.add_argument("--force", action="store_true", help="run even if dt is too large (aliased, meaningless)")
    io.add_config_args(ap)
    args = ap.parse_args()

    cfg = io.load_config(args.target, args.defaults, args)
    temps = args.T or io.temperatures(cfg)
    outdir = os.path.join(args.outdir, cfg["name"])
    os.makedirs(outdir, exist_ok=True)

    rows, spectra = [], {}
    for T in temps:
        print(f"\n=== {cfg['name']}  T = {T} K ===")
        li, dt_ps = load(cfg, T, args.tmin)
        dt_fs = dt_ps * 1e3
        nyq = 0.5 / dt_ps
        print(f"  dump spacing {dt_fs:g} fs -> Nyquist {nyq:.1f} THz;  {li.shape[1]} Li, {li.shape[0]} frames")
        if dt_fs > args.max_dt_fs and not args.force:
            print(f"  STOP: dt > {args.max_dt_fs:g} fs. Li modes (~5-15 THz) are above Nyquist: PDOS aliased, band centre meaningless.\n"
                  f"  Rerun a short NVE with dt <= 5 fs dumps, e.g.  dump 2 all custom 2 dump_phonon.lammpstrj id type xu yu zu\n"
                  f"  (or --force to compute anyway).")
            sys.exit(2)
        fmax = min(args.fmax, 0.8 * nyq)
        f, g, fc, fc_nc, fc2, se, win = analyse(li, dt_ps, args.win_ps, args.fmin, fmax, args.blocks)
        print(f"  VACF window {win} frames = {win * dt_ps:g} ps;  band [{args.fmin:g}, {fmax:g}] THz")
        print(f"  f_c = {fc:.3f} +/- {se:.3f} THz ({fc * THZ_TO_MEV:.2f} meV)   without fmin cut: {fc_nc:.3f} THz   "
              f"at 2x window: {fc2:.3f} THz")
        rows.append(dict(T=T, dt_fs=dt_fs, n_Li=li.shape[1], win_ps=win * dt_ps, fmin_THz=args.fmin, fmax_THz=fmax,
                         fc_THz=fc, fc_se_THz=se, fc_meV=fc * THZ_TO_MEV, fc_nocut_THz=fc_nc, fc_2xwin_THz=fc2))
        m = f <= fmax
        spectra[T] = (f[m], g[m] / (0.5 * (g[m][1:] + g[m][:-1]) * np.diff(f[m])).sum())

    with open(os.path.join(outdir, "phonon_band_center.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    cm = plt.cm.plasma(np.linspace(0.05, 0.85, len(spectra)))
    for c, (T, (f, g)) in zip(cm, spectra.items()):
        axs[0].plot(f, g, color=c, label=f"{T} K")
    axs[0].set_xlabel("f (THz)")
    axs[0].set_ylabel("Li PDOS (area = 1)")
    axs[0].legend(fontsize=8)
    axs[1].errorbar([r["T"] for r in rows], [r["fc_THz"] for r in rows], [r["fc_se_THz"] for r in rows], marker="o", capsize=2)
    axs[1].set_xlabel("T (K)")
    axs[1].set_ylabel("Li band centre (THz)")
    fig.suptitle(f"{cfg['name']}: Li phonon band centre (VACF, classical MD)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pdos_Li.png"))
    plt.close(fig)
    print(f"\n[+] {outdir}/phonon_band_center.csv pdos_Li.png")
    print("Read: lower f_c = softer Li well. Trust only if 'at 2x window' ~ f_c and SE is small. Same protocol for every system.")


if __name__ == "__main__":
    main()
