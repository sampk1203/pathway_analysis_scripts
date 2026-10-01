#!/usr/bin/env python3
"""
03_van_hove_haven.py - Van Hove correlation functions, non-Gaussian parameter and Haven ratio for Li,
straight from the dumps. Framework-COM drift is removed first (same as 00/01).

Definitions (N = number of Li, V = box volume, r_i(t) = drift-corrected UNWRAPPED Li positions)
  self Van Hove     P_s(r,t)  = < delta(r - |r_i(t0+t) - r_i(t0)|) >_{i,t0}          (normalised: int P_s dr = 1)
                    = 4 pi r^2 G_s(r,t). Peaks at the jump lengths; a peak near r=0 that decays while a peak
                    at ~2 A grows = hopping between well-defined sites.
  distinct Van Hove G_d(r,t) = V/(N(N-1)) < sum_{i!=j} delta(r - |r_j(t0+t) - r_i(t0)|) > / (4 pi r^2)
                    (minimum image, box-averaged). At t=0 it is the Li-Li g(r); -> 1 at long t if uncorrelated.
                    Weight at r ~ 0 at t > 0 = "a Li (any Li) is now where Li i was". At LONG t this only reflects the
                    site density (G_d -> rho_site/rho_uniform, and the curves converge to the lattice geometry, not
                    to 1), so it is NOT evidence of concerted motion by itself. Look at SHORT lags (~ one hop time):
                    the nearest-neighbour peak (~2.5 A) decaying while weight at r ~ 0 grows faster than the self
                    part (P_s) can explain is the concerted / replacement signature; quantify it with the hop script.
  non-Gaussian      alpha2(t) = 3 <r^4> / (5 <r^2>^2) - 1     (0 = Gaussian / Fickian; peaks when hopping is heterogeneous)
  tracer D*         MSD_tr(t) = <|r_i(t0+t) - r_i(t0)|^2>            ,  D* = slope / 6
  charge  D_sigma   MSD_col(t) = < |sum_i [r_i(t0+t) - r_i(t0)]|^2 > ,  D_sigma = slope / (6 N)
  Haven ratio       H_R = D* / D_sigma        (H_R = 1: uncorrelated hops; < 1: correlated / concerted)
  conductivity      sigma = n e^2 D_sigma / (k T)   with n = N/V   (Nernst-Einstein value uses D* instead)

Uncertainty: D_sigma comes from ONE collective variable, so it is much noisier than D*. The reported value is the
whole-window one (lags up to --max-lag-frac of the window, MSD fit at t >= --fit-start * t_max). Error bars come from a
moving-block bootstrap over time origins (block length = longest lag by default, --nboot replicates): D*, D_sigma and
H_R are recomputed for every replicate and the bootstrap standard deviation is reported as the +/- error. H_R is the
ratio of the two values of the SAME replicate. If the bootstrap mean differs from the whole-window value by more than one
sd the bootstrap is biased (few independent blocks) and a warning is printed: treat the error as a rough scale only. Rough rule: the relative error of D_sigma is about
sqrt(2 t / (3 T_window)), i.e. ~40 % for a 100 ps window and 25 ps lags; only a longer run or more seeds reduce it.
Trust trends across temperature / composition only when they exceed these intervals.

usage:
  python 03_van_hove_haven.py /path/to/folder                    # all temperatures found in the folder
  python 03_van_hove_haven.py /path/to/folder --T 800 1200 --lags 1 5 10 25 50 --nboot 500
outputs (results/<system>/): vanhove_T<T>K.png, vanhove_T<T>K.npz, haven_summary.csv, haven_summary.png
"""
import argparse
import csv
import os

import numpy as np

import llzo_io as io

E_CHARGE = 1.602176634e-19
KB_J = 1.380649e-23


# ------------------------------------------------------------------ MSD-type quantities
def make_lags(F, max_frac=0.5, fit_start=0.3):
    lmax = max(4, int(F * max_frac))
    a = np.geomspace(1, lmax, 60)
    b = np.linspace(fit_start * lmax, lmax, 60)
    return np.unique(np.round(np.concatenate([a, b])).astype(int).clip(1, lmax))


def per_origin(pos, lags):
    """For every lag (frames): tracer |dr|^2 and |dr|^4 averaged over atoms, and collective |sum dr|^2, for EACH
    time origin. Returns lists of (F-l,) arrays. (Whole-window values = plain means; the bootstrap resamples them.)"""
    tr_o, m4_o, col_o = [], [], []
    for l in lags:
        d = pos[l:] - pos[:-l]                       # (F-l, N, 3)
        d2 = np.einsum("fni,fni->fn", d, d)
        tr_o.append(d2.mean(axis=1))
        m4_o.append((d2 ** 2).mean(axis=1))
        s = d.sum(axis=1)                            # total Li displacement in the framework frame
        col_o.append(np.einsum("fi,fi->f", s, s))
    return tr_o, m4_o, col_o


def loglog_slope(t, y):
    sel = t >= 0.2 * t[-1]
    return float(np.polyfit(np.log(t[sel]), np.log(y[sel]), 1)[0])


def fit_D(t, tr, col, N, fit_start):
    """D* and D_sigma in cm^2/s from MSD slopes over t >= fit_start * t_max (A^2/ps -> cm^2/s: 1e-4)."""
    sel = t >= fit_start * t[-1]
    k_tr = np.polyfit(t[sel], tr[sel], 1)[0]
    k_col = np.polyfit(t[sel], col[sel], 1)[0]
    return k_tr / 6.0 * 1e-4, k_col / (6.0 * N) * 1e-4


def diffusivities(pos, dt, fit_start=0.3, max_frac=0.5):
    """Whole-window D*, D_sigma, MSDs, alpha2 (all origins, all atoms) + per-origin arrays for the bootstrap."""
    F, N = pos.shape[:2]
    lags = make_lags(F, max_frac, fit_start)
    tr_o, m4_o, col_o = per_origin(pos, lags)
    tr = np.array([x.mean() for x in tr_o])
    m4 = np.array([x.mean() for x in m4_o])
    col = np.array([x.mean() for x in col_o])
    a2 = 3.0 * m4 / (5.0 * tr ** 2) - 1.0
    t = lags * dt
    Ds, Dc = fit_D(t, tr, col, N, fit_start)
    return dict(t=t, msd_tr=tr, msd_col=col, alpha2=a2, D_star=Ds, D_sigma=Dc, slope_tr=loglog_slope(t, tr),
                slope_col=loglog_slope(t, col), lags=lags, F=F, N=N, tr_o=tr_o, col_o=col_o)


def bootstrap(res, dt, fit_start, nboot, block_frames, seed=0):
    """Moving-block bootstrap over time origins. Every replicate resamples blocks of `block_frames` consecutive
    origins (same blocks for all lags), refits D* and D_sigma and forms H_R = D*/D_sigma from that replicate.
    Returns bootstrap sd and mean of each quantity and the fraction of replicates with D_sigma <= 0."""
    rng = np.random.default_rng(seed)
    F, N, t = res["F"], res["N"], res["t"]
    block_frames = int(max(1, min(block_frames, F)))
    nb = int(np.ceil(F / block_frames))
    Ds, Dc = [], []
    for _ in range(nboot):
        starts = rng.integers(0, F - block_frames + 1, nb)
        idx = (starts[:, None] + np.arange(block_frames)[None, :]).ravel()
        tr = np.array([o[idx[idx < len(o)]].mean() for o in res["tr_o"]])
        col = np.array([o[idx[idx < len(o)]].mean() for o in res["col_o"]])
        a, b = fit_D(t, tr, col, N, fit_start)
        Ds.append(a)
        Dc.append(b)
    Ds, Dc = np.array(Ds), np.array(Dc)
    H = np.where(Dc > 0, Ds / np.where(Dc > 0, Dc, 1.0), np.nan)
    sd = lambda x: float(np.nanstd(x, ddof=1)) if np.isfinite(x).sum() > 2 else np.nan
    mn = lambda x: float(np.nanmean(x)) if np.isfinite(x).any() else np.nan
    return dict(Ds=sd(Ds), Dc=sd(Dc), H=sd(H), Ds_mean=mn(Ds), Dc_mean=mn(Dc), H_mean=mn(H),
                frac_neg=float((Dc <= 0).mean()), n_blocks=nb, block_frames=block_frames)


# ------------------------------------------------------------------ Van Hove
def van_hove(pos, Lm, dt, lags_ps, dr, rmax, n_orig):
    F, N, _ = pos.shape
    edges = np.arange(0.0, rmax + 0.5 * dr, dr)
    centers = 0.5 * (edges[1:] + edges[:-1])
    shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    V = float(np.prod(Lm))
    out = dict(r=centers, lags_ps=[], Gs=[], Gd=[])

    def distinct(k):
        origins = np.arange(0, F - k, max(1, (F - k) // n_orig))
        cnt = np.zeros(len(centers))
        for t0 in origins:
            dv = pos[t0 + k][None, :, :] - pos[t0][:, None, :]
            dv -= Lm * np.rint(dv / Lm)
            r = np.linalg.norm(dv, axis=-1)
            r[np.diag_indices(N)] = np.inf
            cnt += np.histogram(r.ravel(), edges)[0]
        return cnt / (len(origins) * N * (N - 1) / V * shell)

    out["g0"] = distinct(0)                                        # Li-Li g(r) at t = 0
    for tl in lags_ps:
        k = int(round(tl / dt))
        if k < 1 or k > F // 2:
            print(f"  [skip] lag {tl} ps is outside 1..{F // 2 * dt:.1f} ps")
            continue
        d = np.linalg.norm(pos[k:] - pos[:-k], axis=-1)
        Gs = np.histogram(d.ravel(), edges)[0] / (d.size * dr)
        out["lags_ps"].append(k * dt)
        out["Gs"].append(Gs)
        out["Gd"].append(distinct(k))
    return out


def plot_vanhove(path, title, vh, res, T):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(11, 8), dpi=150)
    cols = plt.cm.viridis(np.linspace(0.05, 0.9, max(1, len(vh["lags_ps"]))))
    r = vh["r"]
    ax = axs[0, 0]
    for c, t, g in zip(cols, vh["lags_ps"], vh["Gs"]):
        ax.plot(r, g, color=c, label=f"{t:g} ps")
    ax.set_xlabel("r (A)")
    ax.set_ylabel("4 pi r^2 G_s(r,t)")
    ax.set_title("self Van Hove")
    ax.legend(fontsize=8)
    ax = axs[0, 1]
    ax.plot(r, vh["g0"], color="k", lw=1.2, label="t = 0 (Li-Li g(r))")
    for c, t, g in zip(cols, vh["lags_ps"], vh["Gd"]):
        ax.plot(r, g, color=c, label=f"{t:g} ps")
    ax.axhline(1, color="0.6", lw=0.6)
    ax.set_ylim(0, 3)                       # the first bins (tiny shell volume) are very noisy: clip them
    ax.set_xlabel("r (A)")
    ax.set_ylabel("G_d(r,t)")
    ax.set_title("distinct Van Hove (clipped at 3)")
    ax.legend(fontsize=8)
    ax = axs[1, 0]
    ax.loglog(res["t"], res["msd_tr"], label="tracer  <|dr|^2>")
    ax.loglog(res["t"], res["msd_col"] / vh["N"], label="collective / N")
    ax.set_xlabel("t (ps)")
    ax.set_ylabel("MSD (A^2)")
    ax.set_title(f"MSD: H_R = D*/D_sigma = {res['D_star'] / res['D_sigma']:.2f}" if res["D_sigma"] > 0
                 else "MSD (D_sigma <= 0: collective MSD too noisy)")
    ax.legend(fontsize=8)
    ax = axs[1, 1]
    ax.semilogx(res["t"], res["alpha2"], color="C3")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlabel("t (ps)")
    ax.set_ylabel("alpha_2(t)")
    ax.set_title("non-Gaussian parameter")
    fig.suptitle(f"{title}   T = {T} K")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ main
def arrhenius(T, y, yerr=None):
    x = 1.0 / (io.KB_EV * np.asarray(T, float))
    ly = np.log(y)
    p, cov = np.polyfit(x, ly, 1, cov=True) if len(T) > 2 else (np.polyfit(x, ly, 1), np.full((2, 2), np.nan))
    return -p[0], float(np.sqrt(cov[0, 0])), p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="folder with the dumps + element_list (or a settings .yaml)")
    ap.add_argument("--defaults", help="settings file (default: analysis_defaults.yaml)")
    ap.add_argument("--T", type=int, nargs="*", help="temperatures (default: all)")
    ap.add_argument("--dump", help="explicit dump file (only with a single --T)")
    ap.add_argument("--tmin", type=float, help="override analysis-window start (ps)")
    ap.add_argument("--lags", type=float, nargs="*", default=[1, 2, 5, 10, 20, 50],
                    help="Van Hove lag times, ps (default 1 2 5 10 20 50; must be <= half the window)")
    ap.add_argument("--dr", type=float, default=0.05, help="Van Hove bin width, A")
    ap.add_argument("--n-orig", type=int, default=200, help="max time origins for the distinct part")
    ap.add_argument("--nboot", type=int, default=300, help="bootstrap replicates for the D*, D_sigma, H_R intervals")
    ap.add_argument("--boot-block-ps", type=float, help="bootstrap block length, ps (default: longest MSD lag)")
    ap.add_argument("--fit-start", type=float, default=0.3, help="MSD fit uses t >= fit_start * t_max")
    ap.add_argument("--max-lag-frac", type=float, default=0.25,
                    help="longest MSD lag as a fraction of the window (default 0.25). Longer lags have few "
                         "independent origins, so D_sigma gets very noisy; shorter lags need the motion to be "
                         "already diffusive (check the MSD log-log slope printed below).")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    cfg = io.load_config(args.target, args.defaults)
    temps = args.T or io.temperatures(cfg)
    if args.dump and len(temps) != 1:
        ap.error("--dump needs exactly one --T")
    outdir = os.path.join(args.outdir, cfg["name"])
    os.makedirs(outdir, exist_ok=True)
    rows = []

    for T in temps:
        path = args.dump or io.dump_path(cfg, T)
        print(f"\n=== {cfg['name']}  T = {T} K  ({os.path.basename(path)}) ===")
        traj = io.prepare(cfg, path, args.tmin)
        lm = io.li_mask(cfg, traj)
        drift = io.framework_drift(cfg, traj)
        li = traj.pos[:, lm, :] - drift[:, None, :]
        N = li.shape[1]
        dt = float(np.median(np.diff(traj.time_ps)))
        Lm = traj.L.mean(axis=0)
        V = float(np.prod(Lm))
        window = traj.time_ps[-1] - traj.time_ps[0]

        res = diffusivities(li, dt, args.fit_start, args.max_lag_frac)
        block = int(round(args.boot_block_ps / dt)) if args.boot_block_ps else int(res["lags"][-1])
        bs = bootstrap(res, dt, args.fit_start, args.nboot, block)
        Ds, Dc = res["D_star"], res["D_sigma"]
        H = Ds / Dc if Dc > 0 else float("nan")
        print(f"  N_Li = {N}   window = {window:.0f} ps   MSD lags up to {res['t'][-1]:.1f} ps, "
              f"fit t >= {args.fit_start * res['t'][-1]:.1f} ps")
        print(f"  D*      = {Ds:.3e} +/- {bs['Ds']:.1e} cm2/s   (bootstrap sd)")
        print(f"  D_sigma = {Dc:.3e} +/- {bs['Dc']:.1e} cm2/s")
        print(f"  H_R     = {H:.2f} +/- {bs['H']:.2f}   ({bs['n_blocks']} bootstrap blocks of {block * dt:.1f} ps)")
        for nm, est, bm, sdv in (("D*", Ds, bs["Ds_mean"], bs["Ds"]), ("D_sigma", Dc, bs["Dc_mean"], bs["Dc"]),
                                 ("H_R", H, bs["H_mean"], bs["H"])):
            if np.isfinite(sdv) and np.isfinite(bm) and abs(bm - est) > sdv:
                print(f"  [WARN] bootstrap mean of {nm} ({bm:.3g}) differs from the estimate ({est:.3g}) by more than "
                      f"one sd: few independent blocks, treat the +/- as a rough scale")
        if bs["n_blocks"] < 4:
            print("  [WARN] fewer than 4 bootstrap blocks: intervals are unreliable (window too short for these lags)")
        if bs["frac_neg"] > 0.05:
            print(f"  [WARN] D_sigma <= 0 in {100 * bs['frac_neg']:.0f} % of bootstrap replicates: H_R not resolvable")
        for name, s in (("tracer", res["slope_tr"]), ("collective", res["slope_col"])):
            if abs(s - 1) > 0.25:
                print(f"  [WARN] {name} MSD log-log slope = {s:.2f} (diffusive regime ~1): D from this window is "
                      f"not converged")
        if Dc <= 0:
            print("  [WARN] D_sigma <= 0: collective MSD slope not resolvable in this window")

        rmax = min(6.0, float(Lm.min()) / 2 - 1e-6)
        vh = van_hove(li, Lm, dt, args.lags, args.dr, rmax, args.n_orig)
        vh["N"] = N
        tag = f"T{T}K"
        plot_vanhove(os.path.join(outdir, f"vanhove_{tag}.png"), cfg["name"], vh, res, T)
        np.savez_compressed(os.path.join(outdir, f"vanhove_{tag}.npz"), r=vh["r"], g0=vh["g0"],
                            lags_ps=np.array(vh["lags_ps"]), Gs=np.array(vh["Gs"]), Gd=np.array(vh["Gd"]),
                            t=res["t"], msd_tr=res["msd_tr"], msd_col=res["msd_col"], alpha2=res["alpha2"])

        n_m3 = N / V * 1e30
        to_sigma = lambda D: n_m3 * E_CHARGE ** 2 * (D * 1e-4) / (KB_J * T) / 100.0     # S/cm
        rows.append(dict(system=cfg["name"], T=T, N_Li=N, window_ps=window, D_tracer_cm2s=Ds, D_tracer_sd=bs["Ds"],
                         D_sigma_cm2s=Dc, D_sigma_sd=bs["Dc"], H_R=H, H_R_sd=bs["H"], H_R_boot_mean=bs["H_mean"],
                         boot_frac_Dsigma_neg=bs["frac_neg"], sigma_from_Dsigma_S_cm=to_sigma(Dc),
                         sigma_NE_from_Dtracer_S_cm=to_sigma(Ds), slope_tracer=res["slope_tr"],
                         slope_collective=res["slope_col"], alpha2_max=float(res["alpha2"].max())))

    with open(os.path.join(outdir, "haven_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[+] {os.path.join(outdir, 'haven_summary.csv')}")

    # ---- across temperatures
    Tarr = np.array([r["T"] for r in rows])
    good = np.array([r["D_sigma_cm2s"] > 0 and r["D_tracer_cm2s"] > 0 for r in rows])
    if good.sum() >= 3:
        Ea_s, e_s, ps = arrhenius(Tarr[good], np.array([r["D_tracer_cm2s"] for r in rows])[good])
        Ea_c, e_c, pc = arrhenius(Tarr[good], np.array([r["D_sigma_cm2s"] for r in rows])[good])
        Ea_sig, e_sig, _ = arrhenius(Tarr[good], np.array([r["sigma_from_Dsigma_S_cm"] * r["T"] for r in rows])[good])
        print(f"  Ea(D*) = {Ea_s:.3f} +/- {e_s:.3f} eV   Ea(D_sigma) = {Ea_c:.3f} +/- {e_c:.3f} eV   "
              f"Ea(sigma T) = {Ea_sig:.3f} +/- {e_sig:.3f} eV   (regression error only)")
    Hs = np.array([r["H_R"] for r in rows])
    Hs = Hs[np.isfinite(Hs)]
    if len(Hs) >= 3:
        print(f"  H_R over temperatures: mean {Hs.mean():.2f}, sd {Hs.std(ddof=1):.2f}, s.e. {Hs.std(ddof=1) / np.sqrt(len(Hs)):.2f}"
              f"   (only meaningful if H_R is T-independent; the runs are independent trajectories)")
    if len(rows) >= 2:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
        x = 1000.0 / Tarr
        ax = axs[0]
        ax.errorbar(x, [r["D_tracer_cm2s"] for r in rows], [r["D_tracer_sd"] for r in rows], fmt="o", color="C0",
                    label="D* (whole window +/- bootstrap sd)")
        ax.errorbar(x, [max(r["D_sigma_cm2s"], 1e-12) for r in rows], [r["D_sigma_sd"] for r in rows], fmt="s",
                    color="C1", label="D_sigma (whole window +/- bootstrap sd)")
        ax.set_yscale("log")
        ax.set_xlabel("1000 / T (1/K)")
        ax.set_ylabel("D (cm2/s)")
        ax.legend(fontsize=8)
        ax = axs[1]
        ax.errorbar(Tarr, [r["H_R"] for r in rows], [r["H_R_sd"] for r in rows], fmt="o", color="C2",
                    label="H_R whole window +/- bootstrap sd")
        ax.axhline(1, color="0.6", lw=0.6)
        ax.set_xlabel("T (K)")
        ax.set_ylabel("Haven ratio D*/D_sigma")
        ax.legend(fontsize=8)
        fig.suptitle(cfg["name"])
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "haven_summary.png"))
        plt.close(fig)
        print(f"[+] {os.path.join(outdir, 'haven_summary.png')}")


if __name__ == "__main__":
    main()
