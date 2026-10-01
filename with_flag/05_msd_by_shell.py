#!/usr/bin/env python3
"""
05_msd_by_shell.py - does a dopant speed up or slow down nearby Li?  NO sites, NO hop definition.

For every Li at every time origin t0 the distance d0 to the dopant (minimum image, dopant position at t0) puts it into
a shell (--shells, default 3 5 7 A -> 0-3, 3-5, 5-7, >7; shells, far reference and error bars are the same in 04, 05, 06:
llzo_io.py).  Reference = each dopant alone and 'any' (nearest dopant); far = >7 A from EVERY dopant (--far self: from that
dopant only). Then
    MSD_shell(tau) = < |r(t0 + tau) - r(t0)|^2 >   over all Li and origins of that shell
(unwrapped positions, framework-COM drift removed, same as 00-04). The number to read is
    ratio(tau) = MSD_shell(tau) / MSD_far(tau)          < 1 : Li starting near the dopant move less (dopant slows)
                                                        > 1 : they move more
where 'far' is the outermost shell. Also printed: density ratio = mean Li count in the shell / count expected for a
uniform Li density in the same region (< 1 : Li depleted around the dopant; volumes from a grid, so overlap of shells of
several dopants and the far exclusion are handled).

Why this settles the site-based conflict: 04 counts hops between sites; near a dopant the ideal sites fit badly
(rattling counted as hops) and peak sites have very few hops. MSD needs neither.

Error bars: time origins are split into --blocks contiguous blocks; +/- = delete-one-block jackknife (MSD, the ratio to far
and the density ratio are each jackknifed directly). It does not include the small number of Li in the inner shell (n_Li is
printed: read the ratio with that in mind).

Limits: a shell is defined by the START distance; Li leave the shell during long tau, so large tau mixes shells (use
tau of a few ps). Local structure differs between regions, so a ratio far from 1 can also be geometry, not the
dopant (no control region is computed).

usage:
  python 05_msd_by_shell.py /path/to/folder
  python 05_msd_by_shell.py /path/to/folder --T 800 1200 --lags 1 5 10 --shells 3 5 7
outputs (results/<s>/): msd_shell.csv, msd_shell_ratio.png
"""
import argparse
import csv
import os

import numpy as np

import llzo_io as io


def load(cfg, T, tmin):
    traj = io.prepare(cfg, io.dump_path(cfg, T), tmin)
    drift = io.framework_drift(cfg, traj)
    lm = io.li_mask(cfg, traj)
    li = traj.pos[:, lm, :] - drift[:, None, :]
    dops, cnt = [], {}
    for t in cfg["dopant_types"]:
        for i in np.where(traj.types == t)[0]:
            el = cfg["types"][t]
            cnt[el] = cnt.get(el, 0) + 1
            dops.append((f"{el}{cnt[el]}", traj.pos[:, i, :] - drift))
    return li, dops, traj.L.mean(axis=0), float(np.median(np.diff(traj.time_ps)))


def shell_msd(li, sh, K, lag_frames, nblocks):
    """li (F,N,3) unwrapped; sh (F,N) shell of every Li in every frame (-1 = excluded). Returns {lag: (msd (K,), se, ratio to
    far (K,), se)} with delete-one-block jackknife errors over the time origins (NaN when the lag leaves too few origins)."""
    K_ = K
    F = li.shape[0]
    out = {}
    for lag in lag_frames:
        if lag >= F - nblocks:
            nan = np.full(K_, np.nan)
            out[lag] = (nan, nan, nan, nan)
            continue
        d2 = ((li[lag:] - li[:-lag]) ** 2).sum(axis=-1)    # (F-lag,N)
        lab = sh[:-lag]
        S = io.block_counts(lab, K_, nblocks, values=d2)
        C = io.block_counts(lab, K_, nblocks)
        msd, se = io.jackknife(lambda s_, c_: s_ / c_, S, C)
        rat, rse = io.jackknife(lambda s_, c_: (s_ / c_) / (s_ / c_)[-1], S, C)
        out[lag] = (msd, se, rat, rse)
    return out


def plot_ratio(rows, lags, names, path):
    """One panel per (dopant, tau): x = T, one line per shell (colour), y = MSD(shell)/MSD(far). One shared legend that
    also gives the mean number of Li in each shell (small n = noisy)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labs = sorted({r["dopant"] for r in rows})
    K = len(names)
    cols = [plt.cm.viridis(k / max(K - 2, 1)) for k in range(K - 1)]
    fig, axs = plt.subplots(len(labs), len(lags), figsize=(4.2 * len(lags), 3.3 * len(labs) + 0.9), dpi=150,
                            squeeze=False, sharex=True, sharey=True)      # ONE y axis: panels comparable by eye
    for ri, lab in enumerate(labs):
        for ci, tau in enumerate(lags):
            a = axs[ri][ci]
            for k in range(K - 1):
                sel = [r for r in rows if r["dopant"] == lab and r["shell"] == names[k] and abs(r["tau_ps"] - tau) < 0.06]
                if sel:
                    a.errorbar([r["T"] for r in sel], [r["ratio_to_far"] for r in sel], [r["ratio_se"] for r in sel],
                               marker="o", ms=4, capsize=2, color=cols[k])
            a.axhline(1, color="0.5", lw=0.8)
            a.set_title(f"{lab},  tau = {tau:g} ps", fontsize=10)
            if ri == len(labs) - 1:
                a.set_xlabel("T (K)")
            if ci == 0:
                a.set_ylabel(f"MSD(shell) / MSD({names[-1]})")
    from matplotlib.lines import Line2D
    h = []
    for k in range(K - 1):
        n = np.mean([r["n_Li"] for r in rows if r["shell"] == names[k]])
        h.append(Line2D([], [], color=cols[k], marker="o", lw=2, label=f"Li starting {names[k]} from dopant (mean {n:.1f} Li)"))
    h.append(Line2D([], [], color="0.5", lw=1, label=f"1 = same as far Li ({names[-1]})"))
    fig.legend(handles=h, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle("Li mobility near each dopant relative to far Li  (< 1 = slower near the dopant; error bars = block jackknife)", fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="folder with the dumps + element_list (or a settings .yaml)")
    ap.add_argument("--defaults")
    ap.add_argument("--T", type=int, nargs="*")
    ap.add_argument("--tmin", type=float)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--shells", type=float, nargs="*", default=list(io.DEFAULT_SHELLS), help="shell edges, A (same default in 04, 05, 06)")
    ap.add_argument("--far", choices=["all", "self"], default="all",
                    help="far shell = farther than the last edge from EVERY dopant (all) or from that dopant only (self)")
    ap.add_argument("--lags", type=float, nargs="*", default=[1.0, 5.0, 10.0], help="tau values, ps")
    ap.add_argument("--blocks", type=int, default=io.DEFAULT_BLOCKS, help="time blocks for the jackknife error bars")
    io.add_config_args(ap)
    args = ap.parse_args()

    cfg = io.load_config(args.target, args.defaults, args)
    temps = args.T or io.temperatures(cfg)
    outdir = os.path.join(args.outdir, cfg["name"])
    os.makedirs(outdir, exist_ok=True)
    edges = np.array(args.shells, float)
    names = io.shell_names(edges)
    K = len(names)

    rows = []
    for T in temps:
        print(f"\n=== {cfg['name']}  T = {T} K ===")
        li, dops, L, dt = load(cfg, T, args.tmin)
        if not dops:
            raise SystemExit("no dopant atoms in this system: nothing to compare")
        lag_frames = [max(1, int(round(x / dt))) for x in args.lags]
        F, N = li.shape[:2]
        Lf = np.broadcast_to(L, (F, 3))
        dists = {lab: io.dist_to(li, dop, Lf) for lab, dop in dops}
        dnear = np.min(list(dists.values()), axis=0)
        dists = {"any": dnear, **dists}
        vf = io.shell_volume_fractions({lab: dop.mean(axis=0) for lab, dop in dops}, L, edges, args.far)
        frb = np.bincount(io.block_id(np.arange(F), F, args.blocks), minlength=args.blocks)[:, None].astype(float)
        for lab, d_ in dists.items():
            sh = io.shell_labels(d_, dnear, edges, args.far)
            res = shell_msd(li, sh, K, lag_frames, args.blocks)
            C = io.block_counts(sh, K, args.blocks)
            n_mean = C.sum(0) / F
            expect = N * vf[lab]
            dens, dens_se = io.jackknife(lambda c_, f_: c_ / f_ / expect, C, frb)
            print(f"  {lab}: Li per shell (mean) and density ratio to uniform (+/- jackknife SE)")
            print("    " + "   ".join(f"{names[k]}: {n_mean[k]:.2f} ({dens[k]:.2f} +/- {dens_se[k]:.2f}x)" for k in range(K)))
            for lag, lf in zip(args.lags, lag_frames):
                msd, se, rat, rse = res[lf]
                print(f"    tau = {lf * dt:g} ps   MSD (A^2) +/- SE   ratio to far +/- SE")
                for k in range(K):
                    print(f"      {names[k]:>8}   {msd[k]:8.3f} +/- {se[k]:.3f}   "
                          + ("     1 (reference)" if k == K - 1 else f"{rat[k]:5.2f} +/- {rse[k]:.2f}"))
                    rows.append(dict(T=T, dopant=lab, tau_ps=lf * dt, shell=names[k], n_Li=n_mean[k], density_ratio=dens[k],
                                     density_ratio_se=dens_se[k], msd_A2=msd[k], msd_se=se[k], ratio_to_far=rat[k],
                                     ratio_se=rse[k], blocks=args.blocks))
    with open(os.path.join(outdir, "msd_shell.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    plot_ratio(rows, args.lags, names, os.path.join(outdir, "msd_shell_ratio.png"))
    print(f"\n[+] {outdir}/msd_shell.csv msd_shell_ratio.png")
    print("Read: ratio < 1 = Li near the dopant move less than far Li; > 1 = more. Check n_Li (few Li -> noisy) and "
          "whether the sign is the same at every tau and T.")


if __name__ == "__main__":
    main()
