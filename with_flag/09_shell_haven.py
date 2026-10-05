#!/usr/bin/env python3
"""
09_shell_haven.py - Haven ratio per distance shell around the dopant (0-3, 3-5, 5-7, >7 A by default). Dumps only, NO sites.

Same shells, 'any'/per-dopant references, far definition and block-jackknife errors as 05 (llzo_io.py).
For every time origin t0 each Li is put in a shell by its distance to the dopant at t0 (the START shell, as in 05). For one
shell S (the Li that are in S at t0) and one lag tau:
    tracer      MSD_tr  = < sum_{i in S} |dr_i|^2 >_t0                          dr_i = r_i(t0+tau) - r_i(t0)
    collective  MSD_col = < | sum_{i in S} dr_i |^2 >_t0                        (the SAME Li, summed vector first)
    H_R(S, tau) = MSD_tr / MSD_col  =  D*_S / D_sigma,S                          (the N_S in D_sigma = MSD_col / (6 N_S t) cancels)
H_R = 1: the Li of the shell move independently; < 1: their displacements are positively correlated (concerted / knock-on).
Also computed for ALL Li together with the same tau (grey line in the figure) - the reference to compare the shells with.
Error bars: delete-one-block jackknife over --blocks time blocks (numerator and denominator are additive sums per block).

READ THIS BEFORE TRUSTING A NUMBER
 * It is an INTRA-shell Haven ratio: correlations between a Li inside the shell and a Li in another shell are not in it. It says
   "do the Li that start in this shell move together", not "what is the Haven ratio of the material there". (A true local Haven
   ratio does not exist: sigma is a property of the whole sample.)
 * A shell with n Li on average: for n ~ 1 the sum is one vector, so MSD_col ~ MSD_tr and H_R -> 1 by construction. Cells with a
   mean of fewer than --min-n Li are drawn as OPEN symbols; do not read a number from them. Ga 0-3 A holds ~0-3 Li.
 * H_R(tau) is a ratio at ONE lag, not a slope fit as in 03. At short tau (a few vibrational periods) both MSDs are the same
   vibration, so H_R -> 1; the value converges (if it does) only for tau >> the hop time. Compare shells at the same tau and look at
   the largest tau. At large tau the Li have left the shell they started in (the shell is the START shell), mixing the shells.
 * D_sigma is one collective variable and is much noisier than D*: with 5 blocks the SE itself is only a rough scale.
 * Li are removed from a shell only by moving; there is no exclusion of Li near the box edge beyond the minimum image.

usage:
  python 09_shell_haven.py /path/to/folder [--T 800 1200] [--lags 2 5 10 20] [--shells 3 5 7] [--far all|self]
  (same system flags as the other scripts: --elements --dopants --dump-pattern ...)
outputs (<outdir>/<system>/): haven_shell.csv, haven_shell.png (+ .svg copies)
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


def shell_sums(d, sh, K, nblocks):
    """d (F-lag,N,3) displacements, sh (F-lag,N) start shell (-1 excluded; K-th extra column = ALL Li).
    Returns per-block sums NUM (nb,K+1) = sum |dr|^2, DEN (nb,K+1) = sum |sum dr|^2, CNT (nb,K+1) = Li-origins, ORG (nb,) origins."""
    Fo = d.shape[0]
    d2 = np.einsum("fni,fni->fn", d, d)
    blk = io.block_id(np.arange(Fo), Fo, nblocks)
    NUM = np.zeros((nblocks, K + 1))
    DEN = np.zeros((nblocks, K + 1))
    CNT = np.zeros((nblocks, K + 1))
    for k in range(K + 1):
        m = (sh == k) if k < K else np.ones(sh.shape, bool)
        s = np.einsum("fni,fn->fi", d, m.astype(float))
        num_o = (d2 * m).sum(axis=1)
        den_o = np.einsum("fi,fi->f", s, s)
        cnt_o = m.sum(axis=1).astype(float)
        NUM[:, k] = np.bincount(blk, weights=num_o, minlength=nblocks)
        DEN[:, k] = np.bincount(blk, weights=den_o, minlength=nblocks)
        CNT[:, k] = np.bincount(blk, weights=cnt_o, minlength=nblocks)
    ORG = np.bincount(blk, minlength=nblocks).astype(float)
    return NUM, DEN, CNT, ORG


def plot(rows, lags, names, path, min_n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    labs = list(dict.fromkeys(r["dopant"] for r in rows))
    K = len(names)
    cols = [plt.cm.viridis(k / max(K - 2, 1)) for k in range(K - 1)] + ["k"]
    fig, axs = plt.subplots(len(labs), len(lags), figsize=(4.4 * len(lags), 3.3 * len(labs) + 1.2), dpi=150, squeeze=False,
                            sharex=True, sharey=True)
    for ri, lab in enumerate(labs):
        for ci, tau in enumerate(lags):
            a = axs[ri][ci]
            for k in range(K):
                sel = sorted([r for r in rows if r["dopant"] == lab and r["shell"] == names[k] and abs(r["tau_ps"] - tau) < 0.06],
                             key=lambda r: r["T"])
                if not sel:
                    continue
                a.errorbar([r["T"] for r in sel], [r["H_R"] for r in sel], [r["H_R_se"] for r in sel], marker="o", ms=4, capsize=2,
                           color=cols[k], ls="--" if k == K - 1 else "-")
                lo = [r for r in sel if r["n_Li_mean"] < min_n]
                if lo:
                    a.plot([r["T"] for r in lo], [r["H_R"] for r in lo], ls="", marker="o", ms=5, mfc="white", mec=cols[k], zorder=5)
            allr = sorted([r for r in rows if r["dopant"] == lab and r["shell"] == "all Li" and abs(r["tau_ps"] - tau) < 0.06],
                          key=lambda r: r["T"])
            if allr:
                a.plot([r["T"] for r in allr], [r["H_R"] for r in allr], color="C3", lw=1.2, ls=":", marker="x", ms=4)
            a.axhline(1, color="0.5", lw=0.8)
            a.set_title(f"{lab},  tau = {tau:g} ps", fontsize=10)
            if ri == len(labs) - 1:
                a.set_xlabel("T (K)")
            if ci == 0:
                a.set_ylabel("Haven ratio  MSD_tr / MSD_col  (Li of the shell)", fontsize=8)
    h = []
    for k in range(K):
        n = np.nanmean([r["n_Li_mean"] for r in rows if r["shell"] == names[k]])
        h.append(Line2D([], [], color=cols[k], marker="o", lw=2, ls="--" if k == K - 1 else "-",
                        label=f"Li starting {names[k]} from dopant (mean {n:.1f} Li)" + (" (far)" if k == K - 1 else "")))
    h.append(Line2D([], [], color="C3", ls=":", marker="x", label="all Li together (same tau)"))
    h.append(Line2D([], [], color="0.4", marker="o", mfc="white", ls="", label=f"open symbol: < {min_n:g} Li in the shell on average (H_R -> 1 by construction)"))
    fig.legend(handles=h, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle("Intra-shell Haven ratio by distance to the dopant (1 = Li move independently, < 1 = correlated; error bars = block jackknife)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="folder with the dumps + element_list (or a settings .yaml)")
    ap.add_argument("--defaults")
    ap.add_argument("--T", type=int, nargs="*")
    ap.add_argument("--tmin", type=float)
    ap.add_argument("--outdir", default="haven09")
    ap.add_argument("--shells", type=float, nargs="*", default=list(io.DEFAULT_SHELLS), help="shell edges, A (same default in 04-07)")
    ap.add_argument("--far", choices=["all", "self"], default="all")
    ap.add_argument("--lags", type=float, nargs="*", default=[2.0, 5.0, 10.0], help="tau values, ps")
    ap.add_argument("--blocks", type=int, default=io.DEFAULT_BLOCKS)
    ap.add_argument("--min-n", type=float, default=3.0, help="mean Li in a shell below this: open symbol / flagged in the log")
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
        F, N = li.shape[:2]
        Lf = np.broadcast_to(L, (F, 3))
        dists = {lab: io.dist_to(li, dop, Lf) for lab, dop in dops}
        dnear = np.min(list(dists.values()), axis=0)
        dists = {"any": dnear, **dists}
        for tau in args.lags:
            lag = max(1, int(round(tau / dt)))
            if lag >= F - args.blocks * 2:
                print(f"  [skip] tau = {tau} ps is too long for this window")
                continue
            d = li[lag:] - li[:-lag]
            for lab, d_ in dists.items():
                sh = io.shell_labels(d_, dnear, edges, args.far)[:-lag]
                NUM, DEN, CNT, ORG = shell_sums(d, sh, K, args.blocks)
                H, Hse = io.jackknife(lambda n_, d_s: n_ / d_s, NUM, DEN)
                nmean = CNT.sum(0) / ORG.sum()
                print(f"  [{lab}] tau = {lag * dt:g} ps    shell      <Li>    H_R +/- jackknife SE")
                for k in range(K + 1):
                    nm = names[k] if k < K else "all Li"
                    flag = "  <- few Li: H_R -> 1 by construction" if (k < K and nmean[k] < args.min_n) else ""
                    print(f"        {nm:>16} {nmean[k]:8.2f}    {H[k]:6.2f} +/- {Hse[k]:.2f}{flag}")
                    rows.append(dict(T=T, dopant=lab, tau_ps=lag * dt, shell=nm, n_Li_mean=float(nmean[k]), H_R=float(H[k]),
                                     H_R_se=float(Hse[k]), msd_tr_A2=float(NUM[:, k].sum() / max(CNT[:, k].sum(), 1)),
                                     msd_col_per_Li_A2=float(DEN[:, k].sum() / max(CNT[:, k].sum(), 1)), blocks=args.blocks))
    if not rows:
        raise SystemExit("nothing computed")
    with open(os.path.join(outdir, "haven_shell.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    lags = [t for t in args.lags if any(abs(r["tau_ps"] - t) < 0.06 for r in rows)]
    plot(rows, lags, names, os.path.join(outdir, "haven_shell.png"), args.min_n)
    print(f"\n[+] {outdir}/haven_shell.csv haven_shell.png haven_shell.svg")
    print("Read: H_R(shell) < 1 = the Li that start in this shell move in a correlated way. INTRA-shell only. Open symbols / '<- few Li' = "
          "the number is not usable. Compare with the red 'all Li' line, and use the largest tau; at short tau every H_R -> 1.")


if __name__ == "__main__":
    main()
