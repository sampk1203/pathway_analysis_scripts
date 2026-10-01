#!/usr/bin/env python3
"""
06_li_econ.py - Li-O effective coordination number (ECoN) near the dopants.  Site-free (part A) + site-resolved (part B).

ECoN (Hoppe): for each Li, w_i = exp[1 - (l_i / l_av)^6] over the O neighbours, l_av = sum(l_i w_i) / sum(w_i) iterated to
convergence, ECoN = sum(w_i).  No cutoff to tune: a regular tetrahedron gives 4.0, one long bond gives ~3.6-3.8.  Also
reported: l_av (weighted mean Li-O bond, A) and the nearest Li-O distance.  Per Li, per frame, minimum image, O within --rmax.

PART A  (dumps only)  - does the dopant change the local oxygen environment of nearby Li?  NO sites, NO hop definition.
  Li-frames are grouped by their distance to the dopant in that frame (--shells, default 3 5 7 A -> 0-3, 3-5, 5-7, >7;
  shells, far reference and error bars are defined ONCE in llzo_io.py and are the same in 04, 05 and 06).
  Shown: shell mean minus far mean (dECoN, dl_av).  --far all (default): 'far' = farther than the last edge from EVERY
  dopant (cleanest reference); --far self: only from that dopant (same definition as 05).  dopant 'any' = nearest dopant.
  Error bars: delete-one-block jackknife over --blocks contiguous time blocks (the difference to far is jackknifed
  directly). It does not include the small number of Li in the inner shell: read n_Li.
  outputs: econ_shell.csv, econ_shell.png (dECoN and dl_av vs T), econ_hist.png (ECoN distributions at --hist-T)

PART A2 (dumps only, own detector, NOTHING from 04)  - site-free hops: back-hop %, hop rate / far, non-back rate / far.
  Anchor detector (see free_hops): coarse-grained Li position (--hop-win-ps), running-mean anchor per dwell, hop when the Li
  stays > --hop-d A from its anchor for --hop-confirm windows; back-hop = new anchor within --hop-d of the previous dwell's
  anchor.  Shell methods: free_at_hop (Li itself in the hop frame) and free_left (anchor the Li leaves).  Same shells, far
  reference, rates and jackknife as 04 (llzo_io.hop_shell_rows) but a different hop definition = consistency check.
  outputs: free_hop_shell_stats.csv, free_back_hop_shell.png, free_hop_rate_ratio_shell.png, free_nonback_ratio_shell.png,
           hop_dip.csv, econ_hopdip.png (ECoN dip during the site-free hops), consistency_04_vs_06.png (only with --hops-dir:
           reads 04's hop_shell_stats.csv AFTER the fact and overlays it; no 04 number enters any 06 number)

PART A3 (dumps only) - EVERY species pair (centre X -> neighbour Y), --pair-stride frames:
  ECoN, l_av, d_min (ECoN with host neighbours Li/La/Zr/O) and partial g(r) (all neighbours incl. dopants).  Centres: each
  host element split into the usual distance shells around each dopant, and each dopant atom on its own.
  outputs: pair_env_shell.csv, pair_env_dopant.csv, pair_rdf.csv, pair_dECoN_<ref>.png, pair_dlav_<ref>.png,
           pair_ddmin_<ref>.png, pair_rdf_<ref>_T<hist-T>K.png, pair_dopant_env.png, pair_rdf_dopant.png

PART B  (needs --hops-dir = the 04 output folder, e.g. hops_cif/<system>; same --tmin and --T as the 04 run)
  Uses the committed site of every Li in every frame from 04's state_T<T>K.npz (written by the current 04; includes the Li
  that never hop).  With an older 04 only hops_T<T>K.csv exists: then the site is known only from a Li's first hop on, Li
  that never hop are left out, and the far-frame reference is biased towards Li that hop.  The hop list is still used for
  the hop dip and the dwell times.
  1. characteristic ECoN of each site type (far frames = pristine reference)          -> econ_by_kind.csv
  2. 'off-site-like' fraction: frames whose ECoN is > --zcut sd from the far-frame value of the site type the Li is
     assigned to, per shell.  Baseline = the far shell.  A larger fraction near a dopant means the assigned site does
     not describe the Li's environment there (site assignment unreliable)                 -> econ_offsite.csv/.png
  3. (moved to part A2, uses the site-free hops) hop 'dip': ECoN just before / after a hop vs its minimum during the hop (window --dip-w frames).  A real hop
     between sites passes through a lower-ECoN transit state (dip > 0); flicker of a Li sitting between two sites
     shows little dip.  Median +/- jackknife SE per shell (Li distance to the nearest dopant in the hop frame);
     right panel = back-hop %                                                             -> hop_dip.csv, econ_hopdip.png
  3b. (moved) hop statistics per shell are now computed site-free in part A2; 04 keeps its own site-based version.
  4. link to dynamics: mean ECoN of a site vs the barrier out of it (barriers_T*.csv from 04), and ECoN during a dwell vs
     the dwell time, per site type (Spearman)                                            -> site_econ_T*.csv, dwell_econ.csv, econ_link.png

Limits: ECoN values at 800-1200 K are broad, read differences between shells and temperatures, not absolute numbers.
The dip and off-site fractions are heuristics: only compare them with the far-shell baseline.  A local-environment
difference is a candidate explanation for a slowdown, not proof of it.  The site state comes from 04's committed hops
(min-residence); the first dwell of each Li (censored by the window start) is excluded from the dwell statistics.

usage:
  python 06_li_econ.py /path/to/folder
  python 06_li_econ.py /path/to/folder --hops-dir hops_cif/MD_run_1Ga_2Ru --T 800 1200
outputs (results/<s>/): see above
"""
import argparse
import csv
import os
import warnings

import numpy as np

import llzo_io as io

KIND_COL = {"24d": "#1f77b4", "96h": "#ff7f0e"}


# ---------------------------------------------------------------- loading
def load(cfg, T, tmin):
    traj = io.prepare(cfg, io.dump_path(cfg, T), tmin)
    drift = io.framework_drift(cfg, traj)
    lm = io.li_mask(cfg, traj)
    o_types = [t for t, el in cfg["types"].items() if el == "O"]
    if not o_types:
        raise SystemExit(f"no oxygen type in cfg['types'] = {cfg['types']}")
    om = np.isin(traj.types, o_types)
    li = traj.pos[:, lm, :] - drift[:, None, :]
    ox = traj.pos[:, om, :] - drift[:, None, :]
    dops, cnt = [], {}
    for t in cfg["dopant_types"]:
        for i in np.where(traj.types == t)[0]:
            el = cfg["types"][t]
            cnt[el] = cnt.get(el, 0) + 1
            dops.append((f"{el}{cnt[el]}", traj.pos[:, i, :] - drift))
    return dict(li=li, ox=ox, dops=dops, L=np.asarray(traj.L, float), time=np.asarray(traj.time_ps, float),
                pos=traj.pos - drift[:, None, :], els=np.array([cfg["types"][int(t)] for t in traj.types]),
                dt=float(np.median(np.diff(traj.time_ps))))


# ---------------------------------------------------------------- ECoN
def econ_frames(li, ox, L, rmax, chunk=40):
    """econ, l_av, d_min, each (F,N).  L is (F,3).  O farther than rmax are ignored (the nearest O is always kept)."""
    F, N, _ = li.shape
    econ, lav, dmin = np.empty((F, N)), np.empty((F, N)), np.empty((F, N))
    for a in range(0, F, chunk):
        b = min(F, a + chunk)
        d = li[a:b, :, None, :] - ox[a:b, None, :, :]
        Lc = L[a:b][:, None, None, :]
        d -= Lc * np.rint(d / Lc)
        D = np.sqrt((d * d).sum(-1))
        dm = D.min(-1)
        D = np.where(D > np.maximum(rmax, dm[..., None]), 100.0, D)
        la = dm.copy()
        for _ in range(60):
            w = np.exp(1.0 - (D / la[..., None]) ** 6)
            new = (D * w).sum(-1) / w.sum(-1)
            done = np.abs(new - la).max() < 1e-7
            la = new
            if done:
                break
        w = np.exp(1.0 - (D / la[..., None]) ** 6)
        econ[a:b], lav[a:b], dmin[a:b] = w.sum(-1), la, dm
    return econ, lav, dmin


def shell_stats(V, sh, K, nblocks):
    """Per shell: mean over all Li-frames, jackknife SE (over contiguous time blocks), mean number of Li per frame, and
    mean(shell) - mean(far) with its jackknife SE (far = last category). sh: (F,N) shell index, -1 = ignored."""
    S = io.block_counts(sh, K, nblocks, values=V)
    C = io.block_counts(sh, K, nblocks)
    m, se = io.jackknife(lambda s_, c_: s_ / c_, S, C)
    d, d_se = io.jackknife(lambda s_, c_: s_ / c_ - (s_ / c_)[-1], S, C)
    return m, se, C.sum(0) / V.shape[0], d, d_se


# ---------------------------------------------------------------- 04 outputs
def read_rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def spear(x, y):
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return np.nan, np.nan
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return np.nan, np.nan
    r = spearmanr(x[ok], y[ok])
    return float(r[0]), float(r[1])


def build_state(ev, F, N):
    st = np.full((F, N), -1, int)
    by = {}
    for i, f, a, b in ev:
        by.setdefault(i, []).append((f, a, b))
    for i, lst in by.items():
        st[:lst[0][0], i] = lst[0][1]
        for k, (f, a, b) in enumerate(lst):
            f1 = lst[k + 1][0] if k + 1 < len(lst) else F
            st[f:f1, i] = b
    return st


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- plots
def _cols(K, plt):
    return [plt.cm.viridis(k / max(K - 2, 1)) for k in range(K - 1)]


def _legend(fig, names, cols, nmean, plt, far_dashed=False, extra=None):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=cols[k], marker="o", lw=2, label=f"Li {names[k]} from dopant (mean {nmean[k]:.1f} Li)")
         for k in range(len(names) - 1)]
    if far_dashed:
        h.append(Line2D([], [], color="k", ls="--", marker="o", lw=1.5, label=f"far Li ({names[-1]}), baseline"))
    if extra:
        h += extra
    fig.legend(handles=h, loc="lower center", ncol=2, fontsize=8, frameon=False)


def plot_delta(rows, names, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labs = list(dict.fromkeys(r["dopant"] for r in rows))
    K = len(names)
    cols = _cols(K, plt)
    fig, axs = plt.subplots(len(labs), 2, figsize=(9.5, 3.0 * len(labs) + 1.0), dpi=150, squeeze=False, sharex=True, sharey="col")   # one y axis per column: rows comparable
    for ri, lab in enumerate(labs):
        for ci, (key, yl) in enumerate([("dECoN", "ECoN(shell) - ECoN(far)"), ("dlav_A", "l_av(shell) - l_av(far)  (A)")]):
            a = axs[ri][ci]
            for k in range(K - 1):
                sel = [r for r in rows if r["dopant"] == lab and r["shell"] == names[k]]
                if sel:
                    a.errorbar([r["T"] for r in sel], [r[key] for r in sel], [r[key + "_se"] for r in sel], marker="o", ms=4,
                               capsize=2, color=cols[k])
            a.axhline(0, color="0.5", lw=0.8)
            a.set_title(f"{lab}: {yl.split(' (')[0]}", fontsize=10)
            a.set_ylabel(yl, fontsize=8)
            if ri == len(labs) - 1:
                a.set_xlabel("T (K)")
    nm = [np.mean([r["n_Li"] for r in rows if r["shell"] == names[k]]) for k in range(K)]
    _legend(fig, names, cols, nm, plt)
    fig.suptitle("Li-O environment near each dopant relative to far Li (site-free)", fontsize=11)
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)


def plot_hist(raw, T, names, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labs = list(raw["sh"])
    K = len(names)
    cols = _cols(K, plt) + ["k"]
    e = raw["econ"]
    bins = np.linspace(np.nanpercentile(e, 0.2), np.nanpercentile(e, 99.8), 45)
    fig, axs = plt.subplots(1, len(labs), figsize=(4.0 * len(labs), 3.6), dpi=150, squeeze=False, sharey=True)
    for a, lab in zip(axs[0], labs):
        sh = raw["sh"][lab]
        for k in range(K):
            v = e[sh == k]
            if v.size > 20:
                a.hist(v, bins=bins, density=True, histtype="step", lw=1.6, color=cols[k], ls="--" if k == K - 1 else "-",
                       label=f"{names[k]}  (n={v.size / e.shape[0]:.1f} Li)")
        a.set_title(f"{lab}, {T} K", fontsize=10)
        a.set_xlabel("Li-O ECoN")
        a.legend(fontsize=7)
    axs[0][0].set_ylabel("density")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_offsite(rows, names, zcut, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labs = list(dict.fromkeys(r["dopant"] for r in rows))
    K = len(names)
    cols = _cols(K, plt)
    fig, axs = plt.subplots(1, len(labs), figsize=(4.0 * len(labs), 4.2), dpi=150, squeeze=False, sharey=True)
    for a, lab in zip(axs[0], labs):
        for k in range(K):
            sel = [r for r in rows if r["dopant"] == lab and r["shell"] == names[k]]
            if sel:
                kw = dict(color="k", ls="--") if k == K - 1 else dict(color=cols[k])
                a.errorbar([r["T"] for r in sel], [r["off_frac"] for r in sel], [r["off_frac_se"] for r in sel], marker="o",
                           ms=4, capsize=2, **kw)
        a.set_title(lab, fontsize=10)
        a.set_xlabel("T (K)")
    axs[0][0].set_ylabel(f"fraction of frames with |ECoN - ref| > {zcut:g} sd")
    nm = [np.mean([r["n_Li"] for r in rows if r["shell"] == names[k]]) for k in range(K)]
    _legend(fig, names, cols, nm, plt, far_dashed=True)
    fig.suptitle("Frames whose Li-O environment does not match the assigned site type\n"
                 "(higher near a dopant than far = site assignment unreliable there)", fontsize=9)
    fig.tight_layout(rect=(0, 0.11, 1, 0.92))
    fig.savefig(path)
    plt.close(fig)


def plot_hopdip(rows, hs_rows, names, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    K = len(names)
    cols = _cols(K, plt) + ["k"]
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.4), dpi=150, sharex=True)
    for k in range(K):
        kw = dict(color=cols[k], ls="--" if k == K - 1 else "-", marker="o", ms=4, capsize=2)
        sel = sorted([r for r in rows if r["shell"] == names[k]], key=lambda r: r["T"])
        if sel:
            axs[0].errorbar([r["T"] for r in sel], [r["dip_median"] for r in sel], [r["dip_median_se"] for r in sel],
                            label=f"{names[k]} (n hops {sum(r['n_hops'] for r in sel)})", **kw)
        bs = sorted([r for r in hs_rows if r["ref"] == "any" and r["method"] == "free_at_hop" and r["shell"] == names[k]],
                    key=lambda r: r["T"])
        if bs:
            axs[1].errorbar([r["T"] for r in bs], [r["back_pct"] for r in bs], [r["back_pct_se"] for r in bs], **kw)
    axs[0].set_ylabel("ECoN dip during the hop (median +/- SE)", fontsize=9)
    axs[1].set_ylabel("back-hops (% of hops with a previous hop)", fontsize=9)
    for a in axs:
        a.set_xlabel("T (K)")
    axs[0].legend(fontsize=7, title="Li distance to nearest dopant in the hop frame", title_fontsize=7)
    fig.suptitle("Hops near a dopant: small dip + many back-hops = flicker, not transport\n"
                 "(right panel = free_back_hop_shell.png, ref any, method free_at_hop; hops from the site-free detector)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path)
    plt.close(fig)


def plot_link(link, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ts = list(link)
    fig, axs = plt.subplots(2, len(Ts), figsize=(max(3.3 * len(Ts), 7.5), 6.4), dpi=150, squeeze=False)
    kinds = sorted({k for T in Ts for k in list(link[T]["bar"][2]) + list(link[T]["dw"][2])})
    kc = {k: KIND_COL.get(k, "0.4") for k in kinds}
    for ci, T in enumerate(Ts):
        x, y, kk = link[T]["bar"]
        a = axs[0][ci]
        if len(x):
            a.scatter(x, y, s=8, c=[kc[k] for k in kk], alpha=0.6, edgecolor="none")
        a.set_title(f"{T} K", fontsize=10)
        a.set_xlabel("mean ECoN of the site the Li leaves")
        if ci == 0:
            a.set_ylabel("barrier out of the site (eV)")
        x, y, kk = link[T]["dw"]
        a = axs[1][ci]
        if len(x):
            a.scatter(x, np.log10(y), s=8, c=[kc[k] for k in kk], alpha=0.5, edgecolor="none")
        a.set_xlabel("mean ECoN during the dwell")
        if ci == 0:
            a.set_ylabel("log10 dwell (ps)")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=kc[k], label=k) for k in kinds], loc="lower center",
               ncol=len(kinds), fontsize=8, frameon=False)
    fig.suptitle("Local Li-O environment vs barrier (top) and dwell time (bottom)\ncolour = site type of the site the Li leaves",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- site-free hop detector (independent of 04)
def free_hops(li, win, dhop, confirm):
    """Anchor / dwell-cluster hop detector. NO site list, NO 04 output, only the unwrapped Li trajectory.
    Positions are averaged over `win` frames (kills vibration).  Every Li has an anchor = running mean of its current
    dwell.  A hop is declared when the coarse position stays farther than `dhop` (A) from the anchor for `confirm`
    consecutive coarse frames (min residence); the new anchor is the mean of those frames.  Back-hop = the new anchor
    lies within `dhop` of the anchor of the PREVIOUS dwell of that Li (it went back where it came from).
    Returns dict: li, f (fine frame where the excursion started), left (3-vector: anchor the Li left), is_back, elig,
    anchors (Fc,N,3) anchor in force at every coarse frame, Fc."""
    F, N, _ = li.shape
    Fc = F // win
    xc = li[:Fc * win].reshape(Fc, win, N, 3).mean(axis=1)
    anchor, cnt = xc[0].copy(), np.ones(N)
    prev = np.full((N, 3), np.nan)
    out, osum = np.zeros(N, int), np.zeros((N, 3))
    anchors = np.empty((Fc, N, 3))
    anchors[0] = anchor
    E = dict(li=[], f=[], left=[], is_back=[], elig=[])
    for c in range(1, Fc):
        x = xc[c]
        far_ = np.linalg.norm(x - anchor, axis=1) > dhop
        stay = ~far_
        out[stay], osum[stay] = 0, 0.0                       # excursion aborted: it was a fluctuation
        anchor[stay] += (x[stay] - anchor[stay]) / (cnt[stay] + 1)[:, None]
        cnt[stay] += 1
        out[far_] += 1
        osum[far_] += x[far_]
        done = np.where(far_ & (out >= confirm))[0]
        if len(done):
            new = osum[done] / out[done][:, None]
            has = np.isfinite(prev[done, 0])
            back = has & (np.linalg.norm(new - prev[done], axis=1) < dhop)
            f0 = (c - out[done] + 1) * win + win // 2
            for k, i in enumerate(done):
                E["li"].append(i); E["f"].append(min(int(f0[k]), Fc * win - 1)); E["left"].append(anchor[i].copy())
                E["is_back"].append(bool(back[k])); E["elig"].append(bool(has[k]))
            prev[done] = anchor[done]
            anchor[done] = new
            cnt[done] = out[done]
            out[done], osum[done] = 0, 0.0
        anchors[c] = anchor
    o = {k: np.array(v) for k, v in E.items()}
    if not len(o["li"]):
        o = dict(li=np.zeros(0, int), f=np.zeros(0, int), left=np.zeros((0, 3)), is_back=np.zeros(0, bool),
                 elig=np.zeros(0, bool))
    o["anchors"], o["Fc"] = anchors, Fc
    return o


def econ_weights(li_f, ox_f, L_f, rmax):
    """ECoN weights w_ij = exp[1-(l_ij/l_av,i)^6] of every Li i to every O j in ONE frame, (N,No)."""
    d = li_f[:, None, :] - ox_f[None, :, :]
    d -= L_f * np.rint(d / L_f)
    D = np.sqrt((d * d).sum(-1))
    dm = D.min(-1)
    D = np.where(D > np.maximum(rmax, dm)[:, None], 100.0, D)
    la = dm.copy()
    for _ in range(60):
        w = np.exp(1.0 - (D / la[:, None]) ** 6)
        new = (D * w).sum(-1) / w.sum(-1)
        done = np.abs(new - la).max() < 1e-7
        la = new
        if done:
            break
    return np.exp(1.0 - (D / la[:, None]) ** 6)


def wjac(a, b):
    """Weighted Jaccard similarity of two ECoN weight vectors per Li: sum min / sum max.  1 = same O set, 0 = disjoint."""
    return np.minimum(a, b).sum(-1) / np.maximum(a, b).sum(-1)


def econ_hops(li, ox, L, win, sim_thr, confirm, rmax, nsub=3):
    """O-environment hop detector (third, independent definition).  NO site list, NO displacement threshold: a Li hops when
    the SET of O atoms it is bonded to changes.  Each Li carries an anchor = running mean (over its current dwell) of the
    ECoN weight vector over ALL O atoms (window-averaged over `nsub` frames).  A hop is committed when the weighted-Jaccard
    similarity between the current window and the anchor stays < sim_thr for `confirm` consecutive windows; the new anchor
    is the mean of those windows.  Back-hop = the new O-set is again similar (>= sim_thr) to the anchor of the PREVIOUS
    dwell.  Same output dict as free_hops (positions are the window-mean Li positions, used only to give a dwell a distance
    to the dopant)."""
    F, N, _ = li.shape
    Fc = F // win
    xc = li[:Fc * win].reshape(Fc, win, N, 3).mean(axis=1)

    def wbar(c):
        fs = np.unique(np.linspace(c * win, c * win + win - 1, nsub).astype(int))
        return np.mean([econ_weights(li[f], ox[f], L[f], rmax) for f in fs], axis=0)

    A = wbar(0)
    Apos, cnt = xc[0].copy(), np.ones(N)
    prevA, hasprev = np.zeros_like(A), np.zeros(N, bool)
    out, osum, xsum = np.zeros(N, int), np.zeros_like(A), np.zeros((N, 3))
    anchors = np.empty((Fc, N, 3))
    anchors[0] = Apos
    E = dict(li=[], f=[], left=[], is_back=[], elig=[])
    for c in range(1, Fc):
        W = wbar(c)
        away = wjac(W, A) < sim_thr
        stay = ~away
        out[stay], osum[stay], xsum[stay] = 0, 0.0, 0.0
        A[stay] += (W[stay] - A[stay]) / (cnt[stay] + 1)[:, None]
        Apos[stay] += (xc[c][stay] - Apos[stay]) / (cnt[stay] + 1)[:, None]
        cnt[stay] += 1
        out[away] += 1
        osum[away] += W[away]
        xsum[away] += xc[c][away]
        done = np.where(away & (out >= confirm))[0]
        if len(done):
            newA = osum[done] / out[done][:, None]
            has = hasprev[done]
            back = has & (wjac(newA, prevA[done]) >= sim_thr)
            f0 = (c - out[done] + 1) * win + win // 2
            for k, i in enumerate(done):
                E["li"].append(i); E["f"].append(min(int(f0[k]), Fc * win - 1)); E["left"].append(Apos[i].copy())
                E["is_back"].append(bool(back[k])); E["elig"].append(bool(has[k]))
            prevA[done], hasprev[done] = A[done], True
            A[done] = newA
            Apos[done] = xsum[done] / out[done][:, None]
            cnt[done] = out[done]
            out[done], osum[done], xsum[done] = 0, 0.0, 0.0
        anchors[c] = Apos
    o = {k: np.array(v) for k, v in E.items()}
    if not len(o["li"]):
        o = dict(li=np.zeros(0, int), f=np.zeros(0, int), left=np.zeros((0, 3)), is_back=np.zeros(0, bool),
                 elig=np.zeros(0, bool))
    o["anchors"], o["Fc"] = anchors, Fc
    return o


def hop_stats(T, dt, ev, prefix, dops, L, dists, dnear, edges, names, far, nblocks, win):
    """Hop statistics per distance shell for the events `ev` of a site-free detector (free_hops or econ_hops), rows in the
    schema of io.hop_shell_rows.  Methods <prefix>_at_hop (shell of the Li in the hop frame) and <prefix>_left (shell of the
    dwell position it leaves; Li-time denominator = time spent on dwell positions of that shell).  The trajectory is cut to a
    whole number of `win` blocks.  Returns (rows, Fp)."""
    K = len(names)
    Fp = ev["Fc"] * win
    Lm = L[:Fp].mean(axis=0)
    rows = []
    dpos = {lab: pos[:Fp].mean(axis=0) for lab, pos in dops}
    def anchor_d(pts, p):
        d = pts - p
        d -= Lm * np.rint(d / Lm)
        return np.linalg.norm(d, axis=-1)
    A = ev["anchors"]                                                     # (Fc,N,3)
    Ad = {lab: anchor_d(A, p) for lab, p in dpos.items()}                  # (Fc,N)
    Ld = {lab: anchor_d(ev["left"], p) if len(ev["li"]) else np.zeros(0) for lab, p in dpos.items()}
    Ad_near = np.min(list(Ad.values()), axis=0)
    Ld_near = np.min(list(Ld.values()), axis=0) if len(ev["li"]) else np.zeros(0)
    Ad, Ld = {"any": Ad_near, **Ad}, {"any": Ld_near, **Ld}
    dn = dnear[:Fp]
    for ref in Ad:
        lab_f = io.shell_labels(dists[ref][:Fp], dn, edges, far)                             # (Fp,N) Li itself
        bc = io.block_counts(lab_f, K, nblocks)
        cat = lab_f[ev["f"], ev["li"]] if len(ev["li"]) else np.zeros(0, int)
        rows += io.hop_shell_rows(T, ref, f"{prefix}_at_hop", names, cat, ev["f"], ev["is_back"], ev["elig"], bc * dt,
                                  bc.sum(0) / Fp, Fp, nblocks)
        lab_c = io.shell_labels(Ad[ref], Ad_near, edges, far)                                # (Fc,N) anchors in force
        lab_a = np.repeat(lab_c, win, axis=0)                                                 # (Fp,N)
        bc = io.block_counts(lab_a, K, nblocks)
        cat = io.shell_labels(Ld[ref], Ld["any"], edges, far) if len(ev["li"]) else np.zeros(0, int)
        rows += io.hop_shell_rows(T, ref, f"{prefix}_left", names, cat, ev["f"], ev["is_back"], ev["elig"], bc * dt,
                                  bc.sum(0) / Fp, Fp, nblocks)
    return rows, Fp


def compare_04(rows_06, hops_dir, names, path):
    """CONSISTENCY CHECK ONLY: reads 04's hop_shell_stats.csv AFTER all analyses are done and overlays the hop definitions
    (ref 'any'): 04 site-based (solid), 06 anchor/displacement (dashed), 06 O-environment (dotted).  Nothing from 04 enters
    any 06 number."""
    p = os.path.join(hops_dir, "hop_shell_stats.csv")
    if not os.path.exists(p):
        print(f"  [compare] {p} not found: run the current 04 first")
        return
    r4 = [r for r in read_rows(p) if r["ref"] == "any"]
    r6 = [r for r in rows_06 if r["ref"] == "any"]
    groups = [("left", ("site_left", "free_left", "econ_left")), ("at hop", ("li_at_hop", "free_at_hop", "econ_at_hop"))]
    sty = [("-", "o"), ("--", "s"), (":", "^")]
    keys = [("back_pct", "back-hops (% of hops with a previous hop)", False), ("rate_ratio_to_far", "hop rate(shell)/rate(far)", True),
            ("nonback_ratio_to_far", "non-back rate(shell)/(far)", True)]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    K = len(names)
    cols = _cols(K, plt) + ["k"]
    Ts = sorted({float(r["T"]) for r in r6})
    fig, axs = plt.subplots(len(keys), 2, figsize=(10.5, 3.1 * len(keys) + 1.0), dpi=150, squeeze=False, sharex=True)
    print("\n  consistency of hop definitions, ref any. back-hop % per T (04 site | 06 anchor | 06 O-environment):")
    for ci, (gname, meths) in enumerate(groups):
        for ri, (key, yl, skip) in enumerate(keys):
            a = axs[ri][ci]
            for k in range(K - (1 if skip else 0)):
                for mi, meth in enumerate(meths):
                    src = r4 if mi == 0 else r6
                    sel = sorted([r for r in src if r["method"] == meth and r["shell"] == names[k]], key=lambda r: float(r["T"]))
                    if not sel:
                        continue
                    se = "back_pct_se" if key == "back_pct" else key.replace("ratio_to_far", "ratio_se")
                    a.errorbar([float(r["T"]) for r in sel], [fnum(r[key]) for r in sel], [fnum(r.get(se)) for r in sel],
                               color=cols[k], ls=sty[mi][0], marker=sty[mi][1], ms=4, capsize=2,
                               mfc=cols[k] if mi == 0 else "white")
            if skip:
                a.axhline(1, color="0.5", lw=0.8)
            a.set_ylabel(yl, fontsize=8)
            if ri == 0:
                a.set_title(f"shell by: {gname}", fontsize=9)
            if ri == len(keys) - 1:
                a.set_xlabel("T (K)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=cols[k], lw=2, label=names[k]) for k in range(K)]
    h += [Line2D([], [], color="0.3", ls=sty[i][0], marker=sty[i][1], mfc="0.3" if i == 0 else "white", label=lab)
          for i, lab in enumerate(("04 site-based", "06 anchor (displacement)", "06 O-environment (ECoN set)"))]
    fig.legend(handles=h, loc="lower center", ncol=min(len(h), 4), fontsize=8, frameon=False)
    fig.suptitle("Consistency of independent hop definitions (same trend in all = robust; disagreement = definition-dependent)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)
    for gname, meths in groups:
        for k in range(K):
            cells = []
            for T in Ts:
                v = []
                for mi, meth in enumerate(meths):
                    src = r4 if mi == 0 else r6
                    x = [r for r in src if r["method"] == meth and r["shell"] == names[k] and float(r["T"]) == T]
                    v.append(f"{fnum(x[0]['back_pct']):5.1f}" if x else "  n/a")
                cells.append(f"{T:.0f}K " + "|".join(v))
            print(f"    {gname:>6} {names[k]:>8}: " + "   ".join(cells))


# ---------------------------------------------------------------- all species pairs
def econ_from_D(D, rmax):
    """D (n,k) ascending distances (inf = no neighbour). ECoN, l_av, d_min (n,), NaN where no neighbour at all."""
    ok = np.isfinite(D[:, 0])
    D = np.where(np.isfinite(D), D, 100.0)
    dm = D[:, 0]
    D = np.where(D > np.maximum(rmax, dm)[:, None], 100.0, D)
    la = dm.copy()
    for _ in range(60):
        w = np.exp(1.0 - (D / la[:, None]) ** 6)
        new = (D * w).sum(-1) / w.sum(-1)
        done = np.abs(new - la).max() < 1e-7
        la = new
        if done:
            break
    w = np.exp(1.0 - (D / la[:, None]) ** 6)
    econ = w.sum(-1)
    return np.where(ok, econ, np.nan), np.where(ok, la, np.nan), np.where(ok, dm, np.nan)


def pair_env(pos, els, dops, L, edges, far, nblocks, stride, rmax_o, rmax_x, rdf_rmax=6.0, dr=0.05, kE=14, kq=96):
    """ECoN / l_av / d_min and g(r) for EVERY ordered species pair (centre X -> neighbour Y).
    pos (F,N,3) drift-removed unwrapped, els (N,) element strings, dops [(label, (F,3))], L (F,3).
    Centres: each host element (Li La Zr O) split into distance shells around the dopants (ref 'any' + every dopant),
    and each dopant atom on its own (no shells).  Neighbours: every element.  ECoN only with host-element neighbours
    (a single dopant atom has no meaningful coordination number); g(r) with all neighbours.
    Returns dict(rows=..., dop_rows=..., rdf=..., groups=..., refs=..., r=...)."""
    from scipy.spatial import cKDTree
    F = pos.shape[0]
    fr = np.arange(0, F, stride)
    Fs = len(fr)
    K = len(edges) + 1
    names = io.shell_names(edges)
    dlabs = [lab for lab, _ in dops]
    del_ = {lab: lab.rstrip("0123456789") for lab in dlabs}
    hosts = [e for e in ("Li", "La", "Zr", "O") if (els == e).any()]
    allsp = hosts + [e for e in dict.fromkeys(del_.values())]
    idx = {e: np.where(els == e)[0] for e in allsp}
    dpos = {lab: p for lab, p in dops}
    # dopant atom index: the atoms of element e that coincide with the trajectory of the dopant (match by first frame)
    dat = {}
    for lab, p in dops:
        cand = idx[del_[lab]]
        j = cand[np.argmin(np.linalg.norm(pos[0, cand] - p[0], axis=1))]
        dat[lab] = int(j)
    refs = ["any"] + dlabs
    # ---- shells of host centres
    sh = {}
    for e in hosts:
        dd = {lab: io.dist_to(pos[fr][:, idx[e], :], dpos[lab][fr], L[fr]) for lab in dlabs}
        dn = np.min(list(dd.values()), axis=0)
        dd = {"any": dn, **dd}
        sh[e] = {ref: io.shell_labels(d, dn, edges, far) for ref, d in dd.items()}
    nb = int(round(rdf_rmax / dr))
    rmid = (np.arange(nb) + 0.5) * dr
    groups = [(e, idx[e], "host") for e in hosts] + [(lab, np.array([dat[lab]]), "dopant") for lab in dlabs]
    econ = {(g, y): [np.full((Fs, len(ix)), np.nan) for _ in range(3)] for g, ix, kind in groups for y in hosts}
    hist, nC = {}, {}
    for g, ix, kind in groups:
        for y in allsp:
            for ref in (refs if kind == "host" else [g]):
                hist[(g, y, ref)] = np.zeros((K if kind == "host" else 1, nb))
                nC[(g, y, ref)] = np.zeros(K if kind == "host" else 1)
    vol = np.mean(np.prod(L[fr], axis=1))
    for t, f in enumerate(fr):
        Lf = L[f]
        P = pos[f] - Lf * np.floor(pos[f] / Lf)
        P[P >= Lf] = 0.0
        trees = {e: cKDTree(P[idx[e]], boxsize=Lf) for e in allsp}
        for g, ix, kind in groups:
            el_g = g if kind == "host" else del_[g]
            for y in allsp:
                selfy = 1 if el_g == y else 0
                k = min(kq, len(idx[y]) - selfy)
                if k < 1:
                    continue
                d, _ = trees[y].query(P[ix], k=k + selfy, distance_upper_bound=rdf_rmax, workers=-1)
                d = d.reshape(len(ix), -1)[:, selfy:]
                if y in hosts:
                    e_, l_, m_ = econ_from_D(d[:, :kE], rmax_o if y == "O" else rmax_x)
                    econ[(g, y)][0][t], econ[(g, y)][1][t], econ[(g, y)][2][t] = e_, l_, m_
                valid = np.isfinite(d)
                bins = np.where(valid, d / dr, 0).astype(int)
                valid &= bins < nb
                for ref in (refs if kind == "host" else [g]):
                    lab = sh[g][ref][t] if kind == "host" else np.zeros(len(ix), int)
                    Kc = K if kind == "host" else 1
                    ok = valid & (lab >= 0)[:, None]
                    hist[(g, y, ref)] += np.bincount((lab[:, None] * nb + bins)[ok], minlength=Kc * nb).reshape(Kc, nb)
                    nC[(g, y, ref)] += np.bincount(lab[lab >= 0], minlength=Kc)
    # ---- statistics
    rows, dop_rows = [], []
    for (g, y), (E, La, Dm) in econ.items():
        kind = "host" if g in hosts else "dopant"
        if kind == "host":
            for ref in refs:
                s_ = sh[g][ref]
                good = np.isfinite(E)
                s_ = np.where(good, s_, -1)
                st = {q: shell_stats(np.where(good, V, 0.0), s_, K, nblocks) for q, V in (("econ", E), ("lav", La), ("dmin", Dm))}
                for k in range(K):
                    r = dict(centre=g, neighbour=y, ref=ref, shell=names[k], n_centres=st["econ"][2][k])
                    for q, key in (("econ", "ECoN"), ("lav", "lav_A"), ("dmin", "dmin_A")):
                        m, s, _, dm_, ds = st[q]
                        r[key], r[key + "_se"] = m[k], s[k]
                        r["d" + key], r["d" + key + "_se"] = dm_[k], ds[k]
                    rows.append(r)
        else:
            good = np.isfinite(E)
            s_ = np.where(good, 0, -1)
            r = dict(centre=g, neighbour=y)
            for q, key, V in (("econ", "ECoN", E), ("lav", "lav_A", La), ("dmin", "dmin_A", Dm)):
                m, s, _, _, _ = shell_stats(np.where(good, V, 0.0), s_, 1, nblocks)
                r[key], r[key + "_se"] = m[0], s[0]
            dop_rows.append(r)
    rdf = {}
    for (g, y, ref), H in hist.items():
        rho = (len(idx[y]) - (1 if (g if g in hosts else del_[g]) == y else 0)) / vol
        shellv = 4 * np.pi * rmid ** 2 * dr
        n = nC[(g, y, ref)][:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            rdf[(g, y, ref)] = np.where(n > 0, H / (n * shellv[None] * rho), np.nan)
    return dict(rows=rows, dop_rows=dop_rows, rdf=rdf, hosts=hosts, allsp=allsp, dlabs=dlabs, refs=refs, r=rmid,
                nC=nC, Fs=Fs)


def plot_pair_delta(rows, hosts, ref, names, key, ylabel, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    K = len(names)
    cols = _cols(K, plt)
    fig, axs = plt.subplots(len(hosts), len(hosts), figsize=(3.1 * len(hosts), 2.6 * len(hosts) + 0.9), dpi=130, squeeze=False,
                            sharex=True)
    for i, x in enumerate(hosts):
        for j, y in enumerate(hosts):
            a = axs[i][j]
            for k in range(K - 1):
                sel = sorted([r for r in rows if r["centre"] == x and r["neighbour"] == y and r["ref"] == ref and r["shell"] == names[k]],
                             key=lambda r: r["T"])
                if sel:
                    a.errorbar([r["T"] for r in sel], [r[key] for r in sel], [r[key + "_se"] for r in sel], marker="o", ms=3,
                               capsize=2, color=cols[k], lw=1)
            a.axhline(0, color="0.5", lw=0.7)
            a.set_title(f"{x} -> {y}", fontsize=9)
            if i == len(hosts) - 1:
                a.set_xlabel("T (K)", fontsize=8)
            if j == 0:
                a.set_ylabel(ylabel, fontsize=7)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=cols[k], marker="o", label=f"centre {names[k]} from {ref}") for k in range(K - 1)],
               loc="lower center", ncol=K - 1, fontsize=8, frameon=False)
    fig.suptitle(f"{key}: shell minus far, every host pair (centre -> neighbour), reference {ref}", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)


def plot_pair_dopant(dop_rows, hosts, dlabs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    qs = [("ECoN", "ECoN"), ("lav_A", "l_av (A)"), ("dmin_A", "nearest distance (A)")]
    fig, axs = plt.subplots(len(qs), len(hosts), figsize=(3.2 * len(hosts), 2.7 * len(qs) + 0.5), dpi=130, squeeze=False, sharex=True)
    cm = plt.cm.tab10
    for ri, (key, yl) in enumerate(qs):
        for j, y in enumerate(hosts):
            a = axs[ri][j]
            for c, lab in enumerate(dlabs):
                sel = sorted([r for r in dop_rows if r["centre"] == lab and r["neighbour"] == y], key=lambda r: r["T"])
                if sel:
                    a.errorbar([r["T"] for r in sel], [r[key] for r in sel], [r[key + "_se"] for r in sel], marker="o", ms=3,
                               capsize=2, color=cm(c), label=lab)
            a.set_title(f"dopant -> {y}", fontsize=9)
            if j == 0:
                a.set_ylabel(yl, fontsize=8)
            if ri == len(qs) - 1:
                a.set_xlabel("T (K)", fontsize=8)
    axs[0][0].legend(fontsize=7)
    fig.suptitle("Environment of each dopant atom itself (mean over frames, block jackknife)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)


def plot_pair_rdf(res, ref, T, names, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hosts, allsp, r = res["hosts"], res["allsp"], res["r"]
    K = len(names)
    cols = _cols(K, plt) + ["k"]
    fig, axs = plt.subplots(len(hosts), len(allsp), figsize=(2.9 * len(allsp), 2.3 * len(hosts) + 0.9), dpi=130, squeeze=False,
                            sharex=True)
    for i, x in enumerate(hosts):
        for j, y in enumerate(allsp):
            a = axs[i][j]
            g = res["rdf"].get((x, y, ref))
            if g is not None:
                for k in range(K):
                    if np.isfinite(g[k]).any():
                        a.plot(r, g[k], color=cols[k], ls="--" if k == K - 1 else "-", lw=1.1)
            a.axhline(1, color="0.7", lw=0.5)
            a.set_title(f"{x} -> {y}", fontsize=8)
            if j == 0:
                a.set_ylabel("g(r)", fontsize=8)
            if i == len(hosts) - 1:
                a.set_xlabel("r (A)", fontsize=8)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=cols[k], ls="--" if k == K - 1 else "-", label=f"centres {names[k]} from {ref}") for k in range(K)],
               loc="lower center", ncol=K, fontsize=8, frameon=False)
    fig.suptitle(f"Partial g(r), every host centre -> every species, split by distance to {ref}, T = {T} K (noisy in the inner shell)", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)


def plot_dopant_rdf(res_by_T, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ts = list(res_by_T)
    r0 = next(iter(res_by_T.values()))
    dlabs, allsp, r = r0["dlabs"], r0["allsp"], r0["r"]
    fig, axs = plt.subplots(len(dlabs), len(allsp), figsize=(2.9 * len(allsp), 2.4 * len(dlabs) + 0.8), dpi=130, squeeze=False, sharex=True)
    cm = plt.cm.plasma(np.linspace(0.05, 0.85, len(Ts)))
    for i, lab in enumerate(dlabs):
        for j, y in enumerate(allsp):
            a = axs[i][j]
            for c, T in zip(cm, Ts):
                g = res_by_T[T]["rdf"].get((lab, y, lab))
                if g is not None and np.isfinite(g[0]).any():
                    a.plot(r, g[0], color=c, lw=1, label=f"{T} K")
            a.axhline(1, color="0.7", lw=0.5)
            a.set_title(f"{lab} -> {y}", fontsize=8)
            if j == 0:
                a.set_ylabel("g(r)", fontsize=8)
            if i == len(dlabs) - 1:
                a.set_xlabel("r (A)", fontsize=8)
    axs[0][0].legend(fontsize=6)
    fig.suptitle("g(r) around each dopant atom, every species, all T (single atom: noisy)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="folder with the dumps + element_list (or a settings .yaml)")
    ap.add_argument("--defaults")
    ap.add_argument("--T", type=int, nargs="*")
    ap.add_argument("--tmin", type=float)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--shells", type=float, nargs="*", default=list(io.DEFAULT_SHELLS), help="shell edges, A (same default in 04, 05, 06)")
    ap.add_argument("--blocks", type=int, default=io.DEFAULT_BLOCKS, help="time blocks for the jackknife error bars")
    ap.add_argument("--rmax", type=float, default=3.5, help="Li-O candidates within this distance, A")
    ap.add_argument("--far", choices=["all", "self"], default="all", help="far reference: > last edge from every dopant / from that dopant")
    ap.add_argument("--hist-T", type=int, help="temperature for econ_hist.png (default: lowest)")
    ap.add_argument("--hops-dir", help="04 output folder (sites.csv, hops_T*.csv, barriers_T*.csv): enables part B")
    ap.add_argument("--zcut", type=float, default=2.0, help="off-site-like threshold, sd of the far-frame ECoN of the site type")
    ap.add_argument("--dip-w", type=int, default=3, help="frames each side of a hop for the ECoN dip")
    ap.add_argument("--min-site-frames", type=int, default=20, help="min Li-frames on a site for its mean ECoN")
    ap.add_argument("--hop-d", type=float, default=1.2, help="site-free hops: Li must move farther than this (A) from its dwell anchor")
    ap.add_argument("--hop-win-ps", type=float, default=0.5, help="site-free hops: positions averaged over this window (ps)")
    ap.add_argument("--hop-confirm", type=int, default=2, help="site-free hops: windows the Li must stay away (min residence)")
    ap.add_argument("--econ-sim", type=float, default=0.5, help="O-environment hops: weighted-Jaccard similarity below which the O set has changed")
    ap.add_argument("--econ-sub", type=int, default=3, help="O-environment hops: frames per window used for the ECoN weights (cost)")
    ap.add_argument("--no-econ-hops", action="store_true", help="skip the O-environment hop detector (slowest part)")
    ap.add_argument("--pair-stride", type=int, default=10, help="species pairs: use every n-th frame")
    ap.add_argument("--pair-rmax", type=float, default=5.0, help="species pairs: ECoN candidates within this distance for non-O neighbours, A")
    ap.add_argument("--no-pairs", action="store_true", help="skip the all-species-pair part")
    args = ap.parse_args()

    cfg = io.load_config(args.target, args.defaults)
    temps = args.T or io.temperatures(cfg)
    outdir = os.path.join(args.outdir, cfg["name"])
    os.makedirs(outdir, exist_ok=True)
    edges = np.array(args.shells, float)
    names = io.shell_names(edges)
    K = len(names)
    hist_T = args.hist_T or temps[0]

    kind_of, site_rows = None, None
    if args.hops_dir:
        sp = os.path.join(args.hops_dir, "sites.csv")
        if not os.path.exists(sp):
            raise SystemExit(f"{sp} not found: --hops-dir must be the 04 output folder (results of 04_sites_hops.py)")
        site_rows = read_rows(sp)
        kind_of = [r["kind"].split()[0] for r in site_rows]
        ukinds = sorted(set(kind_of))
        kid = np.array([ukinds.index(k) for k in kind_of])
        print(f"[part B] site types from {sp}: " + ", ".join(f"{k} ({kind_of.count(k)})" for k in ukinds))

    rowsA, rows_kind, rows_off, rows_dip, rows_dw, link, raw_hist = [], [], [], [], [], {}, None
    rows_free, rows_econ, rows_pair, rows_pdop, pair_res = [], [], [], [], {}
    for T in temps:
        print(f"\n=== {cfg['name']}  T = {T} K ===")
        dat = load(cfg, T, args.tmin)
        li, ox, L, dt = dat["li"], dat["ox"], dat["L"], dat["dt"]
        F, N, _ = li.shape
        if not dat["dops"]:
            raise SystemExit("no dopant atoms in this system: nothing to compare")
        print(f"  frames {F}, Li {N}, O {ox.shape[1]}: computing ECoN (rmax {args.rmax:g} A) ...")
        econ, lav, dmin = econ_frames(li, ox, L, args.rmax)
        print(f"  all Li: ECoN {econ.mean():.2f} +/- {econ.std():.2f} (sd over Li-frames), l_av {lav.mean():.3f} A, "
              f"nearest Li-O {dmin.mean():.3f} A")
        dists = {lab: io.dist_to(li, pos, L) for lab, pos in dat["dops"]}
        dnear = np.min(list(dists.values()), axis=0)
        dists = {"any": dnear, **dists}
        sh = {lab: io.shell_labels(d, dnear, edges, args.far) for lab, d in dists.items()}
        if T == hist_T:
            raw_hist = dict(econ=econ, sh=sh, T=T)

        # ---------------- part A
        for lab in dists:
            st = {q: shell_stats(v, sh[lab], K, args.blocks) for q, v in (("econ", econ), ("lav", lav), ("dmin", dmin))}
            nmean = st["econ"][2]
            print(f"  {lab}: Li per shell (mean) " + "   ".join(f"{names[k]}: {nmean[k]:.1f}" for k in range(K)))
            print(f"      shell       ECoN            dECoN vs far      l_av (A)        dl_av (A)")
            far = K - 1
            for k in range(K):
                r = dict(T=T, dopant=lab, shell=names[k], n_Li=nmean[k])
                for q, key in (("econ", "ECoN"), ("lav", "lav_A"), ("dmin", "dmin_A")):
                    m, s, _, dm, ds = st[q]
                    r[key], r[key + "_se"] = m[k], s[k]
                    r["d" + key], r["d" + key + "_se"] = dm[k], ds[k]
                rowsA.append(r)
                print(f"    {names[k]:>8}   {r['ECoN']:6.3f} +/- {r['ECoN_se']:.3f}   {r['dECoN']:+6.3f} +/- {r['dECoN_se']:.3f}   "
                      f"{r['lav_A']:6.3f} +/- {r['lav_A_se']:.3f}   {r['dlav_A']:+6.3f} +/- {r['dlav_A_se']:.3f}")

        # ---------------- part A2: SITE-FREE hops (own detector, nothing from 04) -> hop statistics + ECoN dip
        win = max(1, int(round(args.hop_win_ps / dt)))
        evf = free_hops(li, win, args.hop_d, args.hop_confirm)
        rows_T, Fp = hop_stats(T, dt, evf, "free", dat["dops"], L, dists, dnear, edges, names, args.far, args.blocks, win)
        rows_free += rows_T
        nh = len(evf["li"])
        print(f"  [site-free hops, anchor/displacement] window {win} frames, d = {args.hop_d:g} A, confirm {args.hop_confirm} windows: "
              f"{nh} hops, {len(set(evf['li'].tolist()))} of {N} Li hop, "
              f"back-hops {100 * evf['is_back'].sum() / max(evf['elig'].sum(), 1):.1f} % (all shells)")
        io.print_hop_shell(rows_T, T)
        if not args.no_econ_hops:
            print(f"  [site-free hops, O-environment] similarity threshold {args.econ_sim:g}, {args.econ_sub} frames per window ...")
            eve = econ_hops(li, ox, L, win, args.econ_sim, args.hop_confirm, args.rmax, args.econ_sub)
            rows_E, _ = hop_stats(T, dt, eve, "econ", dat["dops"], L, dists, dnear, edges, names, args.far, args.blocks, win)
            rows_econ += rows_E
            print(f"  [site-free hops, O-environment] {len(eve['li'])} hops, {len(set(eve['li'].tolist()))} of {N} Li hop, "
                  f"back-hops {100 * eve['is_back'].sum() / max(eve['elig'].sum(), 1):.1f} % (all shells)")
            io.print_hop_shell(rows_E, T)
        w = args.dip_w
        buck = {k: [] for k in range(K)}
        bblk = {k: [] for k in range(K)}
        for i, f in zip(evf["li"], evf["f"]):
            if f - w < 0 or f + w > F:
                continue
            seg = econ[f - w:f + w, i]
            dip = min(econ[f - w:f, i].mean(), econ[f:f + w, i].mean()) - seg.min()
            k = int(sh["any"][f, i])
            if k < 0:
                continue
            buck[k].append(dip)
            bblk[k].append(int(io.block_id(f, F, args.blocks)))
        print(f"  hop dip (ECoN before/after minus its minimum during the hop, window {w} frames; site-free hops) by distance of the Li "
              f"to the nearest dopant (median +/- jackknife SE; IQR in hop_dip.csv):")
        for k in range(K):
            v = np.array(buck[k])
            if len(v) >= 5:
                q = np.percentile(v, [25, 50, 75])
                med, med_se = io.jackknife_events(v, np.array(bblk[k]), args.blocks)
                rows_dip.append(dict(T=T, shell=names[k], n_hops=len(v), dip_median=med, dip_median_se=med_se, dip_q25=q[0],
                                     dip_q75=q[2]))
                print(f"    {names[k]:>8}: {len(v):5d} hops   dip median {med:.3f} +/- {med_se:.3f} (IQR {q[0]:.3f}-{q[2]:.3f})")

        # ---------------- part A3: every species pair (centre -> neighbour), site-free
        if not args.no_pairs:
            print(f"  [pairs] ECoN / l_av / d_min / g(r) for every species pair, every {args.pair_stride}th frame ...")
            res = pair_env(dat["pos"], dat["els"], dat["dops"], L, edges, args.far, args.blocks, args.pair_stride,
                           args.rmax, args.pair_rmax)
            pair_res[T] = res
            for r in res["rows"]:
                rows_pair.append(dict(T=T, **r))
            for r in res["dop_rows"]:
                rows_pdop.append(dict(T=T, **r))
            for r in res["dop_rows"]:
                print(f"    dopant {r['centre']:>4} -> {r['neighbour']:<2}: ECoN {r['ECoN']:5.2f} +/- {r['ECoN_se']:.2f}   "
                      f"l_av {r['lav_A']:.3f} A   nearest {r['dmin_A']:.3f} A")

        # ---------------- part B
        if not args.hops_dir:
            continue
        hp = os.path.join(args.hops_dir, f"hops_T{T}K.csv")
        if not os.path.exists(hp):
            print(f"  [part B] {hp} not found: skipped for this T")
            continue
        ev = []
        for r in read_rows(hp):
            fr_ = int(r["frame"]) if r.get("frame") not in (None, "") else int(round(float(r["t_ps"]) / dt))   # 'frame' written by the current 04
            ev.append((int(r["li"]), fr_, int(r["site_from"]), int(r["site_to"])))
        ev.sort(key=lambda x: (x[0], x[1]))
        if not ev or max(x[1] for x in ev) >= F or max(x[0] for x in ev) >= N or max(max(x[2], x[3]) for x in ev) >= len(kind_of):
            print("  [part B] hops file does not match this trajectory window (use the same --tmin/--T as the 04 run): skipped")
            continue
        state, full = None, False
        stp = os.path.join(args.hops_dir, f"state_T{T}K.npz")
        if os.path.exists(stp):
            zs = np.load(stp)
            if zs["state"].shape == (F, N) and int(zs["n_sites"]) == len(kind_of):
                state, full = zs["state"].astype(int), True
            else:
                print(f"  [part B] {stp} has shape {zs['state'].shape}, expected {(F, N)}: not the same --tmin/--T as the 04 run; "
                      "falling back to the hop list")
        if state is None:
            state = build_state(ev, F, N)
        kidx = np.where(state >= 0, kid[np.maximum(state, 0)], -1)
        print(f"  [part B] {len(ev)} hops, {len(set(x[0] for x in ev))} of {N} Li hop at least once; "
              + ("site of every Li in every frame from state_T%dK.npz (Li that never hop included)" % T if full else
                 "hop list only: Li that never hop have no known site and are left out (rerun 04 to get state_T%dK.npz)" % T)
              + f"; frames with a known site: {100 * (state >= 0).mean():.0f} %")
        far_m = dnear > edges[-1]
        ref = {}
        print("  characteristic ECoN of each site type (sd over Li-frames):")
        for ki, kn in enumerate(ukinds):
            for region, m in (("far", far_m), (f"<{edges[0]:g}A", dnear <= edges[0])):
                mm = (kidx == ki) & m
                if mm.sum() >= 30:
                    rows_kind.append(dict(T=T, kind=kn, region=region, n_frames=int(mm.sum()), ECoN=econ[mm].mean(),
                                          ECoN_sd=econ[mm].std(), lav_A=lav[mm].mean()))
                    print(f"    {kn:>5} {region:>6}: n {int(mm.sum()):6d}   ECoN {econ[mm].mean():.3f} +/- {econ[mm].std():.3f}   "
                          f"l_av {lav[mm].mean():.3f} A")
                    if region == "far":
                        ref[ki] = (econ[mm].mean(), econ[mm].std())
        # off-site-like
        if len(ref) == len(ukinds):
            mu = np.array([ref[k][0] for k in range(len(ukinds))])
            sd = np.array([ref[k][1] for k in range(len(ukinds))])
            z = np.where(kidx >= 0, (econ - mu[np.maximum(kidx, 0)]) / sd[np.maximum(kidx, 0)], 0.0)
            off = (np.abs(z) > args.zcut).astype(float)
            print(f"  off-site-like frames (|ECoN - far value of its site type| > {args.zcut:g} sd), by distance to the nearest dopant:")
            shb = np.where(kidx >= 0, sh["any"], -1)
            m_, s_, n_, _, _ = shell_stats(off, shb, K, args.blocks)
            for k in range(K):
                rows_off.append(dict(T=T, dopant="any", shell=names[k], n_Li=n_[k], off_frac=m_[k], off_frac_se=s_[k]))
                print(f"    {names[k]:>8}: {100 * m_[k]:5.1f} % +/- {100 * s_[k]:.1f}   (Li with a known site per frame {n_[k]:.1f})")
            for lab in [l for l in dists if l != "any"]:
                shb = np.where(kidx >= 0, sh[lab], -1)
                m_, s_, n_, _, _ = shell_stats(off, shb, K, args.blocks)
                for k in range(K):
                    rows_off.append(dict(T=T, dopant=lab, shell=names[k], n_Li=n_[k], off_frac=m_[k], off_frac_se=s_[k]))
        else:
            print("  off-site-like: skipped (a site type has too few far frames for a reference)")

        # site table and barrier link
        m = state >= 0
        idx, ee, ll = state[m], econ[m], lav[m]
        S = len(kind_of)
        n = np.bincount(idx, minlength=S)
        with np.errstate(invalid="ignore", divide="ignore"):
            em = np.bincount(idx, ee, S) / n
            esd = np.sqrt(np.maximum(np.bincount(idx, ee * ee, S) / n - em ** 2, 0))
            lm_ = np.bincount(idx, ll, S) / n
        em = np.where(n >= args.min_site_frames, em, np.nan)
        write_csv(os.path.join(outdir, f"site_econ_T{T}K.csv"),
                  [dict(site=s, kind=kind_of[s], shell_04=site_rows[s].get("shell", ""), n_frames=int(n[s]), ECoN_mean=em[s],
                        ECoN_sd=esd[s], lav_A=lm_[s]) for s in range(S)])
        bx, by_, bk = [], [], []
        bp = os.path.join(args.hops_dir, f"barriers_T{T}K.csv")
        if os.path.exists(bp):
            for r in read_rows(bp):
                s, bar = int(r["site_from"]), fnum(r.get("barrier_eV"))
                if np.isfinite(bar) and np.isfinite(em[s]):
                    bx.append(em[s])
                    by_.append(bar)
                    bk.append(kind_of[s])
            rho, p = spear(bx, by_)
            print(f"  barrier vs mean ECoN of the site left: {len(bx)} directed pairs, Spearman rho = {rho:+.2f} (p = {p:.2g})"
                  "   (rho > 0: tighter/more coordinated site, higher barrier)")
        else:
            print(f"  {bp} not found (04 run with --no-barriers?): barrier link skipped")
        # dwell vs ECoN
        cs = np.vstack([np.zeros((1, N)), np.cumsum(econ, axis=0)])
        dwx, dwy, dwk = [], [], []
        pf = {}
        for i, f, a, b in ev:
            p0 = pf.get(i)
            pf[i] = f
            if p0 is None or f - p0 < 2:              # first dwell of each Li is censored by the window start
                continue
            dwx.append((cs[f, i] - cs[p0, i]) / (f - p0))
            dwy.append((f - p0) * dt)
            dwk.append(kind_of[a])
        link[T] = dict(bar=(np.array(bx), np.array(by_), np.array(bk)), dw=(np.array(dwx), np.array(dwy), np.array(dwk)))
        print("  dwell vs mean ECoN during the dwell (site type of the site left; Spearman with log dwell):")
        dwx, dwy, dwk = np.array(dwx), np.array(dwy), np.array(dwk)
        for kn in ukinds:
            mk = dwk == kn
            if mk.sum() >= 20:
                rho, p = spear(dwx[mk], np.log(dwy[mk]))
                t = np.percentile(dwx[mk], [33.3, 66.7])
                md = [np.median(dwy[mk & (dwx <= t[0])]), np.median(dwy[mk & (dwx > t[0]) & (dwx <= t[1])]),
                      np.median(dwy[mk & (dwx > t[1])])]
                rows_dw.append(dict(T=T, kind=kn, n=int(mk.sum()), rho=rho, p=p, dwell_med_lowECoN=md[0],
                                    dwell_med_midECoN=md[1], dwell_med_highECoN=md[2]))
                print(f"    {kn:>5}: n {int(mk.sum()):5d}   rho = {rho:+.2f} (p = {p:.2g})   median dwell ps (low/mid/high ECoN tercile) "
                      f"{md[0]:.2f} / {md[1]:.2f} / {md[2]:.2f}")

    write_csv(os.path.join(outdir, "econ_shell.csv"), rowsA)
    plot_delta(rowsA, names, os.path.join(outdir, "econ_shell.png"))
    if raw_hist is not None:
        plot_hist(raw_hist, raw_hist["T"], names, os.path.join(outdir, "econ_hist.png"))
    out = "econ_shell.csv econ_shell.png econ_hist.png"
    if rows_kind:
        write_csv(os.path.join(outdir, "econ_by_kind.csv"), rows_kind)
        out += " econ_by_kind.csv"
    if rows_off:
        write_csv(os.path.join(outdir, "econ_offsite.csv"), rows_off)
        plot_offsite([r for r in rows_off], names, args.zcut, os.path.join(outdir, "econ_offsite.png"))
        out += " econ_offsite.csv econ_offsite.png"
    if rows_dip:
        write_csv(os.path.join(outdir, "hop_dip.csv"), rows_dip)
        plot_hopdip(rows_dip, rows_free, names, os.path.join(outdir, "econ_hopdip.png"))
        out += " hop_dip.csv econ_hopdip.png"
    if rows_free:
        out += " " + " ".join(io.write_hop_shell_outputs(rows_free, names, outdir, cfg["name"], prefix="free_"))
    if rows_econ:
        out += " " + " ".join(io.write_hop_shell_outputs(rows_econ, names, outdir, cfg["name"], prefix="oenv_"))
    if rows_free and args.hops_dir:
        compare_04(rows_free + rows_econ, args.hops_dir, names, os.path.join(outdir, "consistency_04_vs_06.png"))
        out += " consistency_04_vs_06.png"
    if pair_res:
        r0 = next(iter(pair_res.values()))
        write_csv(os.path.join(outdir, "pair_env_shell.csv"), rows_pair)
        write_csv(os.path.join(outdir, "pair_env_dopant.csv"), rows_pdop)
        rdf_rows = []
        for T, res in pair_res.items():
            for (g, y, ref), G in res["rdf"].items():
                for k in range(G.shape[0]):
                    for r_, v in zip(res["r"], G[k]):
                        rdf_rows.append(dict(T=T, centre=g, neighbour=y, ref=ref, shell=names[k] if G.shape[0] > 1 else "-", r_A=r_, g=v))
        write_csv(os.path.join(outdir, "pair_rdf.csv"), rdf_rows)
        for ref in r0["refs"]:
            plot_pair_delta(rows_pair, r0["hosts"], ref, names, "dECoN", "ECoN(shell) - ECoN(far)", os.path.join(outdir, f"pair_dECoN_{ref}.png"))
            plot_pair_delta(rows_pair, r0["hosts"], ref, names, "dlav_A", "l_av(shell) - l_av(far)  (A)", os.path.join(outdir, f"pair_dlav_{ref}.png"))
            plot_pair_delta(rows_pair, r0["hosts"], ref, names, "ddmin_A", "d_min(shell) - d_min(far)  (A)", os.path.join(outdir, f"pair_ddmin_{ref}.png"))
            plot_pair_rdf(pair_res[hist_T], ref, hist_T, names, os.path.join(outdir, f"pair_rdf_{ref}_T{hist_T}K.png"))
        plot_pair_dopant(rows_pdop, r0["hosts"], r0["dlabs"], os.path.join(outdir, "pair_dopant_env.png"))
        plot_dopant_rdf(pair_res, os.path.join(outdir, "pair_rdf_dopant.png"))
        out += " pair_env_shell.csv pair_env_dopant.csv pair_rdf.csv pair_dECoN_*.png pair_dlav_*.png pair_ddmin_*.png pair_rdf_*.png pair_dopant_env.png pair_rdf_dopant.png"
    if rows_dw:
        write_csv(os.path.join(outdir, "dwell_econ.csv"), rows_dw)
    if link:
        plot_link(link, os.path.join(outdir, "econ_link.png"))
        out += " dwell_econ.csv econ_link.png site_econ_T*.csv"
    print(f"\n[+] {outdir}/ {out}")
    print("Read: part A dECoN / dl_av != 0 outside the error bars, with the same sign at every T, = the dopant changes "
          "the Li-O environment (check n_Li). Part B: an off-site fraction or hop dip near the dopant that differs from the far "
          "baseline says whether hops there are real transitions or flicker. These are indicators, not proof.")


if __name__ == "__main__":
    main()
