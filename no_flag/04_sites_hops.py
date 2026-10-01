#!/usr/bin/env python3
"""
04_sites_hops.py - Li sites, hops, concerted hops and site-to-site barriers, straight from the dumps.

Pipeline (framework-COM drift removed first, same as 00-03)
  1 sites     Peaks of the Li density pooled over ALL temperatures (one site list per composition, so every T and every
              dopant concentration can be compared site by site). Peaks closer than --merge A are merged.
              --sites-cif CIF: use FIXED crystal sites instead (24d + 96h from the Ia-3d CIF, tiled to the box, aligned
              per T on the La/Zr framework, dopant-occupied sites removed, --merge-pairs optional). Peaks are then only
              a cross-check.
  2 assign    Every Li in every frame -> nearest site (periodic k-d tree). A hop is committed only when the Li has
              stayed at the new site for --min-res ps (default 0.3 ps); shorter visits are vibration/transit, not hops.
  3 hops      Event = (Li, time, site_from, site_to, jump length, dwell time at site_from).
              Rates per site pair: Gamma_ab = n_ab / (Li-time spent on a).
  4 shells    Hop rate, back-hop % and concerted fraction per distance shell (--shells, default 3 5 7 A -> 0-3, 3-5, 5-7, >7;
              far = >7 A from EVERY dopant, --far self = from that dopant only). TWO methods, both reported for 'any'
              (nearest dopant) and for each dopant alone (same code and same definitions as 06, see llzo_io.py):
                site_left : shell = distance of the SITE the Li leaves (fixed site list, time-mean dopant position)
                li_at_hop : shell = distance of the Li itself to the dopant in the hop frame (instantaneous)
              Error bars everywhere = delete-one-block jackknife over --blocks time blocks. (No dopant -> one shell.)
  5 concerted Two hops of DIFFERENT Li are coupled if they start within --tc ps and their sites are within --rc A
              (any of the 4 site pairs). Coupled hops are chained into clusters; size >= 2 = concerted.
              Also: 'refill' = fraction of hops whose vacated site is entered by another Li within --tc ps.
              CONTROL: every Li's hop times are shifted by an independent random circular offset (--nshuf times). This keeps
              each Li's own rate and the spatial distribution of hops but removes cross-Li timing, so the shuffled
              fractions are the chance level. Excess = observed - chance, z = excess / sd(chance).
  6 barriers  (needs results/<system>/density_F_T*K.npz from 01) For every site pair with >= --nmin observed hops:
              bottleneck (minimax) path of F_rel inside a tube of radius --tube A around the straight line a-b;
              barrier_ab = max_path F - F_a. Also nu_eff = Gamma_ab exp(barrier/kT): should be ~ phonon frequency
              (1-10 THz) if the F-barrier and the observed rate are consistent.

Limits you should know:
  * Time resolution = dump spacing (0.1 ps): hops faster than that are seen as one frame, --tc below ~0.3 ps means nothing.
  * F is a smoothed, Li-Li-correlated free energy, not a single-particle NEB barrier; it is a lower bound where smoothing
    raises the density at a saddle. Pairs not connected inside sampled voxels are reported as blocked (barrier >= limit).
  * Concerted fraction has binomial error ~ sqrt(f(1-f)/n_hops) (hops are not fully independent -> optimistic).

usage:
  python 04_sites_hops.py /path/to/folder                       # all temperatures found in the folder
  python 04_sites_hops.py /path/to/folder --T 800 1200 --tc 0.5 --rc 3.5 --min-res 0.3
  python 04_sites_hops.py /path/to/folder --sites-cif cubic_LLZO_wyckoff.cif [--merge-pairs 1.0] [--site-excl 1.0]
  python 04_sites_hops.py /path/to/folder --no-barriers         # skip step 6 (no 01 results needed)
  python 04_sites_hops.py /path/to/folder --sites-cif X.cif --net-center all --net-slab 2 --net-window 9 \\
         [--net-label occ|hops|index|both|none] [--net-bg hopped|geom|none] [--net-wref N] [--bond-cut 2.6]
  Network plots: line width saturates at --net-wref hops (default: 95th percentile of the hopped pairs) so a few flickering pairs
  next to a dopant do not shrink every other bond to a hair; the legend says when it is clipped.  Faint dashed lines = bonds that
  hopped at another T (or, with --net-bg geom, every pair closer than --bond-cut) but not at this T.  Numbers on the sites = mean
  Li on the site (--net-label).  The log prints hop coverage and the busiest bonds with their back-hop share.
outputs (results/<system>/): sites.csv, hops_T<T>K.csv, hop_summary.csv, barriers_T<T>K.csv,
                             hop_shell_stats.csv, back_hop_shell.png, hop_rate_ratio_shell.png, nonback_ratio_shell.png
                             (rows: 'any' + each dopant, columns: the two methods, ONE shared y axis per figure),
                             hops_summary.png, hop_network_T<T>K.png,
                             site_activity_T<T>K.csv (per-site Li, hops, rate, distance to each dopant),
                             state_T<T>K.npz (committed site of every Li in every frame; read by 06)
"""
import argparse
import csv
import os
from collections import Counter

import numpy as np
from scipy.ndimage import generate_binary_structure, gaussian_filter, label, maximum_filter
from scipy.spatial import cKDTree
from scipy.stats import linregress, spearmanr

import llzo_io as io
import llzo_maps as mp
import llzo_sites as ls


GLOSSARY = """
HOW TO READ THIS LOG (also saved as hop_glossary.txt)
 site        density peak; a Li 'sits' on its nearest site. hop = Li stays >= --min-res ps on a new site.
 dwell       time between two hops of the same Li.  High back-hop % = Li rattles between two neighbouring sites;
             the hop rate then overstates net transport.
 shell rate  hops of a shell / (Li-time in that shell) = 'per Li per ps'. Two methods (see step 4): site_left / li_at_hop.
             1/rate = mean time a Li stays before leaving. Rate near dopant < rate far = dopant slows hopping.
 back-hop %  hops that undo the same Li's previous hop / hops that HAVE a previous hop (a Li's first hop cannot be one).
 non-back    hops that do not undo the previous hop; its rate is closer to net transport than the raw rate.
 error bar   delete-one-block jackknife over --blocks contiguous time blocks; < 2 SE = noise; hops of one Li are correlated.
 CHANCE      what you would see if every Li hopped independently. Made by giving each Li's hop times a random
             circular time shift (keeps each Li's own rate and where it hops, destroys who-hops-with-whom).
             Repeated --nshuf times: mean = chance level, sd = its noise.
 EXCESS      observed - chance. Normalised: (obs-chance)/(1-chance) = of the hops that would be lonely by chance,
             the share that instead have a partner = genuine coupling.  z = excess/sd(chance); z < ~3 = noise.
 concerted   a hop of ANOTHER Li starts within --tc ps and its sites are within --rc A of ours.
 cluster     hops chained by that rule (A~B, B~C -> one cluster). Chains grow with hop density, so the LARGEST
             cluster is NOT 'N Li moving together'; compare it with its shuffled value.
 refill      a vacated site is entered by another Li within +/- tc. Chance ~ 1-exp(-arrivals per site per ps x window).
 barrier     F_saddle - F_start on the F map (F = -kT ln rho/rho_site). Directional (a->b differs from b->a).
 nu_eff      rate / exp(-barrier/kT). Transition-state theory: rate = nu exp(-barrier/kT), nu ~ 1-10 THz (rule of
             thumb, order of magnitude). nu_eff << 1 THz: F-barrier too low (smoothing fills saddles) -> rank pairs,
             do not quote as absolute Ea.
"""


# ------------------------------------------------------------------ helpers
def wrap_pos(pos, lo, L):
    """Cartesian coordinates in [0, L) relative to the box origin. pos (F,N,3)."""
    w = (pos - lo) % L
    return np.where(w >= L, w - L, w)


def mi(d, L):
    return d - L * np.rint(d / L)


def load_T(cfg, T, tmin):
    traj = io.prepare(cfg, io.dump_path(cfg, T), tmin)
    lm = io.li_mask(cfg, traj)
    drift = io.framework_drift(cfg, traj)
    li = traj.pos[:, lm, :] - drift[:, None, :]
    L = traj.L.mean(axis=0)
    dops = []
    cnt = {}
    for t in cfg["dopant_types"]:
        for i in np.where(traj.types == t)[0]:
            el = cfg["types"][t]
            cnt[el] = cnt.get(el, 0) + 1
            p = ((traj.pos[:, i, :] - drift).mean(axis=0) - traj.lo) % L
            dops.append(dict(label=f"{el}{cnt[el]}", p=p, x=traj.pos[:, i, :] - drift))   # x: instantaneous, (F,3)
    fw = {}                                           # time-mean La/Zr positions (fractional): reference for crystal sites
    for el in ("La", "Zr"):
        m_ = np.isin(traj.types, [t_ for t_, e_ in cfg["types"].items() if e_ == el])
        if m_.any():
            fw[el] = (((traj.pos[:, m_, :] - drift[:, None, :]).mean(axis=0) - traj.lo) / L) % 1.0
    dt = float(np.median(np.diff(traj.time_ps)))
    return dict(T=T, fw=fw, li=li, lo=traj.lo, L=L, Lf=np.asarray(traj.L, float), dt=dt, window=float(traj.time_ps[-1] - traj.time_ps[0]),
                dops=dops, fold=io.get_fold(cfg, L), time=traj.time_ps)


# ------------------------------------------------------------------ 1 sites
def find_sites(frac_all, Lref, spacing, sigma, radius, ratio, merge):
    n = np.maximum(np.rint(Lref / spacing).astype(int), 4)
    vox = Lref / n
    counts = mp.deposit_cic(frac_all, n)
    rho = gaussian_filter(counts, sigma / vox, mode="wrap")
    mean = counts.sum() / counts.size
    size = tuple(int(2 * max(1, round(radius / v)) + 1) for v in vox)
    pk = (rho == maximum_filter(rho, size=size, mode="wrap")) & (rho >= ratio * mean)
    idx = np.argwhere(pk)
    dens = rho[pk]
    order = np.argsort(-dens)
    xyz = (idx[order] + 0.5) * vox
    keep = []
    tree_pts = []
    for p in xyz:                                     # greedy merge of near-duplicate peaks (plateaus)
        if tree_pts and np.min(np.linalg.norm(mi(np.array(tree_pts) - p, Lref), axis=1)) < merge:
            continue
        keep.append(p)
        tree_pts.append(p)
    return np.array(keep) / Lref                      # fractional coordinates


# ------------------------------------------------------------------ 2/3 assignment and hops
def assign(li, lo, L, sites):
    F, N, _ = li.shape
    w = wrap_pos(li, lo, L).reshape(-1, 3)
    tree = cKDTree(sites % L, boxsize=L)
    d, s = tree.query(w, k=2)
    return s[:, 0].reshape(F, N), d[:, 0].reshape(F, N), (d[:, 1] - d[:, 0]).reshape(F, N)


def commit_hops(s, m):
    """Per Li: hop committed when the nearest site differs from the current one for m consecutive frames.
    Returns list of (frame_of_first_frame_at_new_site, site_from, site_to) and the committed state per frame."""
    F, N = s.shape
    events, state = [], np.empty_like(s)
    for i in range(N):
        col = s[:, i]
        cur, cand, cnt, cand_start = col[0], -1, 0, 0
        last = 0
        seg = []                                      # (start_frame, site)
        seg.append((0, cur))
        for f in range(1, F):
            x = col[f]
            if x == cur:
                cand, cnt = -1, 0
                continue
            if x == cand:
                cnt += 1
            else:
                cand, cnt, cand_start = x, 1, f
            if cnt >= m:
                events.append((i, cand_start, cur, cand))
                seg.append((cand_start, cand))
                cur, cand, cnt = cand, -1, 0
        for k, (f0, site) in enumerate(seg):
            f1 = seg[k + 1][0] if k + 1 < len(seg) else F
            state[f0:f1, i] = site
    events.sort(key=lambda e: (e[1], e[0]))
    return events, state


# ------------------------------------------------------------------ 5 concerted hops
def cluster_sizes(f, li, a, b, win, near):
    """Chain hops of different Li that start within `win` frames and whose sites are neighbours. Returns cluster size
    of every hop."""
    n = len(f)
    order = np.argsort(f, kind="stable")
    fo, lo_, ao, bo = f[order].tolist(), li[order].tolist(), a[order].tolist(), b[order].tolist()
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        j = i + 1
        while j < n and fo[j] - fo[i] <= win:
            if lo_[j] != lo_[i] and (near[ao[i], ao[j]] or near[ao[i], bo[j]] or near[bo[i], ao[j]] or near[bo[i], bo[j]]):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
            j += 1
    roots = [find(i) for i in range(n)]
    c = Counter(roots)
    size = np.empty(n, int)
    size[order] = [c[r] for r in roots]
    return size


def refill_flags(f, li, a, b, win):
    """True for hops i (a->b) such that another Li enters site a within +-win frames."""
    arr = {}
    for k in range(len(f)):
        arr.setdefault(b[k], []).append((f[k], li[k]))
    arr = {s: sorted(v) for s, v in arr.items()}
    out = np.zeros(len(f), bool)
    for k in range(len(f)):
        for fj, lj in arr.get(a[k], []):
            if lj != li[k] and abs(fj - f[k]) <= win:
                out[k] = True
                break
    return out


def concerted_stats(ev, F, win, near, shell_of_event, nshuf, seed):
    f, li, a, b = ev["f"], ev["li"], ev["a"], ev["b"]
    n = len(f)
    res = dict(n=n)
    if n < 5:
        return None
    size = cluster_sizes(f, li, a, b, win, near)
    ref = refill_flags(f, li, a, b, win)
    rng = np.random.default_rng(seed)
    N = int(li.max()) + 1
    fc, fr, sh, mx = [], [], [], []
    nsh = int(shell_of_event.max()) + 1
    for _ in range(nshuf):
        off = rng.integers(0, F, N)
        fs = (f + off[li]) % F
        sz = cluster_sizes(fs, li, a, b, win, near)
        rf = refill_flags(fs, li, a, b, win)
        fc.append((sz >= 2).mean())
        fr.append(rf.mean())
        mx.append(sz.max())
        sh.append([(sz[shell_of_event == k] >= 2).mean() if (shell_of_event == k).any() else np.nan for k in range(nsh)])
    fc, fr, sh = np.array(fc), np.array(fr), np.array(sh)
    fo = float((size >= 2).mean())
    ro = float(ref.mean())
    res.update(f_conc=fo, f_conc_ctrl=float(fc.mean()), f_conc_ctrl_sd=float(fc.std(ddof=1)),
               f_conc_binom_se=float(np.sqrt(fo * (1 - fo) / n)),
               refill=ro, refill_ctrl=float(fr.mean()), refill_ctrl_sd=float(fr.std(ddof=1)),
               mean_cluster=float(size[size >= 2].mean()) if (size >= 2).any() else 0.0, max_cluster=int(size.max()), max_cluster_ctrl=float(np.mean(mx)),
               shell_conc=[float((size[shell_of_event == k] >= 2).mean()) if (shell_of_event == k).any() else np.nan
                           for k in range(nsh)],
               shell_conc_ctrl=[float(np.nanmean(sh[:, k])) if np.isfinite(sh[:, k]).any() else np.nan
                                for k in range(nsh)])
    res["conc_excess_norm"] = (fo - res["f_conc_ctrl"]) / (1.0 - res["f_conc_ctrl"]) if res["f_conc_ctrl"] < 1 else np.nan
    res["z_conc"] = (fo - res["f_conc_ctrl"]) / res["f_conc_ctrl_sd"] if res["f_conc_ctrl_sd"] > 0 else np.nan
    res["z_refill"] = (ro - res["refill_ctrl"]) / res["refill_ctrl_sd"] if res["refill_ctrl_sd"] > 0 else np.nan
    res["cluster_hist"] = {s_: c_ // s_ for s_, c_ in Counter(size[size >= 2].tolist()).items()}   # size -> number of clusters
    return res


# ------------------------------------------------------------------ 6 barriers
def bottleneck(Fsub, src, dst):
    """Smallest h such that voxels with Fsub <= h connect src and dst (26-connectivity). Fsub: inf = unusable.
    Exact: binary search over the sorted distinct F values. Returns (h, ok)."""
    fin = np.isfinite(Fsub)
    if not (fin[src] and fin[dst]):
        return np.nan, False
    st = generate_binary_structure(3, 3)
    vals = np.unique(Fsub[fin])
    lo = int(np.searchsorted(vals, max(Fsub[src], Fsub[dst])))
    hi = len(vals) - 1

    def conn(h):
        lab, _ = label(Fsub <= h, structure=st)
        return lab[src] != 0 and lab[src] == lab[dst]

    if not conn(vals[hi]):
        return np.nan, False
    while lo < hi:
        mid = (lo + hi) // 2
        if conn(vals[mid]):
            hi = mid
        else:
            lo = mid + 1
    return float(vals[lo]), True


def pair_barrier(Frel, L, pa, pb, tube, site_r=0.6):
    """Bottleneck F along the a-b tube. Frel: periodic grid (NaN = unsampled), box L, origin 0. pa, pb: Cartesian
    (A) in [0, L). Returns dict(h, Fa, Fb, ok)."""
    n = np.array(Frel.shape)
    vox = L / n
    d = mi(pb - pa, L)
    pb_u = pa + d                                     # unwrapped partner
    lo_i = np.floor((np.minimum(pa, pb_u) - tube - vox) / vox).astype(int)
    hi_i = np.ceil((np.maximum(pa, pb_u) + tube + vox) / vox).astype(int)
    ax = [np.arange(lo_i[k], hi_i[k] + 1) for k in range(3)]
    sub = Frel[np.ix_(*[a % n[k] for k, a in enumerate(ax)])]
    cx = [(ax[k] + 0.5) * vox[k] for k in range(3)]
    X = np.stack(np.meshgrid(*cx, indexing="ij"), axis=-1)
    ab = pb_u - pa
    t = np.clip(((X - pa) @ ab) / (ab @ ab), 0.0, 1.0)
    dist = np.linalg.norm(X - (pa + t[..., None] * ab), axis=-1)
    Fm = np.where((dist <= tube) & np.isfinite(sub), sub, np.inf)

    def site_voxel(p):
        da = np.linalg.norm(X - p, axis=-1)
        cand = np.where(da <= site_r, Fm, np.inf)
        if not np.isfinite(cand).any():
            return None
        return np.unravel_index(np.argmin(cand), cand.shape)

    sa, sb = site_voxel(pa), site_voxel(pb_u)
    if sa is None or sb is None:
        return dict(h=np.nan, Fa=np.nan, Fb=np.nan, ok=False)
    h, ok = bottleneck(Fm, sa, sb)
    return dict(h=h, Fa=float(Fm[sa]), Fb=float(Fm[sb]), ok=ok)


# ------------------------------------------------------------------ plotting
KIND_COLORS = {"24d": "#1f77b4", "96h": "#ff7f0e", "peak": "0.35"}


def plot_network(path, sites, Lm, occ, nmat, dops, T, title, kinds=None, center=None, slab=0.0, window=0.0, shell_edges=(),
                 label="occ", bg_pairs=None, wref=0.0):
    """Hop network on the site list. Dot colour = site type (24d / 96h / ...), dot size = mean occupancy, line width =
    number of hops between the two sites (either direction, whole window), line colour = type of the two sites (same
    type: that colour, mixed: grey). center = a dopant dict: shift the box periodically so it sits at the centre,
    slab > 0 keeps only sites within +/- slab A of the centre plane (per projection), window > 0 zooms to +/- window A,
    dashed circles = shell edges around the centred dopant. Hops across the box boundary are drawn on both sides.
    bg_pairs (P,2 site indices): site pairs drawn as a faint dashed line even where THIS temperature has no hop, so a bond
    that simply was not visited is not mistaken for a missing bond. wref: hop count at which the line width saturates
    (0 = 95th percentile of the hopped pairs, so a few rattling pairs do not squash every other line to hair width).
    label (only with a slab or a window, otherwise the projection overlaps): 'occ' = mean Li on each site (Li-frames / frames),
    'hops' = hops out of the site, 'index' = row in sites.csv, 'both' = occ + index, 'none'."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    S = len(sites)
    Lm = np.asarray(Lm, float)
    base = np.array([str(k).split()[0] for k in kinds]) if kinds is not None else np.array(["peak"] * S)
    ub = sorted(set(base))
    col = {b: KIND_COLORS.get(b, plt.cm.Set2(k % 8)) for k, b in enumerate(ub)}
    shift = Lm / 2 - center["p"] if center is not None else np.zeros(3)
    pos = (sites + shift) % Lm
    dpos = [((d_["p"] + shift) % Lm, d_["label"]) for d_ in dops]
    tot = nmat + nmat.T
    nout = nmat.sum(axis=1)
    pairs = np.argwhere(np.triu(tot, 1) > 0)
    cnts = tot[pairs[:, 0], pairs[:, 1]] if len(pairs) else np.array([1.0])
    wmax_true = float(max(1.0, cnts.max()))
    wmax = float(wref) if wref and wref > 0 else float(max(1.0, np.percentile(cnts, 95)))   # width saturates here
    els = sorted({lab.rstrip("0123456789") for _, lab in dpos})
    pal = dict(zip(els, ["red", "gold", "cyan", "magenta", "lime"]))

    fig, axs = plt.subplots(1, 3, figsize=(18, 7.0), dpi=130)
    shifts = np.array([(a_, b_, c_) for a_ in (-1, 0, 1) for b_ in (-1, 0, 1) for c_ in (-1, 0, 1)], float) * Lm   # 27 periodic images, index 13 = the box itself
    windowed = window > 0 and center is not None
    for ax, (ix, iy, xl, yl) in zip(axs, [(0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z")]):
        iz = 3 - ix - iy
        lo = np.array([Lm[ix] / 2 - window, Lm[iy] / 2 - window]) if windowed else np.zeros(2)
        hi = np.array([Lm[ix] / 2 + window, Lm[iy] / 2 + window]) if windowed else Lm[[ix, iy]].copy()
        pad = 0.6 if windowed else 0.0

        def inwin(P):
            return ((P[..., ix] >= lo[0] - pad) & (P[..., ix] <= hi[0] + pad) &
                    (P[..., iy] >= lo[1] - pad) & (P[..., iy] <= hi[1] + pad))

        def inslab(P, extra=0.0):
            return np.ones(P.shape[:-1], bool) if slab <= 0 else np.abs(P[..., iz] - Lm[iz] / 2) <= slab + extra

        PI = pos[None, :, :] + shifts[:, None, :]                       # (27,S,3) periodic images of every site
        W, SL = inwin(PI), inslab(PI)
        # bonds: draw every periodic copy of the bond that touches the window (both ends inside the slab)
        def bond_segs(pp):
            a_, b_ = pp[:, 0], pp[:, 1]
            dv = mi(sites[b_] - sites[a_], Lm)
            A = pos[a_][None, :, :] + shifts[:, None, :]
            B = A + dv[None, :, :]
            ok = (inwin(A) | inwin(B)) & inslab(A) & inslab(B)
            si, pi_ = np.where(ok)
            return [[(A[s_, p_, ix], A[s_, p_, iy]), (B[s_, p_, ix], B[s_, p_, iy])] for s_, p_ in zip(si, pi_)], pi_

        if bg_pairs is not None and len(bg_pairs):
            bp = np.asarray(bg_pairs, int)
            bp = bp[tot[bp[:, 0], bp[:, 1]] == 0]                         # only bonds with NO hop at this T
            if len(bp):
                bs, _ = bond_segs(bp)
                if bs:
                    ax.add_collection(LineCollection(bs, linewidths=0.6, colors="0.6", linestyles=(0, (3, 2)), alpha=0.6, zorder=1))
        if len(pairs):
            segs, pi_ = bond_segs(pairs)
            a_, b_ = pairs[:, 0], pairs[:, 1]
            w_all = 0.4 + 6.0 * np.minimum(tot[a_, b_], wmax) / wmax
            c_all = np.array([col[base[a]] if base[a] == base[b] else "0.45" for a, b in pairs], dtype=object)
            if segs:
                ax.add_collection(LineCollection(segs, linewidths=w_all[pi_], colors=list(c_all[pi_]), alpha=0.65, zorder=2))
        osz = 14 + 70 * occ / max(occ.max(), 1e-9)
        if slab > 0:                                                    # sites outside the slab: faint, so the lattice is still visible
            gs, gi = np.where(W & ~SL)
            ax.scatter(PI[gs, gi, ix], PI[gs, gi, iy], s=10, c="0.75", alpha=0.35, edgecolor="none", zorder=1)
        for b in ub:
            ss, ii = np.where(W & SL & (base == b)[None, :])
            ax.scatter(PI[ss, ii, ix], PI[ss, ii, iy], s=osz[ii], c=[col[b]], edgecolor="k", linewidth=0.4, zorder=3)
        for p, lab in dpos:
            Pd = p[None, :] + shifts
            for s_ in np.where(inwin(Pd) & inslab(Pd, 1.0))[0]:
                ax.scatter(Pd[s_, ix], Pd[s_, iy], s=260, marker="*", c=pal[lab.rstrip("0123456789")], edgecolor="k",
                           linewidth=1.0, zorder=6)
                if s_ == 13:
                    ax.annotate(lab, (Pd[s_, ix], Pd[s_, iy]), xytext=(6, 6), textcoords="offset points", fontsize=10,
                                fontweight="bold", zorder=7)
        if center is not None:
            c0 = Lm / 2
            for r in shell_edges:
                ax.add_patch(plt.Circle((c0[ix], c0[iy]), r, fill=False, ls="--", lw=0.7, color="0.4", zorder=1))
        if label != "none" and (slab > 0 or windowed):
            ss, ii = np.where(W & SL)
            for s_, i_ in zip(ss, ii):
                txt = {"occ": f"{occ[i_]:.2f}", "hops": f"{int(nout[i_])}", "index": str(i_),
                       "both": f"{occ[i_]:.2f}\n#{i_}"}[label]
                ax.annotate(txt, (PI[s_, i_, ix], PI[s_, i_, iy]), xytext=(3, -8), textcoords="offset points", fontsize=5.5,
                            color="0.15", zorder=7)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        if not windowed:
            ax.set_xlim(0, Lm[ix])
            ax.set_ylim(0, Lm[iy])
        ax.set_aspect("equal")
        ax.set_xlabel(f"{xl} (A)")
        ax.set_ylabel(f"{yl} (A)")
        if slab > 0:
            ax.set_title(f"bright = sites within +/-{slab:g} A of the plane normal to {'xyz'[iz]}; faint = other sites", fontsize=9)
    h = [Line2D([], [], marker="o", ls="", mfc=col[b], mec="k", ms=8, label=f"{b} site ({int((base == b).sum())})") for b in ub]
    h += [Line2D([], [], color=col[b], lw=3, label=f"{b} <-> {b} hop") for b in ub]
    if len(ub) > 1:
        h += [Line2D([], [], color="0.45", lw=3, label="hop between different types")]
    for n in sorted({max(1, int(round(wmax * f))) for f in (0.1, 0.5, 1.0)}):
        clip = (n == int(round(wmax)) and wmax_true > wmax)
        h.append(Line2D([], [], color="0.3", lw=0.4 + 6.0 * n / wmax,
                        label=f"line width: {n}+ hops (clipped, busiest pair {int(wmax_true)})" if clip else f"line width: {n} hops"))
    if bg_pairs is not None and len(bg_pairs):
        h.append(Line2D([], [], color="0.6", lw=1, ls=(0, (3, 2)), label="bond hopped at another T, not at this T"))
    h += [Line2D([], [], marker="*", ls="", mfc=pal[e], mec="k", ms=13, label=f"{e} (dopant)") for e in els]
    fig.legend(handles=h, loc="lower center", ncol=min(len(h), 6), fontsize=8, frameon=False)
    what = f", centred on {center['label']}" if center is not None else ""
    fig.suptitle(f"{title}  T = {T} K{what}   dot size = mean Li occupancy, dashed circles = shell edges (periodic images drawn)"
                 + {"occ": "; numbers = mean Li on the site", "hops": "; numbers = hops out of the site",
                    "index": "; numbers = site index (sites.csv)", "both": "; numbers = mean Li / #site index",
                    "none": ""}[label if (slab > 0 or windowed) else "none"],
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.11, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)


def plot_summary(path, rows, shell_rows, shell_names, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ts = np.array([r["T"] for r in rows], float)
    fig, axs = plt.subplots(2, 2, figsize=(11, 8), dpi=140)
    ax = axs[0, 0]
    ax.errorbar(1000 / Ts, [r["hop_rate_per_Li_ps"] for r in rows], [r.get("hop_rate_se", np.nan) for r in rows], fmt="o-", capsize=2)
    ax.set_yscale("log")
    ax.set_xlabel("1000 / T (1/K)")
    ax.set_ylabel("hops per Li per ps")
    ax.set_title("total hop rate (Arrhenius), block-jackknife SE")
    ax = axs[0, 1]
    ok = [r for r in rows if "f_conc" in r]
    if ok:
        T_ok = np.array([r["T"] for r in ok])
        ax.errorbar(T_ok, [r["f_conc"] for r in ok], [r["f_conc_binom_se"] for r in ok], fmt="o-", label="observed")
        ax.errorbar(T_ok, [r["f_conc_ctrl"] for r in ok], [r["f_conc_ctrl_sd"] for r in ok], fmt="s--", color="0.4",
                    label="chance (shuffled times)")
        ax.legend(fontsize=8)
    ax.set_xlabel("T (K)")
    ax.set_ylabel("fraction of hops in clusters >= 2")
    ax.set_title("concerted hops vs chance")
    ax = axs[1, 0]
    if ok:
        ax.errorbar(T_ok, [r["refill"] for r in ok], fmt="o-", label="observed")
        ax.errorbar(T_ok, [r["refill_ctrl"] for r in ok], [r["refill_ctrl_sd"] for r in ok], fmt="s--", color="0.4",
                    label="chance")
        ax.legend(fontsize=8)
    ax.set_xlabel("T (K)")
    ax.set_ylabel("fraction of hops whose site is refilled <= tc")
    ax.set_title("knock-on / refill vs chance")
    ax = axs[1, 1]
    for k, sn in enumerate(shell_names):
        rr = [next((s_ for s_ in shell_rows if s_["T"] == T and s_["ref"] == "any" and s_["method"] == "site_left"
                    and s_["shell"] == sn), None) for T in Ts]
        ax.errorbar(Ts, [r["rate_per_Li_ps"] if r else np.nan for r in rr], [r["rate_se"] if r else np.nan for r in rr],
                    fmt="o-", capsize=2, label=sn)
    ax.set_xlabel("T (K)")
    ax.set_ylabel("hops per Li per ps (sites of the shell)")
    ax.set_title("hop rate vs distance of the site to the nearest dopant (block-jackknife SE)")
    ax.legend(fontsize=8)
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="folder with the dumps + element_list (or a settings .yaml)")
    ap.add_argument("--defaults")
    ap.add_argument("--T", type=int, nargs="*")
    ap.add_argument("--tmin", type=float)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--results", help="folder holding 01's results/<system> (default: --outdir)")
    ap.add_argument("--spacing", type=float, default=0.2, help="grid for site finding, A")
    ap.add_argument("--sigma-site", type=float, default=0.35, help="Gaussian sigma for site finding, A")
    ap.add_argument("--site-radius", type=float, default=0.8, help="a site must be the maximum within this radius, A")
    ap.add_argument("--site-ratio", type=float, default=3.0, help="site density >= ratio * uniform density")
    ap.add_argument("--merge", type=float, default=0.8, help="merge site peaks closer than this, A")
    ap.add_argument("--sites-cif", help="pristine Ia-3d garnet CIF (24d + 96h Li positions): use these FIXED crystal sites "
                    "instead of density peaks (dopants come from the dump)")
    ap.add_argument("--merge-pairs", type=float, default=0.0,
                    help="with --sites-cif: merge crystal sites closer than this, A (1.0 turns each 0.8 A 96h pair into one site)")
    ap.add_argument("--site-excl", type=float, default=1.0,
                    help="with --sites-cif: remove crystal sites closer than this to a dopant atom, A")
    ap.add_argument("--min-res", type=float, default=0.3, help="min residence at the new site to count a hop, ps")
    ap.add_argument("--tc", type=float, default=0.5, help="concerted-hop time window, ps")
    ap.add_argument("--rc", type=float, default=3.5, help="concerted-hop site distance, A")
    ap.add_argument("--nshuf", type=int, default=100, help="shuffles for the chance level")
    ap.add_argument("--shells", type=float, nargs="*", default=list(io.DEFAULT_SHELLS), help="dopant-distance shell edges, A "
                    "(same default in 04, 05, 06)")
    ap.add_argument("--far", choices=["all", "self"], default="all",
                    help="far shell = farther than the last edge from EVERY dopant (all) or from that dopant only (self)")
    ap.add_argument("--blocks", type=int, default=io.DEFAULT_BLOCKS, help="time blocks for the jackknife error bars")
    ap.add_argument("--nmin", type=int, default=3, help="min hops per site pair for a barrier")
    ap.add_argument("--tube", type=float, default=1.0, help="tube radius around the a-b line for the barrier, A")
    ap.add_argument("--no-barriers", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--net-center", help="also draw hop_network_T<T>K_<dopant>.png with that dopant (Ga, Ga1 or 'all') at the box centre")
    ap.add_argument("--net-slab", type=float, default=0.0, help="with --net-center: keep only sites within +/- this many A of the centre plane")
    ap.add_argument("--net-window", type=float, default=0.0, help="with --net-center: zoom to +/- this many A around the dopant (0 = full box)")
    ap.add_argument("--net-label", choices=["occ", "hops", "index", "both", "none"], default="occ",
                    help="numbers on the sites of the centred network plots (needs --net-slab or --net-window): occ = mean Li on the "
                         "site, hops = hops out of it, index = row in sites.csv, both = occ + index")
    ap.add_argument("--net-bg", choices=["hopped", "geom", "none"], default="hopped",
                    help="faint dashed bonds in the centred plots where THIS T has no hop: hopped = bonds that hopped at any other T, "
                         "geom = every site pair closer than --bond-cut, none")
    ap.add_argument("--net-wref", type=float, default=0.0,
                    help="hop count at which the network line width saturates (0 = 95th percentile of the hopped pairs)")
    ap.add_argument("--bond-cut", type=float, default=2.6,
                    help="site pairs closer than this (A) count as neighbouring bonds in the coverage printout / --net-bg geom")
    args = ap.parse_args()

    cfg = io.load_config(args.target, args.defaults)
    temps = args.T or io.temperatures(cfg)
    outdir = os.path.join(args.outdir, cfg["name"])
    os.makedirs(outdir, exist_ok=True)
    res_dir = os.path.join(args.results or args.outdir, cfg["name"])

    data = {}
    for T in temps:
        print(f"\n=== load {cfg['name']}  T = {T} K ===")
        data[T] = load_T(cfg, T, args.tmin)
    Lref = data[temps[0]]["L"]
    for T in temps:
        if np.abs(data[T]["L"] / Lref - 1).max() > 0.01:
            print(f"  [WARN] box at {T} K differs >1 % from {temps[0]} K; sites are scaled with the box")

    # ---- 1 sites (pooled over temperatures, fractional coordinates)
    pool = np.concatenate([wrap_pos(d["li"], d["lo"], d["L"]).reshape(-1, 3) / d["L"] for d in data.values()])
    sites_frac = peaks_frac = find_sites(pool, Lref, args.spacing, args.sigma_site, args.site_radius, args.site_ratio, args.merge)
    fold = data[temps[0]]["fold"]
    ncell = int(np.prod(fold))
    sites_by_T = site_kind = None
    if args.sites_cif:
        cs = ls.build(args.sites_cif, data, temps, fold, args.site_excl, args.merge_pairs, peaks_frac)
        sites_frac, sites_by_T, site_kind = cs["frac_ref"], cs["frac_by_T"], cs["kind"]
    S = len(sites_frac)
    n_li = data[temps[0]]["li"].shape[1]
    src = "the crystal CIF (fixed)" if sites_by_T else f"the density pooled over {len(temps)} temperatures"
    print(f"\n[sites] {S} sites from {src} "
          f"= {S / ncell:.0f} per conventional cell (garnet: 24d + 96h = 120); N_Li = {n_li} "
          f"({n_li / S:.2f} Li per site)")
    if sites_by_T and S < n_li:
        print("  [WARN] fewer sites than Li atoms: wrong CIF/fold?")
    if sites_by_T is None and S < n_li * 1.2:
        print("  [WARN] fewer sites than ~1.2 x N_Li: peaks are merged; try a smaller --sigma-site / --site-radius / --merge")

    print(GLOSSARY)
    with open(os.path.join(outdir, "hop_glossary.txt"), "w") as fh:
        fh.write(GLOSSARY)

    shell_edges = list(args.shells)
    shell_names = []
    if data[temps[0]]["dops"]:
        e = [0.0] + shell_edges
        shell_names = [f"{e[k]:g}-{e[k + 1]:g}A" for k in range(len(e) - 1)] + [f">{e[-1]:g}A"]
    else:
        shell_names = ["all"]

    summary_rows, shell_rows, site_table = [], [], None
    net_jobs = []
    barrier_note = None
    for T in temps:
        d = data[T]
        L = d["L"]
        dt = d["dt"]
        F, N, _ = d["li"].shape
        sites = (sites_by_T[T] if sites_by_T else sites_frac) * L
        # site geometry: distance to nearest dopant, shell
        if d["dops"]:
            dd = np.min([np.linalg.norm(mi(sites - dop["p"], L), axis=1) for dop in d["dops"]], axis=0)
            shell = np.searchsorted(shell_edges, dd)
        else:
            dd = np.full(S, np.nan)
            shell = np.zeros(S, int)
        site_d = None
        if d["dops"]:
            site_d = {"any": dd, **{dop["label"]: np.linalg.norm(mi(sites - dop["p"], L), axis=1) for dop in d["dops"]}}
        Dss = np.linalg.norm(mi(sites[:, None] - sites[None], L), axis=-1)
        near = (Dss <= args.rc) & (Dss > 0)

        print(f"\n=== {cfg['name']}  T = {T} K ===")
        s_idx, s_dist, s_gap = assign(d["li"], d["lo"], L, sites)
        m = max(1, int(round(args.min_res / dt)))
        events, state = commit_hops(s_idx, m)
        n_ev = len(events)
        print(f"  frame spacing {dt:.3f} ps, hop committed after {m} frames ({m * dt:.2f} ps); "
              f"in transit (>1.2 A from any site): {100 * (s_dist > 1.2).mean():.1f} % of Li-frames")
        print(f"  assignment quality: mean distance to assigned site {s_dist.mean():.2f} A, within 0.5 A: {100 * (s_dist < 0.5).mean():.0f} %, "
              f"ambiguous (2nd-nearest site < 0.2 A farther): {100 * (s_gap < 0.2).mean():.0f} %")
        sh_f = shell[s_idx]
        print("  assignment by dopant shell (mean dist to site, % ambiguous): " + "; ".join(
            f"{shell_names[k]}: {s_dist[sh_f == k].mean():.2f} A, {100 * (s_gap[sh_f == k] < 0.2).mean():.0f} %"
            for k in range(len(shell_names)) if (sh_f == k).any()))
        if (s_gap < 0.2).mean() > 0.25:
            print("  [WARN] >25 % of Li-frames sit between two sites: rattling across close pairs inflates hop counts; "
                  "compare with --merge-pairs 1.0 (crystal sites) or a larger --merge (peaks)")
        if n_ev < 5:
            print("  [WARN] fewer than 5 hops: nothing to analyse at this T")
            continue
        ev = dict(li=np.array([e[0] for e in events]), f=np.array([e[1] for e in events]),
                  a=np.array([e[2] for e in events]), b=np.array([e[3] for e in events]))
        jump = Dss[ev["a"], ev["b"]]
        # dwell at site_from = frames since the same Li's previous hop (or since window start)
        last = {}
        dwell = np.empty(n_ev)
        for k in range(n_ev):
            dwell[k] = (ev["f"][k] - last.get(ev["li"][k], 0)) * dt
            last[ev["li"][k]] = ev["f"][k]
        rate = n_ev / (N * d["window"])
        is_back, elig = io.back_flags(ev["li"], ev["f"], ev["a"], ev["b"])
        back, tot = int(is_back.sum()), int(elig.sum())
        fb = io.block_id(ev["f"], F, args.blocks)
        blk_h = np.bincount(fb, minlength=args.blocks).astype(float)
        blk_t = N * dt * np.bincount(io.block_id(np.arange(F), F, args.blocks), minlength=args.blocks)
        rate_se = float(io.jackknife(lambda h, t_: h / t_, blk_h, blk_t)[1])
        back_se = float(io.jackknife(lambda b_, e_: 100 * b_ / e_, np.bincount(fb, weights=is_back, minlength=args.blocks),
                                     np.bincount(fb, weights=elig, minlength=args.blocks))[1])
        print(f"  hops: {n_ev}   rate = {rate:.4f} +/- {rate_se:.4f} per Li per ps (jackknife over {args.blocks} blocks)   median jump {np.median(jump):.2f} A   "
              f"jumps > {args.rc:g} A: {(jump > args.rc).sum()}")
        print(f"  mean dwell {dwell.mean():.1f} ps (median {np.median(dwell):.1f})   "
              f"back-hops (undo the Li's previous hop): {100 * back / max(tot, 1):.1f} +/- {back_se:.1f} % ({back} of {tot})")

        # occupancy and rates per site
        occ_frames = np.bincount(state.ravel(), minlength=S)             # Li-frames per site
        occ = occ_frames / F                                              # mean Li per site
        print(f"  sites never occupied in this window: {int((occ_frames == 0).sum())} of {S}")
        nmat = np.zeros((S, S))
        np.add.at(nmat, (ev["a"], ev["b"]), 1)
        out_hops = np.bincount(ev["a"], minlength=S)
        in_hops = np.bincount(ev["b"], minlength=S)
        hop_li = np.bincount(ev["li"], minlength=N)
        n_li_out = np.bincount(np.unique(np.stack([ev["a"], ev["li"]], axis=1), axis=0)[:, 0], minlength=S)
        hl = hop_li[hop_li > 0]
        print(f"  Li that hop at least once: {int((hop_li > 0).sum())} of {N}  (hops per hopping Li: mean {hl.mean():.1f}, "
              f"median {np.median(hl):.0f}; the busiest 10 % of Li make "
              f"{100 * np.sort(hop_li)[::-1][:max(1, N // 10)].sum() / max(hop_li.sum(), 1):.0f} % of all hops)")
        with open(os.path.join(outdir, f"site_activity_T{T}K.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["site", "kind", "shell", "dist_to_dopant_A", "mean_Li_on_site", "hops_out", "hops_in",
                        "distinct_Li_leaving", "rate_per_Li_ps"] + [f"dist_{k_}_A" for k_ in (site_d or {}) if k_ != "any"])
            for i_ in range(S):
                w.writerow([i_, site_kind[i_].split()[0] if site_kind is not None else "peak", shell_names[shell[i_]],
                            f"{dd[i_]:.2f}", f"{occ_frames[i_] / F:.3f}", int(out_hops[i_]), int(in_hops[i_]), int(n_li_out[i_]),
                            f"{out_hops[i_] / (occ_frames[i_] * dt):.4f}" if occ_frames[i_] > 0 else "nan"]
                           + [f"{site_d[k_][i_]:.2f}" for k_ in (site_d or {}) if k_ != "any"])
        # committed site of EVERY Li in EVERY frame (also Li that never hop): 06 reads this instead of the hop list
        np.savez_compressed(os.path.join(outdir, f"state_T{T}K.npz"), state=state.astype(np.int16), dt=dt, n_sites=S)

        # ---- how sparse is the network, and who dominates it?
        tot_m = nmat + nmat.T
        iu = np.triu_indices(S, 1)
        cp = tot_m[iu]
        geo = Dss[iu] <= args.bond_cut
        hopped = cp > 0
        print(f"  hop coverage: {int(hopped.sum())} site pairs have >= 1 hop, {int((cp >= args.nmin).sum())} have >= {args.nmin};  "
              f"{int(geo.sum())} site pairs lie within {args.bond_cut:g} A, {int((hopped & geo).sum())} of them hopped "
              f"({100 * (hopped & geo).sum() / max(geo.sum(), 1):.0f} %), {int((hopped & ~geo).sum())} hopped pairs are farther apart (through-hops)")
        hp_ = cp[hopped]
        print(f"  hops per hopped pair: median {np.median(hp_):.0f}, 90th pct {np.percentile(hp_, 90):.0f}, max {int(hp_.max())};   "
              f"hops per Li in the window: {n_ev / N:.1f}")
        bpair = Counter()
        for k in range(n_ev):
            if is_back[k]:
                bpair[(min(ev["a"][k], ev["b"][k]), max(ev["a"][k], ev["b"][k]))] += 1
        top = np.argsort(-cp)[:5]
        print("  busiest bonds (site: kind, shell, mean Li on it):")
        for o in top:
            x, y = int(iu[0][o]), int(iu[1][o])
            kx, ky = (site_kind[x].split()[0], site_kind[y].split()[0]) if site_kind is not None else ("peak", "peak")
            print(f"    {x:3d} ({kx}, {shell_names[shell[x]]}, {occ[x]:.2f}) - {y:3d} ({ky}, {shell_names[shell[y]]}, {occ[y]:.2f})   "
                  f"{Dss[x, y]:.2f} A   {int(cp[o]):4d} hops   back-hops {100 * bpair[(x, y)] / max(cp[o], 1):3.0f} %")
        n_in = int((shell == 0).sum())
        print(f"    top 5 bonds = {100 * cp[top].sum() / cp.sum():.0f} % of all hops;  inner-shell sites ({shell_names[0]}) are "
              f"{100 * n_in / S:.0f} % of the sites but make {100 * (shell[ev['a']] == 0).mean():.0f} % of the hops")
        print("    -> reading: a bond with many hops and a high back-hop share, next to a dopant, is a Li flickering between two sites "
              "the dopant has distorted (not transport). Ordinary far bonds should each have a similar, modest count; if most far bonds "
              "have 0-3 hops the window is simply too short for a bond-by-bond picture (use per-site rates / MSD instead).")
        with open(os.path.join(outdir, f"hops_T{T}K.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["li", "t_ps", "site_from", "site_to", "jump_A", "dwell_ps", "frame"])
            for k in range(n_ev):
                w.writerow([ev["li"][k], f"{d['time'][ev['f'][k]] - d['time'][0]:.2f}", ev["a"][k], ev["b"][k],
                            f"{jump[k]:.2f}", f"{dwell[k]:.2f}", int(ev["f"][k])])

        # shells: shared definitions and error bars (llzo_io.hop_shell_analysis; 06 calls the same function)
        ev_shell = shell[ev["a"]]
        if d["dops"]:
            li_any = None
            li_d = {}
            for dop in d["dops"]:
                li_d[dop["label"]] = io.dist_to(d["li"], dop["x"], d["Lf"])
                li_any = li_d[dop["label"]] if li_any is None else np.minimum(li_any, li_d[dop["label"]])
            li_d = {"any": li_any, **li_d}
            rows_T = io.hop_shell_analysis(T, dt, ev, shell_names, np.array(shell_edges), args.far, args.blocks,
                                           state=state, site_d=site_d, li_d=li_d)
            shell_rows += rows_T
            io.print_hop_shell(rows_T, T)
            print("    -> reading: rate/far < 1 = dopant slows hops; back - far > 0 = more rattling near the dopant. If site_left and "
                  "li_at_hop disagree, site assignment near the dopant is the weak point (see the assignment table above).")
        else:
            print("  no dopant: one shell, distance tables skipped")

        row = dict(T=T, N_Li=N, window_ps=d["window"], n_hops=n_ev, hop_rate_per_Li_ps=rate, median_jump_A=float(np.median(jump)),
                   hop_rate_se=rate_se, mean_dwell_ps=float(dwell.mean()), back_hop_pct=100 * back / max(tot, 1),
                   back_hop_pct_se=back_se,
                   transit_pct=100 * float((s_dist > 1.2).mean()))

        # concerted
        win = int(round(args.tc / dt))
        cs = concerted_stats(ev, F, win, near, ev_shell, args.nshuf, args.seed)
        if cs:
            row.update({k: v for k, v in cs.items() if k not in ("shell_conc", "shell_conc_ctrl", "cluster_hist", "n")})
            ch = cs["cluster_hist"]
            row.update(clusters_2=ch.get(2, 0), clusters_3=ch.get(3, 0), clusters_4=ch.get(4, 0),
                       clusters_5plus=sum(v for k, v in ch.items() if k >= 5))
            print(f"  concerted (tc = {args.tc:g} ps, rc = {args.rc:g} A): {100 * cs['f_conc']:.1f} % of hops in clusters >= 2 "
                  f"(chance {100 * cs['f_conc_ctrl']:.1f} +/- {100 * cs['f_conc_ctrl_sd']:.1f} %, z = {cs['z_conc']:.1f})")
            print(f"    concerted beyond chance: (obs - chance)/(1 - chance) = {100 * cs['conc_excess_norm']:.0f} % of hops")
            print("    NOTE: the chance level is often large (many hops per window); only the excess over chance means anything.")
            print(f"    clusters by size: {dict(sorted(ch.items()))}   largest {cs['max_cluster']}")
            print(f"    largest cluster: observed {cs['max_cluster']}, shuffled (chance) {cs['max_cluster_ctrl']:.0f}. Clusters are CHAINS, "
                  "so a big number = dense hopping, not N Li moving together; only observed >> chance means something.")
            print(f"    -> reading: if Li hopped independently, {100 * cs['f_conc_ctrl']:.0f} % of hops would still have a neighbour hop by luck. "
                  f"Of the other {100 * (1 - cs['f_conc_ctrl']):.0f} %, {100 * cs['conc_excess_norm']:.0f} % do have a partner: that is real coupling (z > ~3 = not noise).")
            print(f"  refill (vacated site entered by another Li within tc): {100 * cs['refill']:.1f} % "
                  f"(chance {100 * cs['refill_ctrl']:.1f} +/- {100 * cs['refill_ctrl_sd']:.1f} %, z = {cs['z_refill']:.1f})")
            span = (2 * win + 1) * dt
            p_poi = 1.0 - np.exp(-n_ev / S / d["window"] * span)
            print(f"    -> reading: another Li lands on the vacated site within +/-{args.tc:g} ps in {100 * cs['refill']:.0f} % of hops; "
                  f"independent Li would do so {100 * cs['refill_ctrl']:.0f} % (shuffle) / {100 * p_poi:.0f} % (formula 1-exp(-arrivals per site per ps x {span:.1f} ps)). "
                  f"obs/chance = {cs['refill'] / cs['refill_ctrl']:.1f}x = knock-on strength.")
            if len(shell_names) > 1:
                for k, sn in enumerate(shell_names):
                    print(f"    shell {sn:>8}: concerted {100 * cs['shell_conc'][k]:.1f} % (chance {100 * cs['shell_conc_ctrl'][k]:.1f} %)"
                          f"   n_hops = {int((ev_shell == k).sum())}")
        summary_rows.append(row)

        # barriers
        if not args.no_barriers:
            npz = os.path.join(res_dir, f"density_F_T{T}K.npz")
            if not os.path.exists(npz):
                barrier_note = f"{npz} not found: run 01 first (or use --no-barriers)"
            else:
                z = np.load(npz)
                Fr, Lb = z["F_rel"], z["L"]
                kT = io.KB_EV * T
                pw = np.rint(nmat + nmat.T)
                pairs = [(a, b) for a, b in np.argwhere(np.triu(pw) >= args.nmin)]
                # F grid origin is the box origin lo, our sites are relative to lo too
                occ_time = occ_frames * dt                                # Li-ps on each site
                brow = []
                for a, b in pairs:
                    r = pair_barrier(Fr, Lb, sites[a], sites[b], args.tube)
                    for (x, y, Fx) in ((a, b, r["Fa"]), (b, a, r["Fb"])):
                        n_xy = nmat[x, y]
                        gam = n_xy / occ_time[x] if occ_time[x] > 0 else np.nan
                        bar = r["h"] - Fx if r["ok"] else np.nan
                        brow.append(dict(site_from=x, site_to=y, dist_A=float(Dss[x, y]), n_hops=int(n_xy), Gamma_per_ps=gam,
                                         barrier_eV=bar, barrier_kT=bar / kT if r["ok"] else np.nan, blocked=not r["ok"],
                                         shell=shell_names[shell[x]],
                                         nu_eff_THz=gam * np.exp(bar / kT) if r["ok"] and n_xy > 0 else np.nan))
                with open(os.path.join(outdir, f"barriers_T{T}K.csv"), "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(brow[0].keys())) if brow else None
                    if w:
                        w.writeheader()
                        w.writerows(brow)
                good = [b for b in brow if not b["blocked"] and b["n_hops"] > 0]
                nb_ = sum(b["blocked"] for b in brow)
                if good:
                    bb = np.array([b["barrier_eV"] for b in good])
                    nu = np.array([b["nu_eff_THz"] for b in good])
                    row["barrier_median_eV"] = float(np.median(bb))
                    row["barrier_median_kT"] = float(np.median([b["barrier_kT"] for b in good]))
                    row["barrier_p16_eV"], row["barrier_p84_eV"] = (float(x) for x in np.percentile(bb, [16, 84]))
                    row["nu_eff_median_THz"] = float(np.median(nu))
                    print(f"  barriers ({len(pairs)} site pairs with >= {args.nmin} hops): median {np.median(bb):.3f} eV "
                          f"(16-84 %: {row['barrier_p16_eV']:.3f} - {row['barrier_p84_eV']:.3f}), blocked directions: {nb_}, "
                          f"nu_eff median {np.median(nu):.2g} THz (expect ~1-10 if F-barrier and rate agree)")
                    print(f"    -> reading: nu_eff = rate / exp(-barrier/kT). Median {np.median(nu):.2g} THz vs ~1-10 expected: "
                          + ("observed hops are much rarer than the F-barrier predicts -> F-barrier probably too LOW (smoothing fills saddles). "
                             "Rank pairs with it; do not quote as absolute Ea." if np.median(nu) < 0.5 else "F-barrier and observed rate roughly consistent."))
                    if len(good) > 10:
                        gk = np.array([b["barrier_kT"] for b in good])
                        lg = np.log([b["Gamma_per_ps"] for b in good])
                        sp_, lr_ = spearmanr(gk, lg).correlation, linregress(gk, lg)
                        sd_nu = float(np.std(np.log10(nu)))
                        row["barrier_rate_spearman"], row["barrier_rate_slope"], row["log10_nu_sd"] = float(sp_), float(lr_.slope), sd_nu
                        print(f"  F-vs-rate check over {len(good)} directed pairs: Spearman(barrier_kT, ln rate) = {sp_:.2f}, slope = {lr_.slope:.2f} "
                              f"+/- {lr_.stderr:.2f} (theory -1), sd(log10 nu_eff) = {sd_nu:.2f}")
                        print("    -> reading: does the F map RANK the pairs correctly? Spearman near -1 and slope near -1 = yes (high barrier <-> slow pair). "
                              "sd(log10 nu_eff) small (<~0.3 = factor 2) = one common prefactor fits all pairs. Absolute level of nu_eff does not matter for ranking. "
                              "Caveat: F and rate come from the same trajectory, so this is a consistency check, not an independent one.")
                else:
                    print(f"  barriers: none resolved ({nb_} directions blocked/unsampled inside the tube)")

        net_jobs.append(dict(T=T, sites=sites, L=L, occ=occ, nmat=nmat, dops=d["dops"],
                             geo=np.argwhere(np.triu((Dss <= args.bond_cut) & (Dss > 0), 1))))
        if site_table is None:
            site_table = dict(sites=sites, dd=dd, shell=shell)

    # ---- hop networks (drawn after the T loop: a bond that hopped at another T is shown faintly where this T has none)
    used = np.zeros((S, S), bool)
    for jb in net_jobs:
        used |= (jb["nmat"] + jb["nmat"].T) > 0
    bg_hopped = np.argwhere(np.triu(used, 1))
    for jb in net_jobs:
        T, d_ = jb["T"], data[jb["T"]]
        plot_network(os.path.join(outdir, f"hop_network_T{T}K.png"), jb["sites"], jb["L"], jb["occ"], jb["nmat"], jb["dops"], T,
                     cfg["name"], kinds=site_kind, shell_edges=shell_edges, wref=args.net_wref)
        if args.net_center:
            bg = {"hopped": bg_hopped, "geom": jb["geo"], "none": None}[args.net_bg]
            for dop in jb["dops"]:
                if args.net_center in (dop["label"], dop["label"].rstrip("0123456789"), "all"):
                    plot_network(os.path.join(outdir, f"hop_network_T{T}K_{dop['label']}.png"), jb["sites"], jb["L"], jb["occ"],
                                 jb["nmat"], jb["dops"], T, cfg["name"], kinds=site_kind, center=dop, slab=args.net_slab,
                                 window=args.net_window, shell_edges=shell_edges, label=args.net_label, bg_pairs=bg,
                                 wref=args.net_wref)

    if not summary_rows:
        raise SystemExit("no temperature produced enough hops")
    if barrier_note:
        print(f"\n[barriers skipped] {barrier_note}")

    # ---- tables
    keys = []
    for r in summary_rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(outdir, "hop_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(summary_rows)
    shell_out = io.write_hop_shell_outputs(shell_rows, shell_names, outdir, cfg["name"])
    with open(os.path.join(outdir, "sites.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["site", "frac_x", "frac_y", "frac_z", "dist_to_nearest_dopant_A", "shell", "kind"])
        for i in range(S):
            w.writerow([i, *[f"{x:.4f}" for x in sites_frac[i]], f"{site_table['dd'][i]:.2f}", shell_names[site_table['shell'][i]],
                        site_kind[i] if site_kind is not None else "peak"])
    print(f"\n[+] {outdir}/hop_summary.csv sites.csv hops_T*.csv " + " ".join(shell_out) + " " + ("" if args.no_barriers else " barriers_T*.csv"))

    Tarr = np.array([r["T"] for r in summary_rows], float)
    if len(Tarr) >= 3:
        x = 1.0 / (io.KB_EV * Tarr)
        p, cov = np.polyfit(x, np.log([r["hop_rate_per_Li_ps"] for r in summary_rows]), 1, cov=True)
        print(f"  Ea(total hop rate) = {-p[0]:.3f} +/- {np.sqrt(cov[0, 0]):.3f} eV (regression error only; compare Ea(D*) from 03)")
        print("    -> reading: this Ea counts every hop incl. back-and-forth rattling, so it is usually LOWER than Ea(D*) (net transport) from 03.")
        bm = [(r["T"], r["barrier_median_eV"]) for r in summary_rows if "barrier_median_eV" in r]
        if len(bm) > 1:
            bk = [(r["T"], r["barrier_median_kT"]) for r in summary_rows if "barrier_median_kT" in r]
            print("    -> median F-barrier vs T: " + ", ".join(f"{t_:g} K {b_:.3f}" for t_, b_ in bm) + " eV   |   "
                  + ", ".join(f"{t_:g} K {b_:.2f}" for t_, b_ in bk) + " kT")
            print("       test: a real energy barrier is flat in eV; if it is flat in kT (rises ~ proportional to T) it is only a "
                  "density RATIO rho_saddle/rho_site set by geometry/smoothing, not an activation energy.")
    plot_summary(os.path.join(outdir, "hops_summary.png"), summary_rows, shell_rows, shell_names, cfg["name"])
    print(f"[+] {outdir}/hops_summary.png  hop_network_T*.png")


if __name__ == "__main__":
    main()
