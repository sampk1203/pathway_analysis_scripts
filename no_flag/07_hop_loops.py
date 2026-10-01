#!/usr/bin/env python3
"""
07_hop_loops.py - hop statistics of 04 with ALL retracing removed: back-hop degree (palindromes), loop erasure and a
displacement-validated "robust net" hop rate. Read-only on 04. Everything is reported PER TEMPERATURE (nothing is pooled over T).

Problem it solves: 04 counts a hop whenever the nearest ideal site changes, and its back-hop % only sees a->b->a. Near a dopant the
sites fit badly and Li flicker between them, so hop counts there are mostly retracing. Here every Li's hop sequence
(hops_cif/<s>/hops_T<T>K.csv) is analysed as a path on the site graph:
  retrace degree d   hop k completes a palindromic site path of 2d hops (a>b>a = 1, a>b>c>b>a = 2, a>b>a>b>a = 2 ...). 0 = none.
  loop erasure       chronological loop erasure of the path: a hop that arrives at ANY site already on the current kept path
                     (within --maxloop kept steps) closes a loop; every hop of that loop is erased and the path is cut back to that
                     site. Palindromes are the loops that retrace themselves; a>b>c>a is a loop that does not. Surviving hops = NET.
  small displacement a hop is "small" if the Li's mean position over --win-ps after the hop is < --small A from its mean position
                     over --win-ps before it. A real jump between neighbouring sites moves the Li ~1.5-3 A; flicker between
                     near-degenerate sites near a dopant does not. Hops too close to the start/end of the run cannot be tested (NaN,
                     kept as net).
  ROBUST net hop    NET (not in a loop) AND not small. This removes flicker that loop erasure missed (flicker among 3+ sites that
                    never closes a loop within --maxloop, or loops longer than --maxloop).
Per distance shell (Li distance to each dopant in the hop frame; same shells, far reference and jackknife as 04/05/06; llzo_io):
  all/far     hop rate per Li-time in the shell / same for far Li (all 04 hops)
  NET/far     same, loop-erased hops only
  ROBUST/far  same, net AND not-small hops only
Validation columns: %small|net and %small|loop = % of validated net / erased hops with displacement < --small. Loop hops should be
mostly small; net hops mostly not. (n) = number of hops that could be validated.
Errors: +/- = delete-one-block jackknife over --blocks time blocks of that ONE run (hops of one Li are correlated and the far shell is
not independent of the others: treat differences < 2 SE as noise). * = fewer than 20 hops in that cell (SE unreliable).
Limits: loops longer than --maxloop kept steps are not erased (check 4 and 16); the hop list inherits 04's --min-res and site
assignment; a cell like Ga 0-3 A holds ~1 Li, so its counts per temperature are small - more Li-time (longer runs, more dopants) is the
only cure for that, no analysis can add events.
usage: python 07_hop_loops.py /path/to/MD_run --hops-dir hops_cif/MD_run_1Ga_1Ru [--maxloop 8] [--win-ps 0.3] [--small 1.0]
outputs (in --outdir, default --hops-dir): hop_loops.csv (one row per T, dopant, shell), hop_loops.png, hop_flags_T<T>K.csv (one row
per hop: shell, erased, loop length, retrace degree, displacement, robust flag)
"""
import argparse
import csv
import os

import numpy as np

import llzo_io as io

FEW = 20                                         # fewer events than this: open symbol / '*'
MIN_N = getattr(io, 'MIN_N', 5)                  # fewer events than this: no SE (an SE from 0-4 events is meaningless)


def degrees(a, b, dmax):
    """a, b: site_from/site_to of ONE Li's hops (time order). deg[k] = largest d <= dmax such that hops k-d+1..k mirror hops
    k-2d+1..k-d (a palindromic path of 2d hops); 0 = no retrace."""
    n = len(a)
    deg = np.zeros(n, int)
    for k in range(1, n):
        for d in range(min(dmax, (k + 1) // 2), 0, -1):
            lo = k - 2 * d + 1
            ok = all(a[k - d + 1 + j] == b[k - d - j] and b[k - d + 1 + j] == a[k - d - j] for j in range(d))
            ok = ok and all(a[i + 1] == b[i] for i in range(lo, k))
            if ok:
                deg[k] = d
                break
    return deg


def loop_erase(a, b, maxloop):
    """Chronological loop erasure of one Li's site path. Returns (erased (n,) bool, loop_len (n,)): erased = hop belongs to a closed
    loop; loop_len>0 on the hop that closes it (hops in the loop). A break in the path (a[k] != previous b) restarts it."""
    n = len(a)
    erased = np.zeros(n, bool)
    loop_len = np.zeros(n, int)
    st = []                                     # kept path: [site, arrival hop]
    for k in range(n):
        if not st or st[-1][0] != a[k]:
            st = [[a[k], -1]]
        q = -1
        for j in range(len(st) - 2, max(-1, len(st) - 2 - maxloop), -1):
            if st[j][0] == b[k]:
                q = j
                break
        if q >= 0:
            for node in st[q + 1:]:
                erased[node[1]] = True
            erased[k] = True
            loop_len[k] = len(st) - q
            del st[q + 1:]
        else:
            st.append([b[k], k])
    return erased, loop_len


def plot(rows, labs, names, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    K = len(names)
    cols = [plt.cm.viridis(k / max(K - 2, 1)) for k in range(K - 1)]
    panels = (("all_ratio", "all_ratio_se", "n_hops", "all hops / far (as 04)"),
              ("net_ratio", "net_ratio_se", "n_net", "NET (loops erased) / far"),
              ("rob_ratio", "rob_ratio_se", "n_robust", "ROBUST (net, not small) / far"))
    fig, axs = plt.subplots(len(labs), 3, figsize=(12.5, 3.2 * len(labs) + 0.9), dpi=150, squeeze=False, sharex=True, sharey="col")
    for ri, lab in enumerate(labs):
        for ci, (key, se, nkey, ttl) in enumerate(panels):
            a = axs[ri][ci]
            for k in range(K - 1):
                sel = sorted([r for r in rows if r["dopant"] == lab and r["shell"] == names[k]], key=lambda r: r["T"])
                if not sel:
                    continue
                x = [r["T"] for r in sel]
                y = [r[key] for r in sel]
                e = [r[se] if np.isfinite(r[se]) else 0.0 for r in sel]
                a.errorbar(x, y, e, color=cols[k], lw=1.2, capsize=2)
                for r_, y_ in zip(sel, y):
                    a.plot(r_["T"], y_, "o", ms=4, color=cols[k], mfc="white" if r_[nkey] < FEW else cols[k])
            a.axhline(1, color="0.5", lw=0.8)
            a.set_title(f"{lab}: {ttl}", fontsize=9)
            if ri == len(labs) - 1:
                a.set_xlabel("T (K)")
    h = [Line2D([], [], color=cols[k], marker="o", lw=2, label=f"Li {names[k]} from dopant") for k in range(K - 1)]
    h.append(Line2D([], [], color="0.4", marker="o", mfc="white", lw=0, label=f"open: < {FEW} hops"))
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("hop rate relative to far Li, per temperature; error bars = block jackknife within each run", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)


def _ratio(v, se, n, ref):
    if ref:
        return f"{'1 (ref)':>13} "
    if not np.isfinite(v):
        return f"{'n/a':>13} "
    s = f"{v:5.2f}+/-{se:4.2f}" if np.isfinite(se) else f"{v:5.2f}+/- nan"
    return f"{s}{'*' if n < FEW else ' '}"


def _small(p, n):
    return f"{p:5.1f} ({n:3d})" if n else f"{'n/a':>10}"


def print_T(T, rows, a_):
    print(f"\nT = {T} K   maxloop {a_.maxloop}; small = displacement < {a_.small} A over {a_.win_ps} ps; +/- = block jackknife SE; "
          f"* = < {FEW} hops")
    print(f"{'dopant':>6} {'shell':>6} {'hops':>6} {'%loop':>6} {'%palin':>6} {'%nonpal':>7} {'net':>5} {'robust':>6} "
          f"{'all/far':>14} {'NET/far':>14} {'ROBUST/far':>14} {'%small|net (n)':>14} {'%small|loop (n)':>15}")
    for r in rows:
        if not r["n_hops"]:
            continue
        ref = r["is_far"]
        print(f"{r['dopant']:>6} {r['shell']:>6} {r['n_hops']:6d} {r['pct_in_loops']:6.1f} {r['pct_palindromic']:6.1f} "
              f"{r['pct_nonpal']:7.1f} {r['n_net']:5d} {r['n_robust']:6d} "
              f"{_ratio(r['all_ratio'], r['all_ratio_se'], r['n_hops'], ref):>14} "
              f"{_ratio(r['net_ratio'], r['net_ratio_se'], r['n_net'], ref):>14} "
              f"{_ratio(r['rob_ratio'], r['rob_ratio_se'], r['n_robust'], ref):>14} "
              f"{_small(r['pct_small_net'], r['n_val_net']):>14} {_small(r['pct_small_loop'], r['n_val_loop']):>15}")


def definitions(a_, names):
    """Full legend of every quantity in the tables, printed at the end of the log (and saved as hop_loops_definitions.txt)."""
    sh = ", ".join(names)
    return f"""
{'=' * 118}
DEFINITIONS (this run: maxloop {a_.maxloop}, dmax {a_.dmax}, small < {a_.small} A, window {a_.win_ps} ps, {a_.blocks} jackknife blocks, shells {sh})
{'=' * 118}
INPUT
  A hop = one row of hops_cif/<system>/hops_T<T>K.csv (written by 04): Li number, frame, site_from, site_to. 04 declares a hop when a Li's
  nearest ideal crystal site changes (and the Li stays on the new site for 04's --min-res). Each temperature is a separate, independent
  MD run, so everything below is computed and reported per temperature; nothing is pooled over T.
  Each Li's hops are put in time order and treated as a PATH on the graph of sites: site_from -> site_to -> ...

SHELL (rows)  and  DOPANT (rows)
  shell   distance of the Li to the dopant IN THE FRAME OF THE HOP ({sh}). The Li's own position at the hop decides the shell, not the
          site it leaves.  >{a_.shells[-1]:g}A is the FAR reference: Li that are not near the dopant. Every ratio is relative to it.
  dopant  the atom the distance is measured from. Ga1, Ru1 ... are individual dopant atoms; 'any' uses whichever dopant is nearest to the Li.
          The far row is the same Li pool for every dopant (identical numbers in all blocks). With far=all a Li that is far from THIS
          dopant but near ANOTHER one is excluded from both near and far (it is neither).

PATH QUANTITIES (per hop)
  retrace degree d  hop k completes an exact palindrome of 2d hops: a>b>a is d=1, a>b>c>b>a is d=2, a>b>a>b>a is d=2. 0 = no retrace.
                    Checked up to --dmax. A palindrome is an exact return along the same sites.
  loop erasure      chronological. Keep a current path for each Li. A hop that arrives at ANY site already on the current path (looking back at
                    most --maxloop kept steps) closes a LOOP. Every hop of that loop is ERASED and the path is cut back to that site.
                    It is not limited to returning to the start: 1>2, 2>3, 3>4, 4>2 closes the loop 2>3>4>2 (3 hops erased); 1>2 is kept.
                    a>b>a is the shortest loop; a>b>c>a is a loop that is not a palindrome. A hop that does not start where the previous
                    one ended (a gap in the path) restarts the path. Loops longer than --maxloop are NOT erased.
  ERASED / in loop  the hop belongs to a closed loop. Erased does not mean fake: a real jump that is later undone is erased too. It only means
                    the hop contributed no lasting progress along the path.
  NET hop           a hop that survives loop erasure (not in any closed loop): progress along a self-avoiding path.
  small hop         the Li's mean position over {a_.win_ps} ps AFTER the hop is < {a_.small} A from its mean position over {a_.win_ps} ps BEFORE it. A real
                    jump between neighbouring sites moves the Li about 1.5-3 A; flicker between near-degenerate sites (typical close to a
                    dopant, where the sites fit badly) barely moves the mean position. Hops within {a_.win_ps} ps of the start or end of the
                    run cannot be tested (displacement = nan): they are kept as non-small and counted under 'not testable'.
  ROBUST net hop    NET and not small. Removes flicker that loop erasure misses (flicker among 3 or more sites that never closes a loop
                    within --maxloop, and loops longer than --maxloop).

COLUMNS OF THE PER-TEMPERATURE TABLES
  hops          all hops of 04 in that shell at this T.
  %loop         100 * (erased hops) / hops. Both the outward and the return hop of a loop count, so %loop > %palin.
  %palin        100 * (hops with retrace degree >= 1) / hops: exact retraces only.
  %nonpal       100 * (hops that close a loop but are not exact retraces) / hops, e.g. a>b>c>a. Counted on the closing hop only, so small.
  net           number of hops that are not erased (NET hops).
  robust        number of ROBUST net hops (net and not small).
  all/far       (hops in the shell / Li-time in the shell) / (same for the far shell). Li-time = time all Li spent in that shell, so a shell
                holding few Li is not penalised: this is a hop RATE per Li, not a raw count. 1 = same as far Li, >1 = more hops, <1 = fewer.
                Counts all 04 hops, loops included.
  NET/far       the same ratio using NET hops only: how often a Li in this shell makes progress that is not undone, vs far Li.
  ROBUST/far    the same ratio using ROBUST hops only: NET hops that also moved the Li by >= {a_.small} A.
  Reading all/far with NET/far: high all/far with low NET/far = lots of hopping that is undone (retracing / flicker); both near 1 = same
                transport as far Li; both low = genuinely slowed. NET/far and ROBUST/far are the transport-relevant ones; compare them with
                the 05 MSD ratio and the 06 non-back rate. If ROBUST/far is much below NET/far, the net hops in that shell were mostly flicker.
  +/-           delete-one-block jackknife SE over {a_.blocks} time blocks of that single run (a hop list of one Li is correlated, and the far
                shell is not independent of the others): treat differences smaller than 2 SE as noise. With only {a_.blocks} blocks the SE is itself
                uncertain (~35 %). Blank (nan) when the cell has fewer than {MIN_N} events: an SE from 0-4 events is meaningless.
  *             fewer than {FEW} events in that cell (hops for all/far, net hops for NET/far, robust hops for ROBUST/far): the value is
                not reliable. Open symbols in the plot mean the same.
  1 (ref)       the far shell, equal to 1 by definition.
  %small|net (n)   % of the NET hops whose displacement is < {a_.small} A, among the n net hops that could be tested.
  %small|loop (n)  % of the ERASED hops whose displacement is < {a_.small} A, among the n erased hops that could be tested.
                   If erasure separates flicker from real jumps, %small|loop should be high and %small|net low. A high %small|net means net
                   hops are still partly flicker (that is why ROBUST exists). Only hops that could be tested are counted, so n < hops.

THE T=... SUMMARY LINE
  hops = all hops at that T; in loops = erased; palindromic retraces = retrace degree >= 1; net = not erased; robust net = net and
  displacement >= {a_.small} A; not testable = hops too close to the ends of the run for the displacement test.

LIMITS
  Loops longer than --maxloop are not erased (run --maxloop 4 and 16). The 'small' threshold is a judgement (run --small 0.7 and 1.5).
  The hop list inherits 04's --min-res and its site assignment, which fits badly right next to a dopant. A cell such as Ga 0-3 A holds about
  one Li, so its counts per temperature are small; more Li-time (longer runs, more dopants) is the only way to add events. Look for the
  same sign at all temperatures rather than one pooled number. Loop erasure works on site indices, so a genuine path that winds around the
  periodic box and returns to the same site index would also be erased (only non-palindromic loops are affected, a few % of hops).

FILES
  hop_loops.csv           one row per (T, dopant, shell): n_hops, n_loop, n_net, n_robust, pct_in_loops(+_se), pct_palindromic, pct_nonpal,
                          all_/net_/rob_ rate (per Li per ps), ratio and ratio_se, pct_small_net, n_val_net, pct_small_loop, n_val_loop,
                          n_not_testable, maxloop, small_A, win_ps.
  hop_loops.png           all / NET / ROBUST ratio vs T, one line per shell; open symbols = fewer than {FEW} events.
  hop_flags_T<T>K.csv     one row per hop: li, frame, site_from, site_to, shell_any, d_any_A (distance to nearest dopant), in_loop, loop_len
                          (hops in the loop, on the closing hop), retrace_deg, disp_A, small, robust_net (1/0).
{'=' * 118}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target")
    ap.add_argument("--hops-dir", required=True)
    ap.add_argument("--T", type=int, nargs="*")
    ap.add_argument("--tmin", type=float)
    ap.add_argument("--maxloop", type=int, default=8, help="longest loop (kept steps) that is erased")
    ap.add_argument("--dmax", type=int, default=8, help="longest palindrome (degree) checked")
    ap.add_argument("--win-ps", type=float, default=0.3, help="displacement window for the small-hop test")
    ap.add_argument("--small", type=float, default=1.0, help="hops with displacement below this (A) are 'small' (flicker)")
    ap.add_argument("--shells", type=float, nargs="*", default=list(io.DEFAULT_SHELLS))
    ap.add_argument("--far", choices=["all", "self"], default="all")
    ap.add_argument("--blocks", type=int, default=io.DEFAULT_BLOCKS)
    ap.add_argument("--no-flags", action="store_true", help="do not write the per-hop hop_flags_T<T>K.csv files")
    ap.add_argument("--outdir")
    a_ = ap.parse_args()

    cfg = io.load_config(a_.target)
    temps = a_.T or io.temperatures(cfg)
    outdir = a_.outdir or a_.hops_dir
    os.makedirs(outdir, exist_ok=True)
    edges = np.array(a_.shells, float)
    names = io.shell_names(edges)
    K = len(names)
    rows = []
    for T in temps:
        p = os.path.join(a_.hops_dir, f"hops_T{T}K.csv")
        if not os.path.exists(p):
            print("missing", p)
            continue
        traj = io.prepare(cfg, io.dump_path(cfg, T), a_.tmin, verbose=False)
        drift = io.framework_drift(cfg, traj)
        li = traj.pos[:, io.li_mask(cfg, traj), :] - drift[:, None, :]
        F, N = li.shape[:2]
        dt = float(np.median(np.diff(traj.time_ps)))
        nwin = max(1, int(round(a_.win_ps / dt)))
        dops, cnt = {}, {}
        for t in cfg["dopant_types"]:
            for i in np.where(traj.types == t)[0]:
                el = cfg["types"][t]
                cnt[el] = cnt.get(el, 0) + 1
                dops[f"{el}{cnt[el]}"] = traj.pos[:, i, :] - drift
        if not dops:
            raise SystemExit("no dopant atoms: nothing to compare")
        dists = {lab: io.dist_to(li, x, traj.L) for lab, x in dops.items()}
        dnear = np.min(list(dists.values()), axis=0)
        dists = {"any": dnear, **dists}

        H = list(csv.DictReader(open(p)))
        if not H:
            print(f"T={T}: no hops in {p}")
            continue
        L_ = np.array([int(h["li"]) for h in H])
        f_ = np.array([int(h["frame"]) for h in H])
        sa = np.array([int(h["site_from"]) for h in H])
        sb = np.array([int(h["site_to"]) for h in H])
        deg = np.zeros(len(H), int)
        erased = np.zeros(len(H), bool)
        llen = np.zeros(len(H), int)
        order = np.lexsort((f_, L_))
        for grp in np.split(order, np.flatnonzero(np.diff(L_[order])) + 1):
            deg[grp] = degrees(sa[grp], sb[grp], a_.dmax)
            erased[grp], llen[grp] = loop_erase(sa[grp], sb[grp], a_.maxloop)
        disp = np.full(len(H), np.nan)
        for n, (i, f) in enumerate(zip(L_, f_)):
            if f - nwin >= 0 and f + nwin + 1 <= F:
                disp[n] = np.linalg.norm(li[f + 1:f + nwin + 1, i].mean(0) - li[f - nwin:f, i].mean(0))
        valid = np.isfinite(disp)
        small = valid & (disp < a_.small)
        flicker = erased | small                 # not a robust net hop
        nonpal = (llen > 0) & (deg == 0)
        print(f"T={T}: {len(H)} hops, {int(erased.sum())} in loops ({100 * erased.mean():.0f} %), "
              f"{int((deg > 0).sum())} palindromic retraces, {int((~erased).sum())} net, "
              f"{int((~flicker).sum())} robust net (net and displacement >= {a_.small} A; {int((~valid).sum())} hops not testable)")

        rows_T = []
        shell_any = None
        for lab, d_ in dists.items():
            sh = io.shell_labels(d_, dnear, edges, a_.far)
            C = io.block_counts(sh, K, a_.blocks)
            cat = sh[f_, L_]
            if lab == "any":
                shell_any = cat
            kw = dict(litime_blk=C * dt, nmean=C.sum(0) / F, F=F, nblocks=a_.blocks)
            ones = np.ones(len(H), bool)
            r_loop = io.hop_shell_rows(T, lab, "loops", names, cat, f_, erased, ones, **kw)
            r_pal = io.hop_shell_rows(T, lab, "palin", names, cat, f_, deg > 0, ones, **kw)
            r_rob = io.hop_shell_rows(T, lab, "robust", names, cat, f_, flicker, ones, **kw)
            for kk in range(K):
                m = cat == kk
                a1, a2, a3 = r_loop[kk], r_pal[kk], r_rob[kk]
                mn, ml = m & ~erased & valid, m & erased & valid
                n_net_, n_rob_ = a1["n_hops"] - a1["n_back"], a3["n_hops"] - a3["n_back"]
                nse = a1["nonback_ratio_se"] if n_net_ >= MIN_N else np.nan
                rse = a3["nonback_ratio_se"] if n_rob_ >= MIN_N else np.nan
                rows_T.append(dict(
                    T=T, dopant=lab, shell=names[kk], is_far=(kk == K - 1), n_Li_mean=a1["n_Li_mean"], n_hops=a1["n_hops"],
                    n_loop=a1["n_back"], n_net=a1["n_hops"] - a1["n_back"], n_robust=a3["n_hops"] - a3["n_back"],
                    pct_in_loops=a1["back_pct"], pct_in_loops_se=a1["back_pct_se"], pct_palindromic=a2["back_pct"],
                    pct_nonpal=100 * nonpal[m].mean() if m.any() else np.nan,
                    all_rate=a1["rate_per_Li_ps"], all_ratio=a1["rate_ratio_to_far"], all_ratio_se=a1["rate_ratio_se"],
                    net_rate=a1["nonback_rate_per_Li_ps"], net_ratio=a1["nonback_ratio_to_far"], net_ratio_se=nse,
                    rob_rate=a3["nonback_rate_per_Li_ps"], rob_ratio=a3["nonback_ratio_to_far"], rob_ratio_se=rse,
                    pct_small_net=100 * small[mn].mean() if mn.any() else np.nan, n_val_net=int(mn.sum()),
                    pct_small_loop=100 * small[ml].mean() if ml.any() else np.nan, n_val_loop=int(ml.sum()),
                    n_not_testable=int((m & ~valid).sum()), maxloop=a_.maxloop, small_A=a_.small, win_ps=a_.win_ps))
        rows += rows_T
        print_T(T, [r for r in rows_T], a_)

        if not a_.no_flags:
            with open(os.path.join(outdir, f"hop_flags_T{T}K.csv"), "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["li", "frame", "site_from", "site_to", "shell_any", "d_any_A", "in_loop", "loop_len", "retrace_deg",
                            "disp_A", "small", "robust_net"])
                for n in np.argsort(f_, kind="stable"):
                    w.writerow([L_[n], f_[n], sa[n], sb[n], names[shell_any[n]] if shell_any[n] >= 0 else "excluded",
                                f"{dnear[f_[n], L_[n]]:.3f}", int(erased[n]), llen[n], deg[n],
                                f"{disp[n]:.3f}" if valid[n] else "nan", int(small[n]), int(not flicker[n])])

    if not rows:
        raise SystemExit("no hops read")
    labs = list(dict.fromkeys(r["dopant"] for r in rows))
    with open(os.path.join(outdir, "hop_loops.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(rows, labs, names, os.path.join(outdir, "hop_loops.png"))
    print(f"\n[+] {outdir}/hop_loops.csv hop_loops.png" + ("" if a_.no_flags else " hop_flags_T<T>K.csv"))
    txt = definitions(a_, names)
    print(txt)
    with open(os.path.join(outdir, "hop_loops_definitions.txt"), "w") as fh:
        fh.write(txt)
    print("read: NET/far and ROBUST/far are the transport-relevant ratios; each T is an independent run, so look for the same sign "
          "at all T. Full definitions are above and in hop_loops_definitions.txt.")

if __name__ == "__main__":
    main()
