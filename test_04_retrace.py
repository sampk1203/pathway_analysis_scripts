#!/usr/bin/env python3
"""
test_04_retrace.py - back-hop DEGREE from palindromic site visits, and net (non-retraced) hops.  Read-only; writes retrace_degree.csv.

For each Li, its hops in time order are a site path  s0 -> s1 -> s2 ...  A hop k has retrace degree d if the last d hops
exactly reverse, in reverse order, the d hops before them (the site path ... x0 x1 .. xd .. x1 x0 is a palindrome of 2d hops):
    a->b->a                = degree 1  (the old 'back-hop')
    a->b->c->b->a          = degree 2 (hop 4; hop 3 is degree 1)
    a->b->a->b->a          = degree 2 at hop 4 (repeated flipping gets high degree)
degree 0 = no retrace.  (A palindromic site path always has an even number of hops, so degree = hops/2; the 'odd, round up' case
cannot occur. If you count SITES instead, a->b->a has 3 sites -> ceil(3/2) = 2: every degree is then +1. Say if you want that.)
Loops (a->b->c->a) are NOT retraced: degree 0.  Hops need not be close in time (no gap limit).
Net hops = hops left after repeatedly deleting adjacent reverse pairs (a->b followed by b->a) within each Li's path (stack reduction).
Validation printed: % of hops with displacement < --small A (as in test_04_flips.py) among net hops vs among cancelled hops.
usage: python test_04_retrace.py /path/to/MD_run --hops-dir hops_cif/MD_run_1Ga_1Ru
"""
import argparse, csv, os
import numpy as np
import llzo_io as io


def degrees(a, b, dmax):
    """a, b: site_from / site_to of ONE Li's hops, time order. deg[k] = largest d <= dmax with hops k-d+1..k mirroring hops k-2d+1..k-d."""
    n = len(a)
    deg = np.zeros(n, int)
    for k in range(1, n):
        for d in range(min(dmax, (k + 1) // 2), 0, -1):
            lo = k - 2 * d + 1
            ok = all(a[k - d + 1 + j] == b[k - d - j] and b[k - d + 1 + j] == a[k - d - j] for j in range(d))
            ok = ok and all(a[i + 1] == b[i] for i in range(lo, k))      # the 2d hops must be one connected path
            if ok:
                deg[k] = d
                break
    return deg


def cancelled(a, b):
    """stack reduction: a hop that exactly reverses the top of the stack cancels both. True = hop was retraced away."""
    st, can = [], np.zeros(len(a), bool)
    for i in range(len(a)):
        if st and a[i] == b[st[-1]] and b[i] == a[st[-1]]:
            can[st.pop()] = True
            can[i] = True
        else:
            st.append(i)
    return can


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target")
    ap.add_argument("--hops-dir", required=True)
    ap.add_argument("--T", type=int, nargs="*")
    ap.add_argument("--tmin", type=float)
    ap.add_argument("--win-ps", type=float, default=0.3, help="displacement window (as in test_04_flips.py)")
    ap.add_argument("--small", type=float, default=1.0)
    ap.add_argument("--dmax", type=int, default=8, help="longest palindrome checked (degree cap)")
    ap.add_argument("--shells", type=float, nargs="*", default=list(io.DEFAULT_SHELLS))
    a_ = ap.parse_args()

    cfg = io.load_config(a_.target)
    temps = a_.T or io.temperatures(cfg)
    edges = np.array(a_.shells, float)
    names = io.shell_names(edges)
    K = len(names)
    acc = {}
    for T in temps:
        p = os.path.join(a_.hops_dir, f"hops_T{T}K.csv")
        if not os.path.exists(p):
            print("missing", p)
            continue
        traj = io.prepare(cfg, io.dump_path(cfg, T), a_.tmin, verbose=False)
        drift = io.framework_drift(cfg, traj)
        li = traj.pos[:, io.li_mask(cfg, traj), :] - drift[:, None, :]
        F = li.shape[0]
        dt = float(np.median(np.diff(traj.time_ps)))
        k = max(1, int(round(a_.win_ps / dt)))
        dops, cnt = {}, {}
        for t in cfg["dopant_types"]:
            for i in np.where(traj.types == t)[0]:
                el = cfg["types"][t]
                cnt[el] = cnt.get(el, 0) + 1
                dops[f"{el}{cnt[el]}"] = traj.pos[:, i, :] - drift
        H = list(csv.DictReader(open(p)))
        L_ = np.array([int(h["li"]) for h in H])
        f_ = np.array([int(h["frame"]) for h in H])
        sa = np.array([int(h["site_from"]) for h in H])
        sb = np.array([int(h["site_to"]) for h in H])
        isb, _ = io.back_flags(L_, f_, sa, sb)
        deg = np.zeros(len(H), int)
        can = np.zeros(len(H), bool)
        order = np.lexsort((f_, L_))
        cuts = np.flatnonzero(np.diff(L_[order])) + 1
        for grp in np.split(order, cuts):
            deg[grp] = degrees(sa[grp], sb[grp], a_.dmax)
            can[grp] = cancelled(sa[grp], sb[grp])
        dd = {lab: np.zeros(len(H)) for lab in dops}
        disp = np.full(len(H), np.nan)
        for n, (i, f) in enumerate(zip(L_, f_)):
            for lab, x in dops.items():
                v = li[f, i] - x[f]
                v -= traj.L[f] * np.rint(v / traj.L[f])
                dd[lab][n] = np.linalg.norm(v)
            if f - k >= 0 and f + k + 1 <= F:
                disp[n] = np.linalg.norm(li[f + 1:f + k + 1, i].mean(0) - li[f - k:f, i].mean(0))
        dn = np.min(list(dd.values()), axis=0)
        for lab in dops:
            sh = io.shell_labels(dd[lab], dn, edges, "all")
            for s in range(K):
                m = sh == s
                acc.setdefault((lab, s), []).append((deg[m], can[m], disp[m], isb[m].astype(bool)))
        print(f"T={T}: {len(H)} hops, {int((deg > 0).sum())} retraced (deg>=1), {int(isb.sum())} old back-hops, {int((~can).sum())} net")

    print(f"\npooled over T. retrace = degree >= 1; net = survives stack reduction; small = displacement < {a_.small} A over {a_.win_ps} ps")
    hdr = f"{'dopant':>6} {'shell':>6} {'n':>5} {'%old back':>9} {'%retrace':>8} {'%deg1':>6} {'%deg2':>6} {'%deg>=3':>7} {'%net':>6} {'%small|net':>10} {'%small|canc':>11}"
    print(hdr)
    out = []
    for (lab, s), v in sorted(acc.items()):
        dg = np.concatenate([x[0] for x in v])
        cn = np.concatenate([x[1] for x in v])
        ds = np.concatenate([x[2] for x in v])
        ob = np.concatenate([x[3] for x in v])
        if not len(dg):
            continue
        sm = lambda m: (f"{100 * (ds[m & np.isfinite(ds)] < a_.small).mean():8.1f}" if (m & np.isfinite(ds)).sum() else "     n/a")
        r = dict(dopant=lab, shell=names[s], n=len(dg), pct_old_back=100 * ob.mean(), pct_retrace=100 * (dg > 0).mean(),
                 pct_deg1=100 * (dg == 1).mean(), pct_deg2=100 * (dg == 2).mean(), pct_deg3p=100 * (dg >= 3).mean(),
                 pct_net=100 * (~cn).mean())
        out.append(r)
        print(f"{lab:>6} {names[s]:>6} {len(dg):5d} {r['pct_old_back']:9.1f} {r['pct_retrace']:8.1f} {r['pct_deg1']:6.1f} {r['pct_deg2']:6.1f} "
              f"{r['pct_deg3p']:7.1f} {r['pct_net']:6.1f} {sm(~cn):>10} {sm(cn):>11}")
    with open(os.path.join(a_.hops_dir, "retrace_degree.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print("\nread: %retrace > %old back = retracing through 3rd sites that the one-step back-hop missed. %net = hops that are not "
          "undone by later retracing. If %small|net is still high in Ga1 0-3A, loops (a->b->c->a) remain.")


if __name__ == "__main__":
    main()
