#!/usr/bin/env python3
"""
test_04_flips.py - are 04's near-dopant hops real jumps or site-assignment flips?  Read-only, writes nothing but a csv.

For every hop in hops_cif/<s>/hops_T<T>K.csv: Li displacement = |mean pos over k frames AFTER the hop frame - mean pos over
k frames BEFORE it| (k = --win-ps, default 0.3 ps = 04's min-res; unwrapped, framework drift removed). Grouped by the
distance of the Li to each dopant in the hop frame (same shells as 04/05/06).
Real jump: displacement ~ site spacing (median jump_A, ~1.5-3 A). Flip: displacement << site spacing.
Sanity: far shell median displacement should be of order jump_A; if it is ~0 the 'li' column is not the Li index used here.
Prediction if flips: near Ga, a large share of hops has displacement < --small A, much larger than in the far shell.
usage: python test_04_flips.py /path/to/MD_run --hops-dir hops_cif/MD_run_1Ga_1Ru
"""
import argparse, csv, os
import numpy as np
import llzo_io as io

ap = argparse.ArgumentParser()
ap.add_argument("target")
ap.add_argument("--hops-dir", required=True)
ap.add_argument("--T", type=int, nargs="*")
ap.add_argument("--tmin", type=float)
ap.add_argument("--win-ps", type=float, default=0.3)
ap.add_argument("--small", type=float, default=1.0, help="displacement below this (A) = 'did not really move'")
ap.add_argument("--shells", type=float, nargs="*", default=list(io.DEFAULT_SHELLS))
a = ap.parse_args()

cfg = io.load_config(a.target)
temps = a.T or io.temperatures(cfg)
edges = np.array(a.shells, float)
names = io.shell_names(edges)
K = len(names)
acc = {}      # (dopant, shell) -> lists
out = []
for T in temps:
    p = os.path.join(a.hops_dir, f"hops_T{T}K.csv")
    if not os.path.exists(p):
        print("missing", p); continue
    traj = io.prepare(cfg, io.dump_path(cfg, T), a.tmin, verbose=False)
    drift = io.framework_drift(cfg, traj)
    li = traj.pos[:, io.li_mask(cfg, traj), :] - drift[:, None, :]
    F = li.shape[0]
    dt = float(np.median(np.diff(traj.time_ps)))
    k = max(1, int(round(a.win_ps / dt)))
    dops, cnt = {}, {}
    for t in cfg["dopant_types"]:
        for i in np.where(traj.types == t)[0]:
            el = cfg["types"][t]; cnt[el] = cnt.get(el, 0) + 1
            dops[f"{el}{cnt[el]}"] = traj.pos[:, i, :] - drift
    H = list(csv.DictReader(open(p)))
    L_ = np.array([int(h["li"]) for h in H]); f_ = np.array([int(h["frame"]) for h in H])
    sa = np.array([int(h["site_from"]) for h in H]); sb = np.array([int(h["site_to"]) for h in H])
    jump = np.array([float(h["jump_A"]) for h in H])
    isb, _ = io.back_flags(L_, f_, sa, sb)
    n_skip = 0
    dnear = np.full(len(H), np.inf)
    dd = {lab: np.zeros(len(H)) for lab in dops}
    disp = np.full(len(H), np.nan)
    for n, (i, f) in enumerate(zip(L_, f_)):
        for lab, x in dops.items():
            v = li[f, i] - x[f]; v -= traj.L[f] * np.rint(v / traj.L[f]); dd[lab][n] = np.linalg.norm(v)
        if f - k < 0 or f + k + 1 > F:
            n_skip += 1; continue
        disp[n] = np.linalg.norm(li[f + 1:f + k + 1, i].mean(0) - li[f - k:f, i].mean(0))
    dn = np.min(list(dd.values()), axis=0)
    ok = np.isfinite(disp)
    for lab in dops:
        sh = io.shell_labels(dd[lab], dn, edges, "all")
        for s in range(K):
            m = ok & (sh == s)
            acc.setdefault((lab, s), []).append((disp[m], jump[m], isb[m]))
    print(f"T={T}: {len(H)} hops, {n_skip} skipped at window edges, k={k} frames")

print(f"\npooled over T. displacement = |mean(after {a.win_ps} ps) - mean(before)|; small = < {a.small} A")
print(f"{'dopant':>6} {'shell':>7} {'n':>5} {'disp med':>9} {'jump_A med':>10} {'% small':>8} {'% back':>7} {'%small in back':>15} {'%small non-back':>16}")
for (lab, s), v in sorted(acc.items()):
    d = np.concatenate([x[0] for x in v]); j = np.concatenate([x[1] for x in v]); b = np.concatenate([x[2] for x in v]).astype(bool)
    if not len(d):
        continue
    sm = d < a.small
    f = lambda m: f"{100 * sm[m].mean():6.1f}" if m.sum() else "   n/a"
    print(f"{lab:>6} {names[s]:>7} {len(d):5d} {np.median(d):9.2f} {np.median(j):10.2f} {100 * sm.mean():8.1f} {100 * b.mean():7.1f} {f(b):>15} {f(~b):>16}")
    out.append(dict(dopant=lab, shell=names[s], n=len(d), disp_median=np.median(d), jump_median=np.median(j),
                    pct_small=100 * sm.mean(), pct_back=100 * b.mean()))
with open(os.path.join(a.hops_dir, "flip_test.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print("\nread: Ga1 0-3A '% small' >> far (>7A) '% small', disp med < jump_A med  =>  04 near-Ga hops are mostly assignment flips.")
print("      similar to far  =>  hypothesis wrong; 04 and 06 differ for another reason.")
