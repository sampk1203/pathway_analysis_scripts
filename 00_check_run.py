#!/usr/bin/env python3
"""
00_check_run.py - sanity-check each dump before the real analysis.

Reports, inside the analysis window:
  * frames / time range used, box, number of Li, fold factors
  * framework COM drift and how far framework atoms wander from their mean position
    (answers "did the atoms stay where they were?")
  * quick Li tracer D from a multi-origin, drift-corrected MSD, plus a log-log slope
    check that only WARNS.

usage:
  python 00_check_run.py /path/to/Ta81_Ta82_Al1               # folder with dumps + element_list
  python 00_check_run.py /path/to/Ta81_Ta82_Al1 --T 1000 1100
  python 00_check_run.py /path/to/folder --defaults my_defaults.yaml
The system name (= folder name), atom types and dopants come from the folder's element_list.
"""
import argparse
import csv
import os

import numpy as np

import llzo_io as io


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="folder with the dumps + element_list (or a settings .yaml whose dump_dir points there)")
    ap.add_argument("--defaults", help="settings file with run_steps, tmin_ps, ... (default: analysis_defaults.yaml)")
    ap.add_argument("--T", type=int, nargs="*", help="temperatures (default: all in config)")
    ap.add_argument("--dump", help="explicit dump file (only with a single --T)")
    ap.add_argument("--tmin", type=float, help="override analysis-window start (ps)")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    cfg = io.load_config(args.target, args.defaults)
    temps = args.T or io.temperatures(cfg)
    if args.dump and len(temps) != 1:
        ap.error("--dump needs exactly one --T")
    os.makedirs(args.outdir, exist_ok=True)
    rows = []

    for T in temps:
        path = args.dump or io.dump_path(cfg, T)
        print(f"\n=== {cfg['name']}  T = {T} K  ({os.path.basename(path)}) ===")
        traj = io.prepare(cfg, path, args.tmin)
        Lm = traj.L.mean(axis=0)
        fold = io.get_fold(cfg, Lm)
        lm = io.li_mask(cfg, traj)
        n_li = int(lm.sum())
        print(f"  box = {Lm[0]:.3f} x {Lm[1]:.3f} x {Lm[2]:.3f} A   fold = {fold}   N_Li = {n_li}   N_atoms = {len(traj.types)}")

        drift = io.framework_drift(cfg, traj)
        print(f"  framework COM drift over window: max |d| = {np.abs(drift).max():.3f} A")

        # framework atoms: deviation from their own time-averaged position
        fw = ~lm
        p = traj.pos[:, fw, :] - drift[:, None, :]
        dev = np.sqrt(((p - p.mean(axis=0)) ** 2).sum(-1).mean(axis=0))   # (Nfw,)
        fw_types = traj.types[fw]
        worst = 0.0
        for t in sorted(set(fw_types.tolist())):
            d = dev[fw_types == t]
            worst = max(worst, d.max())
            tag = " (dopant)" if t in cfg["dopant_types"] else ""
            print(f"    type {t} {cfg['types'][t]:>2}{tag}: rms displacement from mean position  "
                  f"mean {d.mean():.3f}  max {d.max():.3f} A   (n={len(d)})")
        if worst > 1.5:
            print("  [WARN] a framework atom sits >1.5 A rms from its mean position - possible structural "
                  "rearrangement or a mobile dopant; inspect before trusting site-based analysis.")

        # Li MSD, framework-corrected, multiple time origins
        li_pos = traj.pos[:, lm, :] - drift[:, None, :]
        dtf = float(np.median(np.diff(traj.time_ps)))
        t, msd = io.msd_multi_origin(li_pos, dtf)
        slope = io.loglog_slope_warning(t, msd, label=f"T={T}K")
        sel = t >= 0.3 * t[-1]
        k = np.polyfit(t[sel], msd[sel], 1)[0]           # A^2/ps
        D = k / 6.0 * 1e-4                                # cm^2/s
        print(f"  Li MSD(t_max={t[-1]:.1f} ps) = {msd[-1]:.2f} A^2   log-log slope = {slope:.2f}   D_tracer ~ {D:.2e} cm^2/s")

        rows.append(dict(system=cfg["name"], T=T, frames=len(traj.steps), t_start_ps=traj.time_ps[0],
                         t_end_ps=traj.time_ps[-1], N_Li=n_li, fw_com_drift_max_A=np.abs(drift).max(),
                         fw_worst_rms_A=worst, msd_loglog_slope=slope, D_tracer_cm2s=D))

    out = os.path.join(args.outdir, f"check_{cfg['name']}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[+] summary written to {out}")


if __name__ == "__main__":
    main()
