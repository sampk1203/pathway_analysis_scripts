#!/usr/bin/env python3
"""
02_compare_temps.py - compare the Li free-energy maps of ONE system across temperatures.

Reads results/<system>/density_F_T*K.npz + density_summary_T*K.json written by 01_density_free_energy.py
(no dumps needed). Same layout as the single-T plots (full-box min-F projections along z, y, x), one row per
temperature, one common colour scale.

Two views of the same data (--units both, default):
  eV : F - F_site (eV)         - if the free-energy LANDSCAPE U(r) is temperature independent this does not change
  kT : (F - F_site) / kT       - if the DENSITY RATIO rho(r)/rho_site is temperature independent (e.g. the values
                                 are set by geometry/smoothing tails, not by an energy) this does not change
Look at where the difference maps are ~0 in each view: that tells you which of the two is true, region by region.

Alignment (important for difference maps): all temperatures use ONE shift, computed from the reference
temperature (--ref-T, default lowest), so maps stay aligned voxel-for-voxel.
  --shift SX SY SZ            fixed periodic shift in A
  --center Ga | Ga1 | all     one set of figures per dopant atom, that atom at the box centre
  --slab W                    with --center: min-project only a +/-W A slab through that atom

Outputs (results/<system>/):
  compare_T[_<label>].png, compare_T_kT[_<label>].png     F_rel per temperature
  diff_T[_<label>].png,    diff_T_kT[_<label>].png        difference from the reference temperature
  dopant_rdf_T.png                                         Li-dopant g(r) at every T
  compare_T_table.csv, compare_T_dopants.csv               numbers (also printed)

usage:
  python 02_compare_temps.py results/MD_run_1Ga_1Ru
  python 02_compare_temps.py results/MD_run_1Ga_1Ru --center all --ref-T 800
"""
import argparse
import csv
import glob
import json
import os
import re
import warnings

import numpy as np

import llzo_maps as mp

KB = 8.617333262e-5


def load(folder):
    runs = {}
    for p in glob.glob(os.path.join(folder, "density_F_T*K.npz")):
        T = int(re.search(r"T(\d+)K", p).group(1))
        z = np.load(p)
        js = os.path.join(folder, f"density_summary_T{T}K.json")
        if "F_rel" not in z.files or not os.path.exists(js):
            raise SystemExit(f"{p}: written by an old 01 script (no F_rel / summary). Rerun 01_density_free_energy.py.")
        runs[T] = dict(F=z["F_rel"], rho=z["rho"], L=z["L"], rho_uni=float(z["rho_uniform"]),
                       summ=json.load(open(js)), rdf_r=z["rdf_r"] if "rdf_r" in z.files else None,
                       rdf={k[6:]: z[k] for k in z.files if k.startswith("rdf_g_")})
    if not runs:
        raise SystemExit(f"no density_F_T*K.npz in {folder}")
    return dict(sorted(runs.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="results/<system> folder from 01_density_free_energy.py")
    ap.add_argument("--units", choices=["both", "eV", "kT"], default="both")
    ap.add_argument("--shift", type=float, nargs=3, default=[0, 0, 0], metavar=("SX", "SY", "SZ"))
    ap.add_argument("--center", help="dopant element (Ga), atom label (Ga1) or 'all'")
    ap.add_argument("--slab", type=float, default=0.0, help="with --center: half-thickness (A) of a slab projection")
    ap.add_argument("--ref-T", type=int, help="reference temperature (difference maps, centring); default lowest")
    ap.add_argument("--vmax", type=float, default=0.6, help="eV colour maximum")
    ap.add_argument("--vmax-kT", type=float, default=6.0, help="kT colour maximum")
    ap.add_argument("--dmax", type=float, default=0.15, help="+/- range of eV difference maps")
    ap.add_argument("--dmax-kT", type=float, default=1.5, help="+/- range of kT difference maps")
    args = ap.parse_args()
    if args.slab > 0 and not args.center:
        ap.error("--slab needs --center")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load(args.folder)
    Ts = list(runs)
    T0 = args.ref_T or Ts[0]
    ref = runs[T0]
    L = ref["L"]
    dops0 = ref["summ"]["dopants"]
    print(f"temperatures: {Ts}   reference: {T0} K")

    # ---- views (label, shift)
    if args.center:
        pick = [d["label"] for d in dops0 if args.center in (d["label"], d["element"]) or args.center == "all"]
        if not pick:
            raise SystemExit(f"--center {args.center}: no such dopant. Available: {[d['label'] for d in dops0]}")
        pos0 = {d["label"]: np.array(d["pos_A"]) for d in dops0}
        views = [(lab, L / 2 - pos0[lab]) for lab in pick]
    else:
        views = [(None, np.array(args.shift, float))]
    units = ["eV", "kT"] if args.units == "both" else [args.units]

    CM_F = plt.get_cmap("viridis_r").copy()
    CM_F.set_bad("0.82")
    CM_D = plt.get_cmap("RdBu_r").copy()
    CM_D.set_bad("0.82")
    DCOL, PAL = {}, ["red", "white", "orange", "cyan", "magenta"]

    def scaled(T, unit):
        F = runs[T]["F"]
        return F / (KB * T) if unit == "kT" else F

    def draw(kind, unit, lab, shift):
        n_r = len(Ts)
        fig, axs = plt.subplots(n_r, 3, figsize=(11, 3.3 * n_r), dpi=150, squeeze=False)
        if unit == "eV":
            fr, dr, lbl = (-0.1, args.vmax), args.dmax, "eV"
        else:
            fr, dr, lbl = (-1.0, args.vmax_kT), args.dmax_kT, "kT"
        for ri, T in enumerate(Ts):
            F, sh = mp.apply_shift(scaled(T, unit), L, shift)
            Fr, _ = mp.apply_shift(scaled(T0, unit), L, shift)
            for ci, (axis, xl, yl, ix, iy) in enumerate(mp.PANELS):
                ax = axs[ri, ci]
                pj = mp.project(F, axis, args.slab if lab else 0.0, L)
                if kind == "F":
                    proj, cmap, vmin, vmax = pj, CM_F, fr[0], fr[1]
                else:
                    proj, cmap, vmin, vmax = pj - mp.project(Fr, axis, args.slab if lab else 0.0, L), CM_D, -dr, dr
                im = ax.imshow(np.ma.masked_invalid(proj).T, origin="lower", extent=[0, L[ix], 0, L[iy]], cmap=cmap,
                               vmin=vmin, vmax=vmax, aspect="equal")
                for d in runs[T]["summ"]["dopants"]:
                    p = (np.array(d["pos_A"]) + sh) % L
                    if args.slab > 0 and lab and abs(p[axis] - L[axis] / 2) > args.slab + 1.0:
                        continue
                    ax.scatter(p[ix], p[iy], s=90 if d["label"] == lab else 45, marker="*",
                               color=DCOL.setdefault(d["element"], PAL[len(DCOL) % len(PAL)]), edgecolor="k",
                               linewidth=0.8 if d["label"] == lab else 0.5, zorder=5)
                ax.set_ylabel((f"{T} K\n" if ci == 0 else "") + f"{yl} (A)")
                ax.set_xlabel(f"{xl} (A)")
                if ri == 0:
                    ax.set_title(f"min-F in +/-{args.slab:g} A slab normal to {'xyz'[axis]}" if (args.slab > 0 and lab)
                                 else f"min-F projection along {'xyz'[axis]}", fontsize=10)
            fig.colorbar(im, ax=axs[ri, :], shrink=0.9, pad=0.02,
                         label=(f"(F - F_site) / kT" if unit == "kT" else "F - F_site (eV)") if kind == "F"
                         else f"delta vs {T0} K ({lbl})")
        what = "F - F_site" if unit == "eV" else "(F - F_site)/kT"
        fig.suptitle(f"{os.path.basename(os.path.normpath(args.folder))}: "
                     + (f"{what} (grey = unsampled)" if kind == "F"
                        else f"{what}, difference from {T0} K (grey = unsampled at either T)")
                     + (f", centred on {lab}" if lab else (f", shift {list(shift)} A" if np.any(shift) else "")),
                     y=0.995)
        name = ("compare_T" if kind == "F" else "diff_T") + ("_kT" if unit == "kT" else "") + (f"_{lab}" if lab else "")
        out = os.path.join(args.folder, name + ".png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print("[+]", out)

    for lab, shift in views:
        for unit in units:
            draw("F", unit, lab, shift)
            draw("D", unit, lab, shift)

    # ---- Li-dopant g(r) at every T
    labels = [d["label"] for d in dops0]
    if labels and ref["rdf_r"] is not None:
        fig, axs = plt.subplots(1, len(labels), figsize=(4.6 * len(labels), 3.8), dpi=150, squeeze=False)
        cm = plt.cm.plasma(np.linspace(0.05, 0.85, len(Ts)))
        for a, lab in zip(axs[0], labels):
            for c, T in zip(cm, Ts):
                if lab in runs[T]["rdf"]:
                    a.plot(runs[T]["rdf_r"], runs[T]["rdf"][lab], color=c, label=f"{T} K")
            a.axhline(1, color="0.6", lw=0.6)
            a.set_title(lab)
            a.set_xlabel("r (A)")
            a.set_ylabel("g_Li-dopant(r)")
        axs[0][0].legend(fontsize=8)
        fig.tight_layout()
        out = os.path.join(args.folder, "dopant_rdf_T.png")
        fig.savefig(out)
        plt.close(fig)
        print("[+]", out)

    # ---- tables
    rows, drows = [], []
    for T, r in runs.items():
        F, s = r["F"], r["summ"]
        kT = KB * T
        row = dict(T=T, F_site_over_kT=s["F_site_ref_eV"] / kT, rho_site_over_uniform=float(np.exp(-s["F_site_ref_eV"] / kT)),
                   V_eff_frac=float(np.exp(s["F_site_ref_eV"] / kT)), sampled_frac=s["sampled_fraction"],
                   max_resolvable_dF_eV=s["max_resolvable_dF_eV"], max_resolvable_dF_kT=s["max_resolvable_dF_eV"] / kT,
                   unsampled_below_pct_of_site=100.0 * float(np.exp(-s["max_resolvable_dF_eV"] / kT)),
                   min_neff=s.get("min_neff", np.nan),
                   F_err_median_eV=s.get("F_err_median_network_eV", np.nan),
                   smoothing_sensitivity_eV=s.get("smoothing_sensitivity_eV", np.nan))
        for x in (0.1, 0.2, 0.3):
            row[f"frac_F_below_{x:g}eV"] = float((F < x).mean())
        for x in (1, 2, 3, 4):
            row[f"frac_F_below_{x}kT"] = float((F / kT < x).mean())
        rows.append(row)
        for d in s["dopants"]:
            drows.append(dict(T=T, label=d["label"], closest_Li_A=d["closest_Li_A"], Li_within_2A=d["mean_Li_within_2A"],
                              r_sphere_A=d["r_sphere"], rho_over_uni_sphere=d["rho_over_uni_sphere"],
                              control_median=d["control_median"], control_p16=d["control_p16"],
                              control_p84=d["control_p84"],
                              ratio_to_control=d["rho_over_uni_sphere"] / d["control_median"] if d["control_median"] else np.nan,
                              unsampled_frac_in_sphere=d["unsampled_frac_in_sphere"], rms_disp_A=d["rms_disp_A"]))
    for name, data in (("compare_T_table.csv", rows), ("compare_T_dopants.csv", drows)):
        if data:
            with open(os.path.join(args.folder, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                w.writeheader()
                w.writerows(data)
            print("\n" + name)
            print("  ".join(f"{k[:19]:>19}" for k in data[0]))
            for row in data:
                print("  ".join(f"{v:19.3f}" if isinstance(v, float) else f"{str(v):>19}" for v in row.values()))
    print("\nSampling limit: a voxel is 'sampled' if the sigma-kernel holds >= min_neff raw counts. F_thr - F_site = "
          "max_resolvable_dF_kT (column) = ln(rho_site / rho_thr); unsampled_below_pct_of_site = 100 exp(-that) is the "
          "Li density (% of a site) below which voxels are grey. Values near this limit are detection-limit, not barriers.")
    print("\nNotes: grey/NaN voxels were not sampled: any barrier there only exceeds max_resolvable_dF (~5 kT, which "
          "grows with T in eV). Compare eV-differences (landscape) with kT-differences (density ratio); a real "
          "energy barrier is flat in eV, a geometric/smoothing tail is flat in kT. ratio_to_control < 1: Li depleted "
          "around the dopant relative to ordinary Li sites; closest_Li_A / Li_within_2A come from the raw dump.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
