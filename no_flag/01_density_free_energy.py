#!/usr/bin/env python3
"""
01_density_free_energy.py - periodic Li density and free-energy map, straight from the dump.

Method
  * Li positions relative to the framework centre of mass (drift removed), wrapped into the box, over the
    analysis window only (t >= tmin).
  * density: cloud-in-cell (trilinear) deposition on a fine grid (default 0.2 A voxels; no nearest-voxel
    aliasing) + ADAPTIVE Gaussian smoothing: narrow kernels (sigma_min) where Li is dense, wide (sigma_max) where
    it is sparse. Compared with one fixed sigma this keeps the site peaks sharp and averages noise only where
    needed, and it reduces the spill-over of site density into empty regions (which scales like kT).
    --no-adaptive gives plain fixed-sigma smoothing (sigma = --sigma).
  * F(r) = -kT ln( rho / rho_uniform ),  rho_uniform = N_Li / V. Zero = uniform Li distribution.
  * sampling mask: a voxel is "sampled" if the fixed-sigma kernel around it holds >= --min-neff raw Li counts.
    Unsampled voxels are NaN (grey in plots), never floored. (The floored F is still written to the .dx/.npz as F.)
  * reference: F_site = median F over the density maxima of the pilot (fixed-sigma) map with rho >= site_ratio *
    rho_uniform. F_rel = F - F_site; F_rel / kT is the temperature-scaled version.
  * uncertainty: the window is split into --nblocks-err blocks; F_err = standard error of ln(rho) over blocks
    times kT. A smoothing-sensitivity number (adaptive vs fixed sigma, median |dF_rel| over the network) is printed.
  * dopants: Li-dopant g(r) (box-average Li density as reference), closest Li approach, and the mean rho/rho_uniform
    in a sphere whose radius is the FIRST MINIMUM of that g(r) (or --rdop), next to a control: the same sphere
    around ordinary Li sites.
  * plots: F_rel (eV) and F_rel/kT, same layout always (full-box min projections along z, y, x). Optional:
      --shift SX SY SZ   fixed periodic shift (A), same for every temperature
      --center Ga | Ga1 | all   one figure per dopant atom, that atom at the box centre (reference position taken from
                          the first temperature processed in this call, so all temperatures share one shift)
      --slab W           with --center: min-project only a +/-W A slab through the centred atom

usage:
  python 01_density_free_energy.py /path/to/folder                     # all temperatures
  python 01_density_free_energy.py /path/to/folder --T 800 1200 --center Ga
  options: --spacing 0.2 --sigma 0.5 --sigma-min 0.25 --sigma-max 0.6 --min-neff 10 --units both|eV|kT
"""
import argparse
import json
import os
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

import llzo_io as io
import llzo_maps as mp


# ------------------------------------------------------------------ maps
def _F_from_rho(rho, rho_uni, kT, floor):
    return -kT * np.log(np.maximum(rho, floor * rho_uni) / rho_uni)


def build_maps(li_frac, L, T, args):
    F_, N_ = li_frac.shape[:2]
    n = np.maximum(np.rint(L / args.spacing).astype(int), 4)
    voxel = L / n
    vvol = float(np.prod(voxel))
    pts = li_frac.reshape(-1, 3)
    kT = io.KB_EV * T
    rho_uni = N_ / float(np.prod(L))

    if args.no_adaptive:
        counts = mp.deposit_cic(pts, n)
        pilot = gaussian_filter(counts, args.sigma / voxel, mode="wrap")
        smoothed = pilot
    else:
        smoothed, pilot, counts, _ = mp.adaptive_smooth(pts, n, voxel, args.sigma, args.sigma_min, args.sigma_max)
    rho = smoothed / (F_ * vvol)
    rho_pilot = pilot / (F_ * vvol)
    F = _F_from_rho(rho, rho_uni, kT, args.floor)

    # sampling mask and resolvable limit (fixed-sigma kernel counts)
    Nk = mp.kernel_voxels(args.sigma, voxel)
    sampled = pilot * Nk >= args.min_neff
    F_thr = -kT * np.log(args.min_neff / (Nk * F_ * vvol) / rho_uni)

    # site reference from the pilot map (smoother => maxima are not noise-selected)
    size = tuple(int(2 * max(1, round(args.site_radius / v)) + 1) for v in voxel)
    peaks = (rho_pilot == maximum_filter(rho_pilot, size=size, mode="wrap")) \
        & (rho_pilot >= args.site_ratio * rho_uni) & sampled
    F_site = float(np.median(F[peaks])) if peaks.any() else float(F.min())
    F_rel = np.where(sampled, F - F_site, np.nan)

    # smoothing sensitivity: same pipeline with fixed sigma
    Ff = _F_from_rho(rho_pilot, rho_uni, kT, args.floor)
    Ff_site = float(np.median(Ff[peaks])) if peaks.any() else float(Ff.min())
    Ff_rel = np.where(sampled, Ff - Ff_site, np.nan)
    net = sampled & np.isfinite(F_rel) & (F_rel < 0.3)
    sens = float(np.median(np.abs(F_rel[net] - Ff_rel[net]))) if net.any() and not args.no_adaptive else 0.0

    # block standard error of F
    K = max(2, args.nblocks_err)
    S = np.array([gaussian_filter(mp.deposit_cic(li_frac[idx].reshape(-1, 3), n), args.sigma / voxel, mode="wrap")
                  for idx in np.array_split(np.arange(F_), K)])
    ok = (S > 0).all(axis=0)
    err = kT * np.log(np.where(S > 0, S, 1.0)).std(axis=0, ddof=1) / np.sqrt(K)
    err = np.where(ok, err, np.nan)
    net2 = net & np.isfinite(err) & (F_rel < 0.2)
    err_med = float(np.median(err[net2])) if net2.any() else float("nan")

    site_xyz = (np.argwhere(peaks) + 0.5) * voxel                     # A, relative to box origin
    order = np.argsort(-rho_pilot[peaks])[:500]
    site_xyz = site_xyz[order]
    return dict(counts=counts, rho=rho, rho_pilot=rho_pilot, F=F, F_min0=F - F.min(), F_rel=F_rel, sampled=sampled,
                F_site=F_site, n_sites=int(peaks.sum()), site_xyz=site_xyz, max_resolvable=float(F_thr - F_site),
                n=n, L=L, voxel=voxel, rho_uni=rho_uni, kT=kT, n_frames=F_, n_li=N_, sens=sens, err=err,
                err_med=err_med)


# ------------------------------------------------------------------ dopants
def dopant_atoms(cfg, traj, drift):
    """One dict per dopant atom: label (element + running index in id order), element, atom id, time-averaged
    position relative to the box origin (A, wrapped), rms displacement about it."""
    out, cnt = [], {}
    Lm = traj.L.mean(axis=0)
    for t in cfg["dopant_types"]:
        idx = np.where(traj.types == t)[0]
        idx = idx[np.argsort(traj.ids[idx])]
        for i in idx:
            el = cfg["types"][t]
            cnt[el] = cnt.get(el, 0) + 1
            series = traj.pos[:, i, :] - drift
            mean = series.mean(axis=0)
            rms = float(np.sqrt(((series - mean) ** 2).sum(-1).mean()))
            out.append(dict(label=f"{el}{cnt[el]}", element=el, atom_id=int(traj.ids[i]), index=int(i),
                            p=(mean - traj.lo) % Lm, rms=rms, series=series))
    return out


def li_dopant_rdf(li, dser, Lm, dr=0.05):
    rmax = float(Lm.min()) / 2 - 1e-6
    edges = np.arange(0.0, rmax + 0.5 * dr, dr)
    d = li - dser[:, None, :]
    d -= Lm * np.rint(d / Lm)
    r = np.linalg.norm(d, axis=-1)
    cnt = np.histogram(r.ravel(), edges)[0]
    shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    g = cnt / (li.shape[0] * (li.shape[1] / float(np.prod(Lm))) * shell)
    return 0.5 * (edges[1:] + edges[:-1]), g, float(r.min()), float((r < 2.0).sum(axis=1).mean())


def rdf_shells(r, g, r_lo=1.0):
    """First peak (g>0.5, r>=r_lo) and the first minimum after it, on a 5-bin smoothed g(r)."""
    gs = np.convolve(g, np.ones(5) / 5.0, mode="same")
    ip = None
    for i in range(1, len(gs) - 1):
        if r[i] >= r_lo and gs[i] > gs[i - 1] and gs[i] >= gs[i + 1] and gs[i] > 0.5:
            ip = i
            break
    if ip is None:
        return None, None
    for j in range(ip + 1, len(gs) - 1):
        if gs[j] < gs[j - 1] and gs[j] <= gs[j + 1]:
            return float(r[ip]), float(r[j])
    return float(r[ip]), None


def dopant_stats(maps, dops, li, Lm, args):
    ratio = maps["rho"] / maps["rho_uni"]
    L = maps["L"]
    res = []
    for d in dops:
        r, g, rmin, n2 = li_dopant_rdf(li, d["series"], Lm)
        rp, r_first_min = rdf_shells(r, g)
        rs = args.rdop or r_first_min or 3.0
        src = "--rdop" if args.rdop else ("first g(r) minimum" if r_first_min else "fallback 3.0")
        vals = mp.sphere_values(ratio, L, d["p"], rs)
        smp = mp.sphere_values(maps["sampled"].astype(float), L, d["p"], rs)
        # control: same sphere around ordinary Li sites (not within rs of any dopant)
        ctrl = []
        for s in maps["site_xyz"]:
            far = True
            for d2 in dops:
                dd = s - d2["p"]
                dd -= L * np.rint(dd / L)
                if np.linalg.norm(dd) < rs:
                    far = False
                    break
            if far:
                ctrl.append(float(mp.sphere_values(ratio, L, s, rs).mean()))
        ctrl = np.array(ctrl)
        res.append(dict(label=d["label"], element=d["element"], atom_id=d["atom_id"], pos_A=[float(x) for x in d["p"]],
                        rms_disp_A=d["rms"], closest_Li_A=rmin, mean_Li_within_2A=n2, r_first_peak=rp,
                        r_first_min=r_first_min, r_sphere=float(rs), r_sphere_source=src,
                        rho_over_uni_sphere=float(vals.mean()), rho_over_uni_min_in_sphere=float(vals.min()),
                        unsampled_frac_in_sphere=float(1.0 - smp.mean()),
                        control_median=float(np.median(ctrl)) if len(ctrl) else float("nan"),
                        control_p16=float(np.percentile(ctrl, 16)) if len(ctrl) else float("nan"),
                        control_p84=float(np.percentile(ctrl, 84)) if len(ctrl) else float("nan"),
                        n_control=int(len(ctrl)), rdf_r=r, rdf_g=g))
    return res


# ------------------------------------------------------------------ plotting
def plot_maps(path, F, L, dops_xyz, title, unit, vmin, vmax, shift=None, slab=0.0, centre=None):
    """Full-box min-F projections along z, y, x (same layout as always). F is F_rel (eV) or F_rel/kT."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dops = list(dops_xyz)
    if shift is not None and np.any(np.asarray(shift) != 0):
        F, sh = mp.apply_shift(F, L, shift)
        dops = [(lab, el, (p + sh) % L) for lab, el, p in dops]
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad("0.82")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), dpi=200)
    colors = {}
    for ax, (axis, xl, yl, ix, iy) in zip(axs, mp.PANELS):
        proj = mp.project(F, axis, slab, L)
        im = ax.imshow(np.ma.masked_invalid(proj).T, origin="lower", extent=[0, L[ix], 0, L[iy]], cmap=cmap,
                       vmin=vmin, vmax=vmax, aspect="equal")
        for lab, el, p in dops:
            if slab > 0 and abs(p[axis] - L[axis] / 2) > slab + 1.0:
                continue
            c = colors.setdefault(el, ["red", "white", "orange", "cyan", "magenta"][len(colors) % 5])
            big = lab == centre
            ax.scatter(p[ix], p[iy], s=130 if big else 70, marker="*", color=c, edgecolor="k",
                       linewidth=1.0 if big else 0.6, label=el, zorder=5)
        ax.set_xlabel(f"{xl} (A)")
        ax.set_ylabel(f"{yl} (A)")
        ax.set_title(f"min-F in +/-{slab:g} A slab normal to {'xyz'[axis]}" if slab > 0
                     else f"min-F projection along {'xyz'[axis]}")
    h, l = axs[0].get_legend_handles_labels()
    if h:
        u = dict(zip(l, h))
        axs[0].legend(u.values(), u.keys(), loc="upper right", fontsize=8, title="dopants", framealpha=0.85)
    fig.colorbar(im, ax=axs, shrink=0.85,
                 label=("F - F_site (eV)" if unit == "eV" else "(F - F_site) / kT") + "   [grey = not sampled]")
    sub = []
    if centre:
        sub.append(f"centred on {centre}")
    elif shift is not None and np.any(np.asarray(shift) != 0):
        sub.append(f"periodic shift ({shift[0]:g}, {shift[1]:g}, {shift[2]:g}) A")
    fig.suptitle(title + ("\n" + "; ".join(sub) if sub else ""), y=1.06 if sub else 1.0)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_rdf(path, stats, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=150)
    for s in stats:
        ln, = ax.plot(s["rdf_r"], s["rdf_g"], label=f"{s['label']}  (closest Li {s['closest_Li_A']:.2f} A)")
        ax.axvline(s["r_sphere"], color=ln.get_color(), ls=":", lw=1)
    ax.axhline(1, color="0.6", lw=0.6)
    ax.set_xlabel("r (A)")
    ax.set_ylabel("g_Li-dopant(r)  (vs box-average Li density)")
    ax.set_title(title + "  (dotted: sphere radius used)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_vmd_script(path, T):
    with open(path, "w") as f:
        f.write(f"""# VMD: load Li density / free energy from 01_density_free_energy.py (T = {T} K)
# usage:  vmd -e load_T{T}.tcl        (isovalues are starting points - adjust in Graphics > Representations)
mol new rho_T{T}K.dx type dx waitfor all
mol delrep 0 top
mol representation Isosurface 0.002 0 0 0 1 1
mol color ColorID 1
mol addrep top
mol new Fmin0_T{T}K.dx type dx waitfor all
mol delrep 0 top
mol representation Isosurface 0.15 0 0 0 1 1
mol color ColorID 4
mol addrep top
pbc box -center origin
""")


def edge_note(dops, L, margin=1.5):
    bad, sug = [], np.zeros(3)
    for d in dops:
        dm = np.minimum(d["p"], L - d["p"])
        for k in range(3):
            if dm[k] < margin:
                bad.append(f"{d['label']} ({'xyz'[k]} = {d['p'][k]:.2f})")
                sug[k] = round(L[k] / 2, 1)
    if bad:
        print(f"  [note] {', '.join(sorted(set(bad)))} within {margin} A of a cell edge: cut in the plots. "
              f"Use the same --shift {sug[0]:g} {sug[1]:g} {sug[2]:g} for every T, or --center <element|label>.")


# ------------------------------------------------------------------ driver
DEFAULT_RANGES = {"site": dict(eV=(-0.1, 0.6), kT=(-1.0, 6.0)),
                  "min": dict(eV=(0.0, 0.6), kT=(0.0, 8.0)),
                  "uniform": dict(eV=(-0.3, 0.6), kT=(-3.0, 6.0))}


def run_one(cfg, T, args, REF):
    path = args.dump or io.dump_path(cfg, T)
    print(f"\n=== {cfg['name']}  T = {T} K ===")
    traj = io.prepare(cfg, path, args.tmin)
    lm = io.li_mask(cfg, traj)
    drift = io.framework_drift(cfg, traj)
    li = traj.pos[:, lm, :] - drift[:, None, :]
    frac = io.frac_wrapped(li, traj.lo, traj.L)
    L = traj.L.mean(axis=0)
    maps = build_maps(frac, L, T, args)
    dops = dopant_atoms(cfg, traj, drift)
    stats = dopant_stats(maps, dops, li, L, args) if dops else []
    for d in dops:
        if d["rms"] > 1.0:
            print(f"  [WARN] {d['label']} moves {d['rms']:.2f} A rms about its mean position: centring on it / its "
                  f"time-averaged position is not meaningful")
    outdir = os.path.join(args.outdir, cfg["name"])
    os.makedirs(outdir, exist_ok=True)
    tag = f"T{T}K"
    kT = maps["kT"]

    rdf_kw = {"rdf_r": stats[0]["rdf_r"]} if stats else {}
    rdf_kw.update({f"rdf_g_{s['label']}": s["rdf_g"] for s in stats})
    np.savez_compressed(os.path.join(outdir, f"density_F_{tag}.npz"), rho=maps["rho"], F=maps["F"],
                        F_min0=maps["F_min0"], F_rel=maps["F_rel"], F_err=maps["err"], sampled=maps["sampled"],
                        F_site=maps["F_site"], max_resolvable=maps["max_resolvable"], counts=maps["counts"], L=maps["L"],
                        lo=traj.lo, T=T, rho_uniform=maps["rho_uni"], sigma=args.sigma, spacing=args.spacing, **rdf_kw)
    io.write_dx(os.path.join(outdir, f"rho_{tag}.dx"), maps["rho"], traj.lo, maps["L"], f"Li number density (A^-3), T={T} K")
    io.write_dx(os.path.join(outdir, f"Fmin0_{tag}.dx"), maps["F_min0"], traj.lo, maps["L"], f"F - Fmin (eV), T={T} K")
    write_vmd_script(os.path.join(outdir, f"load_{tag}.tcl"), T)

    # ---- plots
    ref = args.plot_ref
    F_eV = {"uniform": maps["F"], "min": maps["F_min0"], "site": maps["F_rel"]}[ref]
    rg = DEFAULT_RANGES[ref]
    vm = {"eV": (args.vmin if args.vmin is not None else rg["eV"][0], args.vmax if args.vmax is not None else rg["eV"][1]),
          "kT": (args.vmin_kT if args.vmin_kT is not None else rg["kT"][0],
                 args.vmax_kT if args.vmax_kT is not None else rg["kT"][1])}
    units = ["eV", "kT"] if args.units == "both" else [args.units]
    smooth = f"sigma {args.sigma} A" if args.no_adaptive else f"adaptive sigma {args.sigma_min}-{args.sigma_max} A"
    title = (f"{cfg['name']}  T = {T} K   (window {traj.time_ps[0]:.0f}-{traj.time_ps[-1]:.0f} ps, "
             f"{maps['n_frames']} frames, {maps['n_li']} Li, {args.spacing} A voxels, {smooth})")

    views = [(None, args.shift)]
    if args.center:
        if not dops:
            raise SystemExit("--center needs dopants (none in element_list)")
        pick = [d["label"] for d in dops if args.center in (d["label"], d["element"]) or args.center == "all"]
        if not pick:
            raise SystemExit(f"--center {args.center}: no such dopant. Available: {[d['label'] for d in dops]}")
        if not REF:
            REF.update({d["label"]: d["p"].copy() for d in dops})
            print(f"  [center] reference positions taken from T = {T} K; the same shift is used for every T in this call")
        own = {d["label"]: d["p"] for d in dops}
        views = [(lab, L / 2 - REF.get(lab, own[lab])) for lab in pick]
    elif not args.shift and dops:
        edge_note(dops, L)
    xyz = [(d["label"], d["element"], d["p"]) for d in dops]
    for lab, shift in views:
        sfx = f"_{lab}" if lab else ""
        if "eV" in units:
            plot_maps(os.path.join(outdir, f"F_projections_{tag}{sfx}.png"), F_eV, L, xyz, title, "eV", *vm["eV"],
                      shift=shift, slab=args.slab if lab else 0.0, centre=lab)
        if "kT" in units:
            plot_maps(os.path.join(outdir, f"F_projections_kT_{tag}{sfx}.png"), F_eV / kT, L, xyz, title, "kT",
                      *vm["kT"], shift=shift, slab=args.slab if lab else 0.0, centre=lab)
    if stats:
        plot_rdf(os.path.join(outdir, f"dopant_rdf_{tag}.png"), stats, f"{cfg['name']} T = {T} K")

    per_voxel = maps["n_frames"] * maps["n_li"] / maps["counts"].size
    summary = dict(system=cfg["name"], T=T, frames=maps["n_frames"], n_li=maps["n_li"],
                   window_ps=[float(traj.time_ps[0]), float(traj.time_ps[-1])], grid=[int(x) for x in maps["n"]],
                   voxel_A=[float(x) for x in maps["voxel"]], sigma_A=args.sigma, adaptive=not args.no_adaptive,
                   sigma_min_A=args.sigma_min, sigma_max_A=args.sigma_max, kT_eV=kT,
                   rho_uniform_per_A3=maps["rho_uni"], mean_raw_counts_per_voxel=per_voxel,
                   F_range_eV=[float(maps["F"].min()), float(maps["F"].max())], F_site_ref_eV=maps["F_site"],
                   n_site_maxima=maps["n_sites"], sampled_fraction=float(maps["sampled"].mean()),
                   max_resolvable_dF_eV=maps["max_resolvable"], min_neff=args.min_neff,
                   smoothing_sensitivity_eV=maps["sens"], F_err_median_network_eV=maps["err_med"],
                   dopants=[{k: v for k, v in s.items() if k not in ("rdf_r", "rdf_g")} for s in stats])
    with open(os.path.join(outdir, f"density_summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  grid {tuple(int(x) for x in maps['n'])}, mean raw counts/voxel = {per_voxel:.2f}")
    print(f"  F_site = {maps['F_site']:.3f} eV = {maps['F_site'] / kT:.2f} kT (median over {maps['n_sites']} density maxima);"
          f" rho_site/rho_uniform = {np.exp(-maps['F_site'] / kT):.1f}")
    print(f"  sampled volume {100 * summary['sampled_fraction']:.0f}%; barriers resolvable up to ~{maps['max_resolvable']:.2f} eV "
          f"({maps['max_resolvable'] / kT:.1f} kT) above a site (grey = beyond that)")
    print(f"  sampling limit: voxel is sampled if the sigma = {args.sigma} A kernel holds >= {args.min_neff} raw counts -> "
          f"voxels with Li density below {100 * np.exp(-maps['max_resolvable'] / kT):.2f} % of a site are grey "
          f"(ln(rho_site/rho_thr) = {maps['max_resolvable'] / kT:.2f} kT)")
    print(f"  F uncertainty (block SE, network voxels) ~ {maps['err_med']:.3f} eV; smoothing sensitivity "
          f"(adaptive vs fixed sigma) ~ {maps['sens']:.3f} eV")
    fmt = lambda x: "-" if x is None else f"{x:.2f}"
    for s in stats:
        print(f"  {s['label']}: closest Li {s['closest_Li_A']:.2f} A, <Li within 2 A> = {s['mean_Li_within_2A']:.2f}; "
              f"g(r) first peak {fmt(s['r_first_peak'])} A, first min {fmt(s['r_first_min'])} A -> sphere r = {s['r_sphere']:.2f} A "
              f"({s['r_sphere_source']}): rho/rho_uni = {s['rho_over_uni_sphere']:.2f} "
              f"(control Li-site spheres {s['control_median']:.2f}, {s['control_p16']:.2f}-{s['control_p84']:.2f}, "
              f"n={s['n_control']}); unsampled in sphere {100 * s['unsampled_frac_in_sphere']:.0f}%")
    if per_voxel < 0.2:
        print("  [note] sparse raw sampling per voxel: rely on the smoothed map")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="folder with the dumps + element_list (or a settings .yaml whose dump_dir points there)")
    ap.add_argument("--defaults", help="settings file with run_steps, tmin_ps, ... (default: analysis_defaults.yaml)")
    ap.add_argument("--T", type=int, nargs="*")
    ap.add_argument("--dump", help="explicit dump file (only with a single --T)")
    ap.add_argument("--spacing", type=float, default=0.2, help="voxel size, A (default 0.2)")
    ap.add_argument("--sigma", type=float, default=0.5, help="pilot / fixed Gaussian sigma, A (default 0.5)")
    ap.add_argument("--no-adaptive", action="store_true", help="plain fixed-sigma smoothing instead of adaptive")
    ap.add_argument("--sigma-min", type=float, default=0.25, help="adaptive: narrowest kernel, A (default 0.25)")
    ap.add_argument("--sigma-max", type=float, default=0.6, help="adaptive: widest kernel, A (default 0.6)")
    ap.add_argument("--floor", type=float, default=1e-4, help="rho floor as fraction of uniform (only for the floored F)")
    ap.add_argument("--min-neff", type=float, default=10.0,
                    help="min raw counts inside the fixed-sigma kernel for a voxel to count as sampled (default 10)")
    ap.add_argument("--site-ratio", type=float, default=3.0, help="site maxima need rho >= this x uniform (default 3)")
    ap.add_argument("--site-radius", type=float, default=1.0, help="A; maxima are local within this radius (default 1.0)")
    ap.add_argument("--nblocks-err", type=int, default=4, help="time blocks for the F standard error (default 4)")
    ap.add_argument("--rdop", type=float, help="sphere radius (A) around dopants; default: first g(r) minimum")
    ap.add_argument("--plot-ref", choices=["site", "min", "uniform"], default="site",
                    help="zero of the plotted F: 'site' (default, typical Li site), 'min' (deepest voxel), 'uniform'")
    ap.add_argument("--units", choices=["both", "eV", "kT"], default="both", help="plot F in eV, in kT, or both (default)")
    ap.add_argument("--vmin", type=float)
    ap.add_argument("--vmax", type=float)
    ap.add_argument("--vmin-kT", type=float)
    ap.add_argument("--vmax-kT", type=float)
    ap.add_argument("--shift", type=float, nargs=3, metavar=("SX", "SY", "SZ"),
                    help="fixed periodic shift (A) for the plotted map and dopants; same values for every T")
    ap.add_argument("--center", help="dopant element (Ga), atom label (Ga1) or 'all': one figure per atom, centred")
    ap.add_argument("--slab", type=float, default=0.0, help="with --center: half-thickness (A) of a slab projection")
    ap.add_argument("--tmin", type=float, help="override analysis-window start (ps)")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    if args.slab > 0 and not args.center:
        ap.error("--slab needs --center")
    cfg = io.load_config(args.target, args.defaults)
    temps = args.T or io.temperatures(cfg)
    if args.dump and len(temps) != 1:
        ap.error("--dump needs exactly one --T")
    REF = {}
    for T in temps:
        run_one(cfg, T, args, REF)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
