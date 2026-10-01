#!/usr/bin/env bash
# run_pipeline.sh - run 01, 02, 03, 04 (crystal sites), 05, 06 (Li-O ECoN) and 07 (hop loops / net hops) on one or more MD folders.
# usage:  bash run_pipeline.sh [flags] /path/to/MD_run_A [/path/to/MD_run_B ...]
# Each folder needs the dumps + element_list (+ analysis_defaults.yaml if you use one) UNLESS given by flags. Flags (before the folders,
# same for every folder given; each overrides the file/default and is passed to 01 03 04 05 06 07 incl. all sweeps):
#   --dump-dir D         dumps live in D (absolute, or relative to each run folder)
#   --prefix P / --dump-pattern 'dump_{prefix}_{T}K.lammpstrj'   dump file naming
#   --elements "Li La Zr Ga Ru O"   element of LAMMPS atom type 1..N (replaces element_list)
#   --element-list FILE  element_list file elsewhere (absolute, or relative to each run folder)
#   --dopants "Ga Ru"    dopant elements (default: everything not Li La Zr O)
#   --fold "NX NY NZ"    supercell, conventional cells along a b c, e.g. "1 1 2" (default: from the box)
#   --name N             system name for results/<N>, hops_cif/<N> ... (default: run folder name)
#   --scripts DIR        folder with 01-07 + llzo_*.py (default: the folder this script is in)
#   --cif FILE           Li-site CIF for 04 (default below)
# Outputs in each folder:  results/<system>/  (01, 02, 03)   hops_cif/<system>/  (04)   msd05/<system>/  (05)
#                          econ06/<system>/  (06)   loops07/  (07)   logs 01.log ... 07.log
# SENS=1 bash run_pipeline.sh <folders>   : 07 sensitivity only (needs a finished normal run: hops_cif/). Outputs:
#   loops07_ml4/ (--maxloop 4)   loops07_ml16/ (--maxloop 16)   loops07_s07/ (--small 0.7 A)   loops07_s15/ (--small 1.5 A)
#   compare NET/far and ROBUST/far with loops07/ (default --maxloop 8, --small 1.0)
# TEST=1 bash run_pipeline.sh <folders>   : timescale test only (needs a finished normal run: results/, hops_cif/). Outputs:
#   hops_cif_mr1/  04 with --min-res 1.0 ps        econ06_mr1/     06 default (0.5 ps x 2 = 1.0 ps) vs that 04   -> matched 1.0 ps
#   econ06_w03/    06 with 0.15 ps x 2 = 0.3 ps    vs default 04 (0.3 ps)                                         -> matched 0.3 ps
#   econ06_w03_d09/ same + --hop-d 0.9             (short-hop check)
#   loops07_mr1/   07 on the 1.0 ps hop list (hops_cif_mr1) with --win-ps 1.0   -> does the Ga retracing stand out against a cleaner far baseline?
#   read: consistency_04_vs_06.png in each econ06_* folder; if 04 and 06 agree once the min residence is matched, the
#   earlier disagreement was the timescale filter, not the dopant.
# PARAM=1 bash run_pipeline.sh <folders>  : parameter sweeps only (needs a finished normal run: results/, msd05/, hops_cif/, loops07/ = baseline).
#   The normal run now ends with these sweeps (NOSWEEP=1 skips them, NOSWEEP=fast skips only the slow 04 reruns). Each variant has its own
#   folder sweep/<script>_<tag>/ and log sweep_logs/<script>_<tag>.log:  01 sigma / min-neff / spacing / rdop,  03 fit-start / max-lag-frac,
#   05 shells / far / blocks / lags,  06 shells / far / rmax / hop-d / hop window,  07 maxloop / small / dmax / shells / far / blocks / win-ps,
#   04 reruns (min-res 0.5 and 1.0 ps, no pair merging) with matched 06 and 07 on their hop lists.
#   Then sweep_summary.txt / sweep_summary.csv: per quantity the base value, the range over all variants, whether the sign (side of 1) and the
#   shell ordering survive, and warnings for shells with few Li. Only 01, 03, 05 and 07 are summarised (04 and 06 run but are not).
# A failing normal step stops that system (each step has || exit 1; set -e alone is ignored here) and the script goes on to the next folder.
# A failing sweep item is logged to sweep_failed.txt and the sweeps go on.
BASE=/media/sampk/350GB/1_qpivolta/1_Doped_cLLZO
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # scripts folder = where this file lives (override: --scripts DIR)
CIF=$BASE/cubic_LLZO_wyckoff.cif

# ---- system flags -> CFG (passed to every python call) ; --cif and --name are also used here
CFG=()
NAME=
while [ $# -gt 0 ]; do
  case $1 in
    --cif) CIF=$2; shift 2 ;;
    --scripts) S=$(cd "$2" && pwd) || exit 1; shift 2 ;;
    --name) NAME=$2; CFG+=(--name "$2"); shift 2 ;;
    --dump-dir|--prefix|--dump-pattern|--element-list) CFG+=("$1" "$2"); shift 2 ;;
    --elements|--dopants|--fold) read -ra _t <<< "$2"; CFG+=("$1" "${_t[@]}"); shift 2 ;;
    --) shift; break ;;
    --*) echo "unknown flag: $1"; exit 1 ;;
    *) break ;;
  esac
done

[ $# -ge 1 ] || { echo "usage: bash run_pipeline.sh [flags] /path/to/MD_run [more folders]"; exit 1; }
[ -f "$CIF" ] || { echo "CIF not found: $CIF"; exit 1; }
[ -f "$S/04_sites_hops.py" ] || { echo "scripts not found: $S"; exit 1; }

# try <tag> <command...> : run one sweep item, log it, record a failure and carry on (returns 1 on failure)
try() {
  local tag=$1
  shift
  echo "---- sweep $tag"
  if ! "$@" 2>&1 | tee "sweep_logs/$tag.log"; then
    echo "$tag" >> sweep_failed.txt
    echo "!!!! sweep item failed: $tag"
    return 1
  fi
}

sweep_summary() {
python - "$1" <<'PYEOF'
import csv, glob, json, math, os, re, sys
import numpy as np

name = sys.argv[1]
KB = 8.617333262e-5
NAN = float("nan")
fin = lambda x: x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return NAN


def var_dirs(prefix):
    """(variant tag, folder) for the sweep runs of one script; 'base' is added by the caller."""
    return [(os.path.basename(p), p) for p in sorted(glob.glob(f"sweep/{prefix}_*")) if os.path.isdir(p)]


# ---------------------------------------------------------------- readers: each returns records
# record = dict(var, T, g (group key, last item = shell index when shells exist), far, n, <metrics>)
def read01(var, folder):
    out = []
    for p in sorted(glob.glob(os.path.join(folder, "density_summary_T*K.json"))):
        T = int(re.search(r"T(\d+)K", p).group(1))
        s = json.load(open(p))
        for d in s.get("dopants", []):
            cm = d.get("control_median", NAN)
            r = d.get("rho_over_uni_sphere", NAN)
            out.append(dict(var=var, T=T, g=(d["label"],), far=False, n=NAN, rho_over_uni_sphere=r,
                            ratio_to_control=r / cm if fin(cm) and cm else NAN, closest_Li_A=d.get("closest_Li_A", NAN),
                            Li_within_2A=d.get("mean_Li_within_2A", NAN)))
        out.append(dict(var=var, T=T, g=("global",), far=False, n=NAN,
                        rho_site_over_uniform=math.exp(-s["F_site_ref_eV"] / (KB * T)) if "F_site_ref_eV" in s else NAN,
                        sampled_fraction=s.get("sampled_fraction", NAN)))
    return out


def read03(var, folder):
    p = os.path.join(folder, "haven_summary.csv")
    if not os.path.exists(p):
        return []
    R = list(csv.DictReader(open(p)))
    out = [dict(var=var, T=int(r["T"]), g=("all",), far=False, n=NAN, D_tracer_cm2s=fl(r["D_tracer_cm2s"]),
                D_sigma_cm2s=fl(r["D_sigma_cm2s"]), H_R=fl(r["H_R"])) for r in R]
    Ts = np.array([int(r["T"]) for r in R], float)
    D = np.array([fl(r["D_tracer_cm2s"]) for r in R])
    ok = np.isfinite(D) & (D > 0)
    if ok.sum() >= 3:
        slope = np.polyfit(1.0 / Ts[ok], np.log(D[ok]), 1)[0]
        out.append(dict(var=var, T=0, g=("Ea",), far=False, n=NAN, Ea_Dstar_eV=-slope * KB))
    return out


def read05(var, folder):
    p = os.path.join(folder, "msd_shell.csv")
    if not os.path.exists(p):
        return []
    R = list(csv.DictReader(open(p)))
    cnt, size = {}, {}
    for r in R:
        k = (r["T"], r["dopant"], round(fl(r["tau_ps"]), 2))
        size[k] = size.get(k, 0) + 1
    out = []
    for r in R:
        k = (r["T"], r["dopant"], round(fl(r["tau_ps"]), 2))
        i = cnt.get(k, 0)
        cnt[k] = i + 1
        out.append(dict(var=var, T=int(r["T"]), g=(r["dopant"], k[2], i), far=(i == size[k] - 1), n=fl(r["n_Li"]),
                        density_ratio=fl(r["density_ratio"]), ratio_to_far=fl(r["ratio_to_far"])))
    return out


def read07(var, folder):
    p = os.path.join(folder, "hop_loops.csv")
    if not os.path.exists(p):
        return []
    cnt, out = {}, []
    for r in csv.DictReader(open(p)):
        k = (r["T"], r["dopant"])
        i = cnt.get(k, 0)
        cnt[k] = i + 1
        out.append(dict(var=var, T=int(r["T"]), g=(r["dopant"], i), far=(r["is_far"] == "True"), n=fl(r["n_Li_mean"]),
                        net_ratio=fl(r["net_ratio"]), rob_ratio=fl(r["rob_ratio"]), all_ratio=fl(r["all_ratio"]),
                        pct_in_loops=fl(r["pct_in_loops"]), pct_small_net=fl(r["pct_small_net"])))
    return out


# (dataset, baseline folder, sweep prefix(es), reader, metrics, metrics whose reference value is 1, ratio-to-far metrics)
DATA = [
    ("01", f"results/{name}", ["01"], read01,
     ["rho_over_uni_sphere", "ratio_to_control", "closest_Li_A", "Li_within_2A", "rho_site_over_uniform", "sampled_fraction"],
     {"rho_over_uni_sphere", "ratio_to_control"}, set()),
    ("03", f"results/{name}", ["03"], read03, ["D_tracer_cm2s", "D_sigma_cm2s", "H_R", "Ea_Dstar_eV"], set(), set()),
    ("05", f"msd05/{name}", ["05"], read05, ["density_ratio", "ratio_to_far"], {"density_ratio", "ratio_to_far"},
     {"ratio_to_far"}),
    # 07 also on the hop lists of the 04 reruns (min-res / merge-pairs): sweep/07_<tag> and sweep/07_04_<tag>
    ("07", "loops07", ["07"], read07, ["net_ratio", "rob_ratio", "all_ratio", "pct_in_loops", "pct_small_net"],
     {"net_ratio", "rob_ratio", "all_ratio"}, {"net_ratio", "rob_ratio", "all_ratio"}),
]

rows, lown, missing = [], [], []
for ds, basedir, prefs, reader, metrics, ref1, nofar in DATA:
    recs = reader("base", basedir)
    if not recs:
        missing.append(f"{ds}: baseline not found in {basedir}")
        continue
    for pf in prefs:
        for tag, folder in var_dirs(pf):
            sub = folder if ds in ("07",) else os.path.join(folder, name)
            got = reader(tag, sub)
            if not got:
                missing.append(f"{ds}: {tag} produced nothing (failed? see sweep_failed.txt)")
            recs += got
    # ---- small-n warnings (per variant, mean over T)
    nn = {}
    for r in recs:
        if fin(r["n"]) and len(r["g"]) >= 2:
            nn.setdefault((r["var"], r["g"][0], r["g"][-1], r["far"]), []).append(r["n"])
    for (v, dop, si, far), a in sorted(nn.items(), key=str):
        m = float(np.mean(a))
        if (not far and m < 3) or (far and m < 10):
            lown.append((ds, v, dop, si, far, m))
    # ---- per (metric, group): spread over variants, sign agreement, shell ordering
    store = {}
    for m in metrics:
        gs = sorted({r["g"] for r in recs if fin(r.get(m, NAN)) and not (r["far"] and m in nofar)}, key=str)
        for g in gs:
            byv = {}
            for r in recs:
                if r["g"] == g and fin(r.get(m, NAN)) and not (r["far"] and m in nofar):
                    byv.setdefault(r["var"], {})[r["T"]] = r[m]
            if "base" not in byv:
                continue
            base = byv["base"]
            means = {v: float(np.mean(list(d.values()))) for v, d in byv.items()}
            store[(m, g)] = means
            same = tot = 0
            if m in ref1:
                for v, d in byv.items():
                    if v == "base":
                        continue
                    for T, x in d.items():
                        if T in base:
                            tot += 1
                            same += (x - 1.0) * (base[T] - 1.0) > 0
            vals = list(means.values())
            b = means["base"]
            rows.append(dict(dataset=ds, metric=m, group="/".join(str(x) for x in g), base=b, vmin=min(vals), vmax=max(vals),
                             spread_pct=100 * (max(vals) - min(vals)) / abs(b) if b else NAN, n_variants=len(means) - 1,
                             sign_same_pct=100.0 * same / tot if tot else NAN, order=""))
    # ---- shell ordering (non-far shells) in every variant vs baseline, per (metric, dopant[, tau])
    ordr = {}
    for (m, g), means in store.items():
        if m in ref1 and len(g) >= 2 and isinstance(g[-1], int):
            ordr.setdefault((m, g[:-1]), {})[g[-1]] = means
    for (m, gp), bysh in ordr.items():
        idx = sorted(bysh)
        if len(idx) < 2:
            continue
        vs = sorted({v for i in idx for v in bysh[i] if v != "base"})
        ok = n = 0
        for v in vs:
            if all(v in bysh[i] and "base" in bysh[i] for i in idx):
                n += 1
                ok += list(np.argsort([bysh[i]["base"] for i in idx])) == list(np.argsort([bysh[i][v] for i in idx]))
        for r in rows:
            if r["dataset"] == ds and r["metric"] == m and r["group"].startswith("/".join(str(x) for x in gp) + "/"):
                r["order"] = f"{ok}/{n}" if n else ""

with open("sweep_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["dataset", "metric", "group", "base", "vmin", "vmax", "spread_pct", "n_variants",
                                       "sign_same_pct", "order"])
    w.writeheader()
    w.writerows(rows)

L = [f"sweep summary for {name}",
     "base = default run; vmin/vmax = range of the T-mean over base + all variants; spread = (vmax-vmin)/|base|.",
     "sign% = % of (T, variant) values on the same side of 1 as the base value at that T (ratio metrics only).",
     "order = variants (of those run) whose shell ordering equals the base ordering (non-far shells).",
     "group = dopant/tau_ps/shell index (05), dopant/shell index (07), dopant (01), all (03). Shell 0 = innermost.", ""]
cur = None
for r in rows:
    if (r["dataset"], r["metric"]) != cur:
        cur = (r["dataset"], r["metric"])
        L += ["", f"[{r['dataset']}] {r['metric']}",
              f"  {'group':<22}{'base':>11}{'vmin':>11}{'vmax':>11}{'spread%':>9}{'nvar':>5}{'sign%':>7}{'order':>7}"]
    sg = f"{r['sign_same_pct']:7.0f}" if fin(r["sign_same_pct"]) else " " * 7
    L.append(f"  {r['group']:<22}{r['base']:11.4g}{r['vmin']:11.4g}{r['vmax']:11.4g}{r['spread_pct']:9.0f}"
             f"{r['n_variants']:5d}{sg}{r['order']:>7}")
L += ["", "WARNINGS"]
if not lown and not missing:
    L.append("  none")
grp = {}
for ds, v, dop, si, far, m in lown:
    grp.setdefault((ds, dop, si, far), []).append((v, m))
for (ds, dop, si, far), lst in sorted(grp.items(), key=str):
    L.append(f"  [{ds}] {dop} shell {si}{' (far)' if far else ''}: mean n_Li < {10 if far else 3} in "
             + ", ".join(f"{v} ({m:.1f})" for v, m in lst)
             + (" -> far reference thin, every ratio to far is noisy" if far else " -> ratios in this shell are weak"))
for s in missing:
    L.append(f"  {s}")
open("sweep_summary.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
print("\n[+] sweep_summary.csv sweep_summary.txt")
PYEOF
}

# param_sweep: run from inside one system folder ($name set); baseline = the normal run's outputs
param_sweep() {
  local B01="--sigma 0.5 --sigma-min 0.2 --sigma-max 0.6" mr
  mkdir -p sweep sweep_logs
  : > sweep_failed.txt
  # ---- 01 (smoothing, sampling threshold, grid, dopant sphere); baseline = --sigma 0.5 --sigma-min 0.2 --sigma-max 0.6
  try 01_s04 python "$S/01_density_free_energy.py" . "${CFG[@]}" --sigma 0.4 --sigma-min 0.15 --sigma-max 0.5 --outdir sweep/01_s04 || true
  try 01_s06 python "$S/01_density_free_energy.py" . "${CFG[@]}" --sigma 0.6 --sigma-min 0.25 --sigma-max 0.7 --outdir sweep/01_s06 || true
  try 01_ne5 python "$S/01_density_free_energy.py" . "${CFG[@]}" $B01 --min-neff 5 --outdir sweep/01_ne5 || true
  try 01_ne20 python "$S/01_density_free_energy.py" . "${CFG[@]}" $B01 --min-neff 20 --outdir sweep/01_ne20 || true
  try 01_sp25 python "$S/01_density_free_energy.py" . "${CFG[@]}" $B01 --spacing 0.25 --outdir sweep/01_sp25 || true
  try 01_rd25 python "$S/01_density_free_energy.py" . "${CFG[@]}" $B01 --rdop 2.5 --outdir sweep/01_rd25 || true
  try 01_rd35 python "$S/01_density_free_energy.py" . "${CFG[@]}" $B01 --rdop 3.5 --outdir sweep/01_rd35 || true
  # ---- 03 (MSD fit window and lag range)
  try 03_fs02 python "$S/03_van_hove_haven.py" . "${CFG[@]}" --fit-start 0.2 --outdir sweep/03_fs02 || true
  try 03_fs05 python "$S/03_van_hove_haven.py" . "${CFG[@]}" --fit-start 0.5 --outdir sweep/03_fs05 || true
  try 03_ml15 python "$S/03_van_hove_haven.py" . "${CFG[@]}" --max-lag-frac 0.15 --outdir sweep/03_ml15 || true
  try 03_ml40 python "$S/03_van_hove_haven.py" . "${CFG[@]}" --max-lag-frac 0.4 --outdir sweep/03_ml40 || true
  # ---- 05 (shell edges, far reference, jackknife blocks, lags); baseline = --lags 1 5 10
  try 05_shA python "$S/05_msd_by_shell.py" . "${CFG[@]}" --lags 1 5 10 --shells 2.5 4.5 6.5 --outdir sweep/05_shA || true
  try 05_shB python "$S/05_msd_by_shell.py" . "${CFG[@]}" --lags 1 5 10 --shells 3.5 5.5 7.5 --outdir sweep/05_shB || true
  try 05_far python "$S/05_msd_by_shell.py" . "${CFG[@]}" --lags 1 5 10 --far self --outdir sweep/05_far || true
  try 05_bl10 python "$S/05_msd_by_shell.py" . "${CFG[@]}" --lags 1 5 10 --blocks 10 --outdir sweep/05_bl10 || true
  try 05_lag python "$S/05_msd_by_shell.py" . "${CFG[@]}" --lags 2 5 20 --outdir sweep/05_lag || true
  # ---- 06 (dumps only: shells, far, rmax, site-free hop definition)
  try 06_shA python "$S/06_li_econ.py" . "${CFG[@]}" --shells 2.5 4.5 6.5 --no-pairs --no-econ-hops --outdir sweep/06_shA || true
  try 06_shB python "$S/06_li_econ.py" . "${CFG[@]}" --shells 3.5 5.5 7.5 --no-pairs --no-econ-hops --outdir sweep/06_shB || true
  try 06_far python "$S/06_li_econ.py" . "${CFG[@]}" --far self --no-pairs --no-econ-hops --outdir sweep/06_far || true
  try 06_rm30 python "$S/06_li_econ.py" . "${CFG[@]}" --rmax 3.0 --no-pairs --no-econ-hops --outdir sweep/06_rm30 || true
  try 06_rm40 python "$S/06_li_econ.py" . "${CFG[@]}" --rmax 4.0 --no-pairs --no-econ-hops --outdir sweep/06_rm40 || true
  try 06_bl10 python "$S/06_li_econ.py" . "${CFG[@]}" --blocks 10 --no-pairs --no-econ-hops --outdir sweep/06_bl10 || true
  try 06_hd09 python "$S/06_li_econ.py" . "${CFG[@]}" --hop-d 0.9 --no-pairs --no-econ-hops --outdir sweep/06_hd09 || true
  try 06_hd15 python "$S/06_li_econ.py" . "${CFG[@]}" --hop-d 1.5 --no-pairs --no-econ-hops --outdir sweep/06_hd15 || true
  try 06_hw025 python "$S/06_li_econ.py" . "${CFG[@]}" --hop-win-ps 0.25 --hop-confirm 2 --no-pairs --no-econ-hops --outdir sweep/06_hw025 || true
  try 06_hw015 python "$S/06_li_econ.py" . "${CFG[@]}" --hop-win-ps 0.15 --hop-confirm 2 --no-pairs --no-econ-hops --outdir sweep/06_hw015 || true
  # ---- 07 on the default hop list (loop length, flicker threshold, palindrome cap, shells, far, blocks, displacement window)
  try 07_ml4 python "$S/07_hop_loops.py" . "${CFG[@]}" --hops-dir "hops_cif/$name" --maxloop 4 --no-flags --outdir sweep/07_ml4 || true
  try 07_ml16 python "$S/07_hop_loops.py" . "${CFG[@]}" --hops-dir "hops_cif/$name" --maxloop 16 --no-flags --outdir sweep/07_ml16 || true
  try 07_s07 python "$S/07_hop_loops.py" . "${CFG[@]}" --hops-dir "hops_cif/$name" --small 0.7 --no-flags --outdir sweep/07_s07 || true
  try 07_s15 python "$S/07_hop_loops.py" . "${CFG[@]}" --hops-dir "hops_cif/$name" --small 1.5 --no-flags --outdir sweep/07_s15 || true
  try 07_dm4 python "$S/07_hop_loops.py" . "${CFG[@]}" --hops-dir "hops_cif/$name" --dmax 4 --no-flags --outdir sweep/07_dm4 || true
  try 07_shA python "$S/07_hop_loops.py" . "${CFG[@]}" --hops-dir "hops_cif/$name" --shells 2.5 4.5 6.5 --no-flags --outdir sweep/07_shA || true
  try 07_shB python "$S/07_hop_loops.py" . "${CFG[@]}" --hops-dir "hops_cif/$name" --shells 3.5 5.5 7.5 --no-flags --outdir sweep/07_shB || true
  try 07_far python "$S/07_hop_loops.py" . "${CFG[@]}" --hops-dir "hops_cif/$name" --far self --no-flags --outdir sweep/07_far || true
  try 07_bl10 python "$S/07_hop_loops.py" . "${CFG[@]}" --hops-dir "hops_cif/$name" --blocks 10 --no-flags --outdir sweep/07_bl10 || true
  try 07_w015 python "$S/07_hop_loops.py" . "${CFG[@]}" --hops-dir "hops_cif/$name" --win-ps 0.15 --no-flags --outdir sweep/07_w015 || true
  try 07_w06 python "$S/07_hop_loops.py" . "${CFG[@]}" --hops-dir "hops_cif/$name" --win-ps 0.6 --no-flags --outdir sweep/07_w06 || true
  # ---- 04 reruns with other hop definitions (slow; skipped with NOSWEEP=fast), then 07 and 06 on their hop lists with matched windows
  if [ "$NOSWEEP" != fast ]; then
    for mr in 0.5 1.0 nomerge; do
      case $mr in
        0.5)     tag=mr05; a04="--merge-pairs 1.0 --min-res 0.5"; w07=0.5; w06=0.25 ;;
        1.0)     tag=mr10; a04="--merge-pairs 1.0 --min-res 1.0"; w07=1.0; w06=0.5 ;;
        nomerge) tag=nm;   a04="--merge-pairs 0 --min-res 0.3";   w07=0.3; w06=0.15 ;;
      esac
      if try 04_$tag python "$S/04_sites_hops.py" . "${CFG[@]}" --sites-cif "$CIF" $a04 --nshuf 50 --no-barriers --results results --outdir sweep/04_$tag; then
        try 07_04_$tag python "$S/07_hop_loops.py" . "${CFG[@]}" --hops-dir "sweep/04_$tag/$name" --win-ps $w07 --no-flags --outdir sweep/07_04_$tag || true
        try 06_04_$tag python "$S/06_li_econ.py" . "${CFG[@]}" --hops-dir "sweep/04_$tag/$name" --hop-win-ps $w06 --hop-confirm 2 --no-pairs --no-econ-hops \
          --outdir sweep/06_04_$tag || true
      fi
    done
  fi
  [ -s sweep_failed.txt ] && { echo "sweep items that failed:"; cat sweep_failed.txt; }
  sweep_summary "$name" | tee sweep_summary.log || echo "!!!! sweep summary failed"
  return 0
}

for d in "$@"; do
  echo "################ $d"
  (
    set -e -o pipefail
    cd "$d"
    name=$(basename "$(pwd)")
    if [ -n "$NAME" ]; then name=$NAME; fi
    if [ -n "$SENS" ]; then
      mkdir -p loops07_ml4 loops07_ml16 loops07_s07 loops07_s15
      python "$S/07_hop_loops.py" . "${CFG[@]}" --hops-dir "hops_cif/$name" --maxloop 4  --outdir loops07_ml4  | tee 07_ml4.log  || exit 1
      python "$S/07_hop_loops.py" . "${CFG[@]}" --hops-dir "hops_cif/$name" --maxloop 16 --outdir loops07_ml16 | tee 07_ml16.log || exit 1
      python "$S/07_hop_loops.py" . "${CFG[@]}" --hops-dir "hops_cif/$name" --small 0.7 --outdir loops07_s07 --no-flags | tee 07_s07.log || exit 1
      python "$S/07_hop_loops.py" . "${CFG[@]}" --hops-dir "hops_cif/$name" --small 1.5 --outdir loops07_s15 --no-flags | tee 07_s15.log || exit 1
      exit 0
    fi
    if [ -n "$TEST" ]; then
      mkdir -p hops_cif_mr1 econ06_mr1 econ06_w03 econ06_w03_d09
      python "$S/04_sites_hops.py" . "${CFG[@]}" --sites-cif "$CIF" --merge-pairs 1.0 --nshuf 200 --results results --outdir hops_cif_mr1 \
        --min-res 1.0 | tee 04_mr1.log || exit 1
      python "$S/06_li_econ.py" . "${CFG[@]}" --hops-dir "hops_cif_mr1/$name" --outdir econ06_mr1 --no-pairs --no-econ-hops | tee 06_mr1.log || exit 1
      mkdir -p loops07_mr1
      python "$S/07_hop_loops.py" . "${CFG[@]}" --hops-dir "hops_cif_mr1/$name" --win-ps 1.0 --outdir loops07_mr1 --no-flags | tee 07_mr1.log || exit 1
      python "$S/06_li_econ.py" . "${CFG[@]}" --hops-dir "hops_cif/$name" --outdir econ06_w03 --hop-win-ps 0.15 --hop-confirm 2 \
        --no-pairs --no-econ-hops | tee 06_w03.log || exit 1
      python "$S/06_li_econ.py" . "${CFG[@]}" --hops-dir "hops_cif/$name" --outdir econ06_w03_d09 --hop-win-ps 0.15 --hop-confirm 2 \
        --hop-d 0.9 --no-pairs --no-econ-hops | tee 06_w03_d09.log || exit 1
      exit 0
    fi
    if [ -n "$PARAM" ]; then
      param_sweep
      exit 0
    fi
    python "$S/01_density_free_energy.py" . "${CFG[@]}" --center all --sigma 0.5 --sigma-min 0.2 --sigma-max 0.6 | tee 01.log || exit 1
    python "$S/02_compare_temps.py" "results/$name" --center all | tee 02.log || exit 1
    python "$S/03_van_hove_haven.py" . "${CFG[@]}" | tee 03.log || exit 1
    mkdir -p hops_cif msd05 econ06 loops07
    python "$S/04_sites_hops.py" . "${CFG[@]}" --sites-cif "$CIF" --merge-pairs 1.0 --nshuf 200 --results results --outdir hops_cif \
      --net-center all --net-slab 2 --net-window 9 | tee 04.log || exit 1
    python "$S/05_msd_by_shell.py" . "${CFG[@]}" --lags 1 5 10 --outdir msd05 | tee 05.log || exit 1
    python "$S/06_li_econ.py" . "${CFG[@]}" --hops-dir "hops_cif/$name" --outdir econ06 | tee 06.log || exit 1
    python "$S/07_hop_loops.py" . "${CFG[@]}" --hops-dir "hops_cif/$name" --outdir loops07 | tee 07.log || exit 1
    [ "$NOSWEEP" = 1 ] || param_sweep
  ) || echo "!!!! failed in $d (see the last output above)"
done
