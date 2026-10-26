# LLZO hop analysis: results, method, and how to run on other systems

System analysed: `MD_run_1Ga_1Ru` (Ga on 24d Li site, Ru on Zr site, 2 conventional cells, 800-1200 K).

---

## 1. Files

| file | status | what it does |
|---|---|---|
| `04_sites_hops.py` | **changed** | hops, concerted hops, shells, barriers. New: `--sites-cif`, `--merge-pairs`, `--site-excl`, assignment-quality prints |
| `llzo_sites.py` | **new** | fixed crystal Li sites (24d + 96h) from the Ia-3d CIF, aligned to the MD box per T |
| `05_msd_by_shell.py` | **new** | model-free Li mobility (MSD) vs distance from each dopant. No sites, no hops |
| `run_pipeline.sh` | **new** | runs 01, 02, 03, 04 (crystal sites) and 05 on one or more MD folders |
| `01`, `02`, `03`, `llzo_io.py`, `llzo_maps.py` | unchanged | |

Put all files in the same folder (`pathway_analysis_scripts`). Only 04 imports `llzo_sites`. 01 still uses its own density-peak sites (see section 6).

---

## 2. Terms

- **tau (tau)**: a time gap in ps. MSD(tau) = mean of |r(t0+tau) - r(t0)|^2 over Li atoms of a group and over all start times t0.
  Small tau (1 ps) = vibration and rattling. Large tau (5-10 ps) = real travel.
- **MSD** is a per-Li average, so it does not scale with the number of atoms. The number of Li in a shell only sets the noise.
- **Ratio to far**: MSD of a shell divided by MSD of the far shell (> 7 A from that dopant), same T, same tau. 1.00 = same mobility as far Li, < 1 slower, > 1 faster. The far shell is 1.00 by definition.
- **Density ratio (the number with x, e.g. 0.48x)**: mean Li count in the shell divided by the count for a uniform Li density. 1.00x = uniform, 0.48x = half as many Li as uniform.
- **min-res**: a hop counts only if Li stays at the new site at least this long (ps). Default 0.3.
- **tc, rc**: two hops are concerted if they start within tc ps and their sites are within rc A.
- **Refill**: a vacated site is entered by another Li within tc. Reported with a shuffled chance level.
- **Merge pairs**: 96h sites come in pairs 0.8 A apart. `--merge-pairs 1.0` treats each pair as one site. Without it, rattling inside a pair counts as hops.

---

## 3. Results for MD_run_1Ga_1Ru

### 3.1 Site definition checks (04 with `--sites-cif`)

| item | CIF | log |
|---|---|---|
| operators | Ia-3d = 96 | 96 |
| Li positions per cell | 24d + 96h = 120 | 24 + 96 = 120 |
| tiled x(1,1,2) | 240 | 240 |
| Ga on a 24d site | one site removed | 239 (0.02 A from Ga) |
| La / Zr found in MD | 48 / 31 | 48/48, 31/32 (Ru replaces one Zr) |
| Li atoms | 109 | 109 -> 0.46 Li per site (239 sites) |
| merged sites | 47 single 24d + 96 pairs | 143 (72 per cell) |

- Framework fit rms 0.03 A, max 0.07 A at every T.
- 98 % of density peaks have a crystal site within 0.5 A.
- Bug fixed: the framework fit is ambiguous by half a cell (I-centring), which removed the wrong site at 1000-1200 K. After the fix the "dopant-occupied sites differs" WARNs are gone and one `shift moved` line appears per affected T.
- **Unmerged crystal sites are unusable**: median jump 0.81 A (96h pair rattling), 4530 hops against 1953 merged. Always use `--merge-pairs 1.0`.

### 3.2 Merged crystal sites agree with the old peak method (800 -> 1200 K)

| | peaks | crystal, merged |
|---|---|---|
| sites | 136 | 143 |
| hop rate per Li per ps | 0.150 -> 0.272 | 0.179 -> 0.308 |
| refill % (chance) | 24.7 (11.1) -> 43.4 (21.0) | 22.6 (11.0) -> 38.2 (21.0) |
| concerted excess % | 20, 35, 38, 19, 23 | 21, 36, 28, 20, 22 |

### 3.3 Dependence on min-res (merged crystal sites)

| min-res | hops at 800 K | back-hops | rate 800 K | rate 1200 K | Ea(total) |
|---|---|---|---|---|---|
| 0.3 ps | 1953 | 74 % | 0.179 | 0.310 | 0.113 +/- 0.006 eV |
| 0.5 ps | 1130 | 61 % | 0.104 | 0.199 | 0.131 +/- 0.005 eV |
| 1.0 ps | 668 | 45 % | 0.061 | 0.126 | 0.142 +/- 0.008 eV |

- The rate depends on min-res by about 3x. The rise with T (x1.7 to x2.0 from 800 to 1200 K) is robust.
- Ea(total) counts every hop including back-and-forth rattling, so it should be below Ea(D*) from 03. I have not compared with your 03 number.

### 3.4 Concerted and refill (tc / rc scan, merged crystal sites)

- Refill excess above chance: about 10-12 points at 800 K and 15-17 points at 1200 K, at tc 0.3, 0.5 and 1.0. Measured at min-res 0.3 only.
- Concerted excess depends strongly on rc: 8-19 % at rc 2.5 A, 19-41 % at 3.5 A, 45-100 % at 4.5 A. The chance level approaches 1 at large rc and tc, so the normalised value blows up (100 % at tc 1.0, rc 4.5, 1200 K). Negative values at tc 1.0, rc 2.5 mean unmeasurable.
- Only conclusion: hops are coupled more than chance at all tested tc and rc.

### 3.5 Dopant effect on Li mobility

**Site-based near-dopant hop-rate ratios (04) conflict and are not quotable.**
Near Ga / far: peak sites 0.24 -> 0.21; crystal merged 5.6 -> 2.7. The peak value rests on 4 hops at 1 site. The crystal value is inflated by rattling: near Ga, Li sit 0.70-0.75 A from their assigned ideal site (0.45-0.50 A elsewhere), 13-19 % of assignments are ambiguous (3-4 % elsewhere), and the 0-3 A sites are half empty (0.43 Li per site). It falls from 5.6 to 2.3 as min-res goes 0.3 -> 1.0 ps.

**Model-free MSD (05) settles it.**

Ga, MSD ratio to far, 800 -> 1200 K:

| shell | tau = 1 ps | tau = 5 ps | tau = 10 ps |
|---|---|---|---|
| 0-3 A | 1.22, 1.00, 0.99, 0.79, 0.89 | 0.81, 0.47, 0.54, 0.36, 0.52 | 0.52, 0.36, 0.37, 0.31, 0.53 |
| 3-5 A | 0.87, 0.72, 0.84, 0.66, 0.79 | 0.51, 0.39, 0.52, 0.33, 0.53 | 0.41, 0.33, 0.44, 0.27, 0.47 |
| 5-7 A | 0.93-1.08 | 0.78-0.99 | 0.73-0.98 |

- Li starting within 5 A of Ga move about 0.3-0.5 times as far as far Li over 5-10 ps, at every T. **Ga slows nearby Li.**
- Example, 800 K, 3-5 A, tau = 10 ps: MSD 1.06 vs 2.60 A^2 (about 1.0 vs 1.6 A rms).
- The 3-5 A shell (about 8 Li, errors 0.03-0.1) is the reliable one. The 0-3 A shell has only 1.3-1.7 Li (errors 0.1-0.2) but points the same way.
- At 1 ps the 0-3 A shell moves like far Li: local rattling is normal, long-range displacement is suppressed.
- No monotonic temperature trend in the ratio.
- Density around Ga (count vs uniform): 0-3 A 0.48x (800 K) to 0.62x (1200 K); 3-5 A 0.77-0.80x; 5-7 A 1.11x. Li are depleted near Ga, less so when hot.

Ru, MSD ratio to far:
- tau = 1 ps, 0-3 A: 0.90, 0.78, 0.79, 0.76, 0.81 (about 20 % lower, consistent).
- tau = 5-10 ps: 0.6-1.4 with no consistent sign, errors 0.1-0.2. **No clear long-time effect.**
- Density near Ru: about 1.1x (slightly enriched).

Ru acts as a partial control: the Ga slowing is not a generic effect of being near any lattice point. It is partial only, because Ru sits on a Zr site and Ga on a Li site.

---

## 4. What is safe to quote

- Li within 5 A of Ga move about 0.3-0.5x as far as far Li over 5-10 ps, at all T (MSD ratio, one Ga in the box).
- Li density is depleted near Ga: 0.5-0.6x within 3 A, about 0.8x at 3-5 A.
- Ru: no clear long-time effect.
- Refill 10-12 points above chance at 800 K, 15-17 at 1200 K (min-res 0.3).
- Hops are more coupled than chance. No concerted % without its rc.
- Hop rate and Ea(total): state min-res (Ea 0.11-0.14 eV), and compare only with Ea(D*) from 03.

**Do not quote**: any near-dopant hop-rate ratio from 04 (either site definition); the concerted % without rc; any temperature trend in the Ga slowing.

---

## 5. Commands

### 5.0 Paths and the full pipeline (01 -> 05)

Base folder: `/media/sampk/350GB/1_qpivolta/1_Doped_cLLZO`
- scripts: `$BASE/pathway_analysis_scripts`
- CIF (pristine Ia-3d garnet, one conventional cell): `$BASE/cubic_LLZO_wyckoff.cif`

```bash
BASE=/media/sampk/350GB/1_qpivolta/1_Doped_cLLZO
S=$BASE/pathway_analysis_scripts
CIF=$BASE/cubic_LLZO_wyckoff.cif
```

**One system, step by step** (run from its MD folder; `name` is the folder name):
```bash
cd /path/to/MD_run_x
name=$(basename "$(pwd)")

# 01  density + free-energy maps, Li-dopant g(r)          -> results/$name/
python $S/01_density_free_energy.py . --center all --sigma 0.5 --sigma-min 0.2 --sigma-max 0.6 | tee 01.log

# 02  compare temperatures (reads 01's results)           -> results/$name/
python $S/02_compare_temps.py results/$name --center all | tee 02.log

# 03  Van Hove, non-Gaussian, D*, Haven ratio             -> results/$name/
python $S/03_van_hove_haven.py . | tee 03.log

# 04  crystal sites, hops, concerted, barriers (needs 01) -> hops_cif/$name/
mkdir -p hops_cif
python $S/04_sites_hops.py . --sites-cif $CIF --merge-pairs 1.0 --nshuf 200 --results results --outdir hops_cif | tee 04.log

# 05  model-free MSD by distance from each dopant         -> msd05/$name/
mkdir -p msd05
python $S/05_msd_by_shell.py . --lags 1 5 10 --outdir msd05 | tee 05.log
```

**Several systems at once:** `run_pipeline.sh` (same folder as this file) runs all five steps with the paths above.
```bash
bash /path/to/run_pipeline.sh /path/to/sysA/MD_run_x /path/to/sysB/MD_run_y
```

Notes:
- 01 settings (`--sigma 0.5 --sigma-min 0.2 --sigma-max 0.6`) are the ones of your `results_s0.5` run. 01 defaults are 0.5 / 0.25 / 0.6. Sensitivity check: rerun with `--sigma 0.4` and `0.3` (each with `--sigma-max` = sigma + 0.1) into separate `--outdir`s, as in your earlier loop.
- 02 with `--center all` makes one figure set per dopant atom. `--shift` is ignored when `--center` is given, so drop `--shift 0 0 13`. Use `--shift` only without `--center`.
- 02 needs 01's `density_F_T*K.npz` and `density_summary_T*K.json` in `results/$name`.
- 04 reads 01's maps from `--results` (here `results`). If you ran 01 with another `--outdir` (for example `results_s0.5`), pass that to `--results`. 04 writes to `hops_cif/$name` so it never overwrites 01-03 files.
- 03 and 05 need no other step. 04's `--sites-cif` needs no 01 run unless you want barriers.
- 04 and 05 take `--T 800 1200` and `--tmin 150` like 01 and 03. 04 also takes `--min-res`, `--tc`, `--rc`.

### 5.1 Site check, no barriers (fast)

```bash
mkdir -p chk_cif
python $S/04_sites_hops.py . --sites-cif $CIF --merge-pairs 1.0 --no-barriers --nshuf 200 --outdir chk_cif | tee chk_cif/04.log
grep -H "operators\|tiled\|framework fit\|shift moved\|removed\|detected sites\|cross-check\|WARN" chk_cif/04.log
```

### 5.2 Model-free dopant effect (05)

```bash
mkdir -p msd05
python $S/05_msd_by_shell.py . --lags 1 5 10 --outdir msd05 | tee msd05/05.log
grep -B1 -A14 "Li per shell" msd05/05.log | head -80
```

### 5.3 Full 04 run with barriers (needs 01 results)

```bash
ls results*/*/density_F_T*K.npz                 # find where 01 wrote its maps
mkdir -p res_cif
python $S/04_sites_hops.py . --sites-cif $CIF --merge-pairs 1.0 --results results --outdir res_cif | tee res_cif/04.log
```
Use the folder that holds `<system>/density_F_T*K.npz` for `--results` (for example `results_s0.5`).

### 5.4 Robustness scans (optional)

```bash
# hop definition
for m in 0.3 0.5 1.0; do d=chk_cif_mr$m; mkdir -p $d
  python $S/04_sites_hops.py . --sites-cif $CIF --merge-pairs 1.0 --min-res $m --no-barriers --nshuf 50 --outdir $d | tee $d/04.log
done
grep -H "hops:\|back-hops" chk_cif_mr*/04.log

# tc / rc
for tc in 0.3 0.5 1.0; do for rc in 2.5 3.5 4.5; do d=chk_cif_tc${tc}_rc${rc}; mkdir -p $d
  python $S/04_sites_hops.py . --sites-cif $CIF --merge-pairs 1.0 --tc $tc --rc $rc --no-barriers --nshuf 200 --outdir $d | tee $d/04.log
done; done
grep -H "concerted beyond\|refill (" chk_cif_tc*/04.log
```

### 5.5 Loop over several systems

```bash
S=/full/path/to/pathway_analysis_scripts
CIF=/full/path/to/cubic_LLZO_wyckoff.cif
for sys in /path/to/sysA/MD_run_x /path/to/sysB/MD_run_y; do
  ( cd $sys
    mkdir -p chk_cif msd05
    python $S/04_sites_hops.py . --sites-cif $CIF --merge-pairs 1.0 --no-barriers --nshuf 200 --outdir chk_cif | tee chk_cif/04.log
    python $S/05_msd_by_shell.py . --lags 1 5 10 --outdir msd05 | tee msd05/05.log )
done
```

Useful options: `--T 800 1200` (only some temperatures), `--tmin 150` (later window), `--shells 3 5 7` (shell edges in A), `--site-excl 1.0` (remove crystal sites closer than this to a dopant), `--min-res`, `--tc`, `--rc`.

---

## 6. Checks to pass on every new system (from `chk_cif/04.log`)

| check | expected | if not |
|---|---|---|
| `space-group operators` | 96 | wrong CIF |
| `Li positions per cell` | 120 (24d + 96h) | wrong CIF |
| `tiled x(...)` count | 120 x number of cells | wrong fold: set `fold` in the settings file |
| box vs CIF `a` x fold | within a few % (script exits above 10 %) | wrong CIF or fold |
| La count MD vs ideal | equal | wrong fold or CIF |
| Zr count | ideal minus number of Zr-site dopants (Ru here: 31/32) | check dopants |
| framework fit rms | below 0.35 A, max below 1 A | structure changed or wrong CIF |
| `removed` lines | one site per dopant on a Li site; none for a dopant on Zr | other dopant position: check `--site-excl` |
| `detected sites` | 120 x cells minus removed (then merged) | see `WARN` |
| `dopant-occupied sites differs` WARN | absent | a dopant moved between temperatures: check the dump |
| cross-check: peaks with a crystal site within 0.5 A | 70 % or more (98 % here) | Li sits where the CIF has no site (distorted or transformed structure) |
| `assignment quality` ambiguous | below 25 % | rattling inflates hops: use `--merge-pairs 1.0` |
| `median jump` | about 2 A merged (0.8 A means pairs are not merged) | add `--merge-pairs 1.0` |

The Li count expected in the box depends on the doping (here 109 Li + Ga on 240 sites). It is not checked against the CIF automatically. Compare `N_Li` in the `[sites]` line with the Li count of your starting structure.

---

## 7. Caveats and limits

- Each system has one dopant of each kind in one box: one local environment, one trajectory. Error bars in 05 come from time blocks only and do not cover run-to-run scatter.
- The 0-3 A shell holds only 1-3 Li. Prefer the 3-5 A shell for conclusions.
- MSD shells use the START distance. At large tau, Li leave their shell and shells mix. Use tau of a few ps.
- A ratio away from 1 can also be local geometry, not the dopant. Ru is only a partial control. A clean control would be shells around ordinary 24d sites (not written).
- Crystal sites near a dopant are poorly defined (Li relax off the ideal positions). Site-based near-dopant numbers are unreliable in that region.
- 01 uses its own density-peak reference (`F_site`, median F over density maxima). Barriers from 04 read 01's `F_rel`, so their zero point is peak-based even with `--sites-cif`. Not changed.
- The Ga depletion (0.5x within 3 A) fits Ga3+ repelling Li on a vacancy-compensated site. Not checked against the charge balance of the structure.
- The drift removal uses the framework centre of mass, as in 00-03.

---

## 8. Open items

- Summarisation script that reads the logs and states what is happening: deferred.
- Control shells around ordinary 24d sites for the MSD analysis: not written.
- Compare Ea(total) with Ea(D*) from 03: needs your 03 number.
- Optional: use crystal sites in 01 (`F_site` and control spheres). Not done, would change old results.
- Refill and concerted at min-res 0.5 and 1.0: not checked.
