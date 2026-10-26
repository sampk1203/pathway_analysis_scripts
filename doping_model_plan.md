# Plan: dopant-blocking conductivity model for doped LLZO (v2)

Status: PLAN ONLY. No code written yet. Open with Helix: `space-f`, type name, Enter.
Checked in the zip earlier: file names, CSV columns, CLI of 05, helpers in llzo_io. NOT checked: your real CSVs, your dumps.
Everything below marked (guess) or (unverified) is not a result.

---

## 0. Plain summary

- Measure how much each dopant slows Li near it (you have this: y(r), V_A).
- Combine with how much Li is left (n_Li) and how fast Li move without dopant help (D_vac, a measured curve).
- Output: relative bulk conductivity vs dopant amount x, per dopant, with a band.
- It is a semi-empirical trend model. It is NOT a first-principles equation for any dopant at any x.

## 1. Words and symbols

| Symbol | Meaning |
|---|---|
| x | dopant atoms per formula unit: Li(7-q*x) M(x) La3 Zr2 O12. Al 1/16 -> x = 0.0625 |
| q | Li lost per dopant (charge balance): 3 for Al, Ga, Fe; 2 for Mg, Zn. Assumes dopant on a Li site |
| n_Li | Li per formula unit = 7 - q*x |
| N | dopant atoms in the cell = x * (formula units in cell). If the cell is 2x2x4 conventional cells, 16 cells * 8 = 128 f.u., so 1/16 -> 8 Al (cell size UNVERIFIED, read it from the dump) |
| V_fu | volume per formula unit = 2188.24 / 8 = 273.5 A^3 (from your CIF; a = 12.9827) |
| y(r) | MSD(r) / MSD(far): Li mobility at distance r from the nearest dopant, relative to far Li |
| V_A | blocked volume, depth-weighted: integral of (1 - y(r)) 4 pi r^2 dr. A zone where Li is 50 % slower counts as half a frozen sphere |
| D* | Li tracer diffusivity = MSD / 6t |
| D_vac(n_Li) | D* of Li that no dopant slows (far Li), as a function of Li content. It is Li diffusivity. The name D_vac is mine; papers say D(Li content) or effective carrier concentration |
| H_R | Haven ratio = D* / D_sigma (from 03). 1 = Li move independently; < 1 = correlated |

## 2. The math (two formulae + one derived condition)

### Formula A: Nernst-Einstein (the definition of sigma; not my invention)

sigma = n * e^2 * D_sigma / (k T),   D_sigma = D* / H_R,   n = n_Li / V_fu

Ratio to the pure run, same T:

sigma(x) / sigma(0) = [ n_Li(x) / 7 ] * [ D*(x) / D*(0) ] * [ H_R(0) / H_R(x) ]

Three factors: carrier count, mobility, correlation.

### Formula B: my model for D* (the part that is a hypothesis)

D*(x) = D_vac(n_Li) * exp( - x * V_A / V_fu )

Put into A:

sigma_rel(x) = [ n_Li / 7 ] * [ D_vac(n_Li) / D_vac(7) ] * exp( - x V_A / V_fu ) * [ H_R(0) / H_R(x) ]

- Why two formulae: A says what sigma is. B says what I assume D* is. A is textbook; B is the model to be tested. Not two competing models.
- D_vac(n_Li) is a function you measure (a curve), not a number to multiply by n_Li.

### Where the exponential comes from (blocking term)

1. One Li, one dopant at a random spot. Fraction of mobility lost on average = V_A / V_cell. Mobility kept = 1 - V_A / V_cell.
2. N dopants placed independently, slowdowns multiply: (1 - V_A / V_cell)^N.
3. N = x * (f.u.), V_cell = (f.u.) * V_fu, so N / V_cell = x / V_fu. Limit (1 - a/N)^N -> e^(-a) gives exp(-x V_A / V_fu). Cell size cancels, so the same form applies to larger supercells.
4. First-order (small x) version: 1 - x V_A / V_fu. Goes negative at large x, so it cannot be right at large x. Test both.

Assumptions: (i) dopants random and independent; (ii) zones multiply; (iii) the plain average over Li is the right quantity. Real 24d sites are an ordered lattice (i fails). Long-time diffusion through obstacles is an effective-medium problem, not a plain average (iii may fail at high x). Literature on random overlapping obstacles finds a power law over most of the range, not a pure exponential (read the abstract only).
Note: whole-cell MSD at fixed tau IS the plain average over Li, so the closure test (S3) is fair for (iii) at that tau. The long-time sigma is not.

### Derived condition: where sigma peaks (calculus on A + B, ignoring H_R change)

ln sigma = ln n_Li + ln D_vac - x V_A / V_fu + const
d ln sigma / dx = 0  gives

d ln D_vac / dx  =  V_A / V_fu  +  q / (7 - q x)

Left: gain from more vacancies. Right: blocking cost + loss of carriers. This is the "optimum equation". Nobody "said" it; it is calculus on B. With H_R(x) varying, add d ln H_R / dx to the right side.

## 3. Literature status (abstracts/snippets only; NOT full-text read)

Same structure (carriers x mobility, D depends on Li content):
- Lattice-gas MC on garnet, arXiv 1707.00491: non-interacting sigma ~ mole fraction * (1 - fraction), peak at x_Li = 4.5; correlation factor depends on concentration, so optimum depends on interactions.
- Verduzco, Marinero, Strachan, arXiv 2305.08839: diffusivity peaks at Li 6.3-6.5; vacancies vs tetrahedral-site filling; Li content dominates over dopant chemistry at high T.
- Thompson et al., "A Tale of Two Sites": effective carrier concentration depends on site occupancy; peak near Li = 6.5.
- arXiv 1506.07142: Li concentration and arrangement control transport.

Against my model:
- Fu et al. 2026 (Al, Fe, Ga, Zn): total vacancy count is not the carrier concentration; Zn makes dopant-adjacent, mobility-suppressed vacancies; conductivity depends on 24d/96h distribution. So one D_vac curve for all dopants may fail.
- SINTEF study: best Al = 0.2 = minimum to stabilize cubic; dopant's main job is cubic stabilization. Parascos et al.: peak at x_c = 0.18 (critical cubic composition). Model cannot see phase stability.
- Random-obstacle diffusion (PMC3094463): power law, not exp.
- Hilfer & Orbach 1988: dynamic percolation, blockers block carriers (closest analogue to dopant blocking).

Not found: exp(-rho V_A) with an MD-measured y(r) used to predict doping optimum. "Not found" is not "does not exist". The earlier novelty_check.md is weak (duplicate rows, "full text read" claimed for partly read papers): redo before any novelty claim. My earlier statement that the exp form is Campbell's theorem / Forster transfer was from memory and is NOT confirmed.

## 4. Known weaknesses (all of them, in one place)

1. **Vacancies stay near dopants (user observation, all dopants).** Then active vacancies < q per dopant, the trapped fraction depends on dopant, so D_vac may differ per dopant. Also double counting: y(r) already contains slowdown from trapped vacancies, and D_vac(n_Li) assumes all vacancies are active. V_A is NOT the only dopant-specific quantity.
2. Cubic phase stability not in the model. Pure run is forced cubic; real pure LLZO is tetragonal. Real peak is set by stability.
3. H_R changes with concentration (lattice-gas paper). 03 gives H_R with ~40 % error on a 100 ps window.
4. Random placement vs ordered 24d sites.
5. Plain average vs effective medium at long time; exp vs power law.
6. Far Li vanish at high x. At 3/16 dopant spacing ~10 A (hand estimate), so few Li are > 7 A from every dopant. Check far-shell Li count.
7. MSD at one tau is not D. Check diffusive regime (log-log slope ~ 1).
8. D_vac comes from 4 points (0, 1/16, 2/16, 3/16). Peak expected near Li 6.3-6.5, i.e. x ~ 0.17-0.23. Extrapolation.
9. q assumes Li-site dopant. Fe, Zn, Mg site preference unsure.
10. MD = bulk. Experiment = bulk + grain boundaries + density + Li loss.
11. My V_A numbers earlier (Al 280-475 A^3, "blocking costs 20-30 % at x = 0.2") used a guessed width w (guess). Replace with real V_A3 from the csv.

## 5. Inputs

| Input | From | Used for |
|---|---|---|
| `radius_summary.csv` | build_doc.py | folder-level rows: A, r0, w, B, r50, r95, V_A3, r_eff, caps |
| `msd05/<system>/msd_shell.csv` | 05 | far-shell MSD, Li count per shell |
| `whole_msd/<system>/whole_msd.csv` | NEW whole_msd.py | MSD of ALL Li, incl. pure run |
| `haven_summary.csv` | 03 (existing) | D*, D_sigma, H_R per folder and T, with bootstrap sd |
| `doping_map.csv` | you, by hand | folder -> dopant, q, x, n_dopant, n_fu |

```
folder,dopant,q,x,n_dopant,n_fu
pure,none,0,0,0,128
1Al-doped,Al,3,0.0625,8,128
2Al-doped,Al,3,0.125,16,128
3Al-doped,Al,3,0.1875,24,128
5Al-doped,Al,3,0.3125,40,128   <- later; script runs without it
```
(n_dopant / n_fu must equal x: the script checks it. Cell size 128 f.u. is an assumption.)

## 6. Files

| File | Where | Type |
|---|---|---|
| whole_msd.py | with_flag/ | new |
| doping_model.py | next to radius_avg.py (reads CSVs only) | new |
| radius_avg.py | existing | edit: also print w, V_A3, r_eff (free rows) |

No change to build_doc.py, 03, 05, run_pipeline.sh. No README/CHANGELOG exist; usage goes in docstrings.

## 7. whole_msd.py

- CLI like 05: target, --T, --tmin, --lags, --blocks, --outdir whole_msd, io.add_config_args.
- Loader as 05: io.prepare -> framework_drift -> li_mask -> subtract drift.
- Per T and lag: |r(t0+lag) - r(t0)|^2 over all Li and origins; mean + delete-one-block jackknife (io.block_counts, io.jackknife). Same tau frames as 05.
- Works for the pure folder (no dopant). Unsure if load_config accepts an empty dopant list: selftest covers it.
- Optional (decide): also the collective MSD, to get D_sigma directly (noisy, see S6).
- Output: T, tau_ps, n_Li, msd_A2, msd_se, blocks.

## 8. doping_model.py steps

CLI: --map, --root, --radius, [--T] [--tau] [--vfu 273.5] [--host Al] [--out doping_model] [--selftest].

**S1. V_A per folder; radius vs x** (valid fits only: no `*`, A >= 0.1, r95 < r_max_used)
- Per folder, T, tau: A, r0, w, r95, V_A (model-free V_A3 primary; sigmoid analytic 4 pi A B (r0^3/3 + pi^2 w^2 r0/3) cross-check; warn if > 30 % apart), r_eff.
- Plot r95, r50, V_A vs x per dopant, with caps (r_p99_A, dopant_dopant_min_A, half_box_A) as lines. Flat radius with a binding cap is not evidence. (User: radius looks almost flat; this plot is the check.)
- Out: va_table.csv, radius_vs_x.png.

**S2. Mobility table**
- D_far (msd05 'any' far shell), D_all (whole_msd), R_meas = D_all / D_far, far-Li fraction per folder (flag < 0.15 or < 10 far Li per origin).
- Out: mobility_table.csv.

**S3. Closure test (overlap rule)**
- ln(MSD_all / MSD_far) vs x per dopant: exp rule says a straight line with slope -V_A / V_fu. Compare exp, linear (1 - x V_A/V_fu) and a power-law form; show which fits.
- R_pred(x) = exp(-x V_A(1/16 folder) / V_fu) vs R_meas at 2/16, 3/16 (V_A from the lowest x, so it is not circular).
- Out: closure.csv, closure_parity.png, closure_lnR_vs_x.png.

**S4. Host curve D_vac(n_Li) and collapse test**
- Plot ln D_far vs n_Li for ALL dopants on one plot. If Al, Ga, Fe (q=3) and Mg, Zn (q=2) collapse, D_vac is a host property. If not (expected risk: vacancy trapping, Zn), say so in the printout and switch to per-dopant D_vac.
- Fit D_vac for the host (Al): linear and quadratic in n_Li (quadratic has 1 dof on 4 points; show both).
- Transfer: for each other dopant predict D_all(x) = D_vac(n_Li) * R_pred and compare at 1/16, 2/16, 3/16. Leave-one-out for Al.
- Out: transfer.csv, far_vs_nLi.png, transfer_parity.png.

**S5. Active-vacancy check (new, from the vacancy-trapping point)**
- Using existing dumps: count Li-vacancies (empty 24d/96h sites) vs distance from dopant (needs sites from 04 / llzo_sites) and compare with the random expectation. Are vacancies clustered near dopants, per dopant? Estimate the trapped fraction.
- Use: if the trapped fraction differs per dopant, replace n_Li by an effective n_active = 7 - q x * (1 - f_trap) in D_vac. Flag possible double counting with y(r).
- Out: vac_vs_r.png, trapped_fraction.csv. (Optional; start with a quick look, decide after S4.)

**S6. Haven ratio (new): how to handle H_R**
Question: can H_R be fitted vs V_A per dopant? Answer: not per dopant. Reasons: 03 gives one H_R per (folder, T) with ~40 % bootstrap error (100 ps window), and per dopant there are only 4 x points. A trend would be buried in noise.
Do instead:
1. Pool all dopants, all T (careful: H_R may depend on T; include T or fit per T).
2. Regress ln H_R on delta = 7 - n_Li (host effect; lattice-gas paper says H_R depends on Li concentration) and phi = x V_A / V_fu (blocked fraction). Weighted by 03's bootstrap sd.
3. Read the phi coefficient and its CI. If it includes 0: H_R depends on n_Li only (or not at all). Then use H_R(n_Li) or a constant with a +-40 % band, and say so.
4. Report the smallest H_R change the data can detect (about 40 % / sqrt(n)). If the effect is smaller, it cannot be seen.
5. Alternative without H_R: get D_sigma directly from the collective MSD (S7 optional), model D_sigma instead of D*. Noisier but no H_R needed.
Note: 09's H_R is intra-shell, not the whole-cell Haven ratio; do not use it here.
- Out: haven_fit.csv, haven_vs_phi.png.

**S7. sigma_rel(x) and x_opt**
- Grid x = 0 to 0.40, step 0.005. sigma_rel from formula B inside A, normalised to the pure run. H_R from S6 (constant if not significant).
- x_opt per dopant, T, tau, D_vac form (linear/quadratic), block form (exp/linear).
- Band: V_A from 1/16, 2/16, 3/16 folders and bootstrap CI; D_vac forms; H_R band.
- Print cost slope V_A / V_fu and block factor at x_opt; dopant ranking by V_A.
- Mark the cubic-stability window (x_min ~ 0.15-0.20 experiment, external, not from MD) on the plot. A peak below x_min means the model is dominated by stability, not blocking.
- 5/16 present: refit without it, predict it, print the error (held-out test). Absent: print "no high-x data: x_opt is an extrapolation".
- Out: sigma_vs_x.png, x_opt.csv.

## 9. Decision gates

- After S3 and S4: if the closure test fails (exp not better than linear / no straight line), or collapse fails for most dopants, STOP. Do not run 5/16. Report V_A and radius as the result; model not supported.
- If both pass: run Al 5/16.
- If collapse fails only for some dopants: those need their own D_vac (high-x run each) or the S5 correction.

## 10. Runs

- Existing runs (pure, 1/16, 2/16, 3/16 for each dopant): cover S1-S7 except high x.
- New: one Al run at 5/16 (x = 0.31, Li = 6.06). It passes the expected peak and tests the full prediction. Far Li vanish there, so it tests the joint prediction only.
- New dopant later: one dilute run for y(r) and V_A; no high-x run only if collapse passes.
- Co-doping: exp(-sum x_i V_Ai / V_fu), n_Li = 7 - sum q_i x_i. Dopant-dopant interaction untested.

## 11. Dry run and tests (I run these; you run real data)

`doping_model.py --selftest` builds a synthetic tree in a temp dir (truth known):
1. S1: sigmoid y(r) with known r0, w, A; recover V_A within 5 %.
2. S3 discriminates: multiplicative data must pass; min-rule data must fail. Also exp-generated vs linear-generated data: the test must pick the right one.
3. S4: known ln D_vac; Al calibrates; a q = 2 dopant predicted within 5 %. Also a case with a dopant-specific D_vac shift: collapse test must flag it.
4. S6: synthetic H_R with known n_Li dependence and known zero phi dependence; the fit must return phi CI containing 0 at 40 % noise.
5. S7: x_opt equals the analytic optimum of the truth model within one grid step.
6. 5/16 present vs absent: both run; held-out error vs extrapolation warning.
7. Bad inputs: folder missing from map, n_dopant/n_fu != x, too few far Li, missing msd05 file: clear message, no crash.

`whole_msd.py --selftest`: fake LAMMPS dump, random-walking Li with known D (MSD = 6 D tau), one dopant type; second fake with no dopant; MSD within 5 %, SE finite.

Synthetic data tests the code, not the physics. Say that in the summary.

## 12. Decisions made (change any)

- Model-free V_A3 primary; sigmoid-analytic V cross-check.
- V_A for prediction: lowest-x folder per dopant; also test whether V_A changes with x.
- Folder-level radius rows, not pooled element-level rows.
- V_fu fixed 273.5 A^3; lattice change with x ignored.
- n_Li = 7 - q x, dopant on Li sites.
- Conductivity from Nernst-Einstein with H_R from S6, not a measured sigma.
- Result is relative sigma, not S/cm.

## 13. What you get / do not get

Get: relative sigma(x)/sigma(0) per dopant with a band; measured V_A per dopant and ranking; closure, collapse, hold-out tests; a clear pass/fail on each assumption; radius vs x with caps.
Do not get: absolute conductivity; phase stability or solubility limits; grain boundaries; site preference of Fe, Zn, Mg; any dopant at any x (only the tested range).
Fallback if the model fails: plot measured whole-cell MSD vs x per dopant, plus V_A and radius as the measured result. The distance-resolved radius itself looked unpublished in the (weak) novelty check.

## 14. Open questions for you

1. Cell size: how many conventional cells and how many Al per folder? (Check n_dopant / n_fu = x.)
2. Include the collective-MSD / D_sigma option in whole_msd.py?
3. Do the vacancy-clustering check (S5) now or after S4?
4. Same T and tau for all folders? Which tau is diffusive?
