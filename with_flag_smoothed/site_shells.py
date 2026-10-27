#!/usr/bin/env python3
"""site_shells.py - 05's shell MSD and shell Li density, SPLIT by the Li site the nearest dopant occupies (24d / 96h / other).

python site_shells.py {1Al-doped 2Al-doped 3Al-doped} {1Ga-doped 2Ga-doped 3Ga-doped} {1Fe-doped 2Fe-doped 3Fe-doped} --T 800 1200 --lags 1 5 10
python site_shells.py {1Mg-doped Mg=Li24d elements=Li,La,Mg,Zr,O} --site-tol 1.2
python site_shells.py --selftest --cif cubic_LLZO_wyckoff.cif

Groups use the build_doc.py syntax (quote braces if the shell eats them; put options after the groups). Tokens like Al=Li24d are accepted
and ignored (the site is MEASURED here, not stated); elements=Li,La,Zr,Al,O sets the element of LAMMPS type 1..N for that group.

Question: is the Li footprint set by the dopant ELEMENT or by the SITE the dopant sits on? Fe, Mg, Zn move between sites in the runs, so
a footprint averaged over the run mixes site states. Here every frame is split by the state of the dopant that owns the shell.

Method (shells, far reference, ownership, drift removal, error bars = 05_msd_by_shell.py with its defaults: reference 'any', --own start)
  1. Ideal Li sites (24d + 96h) of the pristine Ia-3d CIF are tiled to the box and aligned on the time-mean La/Zr positions
     (rigid shift, llzo_sites.fit_shift, as 04 and build_doc).
  2. Per dopant and frame (dopant position = the smoothed track of 05, --dop-smooth-ps): nearest ideal Li site. State = its kind (24d, 96h),
     or 'other' if the nearest site is farther than --site-tol A. dopant_sites.csv: fraction of frames per state, median distance to the
     nearest site (use it to judge --site-tol), number of site changes (a stay shorter than --min-res-ps is not counted).
  3. Each Li at t0 is put in a shell by its distance to the NEAREST dopant (--shells, default 3 5 7 A) and in the state of THAT dopant
     at t0. Far = farther than the last edge from EVERY dopant; it has no state and is the common reference of all ratios.
        MSD(state, shell, tau) = < |r(t0 + tau) - r(t0)|^2 >    over the Li of that (state, shell)
        ratio = MSD(state, shell) / MSD(far)                    < 1: slower than far Li
  4. Density ratio = mean Li count in (state, shell) / count expected for uniform Li in the same region. The region is the part of the box
     whose nearest dopant is in that state, measured on a 32^3 grid in every sampled frame (--dens-frames frames, spread over the window).
Errors: delete-one-block jackknife over --blocks time blocks (additive sums, as 05). A state that appears in few blocks gets SE = NaN.

Collapse test (pooled over all folders given): per T, tau and near shell, the points (folder, state) with >= --min-n Li per origin; adjusted
R^2 of log(ratio) explained by the dopant ELEMENT, by the STATE, and by both (additive, least squares). State >> element: the footprint
follows the site. Element >> state: it follows the chemistry. Both low: neither explains it (noise or configuration). Points of one folder
share a trajectory and each (T, tau, shell) uses few points, so read the three numbers as a guide, not a test with a p-value.

Output (default outdir = site_shells in the current directory):
  <outdir>/<system>/site_shells.csv   system, element, n_dopant, T, tau_ps, state, shell, n_Li_used, msd_A2, msd_se, ratio_to_far, ratio_se,
                                      density_ratio, density_ratio_se, frac_dopant_frames (share of dopant-frames in this state), few, blocks
  <outdir>/<system>/dopant_sites.csv  per dopant and T: state fractions, median distance to the nearest site, number of site changes
  <outdir>/site_shells_all.csv         every folder
  <outdir>/collapse_stats.csv          n, adjusted R^2 by element / state / both, per (T, tau, shell)
  <outdir>/collapse_T<T>K_tau<tau>ps.png   ratio vs element, colour = state, one panel per near shell; open symbol = fewer than --min-n Li
Cache: <outdir>/<system>/cache/site_shells_T<T>K.json (key = settings, dump + mtime, CIF + mtime); --recompute ignores it.
Needs llzo_io.py, llzo_sites.py in ./with_flag (or --scripts DIR), whole_msd.py next to this file, numpy, scipy, matplotlib; PyYAML only for the selftest.
"""
import argparse, csv, glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
VERSION = 1


def _scripts_dir():
    a = sys.argv
    for i, x in enumerate(a):
        if x == '--scripts' and i + 1 < len(a): return a[i + 1]
        if x.startswith('--scripts='): return x.split('=', 1)[1]
    return os.path.join(HERE, 'with_flag')


SCRIPTS = os.path.abspath(_scripts_dir())
sys.path.insert(0, SCRIPTS)
import numpy as np
import llzo_io as lio
import llzo_sites as ls
from whole_msd import parse_groups, expand, md_dump, md_temps, fname, warn      # same group syntax and dump layout as whole_msd / build_doc

CIF_DEFAULT = '/media/qpivolta/8TB/archive_vm/backup-26-3-26-voltaet/2_debdipto/2x2x4-new/pathway_analysis_scripts/cubic_LLZO_wyckoff.cif'
STATES = ('24d', '96h', 'other')
STATE_COL = {'24d': '#1f77b4', '96h': '#ff7f0e', 'other': '#7f7f7f'}
WARN_RMS, WARN_MAX = 0.35, 1.0                                    # A, as build_doc


# ---------------------------------------------------------------- dopant site state
def site_changes(site, min_res):
    """Number of changes of the nearest-site index, ignoring stays shorter than min_res frames."""
    idx = np.flatnonzero(np.diff(site)) + 1
    starts, ends = np.r_[0, idx], np.r_[idx, len(site)]
    keep = (ends - starts) >= min_res
    s = site[starts][keep]
    return int((np.diff(s) != 0).sum()) if len(s) else 0


# ---------------------------------------------------------------- density of Li per (state, shell)
def density_blocks(cat, st_dop, dxyz, Lf, edges, nb, ncat, fr, n=32):
    """Per time block: Li counts and expected counts (uniform Li) of every category, over the sampled frames fr.
    cat (F,N) category of every Li; st_dop (Nd,F) state of every dopant; dxyz (Nd,F,3); Lf (F,3). -> (counts (nb,ncat), expected (nb,ncat))"""
    F, N = cat.shape
    K1 = len(edges)
    cnt, exp = np.zeros((nb, ncat)), np.zeros((nb, ncat))
    bl = lio.block_id(fr, F, nb)
    u = (np.arange(n) + 0.5) / n
    G = np.stack(np.meshgrid(u, u, u, indexing='ij'), -1).reshape(-1, 3)
    for f, b in zip(fr, bl):
        Lff = Lf[f]
        d = G[:, None, :] * Lff - dxyz[:, f, :][None]
        d -= Lff * np.rint(d / Lff)
        r = np.sqrt((d ** 2).sum(-1))                               # (M, Nd)
        im = r.argmin(axis=1)
        dm = r[np.arange(len(r)), im]
        sh = np.searchsorted(edges, dm)
        gc = np.where(sh < K1, st_dop[im, f] * K1 + sh, ncat - 1)
        exp[b] += N * np.bincount(gc, minlength=ncat) / len(G)
        cnt[b] += np.bincount(cat[f], minlength=ncat)
    return cnt, exp


# ---------------------------------------------------------------- one (folder, T)
def compute(folder, cfg, T, a):
    dump = md_dump(folder, T) or lio.dump_path(cfg, T)
    traj = lio.prepare(cfg, dump, a.tmin, verbose=False)
    if a.tmax is not None:                                          # window = tmin .. tmax (ps after MD start), as whole_msd
        k = traj.time_ps <= a.tmax + 1e-9
        if k.sum() < 10: raise ValueError(f'only {int(k.sum())} frames with t <= {a.tmax} ps')
        traj = lio.Trajectory(steps=traj.steps[k], time_ps=traj.time_ps[k], lo=traj.lo, L=traj.L[k], types=traj.types, ids=traj.ids, pos=traj.pos[k])
    drift = lio.framework_drift(cfg, traj)
    P = traj.pos - drift[:, None, :]
    el = np.array([cfg['types'][int(t)] for t in traj.types])
    L, lo = traj.L.mean(axis=0), traj.lo
    F = len(traj.steps)
    dt = float(np.median(np.diff(traj.time_ps)))

    # ---- ideal Li sites aligned on La/Zr (rigid shift)
    ws = ls.wyckoff_sites(a.cif, lio.get_fold(cfg, L))
    fm = ls._w((P.mean(axis=0) - lo) / L)
    md_fw = {e: fm[el == e] for e in ('La', 'Zr') if (el == e).any() and e in ws['fw']}
    t_fit, rms, mx, _ = ls.fit_shift(md_fw, {e: ws['fw'][e] for e in md_fw}, L)
    if rms > WARN_RMS or mx > WARN_MAX: warn(f'{fname(folder)} T={T}: framework fit rms {rms:.2f} / max {mx:.2f} A: site alignment unreliable')
    sites = ls._w(ws['frac'] + t_fit) * L                           # relative to the box origin, as lio.site_dist_moving wants
    kidx = np.array([STATES.index(k) if k in STATES[:2] else 2 for k in ws['kind']])

    # ---- dopants: nearest site per frame -> state
    dops, cnt = [], {}
    for t in cfg['dopant_types']:
        for i in np.where(traj.types == t)[0]:
            e = cfg['types'][t]
            cnt[e] = cnt.get(e, 0) + 1
            dops.append((f'{e}{cnt[e]}', e, lio.dopant_xyz(cfg, traj, drift, i)))
    if not dops: raise ValueError('no dopant atoms in this folder')
    Nd = len(dops)
    min_res = max(1, int(round(a.min_res_ps / dt)))
    st_dop = np.empty((Nd, F), int)
    site_rows = []
    for k, (lab, e, x) in enumerate(dops):
        d = lio.site_dist_moving(sites, x, lo, L)                   # (F, S)
        j = d.argmin(axis=1)
        dm = d[np.arange(F), j]
        st_dop[k] = np.where(dm <= a.site_tol, kidx[j], 2)
        site_rows.append(dict(T=int(T), dopant=lab, element=e, frac_24d=float((st_dop[k] == 0).mean()), frac_96h=float((st_dop[k] == 1).mean()),
                              frac_other=float((st_dop[k] == 2).mean()), median_dist_A=float(np.median(dm)), n_site_changes=site_changes(j, min_res),
                              min_res_ps=min_res * dt, site_tol_A=a.site_tol))
        del d
    frac_state = [float((st_dop == s).mean()) for s in range(3)]

    # ---- Li: shell by nearest dopant, state of that dopant at t0
    lm = lio.li_mask(cfg, traj)
    li = P[:, lm, :]
    N = li.shape[1]
    edges = np.array(a.shells, float)
    names = lio.shell_names(edges)
    K1 = len(edges)
    ncat = 3 * K1 + 1
    dnear = np.full((F, N), np.inf)
    inear = np.zeros((F, N), int)
    dxyz = np.stack([x for _, _, x in dops])                        # (Nd, F, 3)
    for k in range(Nd):
        d = lio.dist_to(li, dxyz[k], traj.L)
        m = d < dnear
        dnear[m], inear[m] = d[m], k
    sh = np.searchsorted(edges, dnear)
    st = st_dop[inear, np.arange(F)[:, None]]
    cat = np.where(sh < K1, st * K1 + sh, ncat - 1)

    lag_frames = [max(1, int(round(x / dt))) for x in a.lags]
    nb = a.blocks
    msd = {}
    for lag in lag_frames:
        if lag >= F - nb:
            warn(f'{fname(folder)} T={T}: tau = {lag * dt:g} ps needs more than {nb} origin blocks of frames (F = {F}): skipped')
            continue
        d2 = ((li[lag:] - li[:-lag]) ** 2).sum(axis=-1)
        S = lio.block_counts(cat[:-lag], ncat, nb, values=d2)
        C = lio.block_counts(cat[:-lag], ncat, nb)
        m, se = lio.jackknife(lambda s_, c_: s_ / c_, S, C)
        r, rse = lio.jackknife(lambda s_, c_: (s_ / c_) / (s_ / c_)[-1], S, C)
        msd[lag] = (m, se, r, rse, C.sum(0) / (F - lag))
    fr = np.unique(np.linspace(0, F - 1, min(a.dens_frames, F)).astype(int))
    dc, de = density_blocks(cat, st_dop, dxyz, traj.L, edges, nb, ncat, fr)
    dens, dens_se = lio.jackknife(lambda c_, e_: c_ / e_, dc, de)

    rows = []
    for lag in msd:
        m, se, r, rse, nres = msd[lag]
        for c in range(ncat):
            far = c == ncat - 1
            rows.append(dict(T=int(T), tau_ps=lag * dt, state='far' if far else STATES[c // K1], shell=names[-1] if far else names[c % K1],
                             n_Li_used=float(nres[c]), msd_A2=float(m[c]), msd_se=float(se[c]), ratio_to_far=float(r[c]), ratio_se=float(rse[c]),
                             density_ratio=float(dens[c]), density_ratio_se=float(dens_se[c]),
                             frac_dopant_frames=1.0 if far else frac_state[c // K1], few=bool(nres[c] < a.min_n), blocks=nb))
    return dict(rows=rows, site_rows=site_rows, n_dopant=Nd, elements='+'.join(sorted({e for _, e, _ in dops})))


def cache_key(cfg, dump, a):
    return repr((tuple(a.lags), a.blocks, a.tmin, a.tmax, tuple(a.shells), a.site_tol, a.min_res_ps, a.min_n, a.dens_frames, a.cif,
                 os.path.getmtime(a.cif) if os.path.exists(a.cif) else None, dump, os.path.getmtime(dump), cfg.get('run_steps'), cfg.get('dt_ps'),
                 cfg.get('tmin_ps'), cfg.get('dop_smooth_ps'), tuple(sorted(cfg['types'].items())), tuple(cfg['dopant_types']), VERSION))


def _task(x):
    folder, cfg, T, a, outp, key = x
    try:
        res = compute(folder, cfg, T, a)
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        with open(outp, 'w') as fh: json.dump(dict(key=key, **res), fh)
    except (Exception, SystemExit) as e:
        return f'{folder} T={T}: {e}'


# ---------------------------------------------------------------- collapse statistics and figure
def adj_r2(y, *factors):
    """Adjusted R^2 of y explained by the (additive) categorical factors; NaN if too few points."""
    y = np.asarray(y, float)
    cols = [np.ones(len(y))]
    for f in factors:
        f = np.asarray(f)
        lv = sorted(set(f))
        cols += [(f == v).astype(float) for v in lv[1:]]
    X = np.stack(cols, axis=1)
    n, p = X.shape
    sst = ((y - y.mean()) ** 2).sum()
    if n - p < 2 or sst <= 0: return float('nan')
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    ssr = ((y - X @ beta) ** 2).sum()
    return float(1 - (ssr / (n - p)) / (sst / (n - 1)))


def collapse_stats(rows):
    """rows: pooled rows (with 'system', 'element'). One result per (T, tau, near shell)."""
    out = []
    near = [r for r in rows if r['state'] in STATES and not r['few'] and np.isfinite(r['ratio_to_far']) and r['ratio_to_far'] > 0]
    for key in sorted({(r['T'], round(r['tau_ps'], 6), r['shell']) for r in near}):
        pts = [r for r in near if (r['T'], round(r['tau_ps'], 6), r['shell']) == key]
        y = np.log([r['ratio_to_far'] for r in pts])
        el, stt = [r['element'] for r in pts], [r['state'] for r in pts]
        out.append(dict(T=key[0], tau_ps=key[1], shell=key[2], n_points=len(pts), n_elements=len(set(el)), n_states=len(set(stt)),
                        adjR2_element=adj_r2(y, el), adjR2_state=adj_r2(y, stt), adjR2_both=adj_r2(y, el, stt)))
    return out


def plot_collapse(rows, T, tau, names, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    pts = [r for r in rows if r['T'] == T and abs(r['tau_ps'] - tau) < 1e-6 and r['state'] in STATES and np.isfinite(r['ratio_to_far'])]
    if not pts: return False
    els = sorted({r['element'] for r in pts})
    K1 = len(names) - 1
    fig, axs = plt.subplots(1, K1, figsize=(3.6 * K1 + 1.2, 3.8), dpi=150, squeeze=False, sharey=True)
    for k in range(K1):
        ax = axs[0][k]
        for r in (r for r in pts if r['shell'] == names[k]):
            x = els.index(r['element']) + {'24d': -0.2, '96h': 0.0, 'other': 0.2}[r['state']]
            c = STATE_COL[r['state']]
            ax.errorbar(x, r['ratio_to_far'], r['ratio_se'] if np.isfinite(r['ratio_se']) else None, marker='o', ms=5, color=c, mfc='white' if r['few'] else c, capsize=2, lw=0.8)
        ax.axhline(1, color='0.5', lw=0.8)
        ax.set_xticks(range(len(els)), els)
        ax.set_title(f'{names[k]}', fontsize=10)
        if k == 0: ax.set_ylabel(f'MSD(state, shell) / MSD(far),  tau = {tau:g} ps')
    h = [Line2D([], [], color=STATE_COL[s], marker='o', lw=0, label=f'dopant on {s}') for s in STATES]
    h.append(Line2D([], [], color='0.4', marker='o', mfc='white', lw=0, label='open: few Li per origin'))
    fig.legend(handles=h, loc='lower center', ncol=4, fontsize=8, frameon=False)
    fig.suptitle(f'T = {T} K: footprint by element and by the site of the nearest dopant (error bars = block jackknife)', fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)
    return True


# ---------------------------------------------------------------- run
def write_csv(path, rows):
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def run(groups, a):
    todo, plan = [], {}
    for g in groups:
        for f in g['folders']:
            if not os.path.isdir(f):
                warn(f'not a folder: {f}'); continue
            try:
                ga = argparse.Namespace(**vars(a))
                if g['elements']: ga.elements = g['elements']
                cfg = lio.load_config(f, None, ga)
                temps = a.T or md_temps(f) or lio.temperatures(cfg)
            except SystemExit as e:
                warn(f'skipped {f}: {e}'); continue
            odir = os.path.join(a.outdir, cfg['name'])
            for T in temps:
                try: dump = md_dump(f, T) or lio.dump_path(cfg, T)
                except SystemExit as e:
                    warn(f'{f} T={T}: {e}'); continue
                key = cache_key(cfg, dump, a)
                outp = os.path.join(odir, 'cache', f'site_shells_T{T}K.json')
                plan.setdefault((f, cfg['name']), []).append(outp)
                ok = False
                if os.path.exists(outp) and not a.recompute:
                    try: ok = json.load(open(outp))['key'] == key
                    except Exception: ok = False
                if not ok: todo.append((f, cfg, T, a, outp, key))
    print(f'[site_shells] {len(todo)} (folder, T) to compute, {sum(len(v) for v in plan.values()) - len(todo)} cached')
    if a.jobs > 1 and len(todo) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=a.jobs) as ex: errs = list(ex.map(_task, todo))
    else:
        errs = [_task(t) for t in todo]
    for e in errs:
        if e: warn(e)
    pooled = []
    for (f, name), outs in plan.items():
        rows, srows, meta = [], [], None
        for o in outs:
            if not os.path.exists(o): continue
            d = json.load(open(o))
            rows += d['rows']; srows += d['site_rows']; meta = d
        if not rows:
            warn(f'no result for {f}'); continue
        for r in rows: r.update(system=name, element=meta['elements'], n_dopant=meta['n_dopant'])
        rows.sort(key=lambda r: (r['T'], r['tau_ps'], STATES.index(r['state']) if r['state'] in STATES else 9, r['shell']))
        odir = os.path.join(a.outdir, name)
        cols = ['system', 'element', 'n_dopant'] + [k for k in rows[0] if k not in ('system', 'element', 'n_dopant')]
        write_csv(os.path.join(odir, 'site_shells.csv'), [{k: r[k] for k in cols} for r in rows])
        write_csv(os.path.join(odir, 'dopant_sites.csv'), srows)
        report(name, rows, srows)
        print(f'[+] {os.path.join(odir, "site_shells.csv")}  dopant_sites.csv')
        pooled += [{k: r[k] for k in cols} for r in rows]
    if not pooled: return []
    write_csv(os.path.join(a.outdir, 'site_shells_all.csv'), pooled)
    stats = collapse_stats(pooled)
    if stats:
        write_csv(os.path.join(a.outdir, 'collapse_stats.csv'), stats)
        print_collapse(stats)
    names = lio.shell_names(np.array(a.shells, float))
    nfig = 0
    for T in sorted({r['T'] for r in pooled}):
        for tau in sorted({round(r['tau_ps'], 6) for r in pooled}):
            nfig += bool(plot_collapse(pooled, T, tau, names, os.path.join(a.outdir, f'collapse_T{T}K_tau{tau:g}ps.png')))
    print(f'[+] {os.path.join(a.outdir, "site_shells_all.csv")}  collapse_stats.csv  {nfig} collapse_T*_tau*.png')
    return pooled


def report(name, rows, srows):
    print(f'\n=== {name} ===   n_dopant = {rows[0]["n_dopant"]}  element = {rows[0]["element"]}')
    for T in sorted({r['T'] for r in rows}):
        sr = [r for r in srows if r['T'] == T]
        print(f'  T = {T} K   dopant site: ' + '   '.join(f'{r["dopant"]}: 24d {r["frac_24d"]:.2f} / 96h {r["frac_96h"]:.2f} / other {r["frac_other"]:.2f}, '
                                                        f'{r["n_site_changes"]} changes, median dist {r["median_dist_A"]:.2f} A' for r in sr))
        for tau in sorted({r['tau_ps'] for r in rows if r['T'] == T}):
            print(f'    tau = {tau:g} ps   ratio to far (n_Li per origin) by shell')
            for s in STATES:
                rr = [r for r in rows if r['T'] == T and r['tau_ps'] == tau and r['state'] == s]
                if not rr or rr[0]['frac_dopant_frames'] == 0: continue
                print(f'      {s:>5} ({100 * rr[0]["frac_dopant_frames"]:3.0f} % of dopant-frames)  ' + '  |  '.join(
                    f'{r["shell"]}: {r["ratio_to_far"]:.2f}+/-{r["ratio_se"]:.2f} ({r["n_Li_used"]:.1f})' + ('*' if r['few'] else '') for r in rr))
    print('  * = fewer Li per origin than --min-n: ratio is noise, not drawn as a filled symbol and left out of the collapse statistics')


def print_collapse(stats):
    print('\n[collapse] adjusted R^2 of log(ratio): element | state | both   (n points)   one line per T, tau, near shell')
    for s in stats:
        f = lambda v: '  nan' if not np.isfinite(v) else f'{v:5.2f}'
        print(f'  T {s["T"]:>5} tau {s["tau_ps"]:>4g} {s["shell"]:>8}   {f(s["adjR2_element"])} | {f(s["adjR2_state"])} | {f(s["adjR2_both"])}   (n = {s["n_points"]}, '
              f'{s["n_elements"]} elements, {s["n_states"]} states)')
    print('  state > element: footprint follows the site; element > state: it follows the chemistry; both low: neither (noise, configuration). Guide only.')


# ---------------------------------------------------------------- selftest
def _fake_dump(path, rng, cif, modulate, F=600, seg=50, L0=12.9827, n_li=56, sig=0.5, fac=0.2):
    """Cubic cell (fold 1 1 1). La/Zr on the CIF sites (+ a rigid shift), 1 Al that alternates every `seg` frames between a 24d and a 96h site,
    Li random walkers; if modulate: step variance x fac for Li within 4 A of the Al while it is on 24d (so ratio(24d, 0-3 A, 1 frame) = fac)."""
    ws = ls.wyckoff_sites(cif, (1, 1, 1))
    t = np.array([0.11, 0.07, 0.19])
    kinds = np.array(ws['kind'])
    i24 = int(np.flatnonzero(kinds == '24d')[0])
    d = ls._mi(ws['frac'] - ws['frac'][i24]) * L0
    i96 = int(np.flatnonzero(kinds == '96h')[np.argmin(np.linalg.norm(d[kinds == '96h'], axis=1))])
    fixed = {'La': ls._w(ws['fw']['La'] + t) * L0, 'Zr': ls._w(ws['fw']['Zr'] + t) * L0, 'O': rng.uniform(0, L0, (96, 3))}
    on24 = (np.arange(F) // seg) % 2 == 0
    dop = np.where(on24[:, None], ls._w(ws['frac'][i24] + t) * L0, ls._w(ws['frac'][i96] + t) * L0) + rng.normal(0, 0.08, (F, 3))
    li = np.empty((F, n_li, 3))
    li[0] = rng.uniform(0, L0, (n_li, 3))
    for f in range(1, F):
        r = li[f - 1] - dop[f - 1]
        r -= L0 * np.rint(r / L0)
        s = np.full(n_li, sig)
        if modulate and on24[f - 1]: s = np.where(np.linalg.norm(r, axis=1) < 4.0, sig * np.sqrt(fac), sig)
        li[f] = li[f - 1] + rng.normal(0, 1, (n_li, 3)) * s[:, None]
    els = ['Li', 'La', 'Zr', 'Al', 'O']
    n = {'Li': n_li, 'La': len(fixed['La']), 'Zr': len(fixed['Zr']), 'Al': 1, 'O': 96}
    types = np.concatenate([np.full(n[e], i + 1) for i, e in enumerate(els)])
    with open(path, 'w') as fh:
        for k in range(F):
            pos = np.vstack([li[k], fixed['La'], fixed['Zr'], dop[k][None], fixed['O']])
            fh.write(f'ITEM: TIMESTEP\n{k * 1000}\nITEM: NUMBER OF ATOMS\n{len(types)}\nITEM: BOX BOUNDS pp pp pp\n')
            fh.write(''.join(f'0 {L0}\n' for _ in range(3)) + 'ITEM: ATOMS id type xu yu zu\n')
            fh.write(''.join(f'{i + 1} {types[i]} {pos[i, 0]:.5f} {pos[i, 1]:.5f} {pos[i, 2]:.5f}\n' for i in range(len(types))))
    return els, on24


def selftest(cif):
    import tempfile
    rng = np.random.default_rng(3)
    fails = []

    def chk(ok, msg):
        print(f'  [{"ok" if ok else "FAIL"}] {msg}')
        if not ok: fails.append(msg)

    with tempfile.TemporaryDirectory() as tmp:
        for name, mod in (('fake_mod', True), ('fake_null', False)):
            d = os.path.join(tmp, name); os.makedirs(d)
            els, _ = _fake_dump(os.path.join(d, 'dump_lmp_800K.lammpstrj'), rng, cif, mod)
            open(os.path.join(d, 'element_list'), 'w').write(' '.join(els) + '\n')
            open(os.path.join(d, 'analysis_defaults.yaml'), 'w').write('dt_ps: 0.001\nrun_steps: 599000\ntmin_ps: 0\n')
        sys.argv = [sys.argv[0]]
        a = build_parser().parse_args(['--outdir', os.path.join(tmp, 'out'), '--lags', '1', '--cif', cif, '--dens-frames', '600', '--min-n', '1'])
        groups = [dict(folders=[os.path.join(tmp, 'fake_mod'), os.path.join(tmp, 'fake_null')], elements=None)]
        pooled = run(groups, a)
        by = lambda sysn, s, sh, k: next(r[k] for r in pooled if r['system'] == sysn and r['state'] == s and r['shell'] == sh)
        for sysn in ('fake_mod', 'fake_null'):
            sr = list(csv.DictReader(open(os.path.join(tmp, 'out', sysn, 'dopant_sites.csv'))))[0]
            chk(abs(float(sr['frac_24d']) - 0.5) < 0.03 and abs(float(sr['frac_96h']) - 0.5) < 0.03 and float(sr['frac_other']) < 0.02,
                f'{sysn}: dopant state fractions 24d {float(sr["frac_24d"]):.2f} 96h {float(sr["frac_96h"]):.2f} other {float(sr["frac_other"]):.2f} (expect 0.5 / 0.5 / 0)')
            chk(int(sr['n_site_changes']) == 11, f'{sysn}: site changes {sr["n_site_changes"]} (expect 11)')
        r24, r96 = by('fake_mod', '24d', '0-3A', 'ratio_to_far'), by('fake_mod', '96h', '0-3A', 'ratio_to_far')
        chk(abs(r24 - 0.2) < 0.2 * 0.2, f'fake_mod: ratio(24d, 0-3 A) = {r24:.3f} (expect 0.2 +/- 20 %)')
        chk(abs(r96 - 1.0) < 0.2, f'fake_mod: ratio(96h, 0-3 A) = {r96:.3f} (expect 1.0 +/- 0.2)')
        for s in ('24d', '96h'):
            v = by('fake_null', s, '0-3A', 'ratio_to_far'); dv = by('fake_null', s, '0-3A', 'density_ratio')
            chk(abs(v - 1) < 0.2, f'fake_null: ratio({s}, 0-3 A) = {v:.3f} (expect 1 +/- 0.2)')
            chk(abs(dv - 1) < 0.2, f'fake_null: density({s}, 0-3 A) = {dv:.3f} (expect 1 +/- 0.2)')
        print('second run (expect 0 to compute):')
        run(groups, a)
    # collapse statistic on constructed data: the ratio is set by the state, not by the element
    rows = [dict(T=800, tau_ps=1.0, shell='0-3A', state=s, element=e, few=False, ratio_to_far=(0.3 if s == '24d' else 1.0) * np.exp(rng.normal(0, 0.03)))
            for e in ('A', 'B', 'C') for _ in range(3) for s in ('24d', '96h')]
    st = collapse_stats(rows)[0]
    chk(st['adjR2_state'] > 0.9 and st['adjR2_element'] < 0.3, f'collapse: adjR2 state {st["adjR2_state"]:.2f} (expect > 0.9), element {st["adjR2_element"]:.2f} (expect < 0.3)')
    print('\nselftest: ' + ('PASS' if not fails else f'FAIL {fails}') + '  (synthetic data: tests the code, not the physics)')
    sys.exit(1 if fails else 0)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('items', nargs='*', help='groups: {A B C} {D E}')
    ap.add_argument('--scripts', help='folder with llzo_io.py, llzo_sites.py (default: ./with_flag next to this file)')
    ap.add_argument('--cif', default=CIF_DEFAULT, help='pristine Ia-3d garnet CIF (Li, La, Zr, O)')
    ap.add_argument('--T', type=int, nargs='*', help='temperatures (default: every MD-<T>/ dump, else all dumps found)')
    ap.add_argument('--tmin', type=float, help='analysis window start, ps (default: tmin_ps of the settings)')
    ap.add_argument('--tmax', type=float, help='analysis window end, ps after MD start (default: end of the run)')
    ap.add_argument('--shells', type=float, nargs='*', default=list(lio.DEFAULT_SHELLS), help='shell edges, A (same default as 04, 05, 06)')
    ap.add_argument('--lags', type=float, nargs='*', default=[1.0, 5.0, 10.0], help='tau values, ps (default as 05)')
    ap.add_argument('--blocks', type=int, default=lio.DEFAULT_BLOCKS, help='time blocks for the jackknife error bars')
    ap.add_argument('--site-tol', type=float, default=1.0, help='dopant farther than this from every ideal Li site (24d, 96h) counts as state "other", A (check median_dist_A in dopant_sites.csv)')
    ap.add_argument('--min-res-ps', type=float, default=1.0, help='a dopant stay on a site shorter than this is not counted as a site change, ps')
    ap.add_argument('--min-n', type=float, default=3.0, help='mean Li per origin below this: point flagged "few" (open symbol, left out of the collapse statistics)')
    ap.add_argument('--dens-frames', type=int, default=200, help='frames used for the density ratio (grid volume of every state/shell is measured in each of them)')
    ap.add_argument('--outdir', default='site_shells', help='output root; files go to <outdir>/<system>/ and <outdir>/')
    ap.add_argument('--jobs', type=int, default=1, help='parallel (folder, T) computations')
    ap.add_argument('--recompute', action='store_true', help='ignore cached results')
    ap.add_argument('--selftest', action='store_true', help='run on synthetic dumps with a known site-dependent footprint and exit (needs --cif)')
    lio.add_config_args(ap)
    return ap


def main():
    a = build_parser().parse_args()
    if a.selftest:
        cif = next((c for c in (a.cif, os.path.join(HERE, 'cubic_LLZO_wyckoff.cif'), os.path.join(HERE, '..', 'cubic_LLZO_wyckoff.cif')) if os.path.exists(c)), None)
        if not cif: sys.exit('selftest needs the pristine CIF: --cif cubic_LLZO_wyckoff.cif')
        selftest(cif)
    if not os.path.exists(a.cif): sys.exit(f'CIF not found: {a.cif} (--cif)')
    groups = parse_groups(a.items)
    if not groups: sys.exit('no groups given (python site_shells.py {A B C} ...)')
    for g in groups: g['folders'] = expand(g['folders'])
    run(groups, a)


if __name__ == '__main__':
    main()
