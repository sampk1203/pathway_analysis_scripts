#!/usr/bin/env python3
"""predict_D.py - predict the whole-cell Li diffusivity of doped LLZO from the blocking volume V_A, and test it against the null "no blocking".

python predict_D.py {pure 1Al-doped 2Al-doped 3Al-doped} {1Ga-doped 2Ga-doped 3Ga-doped} {1Fe-doped 2Fe-doped 3Fe-doped} {1Mg-doped 2Mg-doped 3Mg-doped} {1Zn-doped 2Zn-doped 3Zn-doped} --radius radius/radius_summary.csv --T 800 1000 1200 --lags 5 10
python predict_D.py --radius radius/radius_summary.csv           (every folder of the map)
python predict_D.py --selftest

Groups use the build_doc.py syntax ({A B C} {D E}); tokens like Al=Li24d and elements=Li,La,Zr,Al,O are accepted and ignored (no dump is read).
Put options after the groups. Reads CSVs only: doping_map.csv (folder,dopant,q,x,n_dopant,n_fu; the pure row has dopant none), whole_msd/<folder>/whole_msd.csv
(whole_msd.py), radius/radius_summary.csv (build_doc.py), optionally msd05/<folder>/msd_shell.csv (05, only for --dvac). The pure folder must be in the map and
have a whole_msd.csv; it is the reference of every group. Use the same window (--tmin/--tmax) and the same --lags in whole_msd.py and 05.

Model (per folder f of dopant d, per T and tau; D = MSD / (6 tau), an effective D: check the log-log slope printed by whole_msd.py)
    Q_meas(f) = MSD_all(f) / MSD_all(pure)                     (same T, tau)
    V_A(f)    = median of V_A of the OTHER folders of the same dopant (same T, tau; valid sigmoid fits only)   -> leave-one-x-out
    Q_pred(f) = exp(-x V_A / V_fu) * F_vac,       D_all_pred = D_pure * Q_pred
    F_vac = 1 by default (far-Li MSD is flat in n_Li within noise). --dvac: F_vac = exp(b (n_Li - 7)), b = slope of ln MSD_far vs n_Li
            from pure + the other folders of d (needs msd05); n_Li = 7 - q x.
The target folder never feeds its own prediction (neither V_A nor F_vac).
Null model: Q = 1 (no dopant effect at all). Per dopant and pooled:
    err = Q_meas - Q_pred (model) and Q_meas - 1 (null);   skill = 1 - mean(err_model^2) / mean(err_null^2)   (> 0: V_A predicts more than "no effect")
    win = fraction of cases with |err_model| < |err_null|;   ok = fraction within 2 sigma or --tol (relative) of Q_meas, for model and null
Cases of one folder at different (T, tau) come from the same trajectory, so they are correlated: the counts overstate the evidence.
--va-source sig (default) uses the sigmoid V_A (smoother); free uses the model-free V_A3 (primary in doping_model.py, often negative or noisy).

Output: <outdir>/predict_D.csv  predict_D_summary.csv  va_used.csv  predict_D_parity.png  predict_D_Q_vs_x.png   (default outdir = predict_D)
Figures show ONE reference case: highest T, tau closest to 10 ps. The csv files hold every (T, tau).
Needs doping_model.py next to this file (loaders, V_A rows), numpy, matplotlib; scipy only for --selftest.
"""
import argparse, glob, math, os, re, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import doping_model as dm

SPEC_RX = re.compile(r'^[A-Z][a-z]?=[A-Za-z0-9_-]+$')          # Al=Li24d (build_doc site spec): ignored here
ELEM_RX = re.compile(r'^elements=([A-Za-z,]+)$')


def warn(msg): print('warning: ' + msg, file=sys.stderr)


def parse_groups(items):
    """-> [{'folders': [...]}] (same syntax as build_doc.py)"""
    groups, cur = [], None

    def add(w):
        if ELEM_RX.match(w) or SPEC_RX.match(w): return
        cur['folders'].append(w)

    for tok in items:
        for part in re.split(r'([{}])', tok):
            if part == '{':
                if cur is not None: sys.exit('nested { not allowed')
                cur = {'folders': []}
            elif part == '}':
                if cur is None: sys.exit('unmatched }')
                groups.append(cur); cur = None
            elif part.strip():
                if cur is None: sys.exit(f'folder outside braces: {part}')
                for w in (part.split() if ('{' in tok or '}' in tok) else [part.strip()]): add(w)
    if cur is not None: sys.exit('missing closing }')
    return [g for g in groups if g['folders']]


def nat(s): return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]
def fname(folder): return os.path.basename(os.path.normpath(folder))


def expand(folders):
    out = []
    for f in folders:
        if any(c in f for c in '*?'):
            hit = sorted((p for p in glob.glob(f) if os.path.isdir(p)), key=lambda p: nat(fname(p)))
            if not hit: warn(f'pattern matched no folder: {f}')
            out += hit
        else:
            out.append(f)
    return out


def keep(a, r):
    if a.T and int(r['T']) not in a.T: return False
    if a.lags and dm.near(a.lags, float(r['tau_ps'])) is None: return False
    return True


def pick(d, T, tau):
    """row of d = {(T, tau): row} at this T and the tau within 6 % of tau, else None"""
    t = dm.near([k[1] for k in d if k[0] == T], tau)
    return None if t is None else d[(T, t)]


# ---------------------------------------------------------------- inputs
def load_inputs(mp, a):
    whole, far = {}, {}
    wdir = a.whole_dir or os.path.join(a.root, 'whole_msd')
    mdir = a.msd05_dir or os.path.join(a.root, 'msd05')
    for f, m in mp.items():
        p = os.path.join(wdir, f, 'whole_msd.csv')
        if not os.path.exists(p): warn(f'{f}: {p} missing -> folder skipped'); continue
        whole[f] = {(int(r['T']), float(r['tau_ps'])): r for r in dm.read_csv(p) if keep(a, r)}
        if not a.dvac or m['dopant'].lower() == 'none': continue
        p = os.path.join(mdir, f, 'msd_shell.csv')
        if not os.path.exists(p): warn(f'{f}: {p} missing -> not used for F_vac'); continue
        d = {}
        for r in dm.read_csv(p, strcols=('dopant', 'shell')):
            if r['dopant'] == 'any' and keep(a, r): d[(int(r['T']), float(r['tau_ps']))] = r      # rows of one (T, tau) are in shell order: the last is far
        far[f] = d
    rad = {}
    if not a.radius or not os.path.exists(a.radius): raise dm.InputError(f'radius_summary.csv not found: {a.radius} (--radius)')
    for r in dm.read_csv(a.radius, strcols=('level', 'name', 'model')):
        if r['level'] != 'folder': continue
        rad.setdefault((r['name'], int(r['T'])), {}).setdefault(float(r['tau_ps']), {})[r['model']] = r
    return rad, whole, far


def select(groups, mp):
    """-> (mp restricted to the grouped folders + the pure folder, {dopant: [folders by x]}, pure folder name)"""
    pure = [f for f, m in mp.items() if m['dopant'].lower() == 'none']
    if not pure: raise dm.InputError('no pure row (dopant none) in the map')
    names = [fname(f) for f in expand([f for g in groups for f in g['folders']])] if groups else list(mp)
    for n in names:
        if n not in mp: warn(f'{n}: not in the map -> skipped')
    sel = {f: mp[f] for f in mp if f in names or f == pure[0]}
    ser = {}
    for f, m in sel.items():
        if m['dopant'].lower() != 'none': ser.setdefault(m['dopant'], []).append(f)
    for d in ser: ser[d].sort(key=lambda f: sel[f]['x'])
    if not ser: raise dm.InputError('no doped folder selected')
    return sel, ser, pure[0]


# ---------------------------------------------------------------- model
def f_vac(dop, f, T, tau, mp, whole, far):
    """exp(b (n_Li(f) - 7)), b = slope of ln MSD_far vs n_Li over pure + the other folders of dop (f excluded). nan if < 2 distinct n_Li."""
    pts = []
    for g, m in mp.items():
        if g == f: continue
        if m['dopant'].lower() == 'none':
            w = pick(whole.get(g, {}), T, tau)
            if w and w['msd_A2'] > 0: pts.append((7.0, math.log(w['msd_A2'])))
        elif m['dopant'] == dop:
            r = pick(far.get(g, {}), T, tau)
            if r and r['msd_A2'] > 0: pts.append((7.0 - m['q'] * m['x'], math.log(r['msd_A2'])))
    if len({round(p[0], 6) for p in pts}) < 2: return float('nan')
    P = np.array(pts)
    b = np.polyfit(P[:, 0], P[:, 1], 1)[0]
    return math.exp(b * ((7.0 - mp[f]['q'] * mp[f]['x']) - 7.0))


def predict(mp, ser, pure, whole, far, va, a):
    rows = []
    nan = float('nan')
    for dop, flist in ser.items():
        for f in flist:
            m = mp[f]
            if f not in whole: continue
            for (T, tau), w in sorted(whole[f].items()):
                wp = pick(whole.get(pure, {}), T, tau)
                if wp is None: continue
                Q = w['msd_A2'] / wp['msd_A2']
                Qse = Q * math.hypot(w['msd_se'] / w['msd_A2'], wp['msd_se'] / wp['msd_A2'])
                vs = [v.get('V_A_use', nan) for v in va if v['dopant'] == dop and v['folder'] != f and v['folder'] in flist and v['T'] == T
                      and abs(v['tau_ps'] - tau) <= 0.06 * max(tau, 1e-9)]
                vs = [v for v in vs if np.isfinite(v)]
                VA = float(np.median(vs)) if vs else nan
                spread = (max(vs) - min(vs)) / VA if len(vs) >= 2 and VA > 0 else nan
                fv = f_vac(dop, f, T, tau, mp, whole, far) if a.dvac else 1.0
                Qp = math.exp(-m['x'] * VA / a.vfu) * fv if np.isfinite(VA) and np.isfinite(fv) else nan
                Dp = wp['msd_A2'] / (6 * tau)
                tol = a.tol * Q
                em, en = Q - Qp, Q - 1.0
                rows.append(dict(folder=f, dopant=dop, q=m['q'], x=m['x'], n_Li_model=7 - m['q'] * m['x'], T=T, tau_ps=tau, n_va_src=len(vs), VA_pred=VA,
                                 VA_spread=spread, F_vac=fv, Q_meas=Q, Q_se=Qse, Q_pred=Qp, err_model=em, err_null=en,
                                 ok_model=int(abs(em) <= max(2 * Qse, tol)) if np.isfinite(Qp) else nan, ok_null=int(abs(en) <= max(2 * Qse, tol)),
                                 D_pure_A2ps=Dp, D_all_obs_A2ps=w['msd_A2'] / (6 * tau), D_all_pred_A2ps=Dp * Qp,
                                 D_all_obs_cm2s=w['msd_A2'] / (6 * tau) * 1e-4, D_all_pred_cm2s=Dp * Qp * 1e-4))
    return rows


def summarise(rows, tol):
    out = []
    for key in sorted({r['dopant'] for r in rows}) + ['ALL']:
        rs = [r for r in rows if (key == 'ALL' or r['dopant'] == key) and np.isfinite(r['Q_pred'])]
        if not rs: out.append(dict(dopant=key, n=0)); continue
        em = np.array([r['err_model'] for r in rs]); en = np.array([r['err_null'] for r in rs])
        mm, mn = float(np.mean(em ** 2)), float(np.mean(en ** 2))
        out.append(dict(dopant=key, n=len(rs), rmse_model=math.sqrt(mm), rmse_null=math.sqrt(mn), skill=1 - mm / mn if mn > 0 else float('nan'),
                        win=float(np.mean(np.abs(em) < np.abs(en))), ok_model=float(np.mean([r['ok_model'] for r in rs])), ok_null=float(np.mean([r['ok_null'] for r in rs])),
                        mean_dev_model_pct=100 * float(np.mean([r['Q_meas'] / r['Q_pred'] - 1 for r in rs])), median_VA_spread=float(np.nanmedian([r['VA_spread'] for r in rs])) if any(np.isfinite(r['VA_spread']) for r in rs) else float('nan')))
    return out


# ---------------------------------------------------------------- output
def print_ref(rows, ref):
    T, tau = ref
    print(f'\n=== Q = D_all / D_pure at the reference case (T = {T} K, tau = {tau:.3g} ps); V_A from the other folders of the same dopant ===')
    print(f' {"folder":<12}{"x":>7}{"n_src":>6}{"V_A":>8}{"spread":>8}{"Q_meas":>8}{"+/-":>7}{"Q_pred":>8}{"err_mod":>9}{"err_null":>9}  flag')
    for r in rows:
        if r['T'] != T or abs(r['tau_ps'] - tau) > 1e-6: continue
        flag = 'no V_A' if not np.isfinite(r['Q_pred']) else ('V_A spread > 50 %' if np.isfinite(r['VA_spread']) and r['VA_spread'] > 0.5 else '')
        print(f' {r["folder"]:<12}{r["x"]:>7.4f}{r["n_va_src"]:>6d}{r["VA_pred"]:>8.0f}{r["VA_spread"]:>8.2f}{r["Q_meas"]:>8.3f}{r["Q_se"]:>7.3f}{r["Q_pred"]:>8.3f}{r["err_model"]:>9.3f}{r["err_null"]:>9.3f}  {flag}')


def print_summary(summ):
    print('\n=== model vs null (every T, tau; cases are correlated) ===')
    print(f' {"dopant":<8}{"n":>4}{"rmse_mod":>10}{"rmse_null":>10}{"skill":>8}{"win":>6}{"ok_mod":>8}{"ok_null":>9}{"dev_%":>8}')
    for s in summ:
        if not s['n']: print(f' {s["dopant"]:<8}{0:>4}   no prediction (no valid V_A from the other folders)'); continue
        print(f' {s["dopant"]:<8}{s["n"]:>4}{s["rmse_model"]:>10.3f}{s["rmse_null"]:>10.3f}{s["skill"]:>8.2f}{s["win"]:>6.2f}{s["ok_model"]:>8.2f}{s["ok_null"]:>9.2f}{s["mean_dev_model_pct"]:>+8.1f}')
    al = [s for s in summ if s['dopant'] == 'ALL' and s['n']]
    if al:
        s = al[0]
        v = 'V_A beats the null (skill > 0, win >= 2/3)' if s['skill'] > 0 and s['win'] >= 2 / 3 else 'V_A is not distinguishable from the null'
        print(f'\nPOOLED: skill = {s["skill"]:.2f}, win = {s["win"]:.2f} -> {v}')
    print('skill > 0: smaller error than Q = 1. The effect is 5-20 % and sigma is 6-13 %: read the per-dopant lines, not only the pooled one.')


def figures(a, rows, ref):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(a.outdir, exist_ok=True)
    T, tau = ref
    rr = [r for r in rows if r['T'] == T and abs(r['tau_ps'] - tau) < 1e-6]
    dops = sorted({r['dopant'] for r in rr})
    col = {d: f'C{i}' for i, d in enumerate(dops)}
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    for d in dops:
        p = [r for r in rr if r['dopant'] == d and np.isfinite(r['Q_pred'])]
        if p: ax.errorbar([r['Q_pred'] for r in p], [r['Q_meas'] for r in p], yerr=[2 * r['Q_se'] for r in p], fmt='o', color=col[d], label=d, capsize=2)
    ax.plot([0.6, 1.2], [0.6, 1.2], 'k--', lw=0.8); ax.axhline(1.0, color='0.6', lw=0.6)
    ax.set_xlabel('Q predicted from V_A (other folders)'); ax.set_ylabel('Q measured = D_all / D_pure (2 sigma)'); ax.legend(frameon=False, fontsize=8)
    ax.set_title(f'T = {T} K, tau = {tau:.3g} ps', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(a.outdir, 'predict_D_parity.png'), dpi=200); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for d in dops:
        p = sorted([r for r in rr if r['dopant'] == d], key=lambda r: r['x'])
        ax.errorbar([r['x'] for r in p], [r['Q_meas'] for r in p], yerr=[2 * r['Q_se'] for r in p], fmt='o', color=col[d], label=d, capsize=2)
        q = [r for r in p if np.isfinite(r['Q_pred'])]
        if q: ax.plot([r['x'] for r in q], [r['Q_pred'] for r in q], 'x--', color=col[d], lw=0.8)
    ax.axhline(1.0, color='0.5', lw=0.8, label='null (Q = 1)')
    ax.set_xlabel('x = n_dopant / n_fu'); ax.set_ylabel('Q = D_all / D_pure'); ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.set_title('points: measured (2 sigma); x: predicted from V_A', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(a.outdir, 'predict_D_Q_vs_x.png'), dpi=200); plt.close(fig)


def analyse(a, groups, plots=True):
    mp0 = dm.load_map(a.map)
    mp, ser, pure = select(groups, mp0)
    rad, whole, far = load_inputs(mp, a)
    if pure not in whole: raise dm.InputError(f'pure folder {pure}: whole_msd.csv missing (needed as the reference)')
    va = dm.step1(mp, rad, whole, a)
    rows = predict(mp, ser, pure, whole, far, va, a)
    summ = summarise(rows, a.tol)
    ref = dm.ref_case([(r['T'], r['tau_ps']) for r in rows if np.isfinite(r['Q_pred'])])
    os.makedirs(a.outdir, exist_ok=True)
    dm.write_csv(os.path.join(a.outdir, 'predict_D.csv'), rows)
    dm.write_csv(os.path.join(a.outdir, 'predict_D_summary.csv'), summ)
    dm.write_csv(os.path.join(a.outdir, 'va_used.csv'), va)
    if plots and ref: figures(a, rows, ref)
    return dict(rows=rows, summ=summ, ref=ref, va=va)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('items', nargs='*', help='groups: {pure A B C} {D E}  (default: every folder of the map)')
    ap.add_argument('--map', default=os.path.join(HERE, 'doping_map.csv'), help='doping_map.csv: folder,dopant,q,x,n_dopant,n_fu (default: next to this file)')
    ap.add_argument('--radius', help='radius_summary.csv of build_doc.py')
    ap.add_argument('--root', default='.', help='folder holding whole_msd/<folder>/whole_msd.csv and msd05/<folder>/msd_shell.csv')
    ap.add_argument('--whole-dir'); ap.add_argument('--msd05-dir')
    ap.add_argument('--T', type=int, nargs='*', help='temperatures (default: everything found)')
    ap.add_argument('--lags', type=float, nargs='*', help='tau values, ps, matched within 6 %% (default: everything found)')
    ap.add_argument('--vfu', type=float, default=dm.VFU, help='volume per formula unit, A^3 (default a^3/8 = 273.5)')
    ap.add_argument('--va-source', choices=['free', 'sig'], default='sig', help='V_A used: sigmoid analytic (sig, default) or model-free V_A3 (free)')
    ap.add_argument('--va-all', action='store_true', help='use V_A even where the sigmoid fit is not valid (V_A <= 0 is never used)')
    ap.add_argument('--dvac', action='store_true', help='multiply by F_vac(n_Li) from the far-Li MSD (needs msd05); default F_vac = 1')
    ap.add_argument('--tol', type=float, default=0.05, help='ok = |Q_meas - Q_pred| within 2 sigma or this fraction of Q_meas (same for the null)')
    ap.add_argument('--outdir', default='predict_D', help='output folder')
    ap.add_argument('--selftest', action='store_true', help='run on synthetic CSVs with a known V_A and exit (needs scipy)')
    return ap


def main():
    a = build_parser().parse_args()
    if a.selftest: selftest()
    try: R = analyse(a, parse_groups(a.items))
    except dm.InputError as e: sys.exit(f'error: {e}')
    if not R['ref']: sys.exit('error: no prediction: no valid V_A from the other folders. See va_used.csv (valid, V_A_use) and the warnings above; --va-all or --va-source free change what counts.')
    print_ref(R['rows'], R['ref']); print_summary(R['summ'])
    print(f'\n[+] {a.outdir}/ predict_D.csv predict_D_summary.csv va_used.csv predict_D_parity.png predict_D_Q_vs_x.png')


# ---------------------------------------------------------------- selftest
def selftest():
    """Synthetic CSVs: Q = exp(-x V / V_fu) exactly, V = 350 (Al), 300 (Ga). Checks the prediction reproduces Q, skill ~ 1, and the leave-one-out V_A."""
    root = tempfile.mkdtemp()
    V = {'Al': 350.0, 'Ga': 300.0}
    mrows = ['folder,dopant,q,x,n_dopant,n_fu', 'pure,none,0,0,0,16']
    names = []
    for d in V:
        for k in (1, 2, 3):
            mrows.append(f'{k}{d}-doped,{d},3,{k / 16},{k},16'); names.append((f'{k}{d}-doped', d, k / 16))
    mp_path = os.path.join(root, 'doping_map.csv')
    open(mp_path, 'w').write('\n'.join(mrows) + '\n')
    pure_msd = {1000: 4.8, 1200: 7.0}
    taus = (5.0, 10.0)
    wr = ['T,tau_ps,n_Li,msd_A2,msd_se,blocks,n_dopant,n_fu,t_start_ps,t_end_ps']
    os.makedirs(os.path.join(root, 'whole_msd', 'pure'))
    for T, m in pure_msd.items():
        for t in taus: wr.append(f'{T},{t},112,{m * t / 10},{0.03 * m * t / 10},5,0,16,100,200')
    open(os.path.join(root, 'whole_msd', 'pure', 'whole_msd.csv'), 'w').write('\n'.join(wr) + '\n')
    rr = ['level,name,T,tau_ps,model,A,r0,w,B,r50,r95,r_p99_A,half_box_A,dopant_dopant_min_A,r_max_used,V_A3,r_eff']
    for f, d, x in names:
        wr = ['T,tau_ps,n_Li,msd_A2,msd_se,blocks,n_dopant,n_fu,t_start_ps,t_end_ps']
        k = int(round(x * 16))
        for T, m in pure_msd.items():
            for t in taus:
                ms = m * t / 10 * math.exp(-x * V[d] / dm.VFU)
                wr.append(f'{T},{t},{112 - 3 * k},{ms},{0.03 * ms},5,{k},16,100,200')
                r0 = dm._solve_r0(V[d])
                rr.append(f'folder,{f},{T},{t},sigmoid,0.8,{r0},0.7,1.0,{r0},8.0,12.0,13.0,9.0,12.0,nan,nan')
                rr.append(f'folder,{f},{T},{t},free,nan,nan,nan,nan,nan,nan,nan,nan,nan,12.0,{V[d]},3.0')
        os.makedirs(os.path.join(root, 'whole_msd', f))
        open(os.path.join(root, 'whole_msd', f, 'whole_msd.csv'), 'w').write('\n'.join(wr) + '\n')
    rad_path = os.path.join(root, 'radius_summary.csv')
    open(rad_path, 'w').write('\n'.join(rr) + '\n')
    ap = build_parser()
    a = ap.parse_args(['--map', mp_path, '--radius', rad_path, '--root', root, '--outdir', os.path.join(root, 'out')])
    R = analyse(a, [], plots=False)
    ok = [r for r in R['rows'] if np.isfinite(r['Q_pred'])]
    assert len(ok) == len(R['rows']) == 2 * 3 * 2 * 2, f'rows: {len(ok)} of {len(R["rows"])}'
    for r in ok:
        assert abs(r['Q_pred'] / r['Q_meas'] - 1) < 1e-6, (r['folder'], r['Q_pred'], r['Q_meas'])
        assert abs(r['VA_pred'] - V[r['dopant']]) < 1e-3, (r['folder'], r['VA_pred'])
    al = [s for s in R['summ'] if s['dopant'] == 'ALL'][0]
    assert al['skill'] > 0.999 and al['win'] == 1.0, al
    a2 = ap.parse_args(['--map', mp_path, '--radius', rad_path, '--root', root, '--outdir', os.path.join(root, 'out2'), '--T', '1200', '--lags', '10'])
    R2 = analyse(a2, parse_groups(['{pure', '1Al-doped', '2Al-doped', '3Al-doped}']), plots=False)
    assert len(R2['rows']) == 3 and all(r['dopant'] == 'Al' for r in R2['rows']), len(R2['rows'])
    print('selftest ok')
    sys.exit(0)


if __name__ == '__main__':
    main()
