#!/usr/bin/env python3
"""doping_model.py - dopant-blocking conductivity model for doped LLZO: steps S1-S4 of doping_model_plan.md (reads CSVs only, no dumps).

python doping_model.py --map doping_map.csv --radius report/radius/radius_summary.csv [--root .] [--T 1000] [--tau 10] [--host Al]
python doping_model.py --selftest

Inputs
  --map      doping_map.csv, written by hand:  folder,dopant,q,x,n_dopant,n_fu   (pure row: pure,none,0,0,0,16)
             x = n_dopant / n_fu is CHECKED (and against n_dopant / n_fu of whole_msd.csv). q = Li lost per dopant (3 Al Ga Fe, 2 Mg Zn).
  --radius   radius_summary.csv of build_doc.py (level = folder rows: sigmoid and free rows).
  --root     folder holding msd05/<folder>/msd_shell.csv (05) and whole_msd/<folder>/whole_msd.csv (whole_msd.py);
             or --msd05-dir / --whole-dir. 05 and whole_msd.py MUST use the same window (--tmin/--tmax) and the same --lags.
  --T --tau  restrict temperatures / tau (ps); tau of the three sources are matched within 6 %. Default: everything found.
             Figures use ONE reference case: highest T, tau closest to 10 ps. csv tables hold every (T, tau).
  --vfu      volume per formula unit, A^3 (default a^3/8 = 273.5)     --host  dopant that calibrates D_vac (default Al)
  --va-all   use V_A even where the sigmoid fit is not valid (default: valid fits only: no '*', A >= 0.1). V_A <= 0 is never used.
  --va-source free|sig   which V_A feeds the prediction (default free = plan; sig = sigmoid analytic, smoother when the model-free integral is noisy)
  --tol      closure: |R_meas - R_pred| / R_meas allowed besides 2 sigma (default 0.05)    --tol-collapse  same for D_far (default 0.10)

S1  V_A and radius vs x     per folder, T, tau: A r0 w B r50 r95, V_A model-free (free row V_A3, primary), V_A from the sigmoid
                            4 pi A B (r0^3/3 + pi^2 w^2 r0/3) (cross-check, warn if > 30 % apart), r_eff, caps. cap_binding = r95 > 0.9 * min(r_p99, half_box):
                            a radius at a cap is a lower bound, not a result.   -> va_table.csv, radius_vs_x.png
S2  mobility table          MSD_all (whole_msd.py), MSD_far (05, shell 'any', last = far; pure folder: all Li), R_meas = MSD_all / MSD_far (same tau, same
                            folder: pure blocking effect), far-Li fraction (flag < 0.15 or < 10 far Li per origin). R_se ignores the (positive) correlation of
                            the two MSD, so it is conservative.   -> mobility_table.csv
S3  closure test            per dopant, T, tau: R_meas(x) fitted with exp(-k x), 1 - k x (1 parameter each, R(0) = 1) and (1 - x/xc)^mu (2 parameters, shown only),
                            chi2 in R. 'better' = lower chi2 of exp vs linear. k_exp * V_fu is the V_A the data want (compare V_A). ln R = a + b x with free
                            a: a = 0 is the multiplicative rule. Prediction from the lowest-x folder: R_pred(x) = exp(-x V_A / V_fu), tested at the other x.
                            -> closure.csv, closure_pred.csv, closure_parity.png, closure_lnR_vs_x.png
S4  host curve + transfer   ln D_far vs n_Li = 7 - q x for all dopants (D = MSD / 6 tau); host (pure + host folders) fitted linear and quadratic in n_Li.
                            Collapse: other dopants' D_far vs the host curve. Transfer: D_all(x) = D_vac(n_Li) R_pred(x) vs measured D_all; the host is
                            predicted leave-one-out.   -> transfer.csv, far_vs_nLi.png, transfer_parity.png
Gate (plan section 9): printed at the end. It is a heuristic on 3-4 points per dopant: read the tables.
Synthetic selftest tests the code, not the physics.
"""
import argparse, csv, math, os, sys, tempfile, warnings
import numpy as np

VFU = 12.9827 ** 3 / 8


class InputError(Exception): pass


def warn(msg): print('warning: ' + msg, file=sys.stderr)


# ---------------------------------------------------------------- io
def _num(v):
    try: return float(v)
    except (TypeError, ValueError): return v


def read_csv(path, strcols=()):
    with open(path, newline='') as fh:
        return [{k: (v if k in strcols else _num(v)) for k, v in r.items()} for r in csv.DictReader(fh)]


def write_csv(path, rows):
    if not rows: return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval='nan'); w.writeheader(); w.writerows(rows)


def near(vals, v, rtol=0.06):
    vals = list(vals)
    if not vals: return None
    b = min(vals, key=lambda u: abs(u - v))
    return b if abs(b - v) <= rtol * max(abs(v), 1e-9) else None


def load_map(path):
    """-> {folder: dict(dopant, q, x, n_dopant, n_fu)}; raises InputError with every problem listed."""
    if not os.path.exists(path): raise InputError(f'map file not found: {path}')
    raw = read_csv(path, strcols=('folder', 'dopant'))
    miss = [c for c in ('folder', 'dopant', 'q', 'x', 'n_dopant', 'n_fu') if not raw or c not in raw[0]]
    if miss: raise InputError(f'{path}: missing column(s) {miss} (need folder,dopant,q,x,n_dopant,n_fu)')
    mp, bad = {}, []
    for r in raw:
        f = str(r['folder']).strip()
        try: q, x, nd, nf = float(r['q']), float(r['x']), float(r['n_dopant']), float(r['n_fu'])
        except (TypeError, ValueError): bad.append(f'{f}: q, x, n_dopant, n_fu must be numbers'); continue
        if nf <= 0 or abs(nd / nf - x) > 1e-6: bad.append(f'{f}: n_dopant / n_fu = {nd:g} / {nf:g} = {nd / nf:.5f} but x = {x:g}')
        mp[f] = dict(folder=f, dopant=str(r['dopant']).strip(), q=q, x=x, n_dopant=nd, n_fu=nf)
    if bad: raise InputError(f'{path}: x does not match n_dopant / n_fu:\n  ' + '\n  '.join(bad))
    return mp


def load_sources(mp, a):
    """-> radius {(name, T): [(tau, sigmoid row, free row)]}, whole {folder: {(T, tau): row}}, msd05 {folder: {(T, tau): far row}}"""
    whole, msd05, rad = {}, {}, {}
    wdir = a.whole_dir or os.path.join(a.root, 'whole_msd')
    mdir = a.msd05_dir or os.path.join(a.root, 'msd05')
    for f, m in mp.items():
        p = os.path.join(wdir, f, 'whole_msd.csv')
        if not os.path.exists(p): warn(f'{f}: {p} missing -> folder skipped'); continue
        rows = read_csv(p)
        whole[f] = {(int(r['T']), float(r['tau_ps'])): r for r in rows if _keep(a, r)}
        r0 = rows[0]
        if abs(r0['n_dopant'] - m['n_dopant']) > 1e-9 or abs(r0['n_fu'] - m['n_fu']) > 1e-9:
            warn(f'{f}: map says n_dopant = {m["n_dopant"]:g}, n_fu = {m["n_fu"]:g}; the dump has {r0["n_dopant"]:g}, {r0["n_fu"]:g}. x is taken from the map: CHECK IT')
        if m['dopant'].lower() == 'none': continue
        p = os.path.join(mdir, f, 'msd_shell.csv')
        if not os.path.exists(p): warn(f'{f}: {p} missing -> no far-Li MSD, folder skipped in S2-S4'); continue
        far = {}
        for r in read_csv(p, strcols=('dopant', 'shell')):
            if r['dopant'] == 'any': far[(int(r['T']), float(r['tau_ps']))] = r           # rows of one (T, tau) are in shell order: the last is far
        msd05[f] = far
    p = a.radius
    if not p or not os.path.exists(p): raise InputError(f'radius_summary.csv not found: {p}')
    for r in read_csv(p, strcols=('level', 'name', 'model')):
        if r['level'] != 'folder': continue
        rad.setdefault((r['name'], int(r['T'])), {}).setdefault(float(r['tau_ps']), {})[r['model']] = r
    return rad, whole, msd05


def _keep(a, r):
    if a.T and int(r['T']) not in a.T: return False
    if a.tau and near(a.tau, float(r['tau_ps'])) is None: return False
    return True


# ---------------------------------------------------------------- S1
def sig_VA(A, r0, w, B): return 4 * math.pi * A * B * (r0 ** 3 / 3 + math.pi ** 2 * w ** 2 * r0 / 3)


def step1(mp, rad, whole, a):
    out = []
    for f, m in mp.items():
        if m['dopant'].lower() == 'none' or f not in whole: continue
        for (T, tau) in sorted(whole[f]):
            d = rad.get((f, T), {})
            t = near(d.keys(), tau)
            row = dict(folder=f, dopant=m['dopant'], q=m['q'], x=m['x'], T=T, tau_ps=tau)
            nan = float('nan')
            if t is None or 'sigmoid' not in d[t] or 'free' not in d[t]:
                out.append(dict(row, valid=0, note='no radius row')); continue
            s, fr = d[t]['sigmoid'], d[t]['free']
            valid = int(s['A'] >= 0.1 and 0 < s['r95'] <= fr['r_max_used'])
            vfree = fr['V_A3']
            vsig = sig_VA(s['A'], s['r0'], s['w'], s['B']) if all(np.isfinite([s['A'], s['r0'], s['w'], s['B']])) else nan
            diff = 100 * (vsig - vfree) / vfree if vfree and np.isfinite(vsig) and np.isfinite(vfree) else nan
            cap = min(s['r_p99_A'], s['half_box_A'])
            src = vsig if a.va_source == 'sig' else vfree
            use = src if ((valid or a.va_all) and np.isfinite(src) and src > 0) else nan          # V_A <= 0 is noise, not a volume
            out.append(dict(row, valid=valid, A=s['A'], r0=s['r0'], w=s['w'], B=s['B'], r50=s['r50'], r95=s['r95'], r_max_used=fr['r_max_used'],
                            r_p99_A=s['r_p99_A'], dopant_dopant_min_A=s['dopant_dopant_min_A'], half_box_A=s['half_box_A'],
                            cap_binding=int(s['r95'] > 0.9 * cap), V_A_free=vfree, V_A_sig=vsig, V_diff_pct=diff, r_eff=fr['r_eff'],
                            V_A_use=use, note=''))
    return out


# ---------------------------------------------------------------- S2
def step2(mp, whole, msd05):
    out = []
    for f, m in mp.items():
        if f not in whole: continue
        none = m['dopant'].lower() == 'none'
        if not none and f not in msd05: continue
        for (T, tau), w in sorted(whole[f].items()):
            if none: far, nfar, ms, mse = None, w['n_Li'], w['msd_A2'], w['msd_se']
            else:
                t = near([k[1] for k in msd05[f] if k[0] == T], tau)
                if t is None: warn(f'{f} T={T} tau={tau:g}: no matching 05 row'); continue
                far = [r for k, r in msd05[f].items() if k[0] == T and k[1] == t][0]
                nfar, ms, mse = far['n_Li_used'], far['msd_A2'], far['msd_se']
            R = w['msd_A2'] / ms
            Rse = R * math.hypot(w['msd_se'] / w['msd_A2'], mse / ms)
            frac = nfar / w['n_Li']
            out.append(dict(folder=f, dopant=m['dopant'], q=m['q'], x=m['x'], n_Li_model=7 - m['q'] * m['x'], T=T, tau_ps=tau, msd_all=w['msd_A2'],
                            msd_all_se=w['msd_se'], msd_far=ms, msd_far_se=mse, n_far_used=nfar, far_frac=frac,
                            far_flag=int(frac < 0.15 or nfar < 10), R_meas=R, R_se=Rse, D_all=w['msd_A2'] / (6 * tau), D_far=ms / (6 * tau)))
    return out


# ---------------------------------------------------------------- S3
def _se(R, se):
    se = np.where(np.isfinite(se) & (se > 0), se, 0.05 * R)
    return np.maximum(se, 0.002 * R)


def fit_blocking(x, R, se):
    from scipy.optimize import curve_fit
    x, R = np.asarray(x, float), np.asarray(R, float)
    se = _se(R, np.asarray(se, float))
    w = 1 / se ** 2
    chi = lambda mod: float(np.sum(((R - mod) / se) ** 2))
    k_lin = float(np.sum(w * x * (1 - R)) / np.sum(w * x * x))
    k0 = float(np.sum(w * x * -np.log(R)) / np.sum(w * x * x))
    try: k_exp = float(curve_fit(lambda xx, k: np.exp(-k * xx), x, R, p0=[k0], sigma=se, absolute_sigma=True)[0][0])
    except Exception: k_exp = k0
    res = dict(k_exp=k_exp, chi2_exp=chi(np.exp(-k_exp * x)), k_lin=k_lin, chi2_lin=chi(1 - k_lin * x),
               xc=float('nan'), mu=float('nan'), chi2_pow=float('nan'), a_lnR=float('nan'), a_lnR_se=float('nan'), b_lnR=float('nan'))
    if len(x) >= 3:
        try:
            p = curve_fit(lambda xx, xc, mu: np.clip(1 - xx / xc, 1e-9, None) ** mu, x, R, p0=[x.max() * 3, 1.0], sigma=se, absolute_sigma=True,
                          bounds=([x.max() * 1.001, 0.05], [50.0, 20.0]))[0]
            res.update(xc=float(p[0]), mu=float(p[1]), chi2_pow=chi(np.clip(1 - x / p[0], 1e-9, None) ** p[1]))
        except Exception: pass
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            c, cov = np.polyfit(x, np.log(R), 1, w=R / se, cov='unscaled')
        res.update(b_lnR=float(c[0]), a_lnR=float(c[1]), a_lnR_se=float(math.sqrt(cov[1, 1])))
    res['better'] = 'exp' if res['chi2_exp'] <= res['chi2_lin'] else 'lin'
    return res


def step3(mp, mob, va, a):
    mi = {(r['folder'], r['T'], r['tau_ps']): r for r in mob}
    vi = {(r['folder'], r['T'], r['tau_ps']): r for r in va}
    fits, preds = [], []
    for dop in sorted({m['dopant'] for m in mp.values() if m['dopant'].lower() != 'none'}):
        fl = sorted([f for f, m in mp.items() if m['dopant'] == dop], key=lambda f: mp[f]['x'])
        for (T, tau) in sorted({(r['T'], r['tau_ps']) for r in mob if r['dopant'] == dop}):
            pts = [(mp[f]['x'], mi[(f, T, tau)], f) for f in fl if (f, T, tau) in mi and np.isfinite(mi[(f, T, tau)]['R_meas']) and mi[(f, T, tau)]['R_meas'] > 0]
            if len(pts) < 2: continue
            x = [p[0] for p in pts]
            ft = fit_blocking(x, [p[1]['R_meas'] for p in pts], [p[1]['R_se'] for p in pts])
            row = dict(dopant=dop, T=T, tau_ps=tau, n_x=len(pts), x_list=' '.join(f'{v:.4g}' for v in x), VA_from_k_exp=ft['k_exp'] * a.vfu, **ft)
            lo = vi.get((pts[0][2], T, tau), {})
            VA = lo.get('V_A_use', float('nan'))
            row['V_A_lowest_x'] = VA
            row['k_pred'] = VA / a.vfu
            fits.append(row)
            for xx, mr, f in pts[1:]:
                Rp = math.exp(-xx * VA / a.vfu) if np.isfinite(VA) else float('nan')
                dev = mr['R_meas'] - Rp
                se = float(_se(np.array([mr['R_meas']]), np.array([mr['R_se']]))[0])
                z = dev / se
                ok = int(np.isfinite(Rp) and (abs(z) <= 2 or abs(dev) / mr['R_meas'] <= a.tol))
                preds.append(dict(dopant=dop, folder=f, x=xx, T=T, tau_ps=tau, V_A_lowest_x=VA, R_pred=Rp, R_meas=mr['R_meas'], R_se=se, dev=dev,
                                  z=z, rel_dev=dev / mr['R_meas'], pass_pred=ok, far_flag=mr['far_flag']))
    return fits, preds


# ---------------------------------------------------------------- S4
def _fit_lnD(n, lnD, s, deg):
    n, lnD = np.asarray(n, float), np.asarray(lnD, float)
    if len(n) <= deg: return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.polyfit(n, lnD, deg, w=1 / np.maximum(s, 0.005))


def step4(mp, mob, preds, a):
    pi = {(r['folder'], r['T'], r['tau_ps']): r for r in preds}
    mi = {(r['folder'], r['T'], r['tau_ps']): r for r in mob}
    host = [f for f, m in mp.items() if m['dopant'] == a.host]
    pure = [f for f, m in mp.items() if m['dopant'].lower() == 'none']
    if not host: raise InputError(f'--host {a.host}: no folder of that dopant in the map')
    out, curves = [], {}
    for (T, tau) in sorted({(r['T'], r['tau_ps']) for r in mob}):
        def pt(f):
            r = mi.get((f, T, tau))
            return None if r is None or not (r['D_far'] > 0) else (7 - mp[f]['q'] * mp[f]['x'], math.log(r['D_far']), r['msd_far_se'] / r['msd_far'])
        base = [f for f in pure + host if pt(f)]
        if len(base) < 3: warn(f'S4 T={T} tau={tau:g}: only {len(base)} host points (pure + {a.host}), skipped'); continue
        P = np.array([pt(f) for f in base])
        cl, cq = _fit_lnD(*P.T, 1), _fit_lnD(*P.T, 2)
        curves[(T, tau)] = (cl, cq, P)
        for f, m in mp.items():
            r = mi.get((f, T, tau))
            if r is None or m['dopant'].lower() == 'none' or not (r['D_far'] > 0): continue
            n, lnD, s = pt(f)
            loo = f in host
            if loo:
                rest = np.array([pt(g) for g in base if g != f])
                fl, fq = _fit_lnD(*rest.T, 1), _fit_lnD(*rest.T, 2)
            else: fl, fq = cl, cq
            pl = float(np.polyval(fl, n)) if fl is not None else float('nan')
            pq = float(np.polyval(fq, n)) if fq is not None else float('nan')
            dev = lambda p: math.exp(lnD - p) - 1 if np.isfinite(p) else float('nan')
            z = (lnD - pl) / max(s, 0.005) if np.isfinite(pl) else float('nan')
            Rp = pi.get((f, T, tau), {}).get('R_pred')
            if Rp is None:                                                # lowest-x folder: its own V_A
                from_v = [p for p in preds if p['dopant'] == m['dopant'] and p['T'] == T and p['tau_ps'] == tau]
                VA = from_v[0]['V_A_lowest_x'] if from_v else float('nan')
                Rp = math.exp(-m['x'] * VA / a.vfu) if np.isfinite(VA) else float('nan')
            Dall_l, Dall_q = math.exp(pl) * Rp, math.exp(pq) * Rp
            out.append(dict(folder=f, dopant=m['dopant'], q=m['q'], x=m['x'], n_Li=n, T=T, tau_ps=tau, host=int(m['dopant'] == a.host), loo=int(loo),
                            D_far_obs=r['D_far'], D_far_pred_lin=math.exp(pl), D_far_pred_quad=math.exp(pq), dev_far_lin=dev(pl), dev_far_quad=dev(pq), z_far_lin=z,
                            collapse_ok=int(m['dopant'] == a.host or abs(z) <= 2 or abs(dev(pl)) <= a.tol_collapse),
                            D_all_obs=r['D_all'], R_pred=Rp, D_all_pred_lin=Dall_l, D_all_pred_quad=Dall_q,
                            dev_all_lin=r['D_all'] / Dall_l - 1, dev_all_quad=r['D_all'] / Dall_q - 1))
    return out, curves


# ---------------------------------------------------------------- gate, printing
def frac(rows, key): return (sum(int(r[key]) for r in rows) / len(rows)) if rows else float('nan')


def verdict(f): return 'n/a' if not np.isfinite(f) else ('supported' if f >= 2 / 3 else 'fails' if f < 1 / 3 else 'mixed')


def gate(fits, preds, tr, mp, a):
    print('\n=== GATE (plan section 9) ===')
    pe = [dict(ok=int(r['better'] == 'exp')) for r in fits]
    fe = frac(pe, 'ok')
    pp = [p for p in preds if np.isfinite(p['R_pred'])]
    fp = frac(pp, 'pass_pred')
    print(f'closure: exp beats linear in {100 * fe:.0f} % of {len(fits)} (dopant, T, tau) fits -> {verdict(fe)}')
    print(f'closure: R_pred (V_A of the lowest-x folder) within 2 sigma or {100 * a.tol:g} % of R_meas in {100 * fp:.0f} % of {len(pp)} cases -> {verdict(fp)}')
    for dop in sorted({p["dopant"] for p in pp}):
        q = [p for p in pp if p['dopant'] == dop]
        print(f'   {dop:>3}: pred ok {100 * frac(q, "pass_pred"):.0f} %   mean R_meas/R_pred - 1 = {100 * np.mean([p["R_meas"] / p["R_pred"] - 1 for p in q]):+.1f} %')
    ot = [r for r in tr if not r['host']]
    fc = frac(ot, 'collapse_ok')
    print(f'collapse: other dopants\' D_far within 2 sigma or {100 * a.tol_collapse:g} % of the {a.host} curve in {100 * fc:.0f} % of {len(ot)} cases -> {verdict(fc)}')
    for dop in sorted({r['dopant'] for r in ot}):
        q = [r for r in ot if r['dopant'] == dop]
        print(f'   {dop:>3} (q = {q[0]["q"]:g}): ok {100 * frac(q, "collapse_ok"):.0f} %   mean D_far/D_curve - 1 = {100 * np.mean([r["dev_far_lin"] for r in q]):+.1f} % (linear)')
    h = [r for r in tr if r['host']]
    if h: print(f'transfer: {a.host} leave-one-out D_all error: median |dev| {100 * np.median([abs(r["dev_all_lin"]) for r in h]):.1f} % (linear D_vac), '
                f'{100 * np.median([abs(r["dev_all_quad"]) for r in h]):.1f} % (quadratic); other dopants {100 * np.median([abs(r["dev_all_lin"]) for r in ot]):.1f} % / '
                f'{100 * np.median([abs(r["dev_all_quad"]) for r in ot]):.1f} %' if ot else '')
    if not np.isfinite(fe) or not np.isfinite(fp) or not np.isfinite(fc): print('GATE: not enough data to decide'); return
    if min(fe, fp) < 1 / 3 or fc < 1 / 3: print('GATE: STOP. Do not run 5/16. Report V_A and radius as the result; the model is not supported.')
    elif min(fe, fp, fc) >= 2 / 3: print('GATE: PASS. Run Al 5/16, then S6 / S7.')
    else: print('GATE: MIXED. Read the per-dopant lines: failing dopants need their own D_vac (high-x run each) or the S5 vacancy correction.')
    print('(heuristic on 3-4 points per dopant; thresholds: >= 2/3 supported, < 1/3 fails)')


def ref_case(cases):
    Ts = sorted({c[0] for c in cases})
    if not Ts: return None
    Tr = Ts[-1]
    taus = sorted({c[1] for c in cases if c[0] == Tr})
    t10 = [t for t in taus if abs(t - 10) < 0.6]
    return Tr, (t10[0] if t10 else taus[-1])


def print_s1(va, ref):
    print('\n=== S1  radius and V_A (reference T = %s K, tau = %s ps) ===' % ref)
    print(f'{"folder":>12} {"x":>6} {"valid":>5} {"r50":>5} {"r95":>5} {"cap":>4} {"V_A free":>9} {"V_A sig":>8} {"r_eff":>6}')
    for r in va:
        if (r['T'], r['tau_ps']) != ref: continue
        if r.get('note'): print(f'{r["folder"]:>12} {r["x"]:>6.4f}   {r["note"]}'); continue
        print(f'{r["folder"]:>12} {r["x"]:>6.4f} {r["valid"]:>5d} {r["r50"]:>5.1f} {r["r95"]:>5.1f} {"CAP" if r["cap_binding"] else "":>4} {r["V_A_free"]:>9.0f} {r["V_A_sig"]:>8.0f} {r["r_eff"]:>6.2f}')
    print('valid = sigmoid fit without "*" and A >= 0.1; CAP = r95 > 0.9 * min(r_p99, half_box): lower bound, not a result. V_A in A^3.')


def print_s2(mob, ref):
    print('\n=== S2  mobility (reference case) ===')
    print(f'{"folder":>12} {"x":>6} {"n_Li":>6} {"MSD_all":>8} {"MSD_far":>8} {"R_meas":>7} {"+/-":>6} {"far Li":>7}')
    for r in mob:
        if (r['T'], r['tau_ps']) != ref: continue
        print(f'{r["folder"]:>12} {r["x"]:>6.4f} {r["n_Li_model"]:>6.3f} {r["msd_all"]:>8.3f} {r["msd_far"]:>8.3f} {r["R_meas"]:>7.3f} {r["R_se"]:>6.3f} {r["n_far_used"]:>5.1f}{"  LOW" if r["far_flag"] else ""}')
    nf = sum(r['far_flag'] for r in mob)
    if nf: print(f'LOW = far Li fraction < 0.15 or < 10 far Li per origin: {nf} of {len(mob)} rows; their R_meas is noisy.')


# ---------------------------------------------------------------- figures
def figures(a, mp, va, mob, fits, preds, tr, curves, ref):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(a.out, exist_ok=True)
    dops = sorted({m['dopant'] for m in mp.values() if m['dopant'].lower() != 'none'})
    col = {d: plt.cm.tab10(i % 10) for i, d in enumerate(dops)}
    # radius vs x
    fig, ax = plt.subplots(1, 3, figsize=(13, 4), dpi=130)
    for d in dops:
        rr = sorted([r for r in va if r['dopant'] == d and (r['T'], r['tau_ps']) == ref and 'A' in r], key=lambda r: r['x'])
        if not rr: continue
        x = [r['x'] for r in rr]
        for k, (key, lab) in enumerate((('r95', 'r95 (A)'), ('r50', 'r50 (A)'), ('V_A_free', 'V_A model-free (A^3)'))):
            y = [r[key] for r in rr]
            ax[k].plot(x, y, '-', color=col[d], label=d)
            ax[k].scatter(x, y, facecolors=[col[d] if r['valid'] else 'none' for r in rr], edgecolors=col[d], zorder=3)
            ax[k].set_ylabel(lab)
        ax[0].plot(x, [r['r_p99_A'] for r in rr], '--', color=col[d], lw=0.8)
        ax[0].plot(x, [r['dopant_dopant_min_A'] for r in rr], ':', color=col[d], lw=0.8)
    hb = [r['half_box_A'] for r in va if 'half_box_A' in r]
    if hb: ax[0].axhline(min(hb), color='k', ls='--', lw=0.8)
    ax[0].set_title('dashed = r_p99, dotted = dopant-dopant min, black = half box (caps)', fontsize=7)
    for a_ in ax: a_.set_xlabel('x (dopant per f.u.)')
    ax[0].legend(fontsize=7)
    fig.suptitle('T = %s K, tau = %s ps; open symbol = fit not valid. A flat radius at a cap is not evidence.' % ref, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, 'radius_vs_x.png')); plt.close(fig)
    # closure ln R vs x
    nc = max(1, len(dops))
    fig, ax = plt.subplots(1, nc, figsize=(3.8 * nc, 3.6), dpi=130, squeeze=False, sharey=True)
    xs = np.linspace(0, max([m['x'] for m in mp.values()] + [0.1]) * 1.1, 100)
    for i, d in enumerate(dops):
        A = ax[0][i]
        pts = sorted([r for r in mob if r['dopant'] == d and (r['T'], r['tau_ps']) == ref], key=lambda r: r['x'])
        ft = [f for f in fits if f['dopant'] == d and (f['T'], f['tau_ps']) == ref]
        if pts:
            A.errorbar([r['x'] for r in pts], [r['R_meas'] for r in pts], [r['R_se'] for r in pts], fmt='o', color=col[d], capsize=2)
        if ft:
            f = ft[0]
            A.plot(xs, np.exp(-f['k_exp'] * xs), '-', color=col[d], label='exp fit')
            A.plot(xs, 1 - f['k_lin'] * xs, '--', color='0.4', label='linear fit')
            if np.isfinite(f['k_pred']): A.plot(xs, np.exp(-f['k_pred'] * xs), ':', color='k', label='pred from lowest x')
        A.set_yscale('log'); A.set_title(d); A.set_xlabel('x'); A.axhline(1, color='0.7', lw=0.6)
        if i == 0: A.set_ylabel('R = MSD_all / MSD_far'); A.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, 'closure_lnR_vs_x.png')); plt.close(fig)
    # parity
    fig, A = plt.subplots(figsize=(4.4, 4.4), dpi=130)
    for d in dops:
        pp = [p for p in preds if p['dopant'] == d and np.isfinite(p['R_pred'])]
        if pp: A.scatter([p['R_pred'] for p in pp], [p['R_meas'] for p in pp], color=col[d], label=d, s=22)
    lim = [0.2, 1.1]; A.plot(lim, lim, 'k-', lw=0.8); A.fill_between(lim, [l * 0.95 for l in lim], [l * 1.05 for l in lim], color='0.85', zorder=0)
    A.set_xlabel('R_pred = exp(-x V_A / V_fu)'); A.set_ylabel('R_meas'); A.legend(fontsize=7); A.set_title('closure (all T, tau)', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, 'closure_parity.png')); plt.close(fig)
    # far vs n_Li
    if ref in curves:
        cl, cq, P = curves[ref]
        fig, A = plt.subplots(figsize=(5, 4), dpi=130)
        nn = np.linspace(6.2, 7.0, 50)
        if cl is not None: A.plot(nn, np.polyval(cl, nn), '-', color='0.4', label=f'{a.host} host, linear')
        if cq is not None: A.plot(nn, np.polyval(cq, nn), '--', color='0.4', label=f'{a.host} host, quadratic')
        for d in dops + ['none']:
            rr = [r for r in mob if r['dopant'] == d and (r['T'], r['tau_ps']) == ref and r['D_far'] > 0]
            if rr: A.errorbar([r['n_Li_model'] for r in rr], [math.log(r['D_far']) for r in rr], [r['msd_far_se'] / r['msd_far'] for r in rr], fmt='o',
                              color=col.get(d, 'k'), label=d if d != 'none' else 'pure', capsize=2)
        A.set_xlabel('n_Li = 7 - q x'); A.set_ylabel('ln D_far (A^2/ps)'); A.legend(fontsize=7)
        A.set_title('collapse test (T = %s K, tau = %s ps)' % ref, fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, 'far_vs_nLi.png')); plt.close(fig)
    # transfer parity
    fig, A = plt.subplots(figsize=(4.4, 4.4), dpi=130)
    for d in dops:
        q = [r for r in tr if r['dopant'] == d]
        if q: A.scatter([r['D_all_pred_lin'] for r in q], [r['D_all_obs'] for r in q], color=col[d], label=d + (' (LOO)' if d == a.host else ''), s=22)
    allv = [r['D_all_obs'] for r in tr] + [r['D_all_pred_lin'] for r in tr]
    if allv:
        lim = [min(allv) * 0.9, max(allv) * 1.1]; A.plot(lim, lim, 'k-', lw=0.8); A.fill_between(lim, [l * 0.9 for l in lim], [l * 1.1 for l in lim], color='0.85', zorder=0)
        A.set_xscale('log'); A.set_yscale('log')
    A.set_xlabel('D_all predicted = D_vac(n_Li) R_pred'); A.set_ylabel('D_all measured'); A.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, 'transfer_parity.png')); plt.close(fig)


# ---------------------------------------------------------------- run
def analyse(a, plots=True):
    mp = load_map(a.map)
    rad, whole, msd05 = load_sources(mp, a)
    gone = sorted({k[0] for k in rad} - set(mp))
    if gone: warn(f'folders in radius_summary.csv but not in the map (ignored): {gone}')
    mp = {f: m for f, m in mp.items() if f in whole}
    if not any(m['dopant'].lower() == 'none' for m in mp.values()): warn('no pure folder with data: D_vac(7) and the host curve lack the x = 0 point')
    va = step1(mp, rad, whole, a)
    vv = [r for r in va if r.get('valid')]
    if vv:
        nd = sum(1 for r in vv if np.isfinite(r['V_diff_pct']) and abs(r['V_diff_pct']) > 30)
        nn = sum(1 for r in vv if r['V_A_free'] <= 0)
        warn(f'S1: of {len(vv)} valid fits, V_A model-free and sigmoid differ by > 30 % in {nd}; model-free V_A <= 0 in {nn}. '
             f'V_A used = {a.va_source} (see V_diff_pct in va_table.csv; --va-source sig|free)')
    mob = step2(mp, whole, msd05)
    fits, preds = step3(mp, mob, va, a)
    tr, curves = step4(mp, mob, preds, a)
    ref = ref_case({(r['T'], r['tau_ps']) for r in mob})
    os.makedirs(a.out, exist_ok=True)
    for name, rows in (('va_table', va), ('mobility_table', mob), ('closure', fits), ('closure_pred', preds), ('transfer', tr)): write_csv(os.path.join(a.out, name + '.csv'), rows)
    if plots and ref: figures(a, mp, va, mob, fits, preds, tr, curves, ref)
    return dict(mp=mp, va=va, mob=mob, fits=fits, preds=preds, tr=tr, curves=curves, ref=ref)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--map', default='doping_map.csv')
    ap.add_argument('--radius', help='radius_summary.csv of build_doc.py')
    ap.add_argument('--root', default='.')
    ap.add_argument('--msd05-dir'); ap.add_argument('--whole-dir')
    ap.add_argument('--T', type=int, nargs='*'); ap.add_argument('--tau', type=float, nargs='*')
    ap.add_argument('--vfu', type=float, default=VFU)
    ap.add_argument('--host', default='Al')
    ap.add_argument('--va-all', action='store_true')
    ap.add_argument('--va-source', choices=['free', 'sig'], default='free', help='V_A used for the prediction: model-free (free row V_A3, plan default) or sigmoid analytic')
    ap.add_argument('--tol', type=float, default=0.05); ap.add_argument('--tol-collapse', type=float, default=0.10)
    ap.add_argument('--out', default='doping_model')
    ap.add_argument('--selftest', action='store_true')
    return ap


def main():
    a = build_parser().parse_args()
    if a.selftest: selftest()
    try: R = analyse(a)
    except InputError as e: sys.exit(f'error: {e}')
    if not R['ref']: sys.exit('error: no (T, tau) with data. S2 needs whole_msd/<folder>/whole_msd.csv for every folder in the map (incl. the pure one) and msd05/<folder>/msd_shell.csv for every doped one; see the warnings above.')
    print_s1(R['va'], R['ref']); print_s2(R['mob'], R['ref'])
    gate(R['fits'], R['preds'], R['tr'], R['mp'], a)
    print(f'\n[+] {a.out}/ va_table.csv mobility_table.csv closure.csv closure_pred.csv transfer.csv radius_vs_x.png closure_lnR_vs_x.png closure_parity.png far_vs_nLi.png transfer_parity.png')


# ---------------------------------------------------------------- selftest
def _solve_r0(V, A=0.8, w=0.7):
    from scipy.optimize import brentq
    return brentq(lambda r0: sig_VA(A, r0, w, 1.0) - V, 0.5, 20)


def synth(root, form='exp', shift=None, V=None, noise=0.003, seed=0, kfac=1.0, skip_msd05=(), few_far=()):
    """Truth: ln D_vac(n) = ln D0 + g (n - 7); R(x) = exp(-x V / V_fu) ('exp'), 1 - kfac x V / V_fu ('lin'), or R(x_lowest) for all x ('min')."""
    import shutil
    for d_ in ('whole_msd', 'msd05'): shutil.rmtree(os.path.join(root, d_), ignore_errors=True)
    rng = np.random.default_rng(seed)
    V = V or {'Al': 300., 'Ga': 250., 'Mg': 350.}
    qd = {'Al': 3, 'Ga': 3, 'Mg': 2}
    D0, g, T = 0.02, 1.2, 1000
    shift = shift or {}
    mp, rad = [('pure', 'none', 0, 0, 0, 16)], []
    os.makedirs(os.path.join(root, 'whole_msd', 'pure'), exist_ok=True)
    hdr = 'T,tau_ps,n_Li,msd_A2,msd_se,blocks,n_dopant,n_fu,t_start_ps,t_end_ps\n'
    wh = {}

    def nz(): return 1 + noise * rng.standard_normal()
    folders = [('pure', 'none', 0)] + [(f'{k}{d}-doped', d, k) for d in qd for k in (1, 2, 3)]
    for f, d, k in folders:
        x = k / 16
        q = qd.get(d, 0)
        n = 7 - q * x
        os.makedirs(os.path.join(root, 'whole_msd', f), exist_ok=True)
        wl, ml = [], []
        for tau in (5.0, 10.0):
            far = 6 * tau * D0 * math.exp(g * (n - 7) + shift.get(d, 0.0)) * nz()
            if d == 'none': R = 1.0
            elif form == 'exp': R = math.exp(-x * V[d] / VFU)
            elif form == 'lin': R = max(1 - kfac * x * V[d] / VFU, 0.05)
            else: R = math.exp(-(1 / 16) * V[d] / VFU)
            allm = far * R * nz() if d != 'none' else far
            wl.append(f'{T},{tau},{round(16 * n)},{allm},{0.004 * allm},5,{k},16,100,200')
            for sh, mult in (('0-3', 0.5), ('3-5', 0.8), ('5-7', 0.95), ('>7', 1.0)):
                ml.append(f'{T},any,{tau},{sh},{3 if sh != ">7" else (4 if f in few_far else 60)},{3 if sh != ">7" else (4 if f in few_far else 60)},start,1,0,{far * mult},{0.004 * far},1,0,5')
        with open(os.path.join(root, 'whole_msd', f, 'whole_msd.csv'), 'w') as fh: fh.write(hdr + '\n'.join(wl) + '\n')
        if d != 'none' and f not in skip_msd05:
            os.makedirs(os.path.join(root, 'msd05', f), exist_ok=True)
            with open(os.path.join(root, 'msd05', f, 'msd_shell.csv'), 'w') as fh:
                fh.write('T,dopant,tau_ps,shell,n_Li,n_Li_used,own,density_ratio,density_ratio_se,msd_A2,msd_se,ratio_to_far,ratio_se,blocks\n' + '\n'.join(ml) + '\n')
        mp.append((f, d, q, x, k, 16)) if d != 'none' else None
        if d != 'none':
            for tau in (5.0, 10.0):
                r0 = _solve_r0(V[d])
                base = dict(level='folder', name=f, T=T, tau_ps=tau, dopant_dopant_min_A=18.0, half_box_A=13.0, r_p99_A=9.0)
                rad.append(dict(base, model='sigmoid', A=0.8, r0=r0, w=0.7, B=1.0, r50=r0, r90=r0 + 1.5, r95=r0 + 2.1, V_A3='', r_max_used=''))
                rad.append(dict(base, model='free', A='', r0='', w='', B='', r50='', r90='', r95='', V_A3=V[d], r_eff=(3 * V[d] / (4 * math.pi)) ** (1 / 3), r_max_used=10.0))
    with open(os.path.join(root, 'doping_map.csv'), 'w') as fh:
        fh.write('folder,dopant,q,x,n_dopant,n_fu\n' + '\n'.join(f'{f},{d},{q},{k / 16 if d != "none" else 0},{nd},{nf}' for f, d, q, k, nd, nf in
                                                                   [(m[0], m[1], m[2], m[4], m[4], m[5]) for m in mp]) + '\n')
    cols = ['level', 'name', 'T', 'tau_ps', 'model', 'A', 'r0', 'w', 'B', 'r50', 'r90', 'r95', 'V_A3', 'r_eff', 'r_max_used', 'dopant_dopant_min_A', 'half_box_A', 'r_p99_A']
    with open(os.path.join(root, 'radius_summary.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval=''); w.writeheader(); w.writerows(rad)


def _args(root, **kw):
    a = build_parser().parse_args(['--map', os.path.join(root, 'doping_map.csv'), '--radius', os.path.join(root, 'radius_summary.csv'),
                                   '--root', root, '--out', os.path.join(root, 'out')])
    for k, v in kw.items(): setattr(a, k, v)
    return a


def selftest():
    fails = []

    def chk(name, ok, info=''):
        print(f'  [{"ok" if ok else "FAIL"}] {name} {info}')
        if not ok: fails.append(name)

    # 1  S1: analytic V_A of the sigmoid vs numerical integral of (1 - y) 4 pi r^2
    r = np.linspace(0, 40, 400001)
    for A, r0, w in ((0.8, 4.0, 0.7), (0.5, 3.0, 0.5), (0.9, 5.0, 1.0)):
        num = float(np.trapz(A / (1 + np.exp((r - r0) / w)) * 4 * np.pi * r ** 2, r)) if hasattr(np, 'trapz') else float(np.trapezoid(A / (1 + np.exp((r - r0) / w)) * 4 * np.pi * r ** 2, r))
        chk('S1 sigmoid V_A analytic vs numeric', abs(sig_VA(A, r0, w, 1) / num - 1) < 0.05, f'(A={A}, r0={r0}, w={w}: {sig_VA(A, r0, w, 1):.0f} vs {num:.0f})')
    with tempfile.TemporaryDirectory() as t:
        # 2  S3 discriminates
        synth(t, 'exp'); R = analyse(_args(t), plots=False)
        chk('S3 exp-generated: exp beats linear everywhere', all(f['better'] == 'exp' for f in R['fits']), f'({len(R["fits"])} fits)')
        chk('S3 exp-generated: R_pred passes everywhere', all(p['pass_pred'] for p in R['preds']), f'({len(R["preds"])} cases)')
        chk('S3 exp-generated: V_A from k_exp within 10 % of the truth', all(abs(f['VA_from_k_exp'] / {'Al': 300, 'Ga': 250, 'Mg': 350}[f['dopant']] - 1) < 0.10 for f in R['fits']))
        chk('S3 exp-generated: intercept of ln R consistent with 0', all(abs(f['a_lnR']) < 3 * f['a_lnR_se'] + 0.01 for f in R['fits']))
        synth(t, 'min'); R = analyse(_args(t), plots=False)
        chk('S3 min-rule data: R_pred must fail at x > lowest', all(not p['pass_pred'] for p in R['preds']), f'({sum(p["pass_pred"] for p in R["preds"])} of {len(R["preds"])} pass)')
        synth(t, 'lin', V={'Al': 800., 'Ga': 700., 'Mg': 900.}); R = analyse(_args(t), plots=False)
        chk('S3 linear-generated: linear beats exp everywhere', all(f['better'] == 'lin' for f in R['fits']))
        # 3  S4
        synth(t, 'exp'); R = analyse(_args(t), plots=False)
        tr = R['tr']
        chk('S4 Ga, Mg D_far collapse onto the Al curve', all(r['collapse_ok'] for r in tr), f'(max |dev| {100 * max(abs(r["dev_far_lin"]) for r in tr):.1f} %)')
        ot = [r for r in tr if not r['host']]
        chk('S4 transfer: q = 2 and q = 3 dopants predicted within 5 %', all(abs(r['dev_all_lin']) < 0.05 for r in ot), f'(max {100 * max(abs(r["dev_all_lin"]) for r in ot):.1f} %)')
        chk('S4 Al leave-one-out within 5 %', all(abs(r['dev_all_lin']) < 0.05 for r in tr if r['host']))
        synth(t, 'exp', shift={'Ga': 0.15}); R = analyse(_args(t), plots=False)
        ga = [r for r in R['tr'] if r['dopant'] == 'Ga']; mg = [r for r in R['tr'] if r['dopant'] == 'Mg']
        chk('S4 dopant-specific D_vac shift (Ga +15 %) is flagged', all(not r['collapse_ok'] for r in ga) and all(r['collapse_ok'] for r in mg))
        # 7  bad inputs
        synth(t, 'exp', skip_msd05=('2Mg-doped',), few_far=('3Ga-doped',)); R = analyse(_args(t), plots=False)
        chk('bad: missing msd05 file -> no crash, folder skipped', not any(r['folder'] == '2Mg-doped' for r in R['mob']))
        chk('bad: too few far Li -> flagged', all(r['far_flag'] for r in R['mob'] if r['folder'] == '3Ga-doped'))
        synth(t, 'exp')
        s = open(os.path.join(t, 'doping_map.csv')).read().replace('1Al-doped,Al,3,0.0625,1,16', '1Al-doped,Al,3,0.0625,2,16')
        open(os.path.join(t, 'doping_map.csv'), 'w').write(s)
        try: analyse(_args(t), plots=False); chk('bad: n_dopant / n_fu != x', False)
        except InputError as e: chk('bad: n_dopant / n_fu != x -> clear message', '1Al-doped' in str(e) and 'n_dopant / n_fu' in str(e))
        synth(t, 'exp')
        s = open(os.path.join(t, 'doping_map.csv')).read().replace('3Mg-doped,Mg,2,0.1875,3,16\n', '')
        open(os.path.join(t, 'doping_map.csv'), 'w').write(s)
        R = analyse(_args(t), plots=False)
        chk('bad: folder missing from the map -> ignored with a warning', not any(r['folder'] == '3Mg-doped' for r in R['mob']))
        synth(t, 'exp'); R = analyse(_args(t), plots=True)
        chk('figures written', all(os.path.exists(os.path.join(t, 'out', f)) for f in ('radius_vs_x.png', 'closure_parity.png', 'closure_lnR_vs_x.png', 'far_vs_nLi.png', 'transfer_parity.png')))
        gate(R['fits'], R['preds'], R['tr'], R['mp'], _args(t))
    print('\nselftest: ' + ('PASS' if not fails else f'FAIL {fails}') + '  (synthetic data: tests the code, not the physics)')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
