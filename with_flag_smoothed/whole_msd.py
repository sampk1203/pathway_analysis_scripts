#!/usr/bin/env python3
"""whole_msd.py - MSD of ALL Li (no shells, no dopant needed), per folder, T and tau. Works on the pure folder too.

python whole_msd.py {pure 1Al-doped 2Al-doped 3Al-doped} {1Ga-doped 2Ga-doped} --T 800 900 --lags 1 5 10
python whole_msd.py {pure 1Al-doped} --tmin 100 --tmax 200      (window 100-200 ps)
python whole_msd.py --selftest

Groups use the build_doc.py syntax. Tokens like Al=Li24d are accepted and ignored; elements=Li,La,Zr,Al,O sets the element of
LAMMPS type 1..N for that group. Quote braces if the shell eats them. Put options after the groups.

What is computed (same loader, drift removal, tau frames and block jackknife as 05_msd_by_shell.py):
    MSD_all(tau) = < |r(t0 + tau) - r(t0)|^2 >    over every Li and every time origin t0
(unwrapped positions, framework-COM drift removed). Time origins are split into --blocks contiguous blocks; msd_se = delete-one-block
jackknife. tau frames = max(1, round(tau / frame spacing)), as 05, so the numbers sit at the same tau as msd05/*/msd_shell.csv.
MSD at one tau is not D: the printout gives the log-log slope over the taus (about 1 = diffusive; --lags needs >= 2 values).

--collective (optional, off by default): also |sum_i dr_i|^2 / N_Li over all Li (displacement of the Li centre of mass, per Li).
Equals MSD_all if Li move independently; collective/MSD_all = D_sigma/D* = 1/H_R in the long-time limit. Noisy: one sample per time origin.

Output: <outdir>/<system>/whole_msd.csv   (default outdir = whole_msd in the current directory; <system> = folder name or --name)
    T, tau_ps, n_Li, msd_A2, msd_se, blocks, n_dopant, n_fu, t_start_ps, t_end_ps, [msd_coll_A2, msd_coll_se]
n_dopant = atoms of the dopant types; n_fu = La atoms / 3 (formula units in the cell, NaN if no La); both read from the dump, for the
n_dopant / n_fu = x check in doping_model.py.
Cache: <outdir>/<system>/cache/whole_msd_T<T>K.json (key = lags, blocks, tmin, dump file + mtime, settings); --recompute ignores it.
Needs llzo_io.py in ./with_flag (or --scripts DIR), numpy; PyYAML only for the selftest.
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

SPEC_RX = re.compile(r'^[A-Z][a-z]?=[A-Za-z0-9_-]+$')          # Al=Li24d (build_doc site spec): ignored here
ELEM_RX = re.compile(r'^elements=([A-Za-z,]+)$')


def warn(msg): print('warning: ' + msg, file=sys.stderr)


def parse_groups(items):
    """-> [{'folders': [...], 'elements': [...] | None}] (same syntax as build_doc.py)"""
    groups, cur = [], None

    def add(w):
        m = ELEM_RX.match(w)
        if m: cur['elements'] = [x.capitalize() for x in m.group(1).split(',') if x]
        elif SPEC_RX.match(w): pass
        else: cur['folders'].append(w)

    for tok in items:
        for part in re.split(r'([{}])', tok):
            if part == '{':
                if cur is not None: sys.exit('nested { not allowed')
                cur = {'folders': [], 'elements': None}
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


def md_dump(folder, T):
    """<folder>/MD-<T>/ dump: prefer a name containing '<T>K', else the only/first *.lammpstrj. None if absent."""
    hit = sorted(glob.glob(os.path.join(folder, f'MD-{T}', '*.lammpstrj')))
    pref = [h for h in hit if f'{T}K' in os.path.basename(h)]
    return (pref or hit or [None])[0]


def md_temps(folder):
    out = []
    for d in glob.glob(os.path.join(folder, 'MD-*')):
        t = os.path.basename(d)[3:]
        if t.isdigit() and md_dump(folder, int(t)): out.append(int(t))
    return sorted(out)


def whole_msd(li, lag_frames, nb, collective=False):
    """li (F,N,3) unwrapped, drift removed. -> {lag: dict(msd, se[, coll, coll_se])} or None where the lag leaves too few origins."""
    F, N = li.shape[:2]
    out = {}
    for lag in lag_frames:
        if lag >= F - nb:
            out[lag] = None
            continue
        dr = li[lag:] - li[:-lag]                                    # (F-lag, N, 3)
        n0 = F - lag
        bid = lio.block_id(np.arange(n0), n0, nb)                    # same origin blocks as 05 (labels = sh[:-lag])
        C = np.bincount(bid, minlength=nb) * float(N)
        S = np.bincount(bid, weights=(dr ** 2).sum(axis=(1, 2)), minlength=nb)
        m, se = lio.jackknife(lambda s_, c_: s_ / c_, S, C)
        r = dict(msd=float(m), se=float(se))
        if collective:
            Sc = np.bincount(bid, weights=(dr.sum(axis=1) ** 2).sum(axis=1), minlength=nb)
            mc, sec = lio.jackknife(lambda s_, c_: s_ / c_, Sc, C)
            r.update(coll=float(mc), coll_se=float(sec))
        out[lag] = r
    return out


def compute(folder, cfg, T, tmin, tmax, lags_ps, nb, collective):
    dump = md_dump(folder, T) or lio.dump_path(cfg, T)
    traj = lio.prepare(cfg, dump, tmin, verbose=False)
    if tmax is not None:                                              # window = tmin .. tmax (ps after MD start)
        k = traj.time_ps <= tmax + 1e-9
        if k.sum() < 10: raise ValueError(f'only {int(k.sum())} frames with t <= {tmax} ps')
        traj = lio.Trajectory(steps=traj.steps[k], time_ps=traj.time_ps[k], lo=traj.lo, L=traj.L[k], types=traj.types, ids=traj.ids, pos=traj.pos[k])
    drift = lio.framework_drift(cfg, traj)
    lm = lio.li_mask(cfg, traj)
    li = traj.pos[:, lm, :] - drift[:, None, :]
    dt = float(np.median(np.diff(traj.time_ps)))
    lag_frames = [max(1, int(round(x / dt))) for x in lags_ps]
    res = whole_msd(li, lag_frames, nb, collective)
    n_dop = int(np.isin(traj.types, cfg['dopant_types']).sum())
    n_la = int(np.isin(traj.types, [t for t, el in cfg['types'].items() if el == 'La']).sum())
    n_fu = n_la / 3.0 if n_la else float('nan')
    rows = []
    for lag in lag_frames:
        r = res[lag]
        if r is None:
            warn(f'{fname(folder)} T={T}: tau = {lag * dt:g} ps needs more than {nb} origin blocks of frames (F = {li.shape[0]}): skipped')
            continue
        row = dict(T=int(T), tau_ps=lag * dt, n_Li=int(li.shape[1]), msd_A2=r['msd'], msd_se=r['se'], blocks=nb, n_dopant=n_dop, n_fu=n_fu,
                   t_start_ps=float(traj.time_ps[0]), t_end_ps=float(traj.time_ps[-1]))
        if collective: row.update(msd_coll_A2=r['coll'], msd_coll_se=r['coll_se'])
        rows.append(row)
    return rows


def cache_key(cfg, dump, tmin, tmax, lags, nb, collective):
    return repr((tuple(lags), nb, tmin, tmax, collective, dump, os.path.getmtime(dump), cfg.get('run_steps'), cfg.get('dt_ps'), cfg.get('tmin_ps'),
                 tuple(sorted(cfg['types'].items())), tuple(cfg['dopant_types']), VERSION))


def _task(a):
    folder, cfg, T, tmin, tmax, lags, nb, coll, outp, key = a
    try:
        rows = compute(folder, cfg, T, tmin, tmax, lags, nb, coll)
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        with open(outp, 'w') as fh: json.dump(dict(key=key, rows=rows), fh)
    except (Exception, SystemExit) as e:
        return f'{folder} T={T}: {e}'


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
                key = cache_key(cfg, dump, a.tmin, a.tmax, a.lags, a.blocks, a.collective)
                outp = os.path.join(odir, 'cache', f'whole_msd_T{T}K.json')
                plan.setdefault((f, cfg['name']), []).append(outp)
                ok = False
                if os.path.exists(outp) and not a.recompute:
                    try: ok = json.load(open(outp))['key'] == key
                    except Exception: ok = False
                if not ok: todo.append((f, cfg, T, a.tmin, a.tmax, a.lags, a.blocks, a.collective, outp, key))
    print(f'[whole_msd] {len(todo)} (folder, T) to compute, {sum(len(v) for v in plan.values()) - len(todo)} cached')
    if a.jobs > 1 and len(todo) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=a.jobs) as ex: errs = list(ex.map(_task, todo))
    else:
        errs = [_task(t) for t in todo]
    for e in errs:
        if e: warn(e)
    out = {}
    for (f, name), outs in plan.items():
        rows = []
        for o in outs:
            if os.path.exists(o): rows += json.load(open(o))['rows']
        if not rows:
            warn(f'no result for {f}'); continue
        rows.sort(key=lambda r: (r['T'], r['tau_ps']))
        odir = os.path.join(a.outdir, name)
        with open(os.path.join(odir, 'whole_msd.csv'), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        out[(f, name)] = rows
        report(name, rows, a.collective)
        print(f'[+] {os.path.join(odir, "whole_msd.csv")}')
    return out


def report(name, rows, collective):
    print(f'\n=== {name} ===   n_Li = {rows[0]["n_Li"]}  n_dopant = {rows[0]["n_dopant"]}  n_fu = {rows[0]["n_fu"]:g}  window {min(r["t_start_ps"] for r in rows):g}-{max(r["t_end_ps"] for r in rows):g} ps')
    for T in sorted({r['T'] for r in rows}):
        rr = [r for r in rows if r['T'] == T]
        s = f'  T = {T} K   ' + '   '.join(f'tau {r["tau_ps"]:g} ps: {r["msd_A2"]:.3f} +/- {r["msd_se"]:.3f} A^2' for r in rr)
        if len(rr) >= 2:
            sl = np.polyfit(np.log([r['tau_ps'] for r in rr]), np.log([r['msd_A2'] for r in rr]), 1)[0]
            s += f'   log-log slope {sl:.2f}' + ('' if abs(sl - 1) <= 0.2 else '  [WARN: not ~1, not diffusive]')
        print(s)
        if collective:
            print('           collective/MSD_all: ' + '   '.join(f'{r["msd_coll_A2"] / r["msd_A2"]:.2f}' for r in rr))


# ---------------------------------------------------------------- selftest
def _fake_dump(path, rng, n_li, with_dop, D, F=400, L=26.0):
    """Li random walk, MSD = 6 D tau exactly in expectation (frame spacing 1 ps); framework fixed; one Al if with_dop."""
    els = ['Li', 'La', 'Zr'] + (['Al'] if with_dop else []) + ['O']
    n = {'Li': n_li, 'La': 24, 'Zr': 16, 'Al': 1, 'O': 96}
    types = np.concatenate([np.full(n[e], i + 1) for i, e in enumerate(els)])
    N = len(types)
    pos0 = rng.uniform(0, L, (N, 3))
    li = pos0[:n_li][None] + np.concatenate([np.zeros((1, n_li, 3)), np.cumsum(rng.normal(0, np.sqrt(2 * D), (F - 1, n_li, 3)), axis=0)])
    pos = np.repeat(pos0[None], F, axis=0)
    pos[:, :n_li] = li
    with open(path, 'w') as fh:
        for k in range(F):
            fh.write(f'ITEM: TIMESTEP\n{k * 1000}\nITEM: NUMBER OF ATOMS\n{N}\nITEM: BOX BOUNDS pp pp pp\n')
            fh.write(''.join(f'0 {L}\n' for _ in range(3)) + 'ITEM: ATOMS id type xu yu zu\n')
            fh.write(''.join(f'{i + 1} {types[i]} {pos[k, i, 0]:.5f} {pos[k, i, 1]:.5f} {pos[k, i, 2]:.5f}\n' for i in range(N)))
    return els


def selftest():
    import tempfile
    rng = np.random.default_rng(1)
    D, fails = 0.5, []
    with tempfile.TemporaryDirectory() as tmp:
        for name, dop in (('fake_dop', True), ('fake_pure', False)):
            d = os.path.join(tmp, name); os.makedirs(d)
            els = _fake_dump(os.path.join(d, 'dump_lmp_800K.lammpstrj'), rng, 64, dop, D)
            open(os.path.join(d, 'element_list'), 'w').write(' '.join(els) + '\n')
            open(os.path.join(d, 'analysis_defaults.yaml'), 'w').write('dt_ps: 0.001\nrun_steps: 399000\ntmin_ps: 0\n')
        sys.argv = [sys.argv[0]]
        ap = build_parser()
        a = ap.parse_args(['--outdir', os.path.join(tmp, 'out'), '--lags', '1', '5', '10', '--collective'])
        groups = [dict(folders=[os.path.join(tmp, 'fake_dop'), os.path.join(tmp, 'fake_pure')], elements=None)]
        res = run(groups, a)
        for (f, name), rows in res.items():
            for r in rows:
                true = 6 * D * r['tau_ps']
                err = abs(r['msd_A2'] - true) / true
                ok = err < 0.05 and np.isfinite(r['msd_se']) and r['msd_se'] > 0
                if r['tau_ps'] == 1.0:                                      # collective: 400 origins at tau = 1 -> ~4 % noise
                    ok = ok and abs(r['msd_coll_A2'] / r['msd_A2'] - 1) < 0.15
                print(f'  [{"ok" if ok else "FAIL"}] {name} tau {r["tau_ps"]:g}: MSD {r["msd_A2"]:.3f} vs {true:.3f} ({100 * err:.1f} %), se {r["msd_se"]:.3f}')
                if not ok: fails.append((name, r['tau_ps']))
            if name == 'fake_dop' and rows[0]['n_dopant'] != 1: fails.append((name, 'n_dopant'))
            if name == 'fake_pure' and rows[0]['n_dopant'] != 0: fails.append((name, 'n_dopant'))
            if abs(rows[0]['n_fu'] - 8.0) > 1e-9: fails.append((name, 'n_fu'))
        if len(res) != 2: fails.append(('folders', len(res)))
        # cache: second run must read, not recompute
        print('second run (expect 0 to compute):')
        run(groups, a)
    print('\nselftest: ' + ('PASS' if not fails else f'FAIL {fails}') + '  (synthetic data: tests the code, not the physics)')
    sys.exit(1 if fails else 0)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('items', nargs='*', help='groups: {A B C} {D E}')
    ap.add_argument('--scripts', help='folder with llzo_io.py (default: ./with_flag next to this file)')
    ap.add_argument('--T', type=int, nargs='*', help='temperatures (default: every MD-<T>/ dump, else all dumps found)')
    ap.add_argument('--tmin', type=float, help='analysis window start, ps (default: tmin_ps of the settings)')
    ap.add_argument('--tmax', type=float, help='analysis window end, ps after MD start (default: end of the run); e.g. --tmin 100 --tmax 200')
    ap.add_argument('--lags', type=float, nargs='*', default=[1.0, 5.0, 10.0], help='tau values, ps (default as 05)')
    ap.add_argument('--blocks', type=int, default=lio.DEFAULT_BLOCKS, help='time blocks for the jackknife error bars')
    ap.add_argument('--outdir', default='whole_msd', help='output root; files go to <outdir>/<system>/')
    ap.add_argument('--jobs', type=int, default=1, help='parallel (folder, T) computations')
    ap.add_argument('--collective', action='store_true', help='also the collective (Li centre-of-mass) MSD per Li')
    ap.add_argument('--recompute', action='store_true', help='ignore cached results')
    ap.add_argument('--selftest', action='store_true', help='run on synthetic dumps with known D and exit')
    lio.add_config_args(ap)
    return ap


def main():
    a = build_parser().parse_args()
    if a.selftest: selftest()
    groups = parse_groups(a.items)
    if not groups: sys.exit('no groups given (python whole_msd.py {A B C} ...)')
    for g in groups: g['folders'] = expand(g['folders'])
    run(groups, a)


if __name__ == '__main__':
    main()
