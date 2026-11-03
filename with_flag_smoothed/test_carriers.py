#!/usr/bin/env python3
"""test_carriers.py - synthetic checks for carriers.py (no MD data needed):  python test_carriers.py   (or pytest).
Toy lattice: 8x8x8 simple-cubic sites (2 A), two 'dopants', Li hop at random but never out of / into sites within 3 A of a dopant (trapped)."""
import argparse, os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
if '--scripts' not in sys.argv: sys.argv += ['--scripts', os.path.join(HERE, 'with_flag')]
import numpy as np
import carriers as ca

L = np.array([16.0] * 3)
mi = lambda d: d - L * np.rint(d / L)


def toy(F=2500, N=250, seed=1):
    rng = np.random.default_rng(seed)
    n, sp = 8, 2.0
    g = np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)]) * sp
    S = len(g)
    kind = np.where((g.sum(1) / sp) % 2 == 0, '24d', '96h')
    dop = np.array([[0, 0, 0], [8, 8, 8]], float)
    dd = np.min([np.linalg.norm(mi(g - d), axis=1) for d in dop], axis=0)
    trap = dd < 3.0
    nbr = [np.where((np.linalg.norm(mi(g - g[s]), axis=1) < 2.1) & (np.arange(S) != s))[0] for s in range(S)]
    occ = np.zeros(S, int); pos = rng.choice(S, N, replace=False); occ[pos] = 1
    state = np.empty((F, N), int); state[0] = pos
    for f in range(1, F):
        cur = state[f - 1].copy()
        for i in rng.permutation(N):
            s = cur[i]
            if trap[s] or rng.random() > 0.08: continue
            t = rng.choice(nbr[s])
            if occ[t] == 0 and not trap[t]: occ[s] = 0; occ[t] = 1; cur[i] = t
        state[f] = cur
    li = np.empty((F, N, 3)); li[0] = g[state[0]]
    for f in range(1, F): li[f] = li[f - 1] + mi(g[state[f]] - g[state[f - 1]])
    li += rng.normal(0, 0.12, li.shape)
    dops = [dict(label=f'X{i}', x=np.tile(d, (F, 1))) for i, d in enumerate(dop)]
    return g, kind, state, li, dops, np.tile(L, (F, 1))


def args(**kw):
    d = dict(car_win_ps=10.0, car_ks=[1, 2, 3, 5], car_kdef=2, car_blocks=5, car_every=5, car_nn=2.1, car_edge_max=2.1, car_bond_cut=2.1)
    d.update(kw)
    return argparse.Namespace(**d)


def get(rows, metric, shell='all', param=''):
    for r in rows:
        if r['metric'] == metric and r['shell'] == shell and r['param'] == param: return r['value']
    raise KeyError((metric, shell, param))


def run(**kw):
    g, kind, state, li, dops, Lf = toy(**kw)
    rows = ca.analyse(li, state, g, kind, dops, L, np.zeros(3), Lf, 0.1, 16, [2.5, 4.0, 5.5], 800, args())
    return rows


def test_trapped_shell_is_less_mobile():
    rows = run()
    inner, far = get(rows, 'f_mobile', '0-2.5A', '2'), get(rows, 'f_mobile', '>5.5A', '2')
    assert inner < 0.05 and far > 0.6 and inner < far
    assert get(rows, 'D_all', '0-2.5A') < 0.05 * get(rows, 'D_all', '>5.5A')


def test_partition_identity():
    rows = run()
    for sh in ('0-2.5A', '2.5-4A', '4-5.5A', '>5.5A', 'all'):
        f, dm, dt_, da = (get(rows, 'f_mobile', sh, '2'), get(rows, 'D_mobile', sh, '2'), get(rows, 'D_trapped', sh, '2'), get(rows, 'D_all', sh))
        if np.isfinite(dm) and np.isfinite(dt_): assert abs(f * dm + (1 - f) * dt_ - da) < 1e-6 * da + 1e-12


def test_k_sweep_monotone():
    rows = run()
    v = [get(rows, 'f_mobile', 'all', str(k)) for k in (1, 2, 3, 5)]
    assert all(v[i] >= v[i + 1] - 1e-12 for i in range(3))


def test_hop_rate_grows_with_empty_neighbours():
    rows = run()
    v = [get(rows, 'hop_rate_per_Li_ps', 'all', p) for p in ('0', '1', '2', '3+')]
    assert v[0] < v[1] < v[2] < v[3]


def test_empty_site_fraction_matches_filling():
    rows = run()
    assert abs(get(rows, 'empty_site_frac', 'all', 'all') - (1 - 250 / 512)) < 1e-6


def test_percolation_periodic():
    n, sp = 6, 2.0
    g = np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)]) * sp
    Lb = np.array([n * sp] * 3)
    D = np.linalg.norm((g[:, None] - g[None]) - Lb * np.rint((g[:, None] - g[None]) / Lb), axis=-1)
    ea, eb = np.where(np.triu((D <= 2.1) & (D > 0), 1))
    keep = np.ones(len(g), bool)
    assert ca.percolate(keep, ea, eb, g, Lb)[2] == 3
    slab = keep & ~(np.isclose(g[:, 0], 0.0))                  # delete one plane: no path round the box along x any more
    sel = slab[ea] & slab[eb]
    assert ca.percolate(slab, ea[sel], eb[sel], g, Lb)[2] < 3
    cut = keep & ~(np.isclose(g[:, 0], 0.0) | np.isclose(g[:, 0], 6.0))
    sel = cut[ea] & cut[eb]
    assert ca.percolate(cut, ea[sel], eb[sel], g, Lb)[1] < len(g)   # two planes: the cluster is smaller than the lattice
    assert ca.percolate(keep, ea[:0], eb[:0], g, Lb)[1] == 1         # no bonds: giant cluster = 1 site


def test_arrhenius_recovers_activation_energy():
    kB = ca.KB
    res = {'s': {T: [dict(shell='all', metric='D_all', param='', value=1e-5 * np.exp(-0.30 / (kB * T)), se=0, n=0)] for T in (600, 700, 800, 900)}}
    r = ca.arrhenius(res, 2)
    assert len(r) == 1 and abs(r[0]['Ea_eV'] - 0.30) < 1e-6


def test_radius_table_filters_invalid_fits():
    try:
        import build_doc as bd, pandas as pd
    except Exception as e:                                       # no python-docx here
        print('skip radius table test:', e); return
    rows = []
    for nm, A in (('Al', 0.5), ('Ca', 0.05)):
        for T in (800, 900):
            b = dict(level='element', name=nm, T=T, tau_ps=5.0)
            rows += [dict(b, model='free', r_max_used=9.0), dict(b, model='sigmoid', A=A, r95=4.0 + T / 1000, r50=2.0, B=1.0)]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'radius_summary.csv'); pd.DataFrame(rows).to_csv(p, index=False)
        head, out = bd.radius_sigmoid_table(p)
    assert [r[0] for r in out] == ['Al'] and out[0][1] == '2'    # Ca: A < 0.1 -> not valid


if __name__ == '__main__':
    for k, v in list(globals().items()):
        if k.startswith('test_'):
            v(); print('ok  ', k)
