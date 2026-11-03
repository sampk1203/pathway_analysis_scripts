#!/usr/bin/env python3
"""carriers.py - why is sigma low?  Four MD-only tests + an Arrhenius split, for every dopant folder (same groups as build_doc.py).

python carriers.py {A B C} {D E Ca=La elements=Li,La,Zr,Ca,O} --outdir carriers_out [--jobs 4]
(build_doc.py calls the same code and puts the figures + one summary table into the report; --no-carriers there skips it.)

NEEDS a finished run_pipeline.sh (04) in every folder: hops_cif/<name>/state_T<T>K.npz (committed site of every Li in every frame) and
sites.csv (+ barriers_T<T>K.csv if 04 ran with barriers). The dumps are read again for the Li / dopant positions. Same shells, dopant smoothing,
time window (--tmin), time blocks and delete-one-block jackknife as 04-09.

1  MOBILE vs TRAPPED Li.  The window is cut into non-overlapping windows of --car-win-ps. In every window a Li is MOBILE if it visits at least
   k OTHER sites (distinct; a back-and-forth between two sites is k = 1). Shell = shell of the Li at the start of the window (as 05, 'start').
   Per T and shell: f_mobile, D_all, D_mobile, D_trapped (Li-window MSD / 6 tau, tau = the window; cm2/s) for k in --car-ks (default 1 2 3 5; headline
   --car-kdef = 2).  f_mobile x D_mobile + (1 - f_mobile) x D_trapped = D_all by construction, so D_all is the quantity to compare with 03's D_tracer
   (printed, if 03 ran).  f_frozen_whole = Li that never change site in the whole window.  If f_mobile hardly moves with k, the split is not
   an artefact of the threshold.
2  VACANCIES / EMPTY SITES.  Li per formula unit (8 f.u. per conventional cell, so doping on La or Zr does not change the f.u. count), Li per site, the
   fraction of EMPTY sites (per site kind 24d / 96h / ..., per shell of the site around the nearest dopant), the number of empty sites among the
   neighbours (sites closer than --car-nn A) of every Li, and the committed-hop rate per Li-ps as a function of that number (0, 1, 2, 3+):
   flat -> hops are not vacancy-limited; steep -> they are.
3  PERCOLATION (periodic: a cluster spans only if its cycles wind round the box, rank 3 of the winding vectors).
   dynamic : graph of site pairs that hopped (jump <= --car-edge-max A) with pair rate >= m x median pair rate, m on a log grid;
             gamma_c_over_med = the largest m at which the network still spans all three axes (higher = more robust, nan = never spans,
             value at the top of the grid is a lower bound).
   static  : geometric bond graph (sites closer than --car-bond-cut A) after removing every site closer than r_block to a dopant (time-median
             distance); r_block_c = the smallest r_block at which the lattice stops spanning; giant-cluster fraction vs r_block.  Compare r_block_c with
             the radius of influence (sigmoid r95) in the report.  Also: the lattice without the 'frozen' sites (occupied > 50 % of the time and not one
             hop in or out of them).
4  SITE-ENERGY HETEROGENEITY.  F_i = -kT ln(occ_i / occ_ref) from the mean occupation of every site (reference = far sites of the same kind),
   sigma_F per shell (std of F_i over sites visited), sigma_F / kT, sigma_excess = sqrt(sigma^2 - sigma_far^2) (the far shell is the sampling-noise
   control), zwanzig = exp(-(sigma/kT)^2) (Gaussian energetic disorder, a ROUGH D/D0 estimate), the fraction of sites never visited, F vs distance
   profile, and the median site-to-site barrier per shell from 04 (barriers_T*.csv).  LIMIT: Li-Li exclusion biases occupancies; sigma_F is an
   EFFECTIVE site-energy spread, not a DFT energy (the dump has no per-atom energies).
5  ARRHENIUS SPLIT (needs >= 3 T): Ea and prefactor of f_mobile, D_mobile, D_all (per shell and 'all').  Higher Ea of D_mobile = landscape / trapping;
   higher Ea of f_mobile = thermally activated carriers; same Ea but lower prefactor = fewer carriers / poor percolation.

Outputs: <folder>/carriers/<name>/carriers_T<T>K.json (cache, --recompute ignores it), <outdir>/carriers_summary_allT.csv (every T), <outdir>/carriers_long.csv (system, T, shell, metric, param,
value, se, n; one row per number), <outdir>/arrhenius.csv, <outdir>/carriers_summary.csv and the figures carriers_*.png/.svg.
Errors: delete-one-block jackknife over --car-blocks time blocks (windows / frames, not Li, are the blocks). Time resolution = dump spacing.
"""
import argparse, csv, glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _scripts_dir():
    a = sys.argv
    for i, x in enumerate(a):
        if x == '--scripts' and i + 1 < len(a): return a[i + 1]
        if x.startswith('--scripts='): return x.split('=', 1)[1]
    return os.path.join(HERE, 'with_flag')


SCRIPTS = os.path.abspath(_scripts_dir())
for _p in (SCRIPTS, HERE):
    if _p not in sys.path: sys.path.insert(0, _p)
import numpy as np
import llzo_io as lio
import matplotlib
import matplotlib.pyplot as plt

KB = lio.KB_EV
NAN = float('nan')
CAR_VERSION = 'carriers-v1'
NEMPTY = 4                                   # hop rate vs empty neighbours: bins 0, 1, 2, 3+
RBLOCKS = (0.0, 1.5, 2.0, 2.5, 3.0) + tuple(float(x) for x in np.round(np.arange(3.5, 28.0 + 1e-9, 0.1), 1))
MULTS = np.geomspace(0.1, 10.0, 21)
UNIT = 1e-4                                  # 1 A^2/ps = 1e-4 cm^2/s


def warn(msg): print('warning: ' + msg, file=sys.stderr)
def fname(folder): return os.path.basename(os.path.normpath(folder))
def nat(s): return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]
def mi(d, L): return d - L * np.rint(d / L)


def add_args(ap):
    g = ap.add_argument_group('carrier analysis (carriers.py)')
    g.add_argument('--car-win-ps', type=float, default=10.0, help='mobile/trapped: window length tau, ps (a Li is judged inside one window)')
    g.add_argument('--car-ks', type=int, nargs='*', default=[1, 2, 3, 5], help='mobile = visits >= k other sites in the window; sweep')
    g.add_argument('--car-kdef', type=int, default=2, help='headline k (must be in --car-ks)')
    g.add_argument('--car-blocks', type=int, default=lio.DEFAULT_BLOCKS, help='jackknife time blocks')
    g.add_argument('--car-every', type=int, default=5, help='site-to-dopant distances every Nth frame (state / occupancy use every frame)')
    g.add_argument('--car-nn', type=float, default=2.6, help='neighbour sites of a Li = sites closer than this, A')
    g.add_argument('--car-edge-max', type=float, default=3.6, help='dynamic network: hopped pairs with jump <= this, A')
    g.add_argument('--car-bond-cut', type=float, default=2.6, help='static lattice: site pairs closer than this are bonded, A')
    g.add_argument('--car-hops', default='hops_cif', help='04 output folder inside every run folder')
    g.add_argument('--car-tref-only', action='store_true', help='figures only at the reference T (default: figures + summary at every T)')
    g.add_argument('--car-tref', type=int, help='T of the summary figures / table (default: highest T found in most systems)')
    return g


# ---------------------------------------------------------------- helpers: jackknife wrappers, percolation
def jk(fn, *blocks):
    return lio.jackknife(fn, *blocks)


def percolate(keep, ea, eb, P, L):
    """Periodic percolation of the graph (nodes = sites where keep, edges ea-eb). Returns (n_nodes, giant size, rank of the giant, max rank).
    rank = rank of the winding vectors of the cycles of a cluster (3 = it connects the box to its own periodic image along a, b and c)."""
    S = len(P)
    adj = [[] for _ in range(S)]
    if len(ea):
        s = np.rint((P[eb] - P[ea]) / L).astype(int)          # image of b seen from a: I_b = I_a - s
        for a_, b_, sv in zip(ea.tolist(), eb.tolist(), s):
            sv = tuple(int(x) for x in sv)
            adj[a_].append((b_, tuple(-x for x in sv)))
            adj[b_].append((a_, sv))
    seen = np.zeros(S, bool)
    n_nodes = int(keep.sum())
    best, best_rank, max_rank = 0, 0, 0
    for r in np.where(keep)[0]:
        if seen[r]: continue
        img = {int(r): (0, 0, 0)}
        seen[r] = True
        stack, size, wind = [int(r)], 1, set()
        while stack:
            u = stack.pop()
            iu = img[u]
            for v, inc in adj[u]:
                nv = (iu[0] + inc[0], iu[1] + inc[1], iu[2] + inc[2])
                if v not in img:
                    img[v] = nv
                    seen[v] = True
                    size += 1
                    stack.append(v)
                else:
                    w = (nv[0] - img[v][0], nv[1] - img[v][1], nv[2] - img[v][2])
                    if w != (0, 0, 0) and len(wind) < 400: wind.add(w)
        rk = int(np.linalg.matrix_rank(np.array(sorted(wind), float))) if wind else 0
        max_rank = max(max_rank, rk)
        if size > best: best, best_rank = size, rk
    return n_nodes, best, best_rank, max_rank


class Rows(list):
    def add(self, metric, shell, value, se=NAN, param='', n=NAN):
        self.append(dict(shell=shell, metric=metric, param=str(param), value=float(value), se=float(se), n=float(n)))


# ---------------------------------------------------------------- the analysis (arrays in, rows out)
def analyse(li, state, sites, kind, dops, L, lo, Lf, dt, n_fu, edges, T, a, barriers=None, log=print):
    """li (F,N,3) drift-free unwrapped Li (dump frame); state (F,N) committed site ids; sites (S,3) relative to lo; kind (S,) str;
    dops [dict(label, x (F,3))] (smoothed, dump frame); L (3,) mean box; Lf (F,3); returns Rows."""
    F, N = state.shape
    S = len(sites)
    nb = int(a.car_blocks)
    kT = KB * T
    state = state.astype(np.int64)
    out = Rows()
    has = bool(dops)
    K = len(edges) + 1 if has else 1
    names = lio.shell_names(edges) if has else ['all']
    ev = max(1, int(a.car_every))
    ks = list(a.car_ks)
    kdef = int(a.car_kdef)
    if kdef not in ks: ks.append(kdef)
    L = np.asarray(L, float)

    # ---- geometry: Li -> nearest dopant (every frame), site -> nearest dopant (every ev-th frame, moving dopant)
    if has:
        dn = np.min([lio.dist_to(li, d['x'], Lf) for d in dops], axis=0)            # (F,N)
        shell_li = np.searchsorted(np.asarray(edges, float), dn)
        fi = np.arange(0, F, ev)
        dd = np.min([lio.site_dist_moving(sites, d['x'][fi], lo, L) for d in dops], axis=0)      # (Fs,S)
        shell_site = np.searchsorted(np.asarray(edges, float), dd)
        dd_static = np.median(dd, axis=0)
        shell_static = np.searchsorted(np.asarray(edges, float), dd_static)
    else:
        shell_li = np.zeros((F, N), int)
        fi = np.arange(0, F, ev)
        shell_site = np.zeros((len(fi), S), int)
        dd_static = np.full(S, np.inf)
        shell_static = np.zeros(S, int)
    Fs = len(fi)
    bf = lio.block_id(np.arange(F), F, nb)[:, None]                                 # (F,1)
    bfi = lio.block_id(fi, F, nb)

    n_dop = len(dops)
    out.add('n_Li', 'all', N); out.add('n_sites', 'all', S); out.add('n_dopants', 'all', n_dop)
    out.add('li_per_fu', 'all', N / n_fu); out.add('li_per_site', 'all', N / S); out.add('n_fu', 'all', n_fu)
    out.add('window_ps', 'all', (F - 1) * dt)
    for k in range(K):
        out.add('n_sites_shell', names[k], int((shell_static == k).sum()))

    # ---- 1  mobile vs trapped
    Wf = max(2, int(round(a.car_win_ps / dt)))
    starts = np.arange(0, F - Wf, Wf)
    nW = len(starts)
    tau = Wf * dt
    if nW < 3:
        warn(f'T={T}: only {nW} windows of {tau:g} ps (run {(F - 1) * dt:g} ps): mobile / trapped skipped, lower --car-win-ps')
    else:
        nbw = min(nb, nW)
        sh_w = np.empty((nW, N), int); d2 = np.empty((nW, N)); nnew = np.empty((nW, N), int)
        for w, s0 in enumerate(starts):
            e0 = s0 + Wf
            sh_w[w] = shell_li[s0]
            d = li[e0] - li[s0]
            d2[w] = (d * d).sum(1)
            nnew[w] = (np.diff(np.sort(state[s0:e0 + 1], axis=0), axis=0) != 0).sum(0)
        idx = (lio.block_id(np.arange(nW), nW, nbw)[:, None] * K + sh_w).ravel()

        def bs(x=None):
            return np.bincount(idx, weights=None if x is None else np.asarray(x, float).ravel(), minlength=nbw * K).reshape(nbw, K)

        def both(fn, *bl):
            """jackknife per shell and for all shells together"""
            r, se = jk(fn, *bl)
            ra, sa = jk(fn, *[b.sum(axis=1, keepdims=True) for b in bl])
            return list(r) + [float(ra[0])], list(se) + [float(sa[0])]

        nt, sa_ = bs(), bs(d2)
        D = lambda s_, n_: s_ / (6.0 * tau * n_) * UNIT
        r, se = both(lambda s_, n_: D(s_, n_), sa_, nt)
        for k, nm_ in zip(range(K + 1), names + ['all']):
            out.add('D_all', nm_, r[k], se[k], n=(nt[:, k].sum() if k < K else nt.sum()) / nW)
        for kk in ks:
            mob = nnew >= kk
            nm, sm = bs(mob), bs(d2 * mob)
            r, se = both(lambda m_, t_: m_ / t_, nm, nt)
            for k, nm_ in zip(range(K + 1), names + ['all']):
                out.add('f_mobile', nm_, r[k], se[k], kk, n=(nt[:, k].sum() if k < K else nt.sum()) / nW)
            r, se = both(lambda s_, m_: D(s_, m_), sm, nm)
            for k, nm_ in zip(range(K + 1), names + ['all']): out.add('D_mobile', nm_, r[k], se[k], kk)
            r, se = both(lambda sa2, sm2, t_, m_: D(sa2 - sm2, t_ - m_), sa_, sm, nt, nm)
            for k, nm_ in zip(range(K + 1), names + ['all']): out.add('D_trapped', nm_, r[k], se[k], kk)
        out.add('window_tau_ps', 'all', tau); out.add('n_windows', 'all', nW)
    nn_whole = (np.diff(np.sort(state, axis=0), axis=0) != 0).sum(0)
    fz = nn_whole == 0
    for k in range(K):
        m = shell_li[0] == k
        out.add('f_frozen_whole', names[k], fz[m].mean() if m.any() else NAN, n=m.sum())
    out.add('f_frozen_whole', 'all', fz.mean(), n=N)

    # ---- 2  vacancies / empty sites
    occ = np.zeros((F, S), np.int16)
    for f in range(F): occ[f] = np.bincount(state[f], minlength=S)
    empty = occ == 0
    kinds = ['all'] + sorted(set(kind))
    sub_e = empty[fi]
    for kd in kinds:
        km = np.ones(S, bool) if kd == 'all' else (kind == kd)
        if not km.any(): continue
        ne_ = np.zeros((nb, K)); nt_ = np.zeros((nb, K))
        for b in range(nb):
            sb = bfi == b
            for k in range(K):
                m = (shell_site[sb] == k) & km[None, :]
                ne_[b, k] = (sub_e[sb] & m).sum(); nt_[b, k] = m.sum()
        r, se = jk(lambda e_, t_: e_ / t_, ne_, nt_)
        ra, sa = jk(lambda e_, t_: e_ / t_, ne_.sum(1, keepdims=True), nt_.sum(1, keepdims=True))
        for k in range(K): out.add('empty_site_frac', names[k], r[k], se[k], kd, n=nt_[:, k].sum() / max(Fs, 1))
        out.add('empty_site_frac', 'all', ra[0], sa[0], kd, n=nt_.sum() / max(Fs, 1))
    Dss = np.linalg.norm(mi(sites[:, None, :] - sites[None, :, :], L), axis=-1)
    A = ((Dss <= a.car_nn) & (Dss > 0)).astype(np.float32)
    EN = empty.astype(np.float32) @ A.T                                              # (F,S) empty neighbours of every site
    nei = np.take_along_axis(EN, state, axis=1)                                      # (F,N) empty neighbours of the site each Li sits on
    deg = A.sum(1)
    out.add('mean_site_degree', 'all', deg.mean())
    idx = (bf * K + shell_li).ravel()
    cnt = np.bincount(idx, minlength=nb * K).reshape(nb, K).astype(float)
    sm_ = np.bincount(idx, weights=nei.ravel(), minlength=nb * K).reshape(nb, K)
    r, se = jk(lambda s_, c_: s_ / c_, sm_, cnt)
    ra, sa = jk(lambda s_, c_: s_ / c_, sm_.sum(1, keepdims=True), cnt.sum(1, keepdims=True))
    for k in range(K): out.add('empty_nbrs_per_Li', names[k], r[k], se[k], n=cnt[:, k].sum() / F)
    out.add('empty_nbrs_per_Li', 'all', ra[0], sa[0], n=N)
    hop = state[1:] != state[:-1]
    npv = np.minimum(np.rint(nei[:-1]).astype(int), NEMPTY - 1)
    i3 = ((bf[:-1] * K + shell_li[:-1]) * NEMPTY + npv).ravel()
    expo = np.bincount(i3, minlength=nb * K * NEMPTY).reshape(nb, K, NEMPTY).astype(float)
    hp = np.bincount(i3[hop.ravel()], minlength=nb * K * NEMPTY).reshape(nb, K, NEMPTY).astype(float)
    r, se = jk(lambda h_, e_: h_ / (e_ * dt), hp, expo)
    ra, sa = jk(lambda h_, e_: h_ / (e_ * dt), hp.sum(1), expo.sum(1))
    for n_ in range(NEMPTY):
        lab = f'{n_}' if n_ < NEMPTY - 1 else f'{n_}+'
        for k in range(K): out.add('hop_rate_per_Li_ps', names[k], r[k, n_], se[k, n_], lab, n=expo[:, k, n_].sum() * dt)
        out.add('hop_rate_per_Li_ps', 'all', ra[n_], sa[n_], lab, n=expo[:, :, n_].sum() * dt)

    # ---- 3  percolation
    P = np.mod(sites, L)
    cnt_pair = np.zeros((S, S))
    ff, ii = np.where(hop)
    np.add.at(cnt_pair, (state[ff, ii], state[ff + 1, ii]), 1)
    pair = cnt_pair + cnt_pair.T
    iu = np.triu_indices(S, 1)
    pc = pair[iu]
    use = (pc > 0) & (Dss[iu] <= a.car_edge_max)
    ea_, eb_ = iu[0][use], iu[1][use]
    rate = pc[use] / max((F - 1) * dt, 1e-9)                                         # hops per ps per pair
    allk = np.ones(S, bool)
    if len(rate) < 5:
        warn(f'T={T}: fewer than 5 hopped pairs: dynamic percolation skipped')
    else:
        med = float(np.median(rate))
        out.add('pair_rate_median_per_ns', 'all', med * 1000.0)
        gc = NAN
        for m in MULTS:
            sel = rate >= m * med
            nn_, g_, rk, _ = percolate(allk, ea_[sel], eb_[sel], P, L)
            out.add('perc_dyn_giant', 'all', g_ / S, param=f'{m:.3g}'); out.add('perc_dyn_rank', 'all', rk, param=f'{m:.3g}')
            if rk == 3: gc = float(m)
        out.add('gamma_c_over_med', 'all', gc)
        out.add('gamma_c_at_grid_top', 'all', float(gc >= MULTS[-1] - 1e-9) if np.isfinite(gc) else 0.0)
    ga, gb = np.where(np.triu((Dss <= a.car_bond_cut) & (Dss > 0), 1))
    n0, g0, rk0, _ = percolate(allk, ga, gb, P, L)
    out.add('perc_lattice_rank', 'all', rk0)
    if rk0 < 3: warn(f'T={T}: the bond graph (bond-cut {a.car_bond_cut:g} A) does not span the box even without blocking: raise --car-bond-cut')
    if has and rk0 == 3:
        rc = NAN
        for rb in RBLOCKS:
            keep = dd_static > rb
            sel = keep[ga] & keep[gb]
            nn_, g_, rk, _ = percolate(keep, ga[sel], gb[sel], P, L)
            out.add('perc_static_giant', 'all', g_ / max(nn_, 1), param=f'{rb:g}')
            out.add('perc_static_rank', 'all', rk, param=f'{rb:g}')
            out.add('perc_static_removed', 'all', 1 - keep.mean(), param=f'{rb:g}')
            if rk < 3 and not np.isfinite(rc): rc = rb
        out.add('r_block_c', 'all', rc)
        out.add('r_block_c_never_lost', 'all', float(not np.isfinite(rc)))
    mean_occ = occ.mean(0)
    hops_site = pair.sum(1)
    froz = (mean_occ > 0.5) & (hops_site == 0)
    if rk0 == 3:
        keep = ~froz
        sel = keep[ga] & keep[gb]
        nn_, g_, rk, _ = percolate(keep, ga[sel], gb[sel], P, L)
        out.add('frozen_site_frac', 'all', froz.mean())
        out.add('perc_nofrozen_giant', 'all', g_ / max(nn_, 1)); out.add('perc_nofrozen_rank', 'all', rk)

    # ---- 4  site-energy heterogeneity
    sock = np.zeros((nb, S))
    for b in range(nb):
        sb = np.where(lio.block_id(np.arange(F), F, nb) == b)[0]
        sock[b] = occ[sb].sum(0)

    def Fsite(tot):
        om = tot / max(F, 1)
        Fi = np.full(S, np.inf)
        for kd in sorted(set(kind)):
            mk = kind == kd
            far = mk & (shell_static == K - 1) if has else mk
            ref = om[far].mean() if far.any() and om[far].mean() > 0 else om[mk].mean()
            if ref <= 0: continue
            with np.errstate(divide='ignore'):
                Fi[mk] = np.where(om[mk] > 0, -kT * np.log(np.where(om[mk] > 0, om[mk], 1.0) / ref), np.inf)
        return Fi

    def sig(tot):
        Fi = Fsite(tot)
        fin = np.isfinite(Fi)
        o = np.full(K + 1, NAN)
        for k in range(K):
            m = fin & (shell_static == k)
            if m.sum() >= 3: o[k] = Fi[m].std()
        if fin.sum() >= 3: o[K] = Fi[fin].std()
        return o

    sg, sge = jk(sig, sock)
    sfar = sg[K - 1] if has else 0.0
    om_tot = sock.sum(0) / max(F, 1)
    for k in range(K + 1):
        nm_ = names[k] if k < K else 'all'
        out.add('F_sigma_eV', nm_, sg[k], sge[k])
        out.add('F_sigma_over_kT', nm_, sg[k] / kT, sge[k] / kT)
        out.add('zwanzig_D_ratio', nm_, np.exp(-(sg[k] / kT) ** 2) if np.isfinite(sg[k]) else NAN)
        if has and k < K - 1 and np.isfinite(sg[k]) and np.isfinite(sfar):
            out.add('F_sigma_excess_eV', nm_, np.sqrt(max(sg[k] ** 2 - sfar ** 2, 0.0)))
        m = (shell_static == k) if k < K else np.ones(S, bool)
        out.add('frac_sites_never_visited', nm_, (om_tot[m] == 0).mean() if m.any() else NAN, n=m.sum())
    Fi = Fsite(sock.sum(0))
    if has:
        edg = np.arange(0.0, min(np.nanmax(dd_static[np.isfinite(dd_static)]), 12.0) + 1.0, 1.0)
        for lo_, hi_ in zip(edg[:-1], edg[1:]):
            m = (dd_static >= lo_) & (dd_static < hi_) & np.isfinite(Fi)
            if m.sum() >= 3:
                out.add('F_profile_mean_eV', 'all', Fi[m].mean(), param=f'{0.5 * (lo_ + hi_):.1f}', n=m.sum())
                out.add('F_profile_std_eV', 'all', Fi[m].std(), param=f'{0.5 * (lo_ + hi_):.1f}', n=m.sum())
    if barriers:
        for k in range(K):
            v = [b for b, s_ in barriers if s_ == names[k]]
            if v: out.add('barrier_median_eV', names[k], np.median(v), np.std(v) / np.sqrt(len(v)), n=len(v))
        if barriers: out.add('barrier_median_eV', 'all', np.median([b for b, _ in barriers]), n=len(barriers))
    return out


# ---------------------------------------------------------------- I/O wrapper: one (folder, T)
def md_dump(folder, T):
    hit = sorted(glob.glob(os.path.join(folder, f'MD-{T}', '*.lammpstrj')))
    pref = [h for h in hit if f'{T}K' in os.path.basename(h)]
    return (pref or hit or [None])[0]


def md_temps(folder):
    out = []
    for d in glob.glob(os.path.join(folder, 'MD-*')):
        t = os.path.basename(d)[3:]
        if t.isdigit() and md_dump(folder, int(t)): out.append(int(t))
    return sorted(out)


def read_sites(path):
    fr, kd = [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            fr.append([float(r['frac_x']), float(r['frac_y']), float(r['frac_z'])])
            kd.append(str(r['kind']).split()[0] if r.get('kind') else 'peak')
    return np.array(fr), np.array(kd)


def read_barriers(path):
    out = []
    if not os.path.exists(path): return out
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                if str(r.get('blocked')) == 'True': continue
                b = float(r['barrier_eV'])
                if np.isfinite(b): out.append((b, r['shell']))
            except (KeyError, ValueError): pass
    return out


def cache_key(a, cfg, dump, hdir, T, tmin):
    st = os.path.join(hdir, f'state_T{T}K.npz')
    return repr((CAR_VERSION, a.car_win_ps, tuple(a.car_ks), a.car_kdef, a.car_blocks, a.car_every, a.car_nn, a.car_edge_max, a.car_bond_cut,
                 tuple(a.shells), tmin, cfg.get('dop_smooth_ps'), tuple(sorted(cfg['types'].items())), dump, os.path.getmtime(dump),
                 os.path.getmtime(st), os.path.getmtime(os.path.join(hdir, 'sites.csv'))))


def compute_T(task):
    """-> error string or None; writes <folder>/carriers/<name>/carriers_T<T>K.json"""
    folder, cfg, T, a, outp, key = task
    try:
        name = cfg['name']
        hdir = os.path.join(folder, a.car_hops, name)
        dump = md_dump(folder, T) or lio.dump_path(cfg, T)
        traj = lio.prepare(cfg, dump, a.tmin, verbose=False)
        drift = lio.framework_drift(cfg, traj)
        li = traj.pos[:, lio.li_mask(cfg, traj), :] - drift[:, None, :]
        L = traj.L.mean(axis=0)
        dops = []
        cnt = {}
        for t in cfg['dopant_types']:
            for i in np.where(traj.types == t)[0]:
                el = cfg['types'][t]; cnt[el] = cnt.get(el, 0) + 1
                dops.append(dict(label=f'{el}{cnt[el]}', x=lio.dopant_xyz(cfg, traj, drift, i)))
        dt = float(np.median(np.diff(traj.time_ps)))
        state = np.load(os.path.join(hdir, f'state_T{T}K.npz'))['state']
        if state.shape != li.shape[:2]:
            return (f'{folder} T={T}: 04 state {state.shape} != dump window {li.shape[:2]} (other --tmin or other dump than 04 used? re-run 04)')
        fr, kind = read_sites(os.path.join(hdir, 'sites.csv'))
        sites = fr * L
        state = state.astype(np.int64)
        idx = np.linspace(0, len(state) - 1, min(len(state), 200)).astype(int)
        shift = mi((li[idx] - traj.lo) - sites[state[idx]], L).reshape(-1, 3).mean(0)
        if np.linalg.norm(shift) > 0.5: warn(f'{folder} T={T}: rigid shift of the 04 sites to the Li is {np.linalg.norm(shift):.2f} A (large)')
        sites = sites + shift
        n_fu = int(np.prod(lio.get_fold(cfg, L))) * 8
        rows = analyse(li, state, sites, kind, dops, L, traj.lo, np.asarray(traj.L, float), dt, n_fu, list(a.shells), T, a,
                       barriers=read_barriers(os.path.join(hdir, f'barriers_T{T}K.csv')))
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        with open(outp, 'w') as fh: json.dump(dict(key=key, rows=list(rows), dt=dt, window_ps=float(traj.time_ps[-1] - traj.time_ps[0])), fh)
        return None
    except Exception as e:                                       # keep the other (folder, T) going
        import traceback
        return f'{folder} T={T}: {type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}'


def collect(groups, a, recompute=False, hdr=print):
    """compute / read the cache for every (folder, T). Returns {system name: {T: rows}} in the order of the groups."""
    todo, plan = [], {}
    for gi, g in enumerate(groups):
        for f in g['folders']:
            if not os.path.isdir(f): continue
            try:
                ga = argparse.Namespace(**vars(a))
                if g.get('elements'): ga.elements = g['elements']
                cfg = lio.load_config(f, None, ga)
                temps = a.T or md_temps(f) or lio.temperatures(cfg)
            except SystemExit as e:
                warn(f'carriers skipped for {f}: {e}'); continue
            if not cfg['dopant_types']: warn(f'{f}: no dopant atoms: shells = one shell, percolation by r_block skipped')
            hdir = os.path.join(f, a.car_hops, cfg['name'])
            plan[f] = (cfg, temps)
            for T in temps:
                if not (os.path.exists(os.path.join(hdir, f'state_T{T}K.npz')) and os.path.exists(os.path.join(hdir, 'sites.csv'))):
                    warn(f'{f} T={T}: no {a.car_hops}/{cfg["name"]}/state_T{T}K.npz + sites.csv (run 04 first)'); continue
                dump = md_dump(f, T) or lio.dump_path(cfg, T)
                outp = os.path.join(f, 'carriers', cfg['name'], f'carriers_T{T}K.json')
                key = cache_key(a, cfg, dump, hdir, T, a.tmin)
                ok = False
                if os.path.exists(outp) and not recompute:
                    try: ok = json.load(open(outp))['key'] == key
                    except Exception: ok = False
                if not ok: todo.append((f, cfg, T, a, outp, key))
    n_all = sum(len(v[1]) for v in plan.values())
    print(f'[carriers] {len(todo)} (folder, T) to compute, {n_all - len(todo)} cached/skipped')
    jobs = getattr(a, 'jobs', 1)
    if jobs > 1 and len(todo) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=jobs) as ex: errs = list(ex.map(compute_T, todo))
    else:
        errs = [compute_T(t) for t in todo]
    for e in errs:
        if e: warn('carriers: ' + e)
    res = {}
    for f, (cfg, temps) in plan.items():
        for T in temps:
            p = os.path.join(f, 'carriers', cfg['name'], f'carriers_T{T}K.json')
            if os.path.exists(p):
                res.setdefault(fname(f), {})[T] = json.load(open(p))['rows']
    return res


# ---------------------------------------------------------------- aggregate: long csv, Arrhenius, summary, figures
def arrhenius(res, kdef):
    """-> rows (system, shell, quantity, Ea_eV, Ea_se, ln_prefactor, n_T, T_list) for f_mobile / D_mobile / D_all."""
    out = []
    for sysn, byT in res.items():
        Ts = sorted(byT)
        for q, prm in (('f_mobile', str(kdef)), ('D_mobile', str(kdef)), ('D_all', '')):
            shells = []
            for T in Ts:
                for r in byT[T]:
                    if r['metric'] == q and r['param'] == prm and r['shell'] not in shells: shells.append(r['shell'])
            for sh in shells:
                x, y = [], []
                for T in Ts:
                    v = [r['value'] for r in byT[T] if r['metric'] == q and r['param'] == prm and r['shell'] == sh]
                    if v and np.isfinite(v[0]) and v[0] > 0: x.append(1.0 / (KB * T)); y.append(np.log(v[0]))
                if len(x) < 3: continue
                pf, cov = np.polyfit(x, y, 1, cov=True) if len(x) > 3 else (np.polyfit(x, y, 1), np.full((2, 2), NAN))
                out.append(dict(system=sysn, shell=sh, quantity=q, Ea_eV=-pf[0], Ea_se=float(np.sqrt(cov[0, 0])), ln_prefactor=pf[1], n_T=len(x),
                                T_list=' '.join(str(int(round(1 / (KB * xx)))) for xx in x)))
    return out


def val(res, sysn, T, metric, shell='all', param='', col='value'):
    for r in res.get(sysn, {}).get(T, []):
        if r['metric'] == metric and r['shell'] == shell and r['param'] == param: return r[col]
    return NAN


def pick_tref(res, a):
    if a.car_tref: return a.car_tref
    cnt = {}
    for byT in res.values():
        for T in byT: cnt[T] = cnt.get(T, 0) + 1
    if not cnt: return None
    mx = max(cnt.values())
    return max(T for T, c in cnt.items() if c >= max(1, mx // 2))


def shells_of(res, sysn, T):
    out = []
    for r in res.get(sysn, {}).get(T, []):
        if r['metric'] == 'f_mobile' and r['shell'] != 'all' and r['shell'] not in out: out.append(r['shell'])
    return out


def _fmt(x, d=3):
    if x is None or not np.isfinite(x): return '-'
    return f'{x:.{d}g}'


def summary_table(res, order, T, kdef):
    """(header, rows): one row per system at T."""
    head = ['system', 'Li/f.u.', 'f_mob all', 'f_mob inner', 'f_mob far', 'D_all (cm2/s)', 'D_mob far', 'empty sites', 'empty nbrs/Li', 'gamma_c/med',
            'r_block_c (A)', 'sigma_F/kT inner', 'sigma_F/kT far', 'Ea D_all', 'Ea D_mob', 'Ea f_mob']
    ar = {}
    for r in arrhenius({s: res[s] for s in order if s in res}, kdef): ar[(r['system'], r['shell'], r['quantity'])] = r['Ea_eV']
    rows = []
    for s in order:
        if T not in res.get(s, {}): continue
        sh = shells_of(res, s, T)
        inner, far = (sh[0], sh[-1]) if len(sh) >= 2 else ('all', 'all')
        k = str(kdef)
        rows.append([s, _fmt(val(res, s, T, 'li_per_fu'), 4), _fmt(val(res, s, T, 'f_mobile', 'all', k)), _fmt(val(res, s, T, 'f_mobile', inner, k)),
                     _fmt(val(res, s, T, 'f_mobile', far, k)), _fmt(val(res, s, T, 'D_all'), 3), _fmt(val(res, s, T, 'D_mobile', far, k), 3),
                     _fmt(val(res, s, T, 'empty_site_frac', 'all', 'all')), _fmt(val(res, s, T, 'empty_nbrs_per_Li')),
                     _fmt(val(res, s, T, 'gamma_c_over_med')), _fmt(val(res, s, T, 'r_block_c')),
                     _fmt(val(res, s, T, 'F_sigma_over_kT', inner)), _fmt(val(res, s, T, 'F_sigma_over_kT', far)),
                     _fmt(ar.get((s, 'all', 'D_all'), NAN), 3), _fmt(ar.get((s, 'all', 'D_mobile'), NAN), 3), _fmt(ar.get((s, 'all', 'f_mobile'), NAN), 3)])
    return head, rows


def _savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def make_figs(res, order, outdir, T, kdef, arr=True):
    os.makedirs(outdir, exist_ok=True)
    systems = [s for s in order if T in res.get(s, {})]
    if not systems: return []
    k = str(kdef)
    cmap = plt.get_cmap('tab10' if len(systems) <= 10 else 'tab20')
    col = {s: cmap(i % cmap.N) for i, s in enumerate(systems)}
    shs = []
    for s in systems:
        for h in shells_of(res, s, T):
            if h not in shs: shs.append(h)
    shs = shs or ['all']
    scol = {h: plt.get_cmap('viridis')(i / max(len(shs) - 1, 1)) for i, h in enumerate(shs)}
    x = np.arange(len(systems))
    figs = []

    def by_shell(ax, metric, param, ylabel, log=False, title=None):
        w = 0.8 / len(shs)
        for j, h in enumerate(shs):
            y = [val(res, s, T, metric, h, param) for s in systems]
            e = [val(res, s, T, metric, h, param, 'se') for s in systems]
            ax.bar(x + (j - (len(shs) - 1) / 2) * w, y, w, yerr=[0 if not np.isfinite(v) else v for v in e], color=scol[h], label=h, error_kw=dict(lw=0.6))
        ax.set_xticks(x); ax.set_xticklabels(systems, rotation=60, ha='right', fontsize=6)
        ax.set_ylabel(ylabel, fontsize=8); ax.set_title(title or metric, fontsize=8)
        if log: ax.set_yscale('log')
        ax.tick_params(labelsize=7)

    fig, axs = plt.subplots(2, 4, figsize=(19, 8.5))
    by_shell(axs[0, 0], 'f_mobile', k, 'fraction of Li windows', title=f'1  mobile fraction (>= {k} other sites / window)')
    axs[0, 0].legend(fontsize=6, title='shell', title_fontsize=6)
    by_shell(axs[0, 1], 'D_mobile', k, 'D of the mobile subset (cm2/s)', True, '1  D mobile')
    by_shell(axs[0, 2], 'empty_site_frac', 'all', 'fraction of sites empty', title='2  empty sites (all kinds)')
    by_shell(axs[0, 3], 'empty_nbrs_per_Li', '', 'empty neighbour sites per Li', title='2  empty neighbours of a Li')
    ax = axs[1, 0]
    ax.bar(x, [val(res, s, T, 'gamma_c_over_med') for s in systems], color=[col[s] for s in systems])
    ax.set_xticks(x); ax.set_xticklabels(systems, rotation=60, ha='right', fontsize=6); ax.set_ylabel('gamma_c / median pair rate', fontsize=8)
    ax.set_title('3  dynamic network: highest rate cut that still spans', fontsize=8); ax.set_yscale('log'); ax.tick_params(labelsize=7)
    ax = axs[1, 1]
    top = float(max(RBLOCKS))
    rc = np.array([val(res, s, T, 'r_block_c') for s in systems])
    nl = np.array([val(res, s, T, 'r_block_c_never_lost') for s in systems]) == 1
    yy = np.where(nl, top, np.nan_to_num(rc, nan=0.0))
    bars = ax.bar(x, yy, color=[col[s] for s in systems])
    for b_, n_ in zip(bars, nl):
        if n_: b_.set_hatch('//'); b_.set_alpha(0.5)
    ax.set_xticks(x); ax.set_xticklabels(systems, rotation=60, ha='right', fontsize=6); ax.set_ylabel('r_block_c (A)', fontsize=8)
    ax.set_title(f'3  static lattice: blocking radius that kills spanning (hatched = never lost, >= {top:g}; 0 = no dopant)', fontsize=8); ax.tick_params(labelsize=7)
    by_shell(axs[1, 2], 'F_sigma_over_kT', '', 'sigma_F / kT', title='4  site-energy spread (effective)')
    ax = axs[1, 3]
    ax.bar(x, [val(res, s, T, 'li_per_fu') for s in systems], color=[col[s] for s in systems]); ax.set_ylim(min(6.0, np.nanmin([val(res, s, T, 'li_per_fu') for s in systems]) - 0.2), None)
    ax.set_xticks(x); ax.set_xticklabels(systems, rotation=60, ha='right', fontsize=6); ax.set_ylabel('Li per f.u.', fontsize=8); ax.set_title('Li content', fontsize=8); ax.tick_params(labelsize=7)
    fig.suptitle(f'carrier analysis at {T} K (error bars: block jackknife)', fontsize=10)
    figs.append(_savefig(fig, os.path.join(outdir, f'carriers_overview_T{T}K.png')))

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    xs = np.array([val(res, s, T, 'li_per_fu') for s in systems])
    for ax, (m, p, sh, yl, lg) in zip(axs.ravel(), [('D_all', '', 'all', 'D_all (cm2/s)', True), ('f_mobile', k, 'all', 'f_mobile (all Li)', False),
                                                      ('empty_site_frac', 'all', 'all', 'empty-site fraction', False), ('gamma_c_over_med', '', 'all', 'gamma_c / median', True),
                                                      ('F_sigma_over_kT', '', 'all', 'sigma_F / kT (all sites)', False), ('hop_rate_per_Li_ps', '3+', 'all', 'hop rate, Li with >= 3 empty nbrs (/Li/ps)', False)]):
        y = np.array([val(res, s, T, m, sh, p) for s in systems])
        ax.scatter(xs, y, c=[col[s] for s in systems], s=30)
        for s, xx, yy in zip(systems, xs, y):
            if np.isfinite(xx) and np.isfinite(yy): ax.annotate(s, (xx, yy), fontsize=5, xytext=(2, 2), textcoords='offset points')
        ax.set_xlabel('Li per f.u.', fontsize=8); ax.set_ylabel(yl, fontsize=8); ax.tick_params(labelsize=7)
        if lg and np.isfinite(y).any() and (y[np.isfinite(y)] > 0).all(): ax.set_yscale('log')
    fig.suptitle(f'does everything collapse onto Li content?  T = {T} K', fontsize=10)
    figs.append(_savefig(fig, os.path.join(outdir, f'carriers_vs_Li_T{T}K.png')))

    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
    for s in systems:
        for ax, (m, par, xv) in zip(axs[:2], [('perc_static_giant', RBLOCKS, 'r_block'), ('perc_dyn_giant', MULTS, 'm')]):
            xx = [p for p in par]
            yy = [val(res, s, T, m, 'all', f'{p:g}' if xv == 'r_block' else f'{p:.3g}') for p in xx]
            if np.isfinite(yy).any(): ax.plot(xx, yy, '-o', ms=3, color=col[s], label=s)
        yy = [val(res, s, T, 'hop_rate_per_Li_ps', 'all', f'{n}' if n < NEMPTY - 1 else f'{n}+') for n in range(NEMPTY)]
        axs[2].plot(range(NEMPTY), yy, '-o', ms=3, color=col[s])
    axs[0].set_xlabel('r_block: sites closer than this to a dopant removed (A)', fontsize=8); axs[0].set_ylabel('giant cluster / remaining sites', fontsize=8)
    axs[0].set_title('3  static percolation of the lattice', fontsize=8)
    axs[1].set_xlabel('m: pairs with rate >= m x median kept', fontsize=8); axs[1].set_xscale('log'); axs[1].set_ylabel('giant cluster / all sites', fontsize=8)
    axs[1].set_title('3  dynamic percolation (hopped pairs)', fontsize=8)
    axs[2].set_xticks(range(NEMPTY)); axs[2].set_xticklabels(['0', '1', '2', '3+']); axs[2].set_xlabel('empty neighbour sites of the Li', fontsize=8)
    axs[2].set_ylabel('hop rate per Li per ps', fontsize=8); axs[2].set_title('2  vacancy-limited hopping?', fontsize=8)
    axs[0].legend(fontsize=5, ncol=2)
    for ax in axs: ax.tick_params(labelsize=7)
    fig.suptitle(f'percolation and vacancy limitation, T = {T} K', fontsize=10)
    figs.append(_savefig(fig, os.path.join(outdir, f'carriers_percolation_T{T}K.png')))

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    for s in systems:
        xx, m_, sd_ = [], [], []
        for r in res[s][T]:
            if r['metric'] == 'F_profile_mean_eV': xx.append(float(r['param'])); m_.append(r['value'])
        sdv = {float(r['param']): r['value'] for r in res[s][T] if r['metric'] == 'F_profile_std_eV'}
        if xx:
            axs[0].plot(xx, m_, '-o', ms=3, color=col[s], label=s)
            axs[1].plot(xx, [sdv.get(v, NAN) for v in xx], '-o', ms=3, color=col[s])
    axs[0].set_ylabel('mean F_i (eV), relative to far sites', fontsize=8); axs[1].set_ylabel('std of F_i in the bin (eV)', fontsize=8)
    for ax in axs: ax.set_xlabel('distance of the site to the nearest dopant (A)', fontsize=8); ax.tick_params(labelsize=7)
    axs[0].legend(fontsize=5, ncol=2)
    fig.suptitle(f'4  effective site free energy vs distance, T = {T} K (F_i = -kT ln occupation)', fontsize=10)
    figs.append(_savefig(fig, os.path.join(outdir, f'carriers_Fprofile_T{T}K.png')))

    ks = sorted({int(r['param']) for s in systems for r in res[s][T] if r['metric'] == 'f_mobile' and r['param']})
    fig, axs = plt.subplots(1, len(shs) + 1, figsize=(3.6 * (len(shs) + 1), 4.2), squeeze=False)
    for ax, h in zip(axs[0], shs + ['all']):
        for s in systems:
            ax.plot(ks, [val(res, s, T, 'f_mobile', h, str(q)) for q in ks], '-o', ms=3, color=col[s], label=s)
        ax.set_title(f'shell {h}', fontsize=8); ax.set_xlabel('k (other sites visited per window)', fontsize=8); ax.tick_params(labelsize=7)
    axs[0, 0].set_ylabel('f_mobile', fontsize=8); axs[0, 0].legend(fontsize=5)
    fig.suptitle(f'1  is the mobile fraction an artefact of k?  T = {T} K', fontsize=10)
    figs.append(_savefig(fig, os.path.join(outdir, f'carriers_ksweep_T{T}K.png')))

    ar = arrhenius({s: res[s] for s in systems if len(res[s]) >= 3}, kdef) if arr else []
    if ar:
        names_ = []
        for r in ar:
            if r['system'] not in names_: names_.append(r['system'])
        fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
        w = 0.27
        for j, q in enumerate(('D_all', 'D_mobile', 'f_mobile')):
            ys = [next((r['Ea_eV'] for r in ar if r['system'] == s and r['shell'] == 'all' and r['quantity'] == q), NAN) for s in names_]
            es = [next((r['Ea_se'] for r in ar if r['system'] == s and r['shell'] == 'all' and r['quantity'] == q), NAN) for s in names_]
            axs[0].bar(np.arange(len(names_)) + (j - 1) * w, ys, w, yerr=[0 if not np.isfinite(v) else v for v in es], label=q, error_kw=dict(lw=0.6))
        axs[0].set_xticks(range(len(names_))); axs[0].set_xticklabels(names_, rotation=60, ha='right', fontsize=6); axs[0].set_ylabel('Ea (eV), all Li', fontsize=8)
        axs[0].legend(fontsize=7); axs[0].set_title('5  Arrhenius split (higher Ea of D_mobile: landscape; of f_mobile: activated carriers)', fontsize=8)
        for s in names_:
            Ts = sorted(res[s]); y = [val(res, s, t, 'D_all') for t in Ts]
            axs[1].plot([1000.0 / t for t in Ts], y, '-o', ms=3, color=col.get(s), label=s)
        axs[1].set_yscale('log'); axs[1].set_xlabel('1000 / T (1/K)', fontsize=8); axs[1].set_ylabel('D_all (cm2/s)', fontsize=8); axs[1].legend(fontsize=5, ncol=2)
        for ax in axs: ax.tick_params(labelsize=7)
        figs.append(_savefig(fig, os.path.join(outdir, 'carriers_arrhenius.png')))
    return figs


def write_outputs(res, order, outdir, a):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'carriers_long.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['system', 'T', 'shell', 'metric', 'param', 'value', 'se', 'n'])
        for s in order:
            for T in sorted(res.get(s, {})):
                for r in res[s][T]: w.writerow([s, T, r['shell'], r['metric'], r['param'], r['value'], r['se'], r['n']])
    ar = arrhenius(res, a.car_kdef)
    with open(os.path.join(outdir, 'arrhenius.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['system', 'shell', 'quantity', 'Ea_eV', 'Ea_se', 'ln_prefactor', 'n_T', 'T_list']); w.writeheader(); w.writerows(ar)
    T = pick_tref(res, a)
    head, rows = summary_table(res, order, T, a.car_kdef) if T else ([], [])
    with open(os.path.join(outdir, 'carriers_summary.csv'), 'w', newline='') as fh:        # reference T only (what build_doc puts in the report)
        w = csv.writer(fh); w.writerow(head); w.writerows(rows)
    allT = sorted({t for s in order for t in res.get(s, {})})                               # every T: one row per (T, system)
    with open(os.path.join(outdir, 'carriers_summary_allT.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        for i, t in enumerate(allT):
            h_, r_ = summary_table(res, order, t, a.car_kdef)
            if i == 0: w.writerow(['T_K'] + h_)
            for r in r_: w.writerow([t] + r)
    return T, head, rows, ar


CAR_TEXT = ('Carrier analysis from the MD (carriers.py). 1: mobile vs trapped Li (a Li is mobile in a window if it visits at least k other sites; the k sweep '
            'shows the split is not a threshold artefact; D = Li-window MSD / 6 tau). 2: Li per formula unit, empty sites, empty neighbours of a Li and the hop '
            'rate as a function of that number. 3: periodic percolation, dynamic (hopped pairs, rate cut) and static (lattice minus sites within r_block of a '
            'dopant; r_block_c = radius at which spanning is lost). 4: effective site free energy F = -kT ln(occupation) relative to far sites, its spread '
            'sigma_F (far shell = sampling-noise control) and the Zwanzig estimate exp(-(sigma/kT)^2). 5: Arrhenius split of f_mobile, D_mobile, D_all. '
            'sigma_F is an effective spread (Li-Li exclusion biases occupancies), not a DFT energy. Table: one row per system at the reference temperature; '
            'f_mob columns use the headline k; Ea in eV over all temperatures (>= 3 needed).')


def run_carriers(groups, a, outdir, recompute=False):
    """Used by build_doc: returns dict(figs, text, head, rows) or None."""
    order = []
    for g in groups:
        for f in g['folders']:
            if os.path.isdir(f): order.append(fname(f))
    res = collect(groups, a, recompute)
    if not res:
        warn('carriers: no result (run 04 first?)'); return None
    T, head, rows, ar = write_outputs(res, order, outdir, a)
    if T is None: return None
    figs = make_figs(res, order, outdir, T, a.car_kdef)                        # reference T + the Arrhenius figure
    if not getattr(a, 'car_tref_only', False):                                  # the same figure set at every other T
        for t in sorted({t for s in order for t in res.get(s, {})}):
            if t != T: figs += make_figs(res, order, outdir, t, a.car_kdef, arr=False)
    print(f'[carriers] {outdir}/carriers_long.csv arrhenius.csv carriers_summary.csv ' + ' '.join(os.path.basename(p) for p in figs))
    return dict(figs=figs, text=CAR_TEXT + f' Reference temperature: {T} K.', head=head, rows=rows, T=T)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('items', nargs='+', help='groups: {A B C} {D E Ca=La elements=Li,La,Zr,Ca,O}')
    ap.add_argument('--outdir', default='carriers_out')
    ap.add_argument('--scripts', help='folder with llzo_io.py ... (default: ./with_flag next to this file)')
    ap.add_argument('--T', type=int, nargs='*', help='temperatures (default: all MD-<T> folders)')
    ap.add_argument('--tmin', type=float, help='window start, ps (as 04; default tmin_ps of the settings)')
    ap.add_argument('--shells', type=float, nargs='*', default=list(lio.DEFAULT_SHELLS), help='shell edges, A')
    ap.add_argument('--jobs', type=int, default=1, help='parallel (folder, T)')
    ap.add_argument('--recompute', action='store_true')
    add_args(ap)
    lio.add_config_args(ap)
    a = ap.parse_args()
    import build_doc as bd                                    # only the group syntax is taken from there
    groups = bd.parse_groups(a.items)
    for g in groups: g['folders'] = bd.expand(g['folders'])
    r = run_carriers(groups, a, a.outdir, a.recompute)
    if r:
        print('\n' + '  '.join(r['head']))
        for row in r['rows']: print('  '.join(row))


if __name__ == '__main__':
    main()
