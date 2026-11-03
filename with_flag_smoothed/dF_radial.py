#!/usr/bin/env python3
"""dF_radial.py - radial Li free-energy shift around dopants, against a geometry-matched pure-cell reference.

python dF_radial.py {pure} {1Al-doped 2Al-doped 3Al-doped Al=Li24d elements=Li,La,Zr,Al,O} {1Ga-doped 2Ga-doped Ga=Li24d} --outdir dfr_out [--T 800 1200] [--jobs 4]

Same group syntax, same system flags (--dump-dir --prefix --dump-pattern --elements --element-list --dopants --fold --name --dop-smooth-ps),
same --scripts / --cif / --T / --tmin / --jobs / --recompute as build_doc.py and carriers.py. STANDALONE: not called by run_pipeline.sh or
build_doc.py. Needs only the raw dumps (+ element_list) of every folder and the pristine Ia-3d CIF; no 01-09 output is read.

WHAT IT COMPUTES  (Boltzmann inversion of the Li density, radially averaged around the dopant)
  F(d)/kT = -ln( <rho>_shell(d) / rho_uniform )         rho_uniform = N_Li / V of that cell.  Dimensionless: no kT in the number.
  dF(d)/kT = F_doped(d)/kT - F_pure(d)/kT               > 0: Li depleted around the dopant (blocking);  < 0: Li enriched (trapping).
  Li density: cloud-in-cell + adaptive Gaussian smoothing, the SAME maps as 01_density_free_energy.py (same defaults). The shell average is
  the volume average of rho over ALL voxels of the shell (unsampled voxels included), i.e. the Li number density of the shell; so no
  sampling mask biases it and no voxel is floored. Sparse bins are flagged instead (see below).
  Is the temperature influence removed?  Only the explicit kT prefactor is. dF/kT = dE/kT - dS/k: a purely energetic dF scales as 1/T, an
  entropic one is T independent. dF_Tscan_<system>.png shows dF/kT and dF in eV (= kT x dF/kT) for every T: the panel in which the curves
  collapse tells which part dominates. Fewer Li in a doped cell (charge compensation) is removed by the rho_uniform normalisation.

REFERENCE  (what makes dF a dopant effect and not lattice geometry)
  The reference is the PURE folder (a folder with no dopant atoms in its element list; or --ref FOLDER), same T. In the pure cell the same
  crystal sites as in the doped cell are taken as virtual dopant positions (site index from the CIF fit, tiled identically; the fit's
  lattice-translation ambiguity moves all centres together, so their relative geometry is kept). Distances, ownership (below) and site
  classes are then computed with exactly the same code in both cells, so lattice shell structure and box-size truncation cancel.
  The virtual centre is an ideal site, the real dopant is its time-averaged position: bins below --rmin (default 1.0 A) are flagged.
  Dopant position: time mean of the unwrapped path. A dopant that moves more than 1 A rms prints a warning (a blurred centre).

CURVES  (per system, T; each for site class 'all' and, unless --no-split, '24d' and '96h')
  Ownership: a voxel belongs to its nearest dopant (periodic minimum image), distance d = distance to that dopant.
  <label>   individual dopant (Al1, Al2, ...): only voxels it owns. With one dopant in the cell there is only 'pooled'.
  pooled    all voxels, d = distance to the nearest dopant. This is the combined / average curve (volume weighted).
  Site class of a voxel = kind (24d / 96h) of the nearest ideal Li site of the CIF, aligned on the La/Zr framework as in 04. It follows the
  lattice, not the occupation (near a dopant ideal sites fit worse, 05's docstring): treat the split near r < 3 A with care.
  Radius range: d from 0 to --rmax (default half the shortest box edge = complete spheres), bins --dr (default 0.25 A).

FLAGS AND ERRORS (columns of dF_radial_long.csv)
  se_kT        block standard error of dF/kT: the window is cut into --blocks (default 4) contiguous time blocks; density is rebuilt per block
               (doped and pure independently): se^2 = var_b(ln rho_doped)/K + var_b(ln rho_pure)/K.
  half_diff_kT |dF(first half of the blocks) - dF(second half)|. NOT CONVERGED (nonconv = 1) when it exceeds --conv-tol (default 0.5 kT) or is
               undefined. A non-ergodic run (Li near a blocker never exchanges with the rest) shows the starting configuration, not an equilibrium.
  sparse       fewer than --min-count raw Li samples (default 20) or fewer than --min-vox voxels (default 20) in the bin of the doped or the pure
               cell: dF there is a LOWER BOUND on the depletion, never infinity. Fmax_kT = largest F/kT that min-count samples could resolve.
  below_rmin   bin ends at or below --rmin.
Summary (dF_summary.csv, pooled curves, bins with r >= rmin, finite, not sparse): dF_max_kT and where, r_thr = outer edge of the last bin with
  |dF| >= --thr kT (default 0.1; censored = 1 if that is the last valid bin: the perturbation reaches the edge, r_thr is a lower bound), frac_nonconv.

OUTPUTS  <outdir>/  dF_radial_long.csv  dF_summary.csv
  dF_<system>_T<T>K.png     per system and T: row 1 F/kT doped (solid) and pure (dashed), row 2 dF/kT: pooled (thick, band = se) + individual dopants
  dF_elements_<site>_T<T>K.png   one line per dopant element = mean over its systems (band = min-max), thin dashed = each system
  dF_Tscan_<system>.png     dF/kT (left) and dF in eV (right) vs d, one line per T (pooled, site 'all')
  Markers: open = sparse (lower bound), x = not converged. Grey = d < rmin.  Every .png also gets an .svg (llzo_svg).
Cache: <folder>/dfr/<name>/ (pure: ref_T<T>K.npz, doped: dfr_T<T>K.json); --recompute ignores it. Cached grids of the pure cell make reruns with
other --dr / --rmax / flags cheap for the reference but the doped cells are read again from their dumps.
Limit: Li-Li exclusion is not separated from 'energy'; F is a free energy (PMF), not a DFT energy.

MOVING CENTRE (default; --static = old fixed time-mean centre, with the 24d/96h split)
  Per dopant the centre is its path smoothed over --dop-smooth-ps (vibration out, hops in). Every frame, all Li are shifted by -centre(t) (periodic),
  so the dopant sits at the origin of a dopant-frame density; shells are distances to the origin. No ownership competition: dopant i uses only the
  complete sphere r < r_own(i) = min(rmax, half the 5th-percentile distance to any other dopant): per-dopant curves do not depend on how many
  dopants the cell has. 'pooled' = volume-weighted merge of the dopants. Only site class 'all' (a static 24d/96h class map has no meaning in a moving frame).
  Reference: each frame the centre is mapped to its nearest ideal site of the dopant's sublattice; the pure-cell density around those virtual
  sites is mixed with the visit fractions (sites visited < --site-wmin of the time are dropped). Frames > 1 A from every such site are counted
  (frac_far in the json 'dopants' entry; warning above 10 %).
  Empty bins are NOT joined by a line any more (the old 24d 'drop' was a straight line across bins without voxels of that class).
MASTER CURVES  dF_master.png / dF_master.csv: per dopant element, site 'all', individual-dopant curves, valid bins only (not sparse, not
  non-converged, r >= rmin). Weights 1/(se^2 + --se-floor^2).
  B: per T the weighted mean of dF/kT over dopants and systems; curve = mean over T, band = min-max over T.
  A: per bin weighted least squares of dF/kT = a + b/T over all points of that element (>= 3 distinct T): dE = k_B b (eV), dS/k = -a, errors from
     the fit covariance x max(1, chi2/dof). dashed in the first panel: A rebuilt at the median T, for comparison with B.
"""
import argparse, csv, glob, json, os, re, sys, warnings

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
import llzo_maps as mp
import llzo_sites as ls
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

NAN = float('nan')
VERSION = 'dfr-v2'
SPLIT = (('24d', 0), ('96h', 1))
SUBLAT = {'li24d': ('Li', '24d'), 'li96h': ('Li', '96h'), 'li': ('Li', None), 'la': ('La', None), 'zr': ('Zr', None)}
DEFAULT_SPEC = ('Li', '24d')
FIELDS = ['system', 'element', 'n_dopant', 'T', 'who', 'site', 'r_lo', 'r_hi', 'r_mid', 'n_vox', 'vol_A3', 'cnt_doped', 'cnt_pure',
          'F_kT', 'Fref_kT', 'dF_kT', 'dF_eV', 'se_kT', 'half1_kT', 'half2_kT', 'half_diff_kT', 'Fmax_kT', 'sparse', 'nonconv', 'below_rmin']


def warn(msg): print('warning: ' + msg, file=sys.stderr)
def fname(folder): return os.path.basename(os.path.normpath(folder))
def nat(s): return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def add_args(ap):
    g = ap.add_argument_group('radial dF (dF_radial.py)')
    g.add_argument('--outdir', default='dfr_out', help='figures + csv go here (default dfr_out)')
    g.add_argument('--ref', help='pure-cell folder used as reference (default: the first folder of the groups without dopant atoms)')
    g.add_argument('--spacing', type=float, default=0.2, help='voxel size, A (as 01)')
    g.add_argument('--sigma', type=float, default=0.5, help='pilot / fixed Gaussian sigma, A (as 01)')
    g.add_argument('--sigma-min', type=float, default=0.25, help='adaptive: narrowest kernel, A (as 01)')
    g.add_argument('--sigma-max', type=float, default=0.6, help='adaptive: widest kernel, A (as 01)')
    g.add_argument('--no-adaptive', action='store_true', help='fixed-sigma smoothing (sigma = --sigma) instead of adaptive')
    g.add_argument('--dr', type=float, default=0.25, help='radial bin width, A')
    g.add_argument('--rmax', type=float, help='largest distance, A (default: half the shortest box edge)')
    g.add_argument('--rmin', type=float, default=1.0, help='bins ending at or below this are flagged / not summarised, A')
    g.add_argument('--blocks', type=int, default=4, help='time blocks for the error and the half-split check (>= 2)')
    g.add_argument('--min-count', type=float, default=20.0, help='bins with fewer raw Li samples are flagged sparse (lower bound)')
    g.add_argument('--min-vox', type=int, default=20, help='bins with fewer voxels are flagged sparse')
    g.add_argument('--conv-tol', type=float, default=0.5, help='NOT CONVERGED when the two halves differ by more than this, kT')
    g.add_argument('--thr', type=float, default=0.1, help='|dF| threshold of r_thr, kT')
    g.add_argument('--no-split', action='store_true', help='static mode only: skip the 24d / 96h split (default: all, 24d and 96h)')
    g.add_argument('--static', action='store_true', help='old behaviour: fixed time-mean dopant centre, ownership shells, 24d/96h split')
    g.add_argument('--site-wmin', type=float, default=0.02, help='moving mode: reference sites visited less than this fraction of the frames are dropped')
    g.add_argument('--se-floor', type=float, default=0.05, help='master curves: floor added (in quadrature) to se_kT before weighting, kT')
    return g


# ---------------------------------------------------------------- density (same maps as 01)
def grid_n(L, spacing):
    return np.maximum(np.rint(np.asarray(L, float) / spacing).astype(int), 4)


def density(frac, n, L, a):
    """frac (F,N,3) in [0,1) -> (rho in A^-3 per voxel, raw CIC counts per voxel)."""
    F_ = frac.shape[0]
    voxel = np.asarray(L, float) / n
    vvol = float(np.prod(voxel))
    pts = frac.reshape(-1, 3)
    if a.no_adaptive:
        counts = mp.deposit_cic(pts, n)
        sm = gaussian_filter(counts, a.sigma / voxel, mode='wrap')
    else:
        sm, _, counts, _ = mp.adaptive_smooth(pts, n, voxel, a.sigma, a.sigma_min, a.sigma_max)
    return sm / (F_ * vvol), counts


def make_rec(frac, L, a):
    """Record of one cell: full-window density, per-block densities, raw counts. frac (F,N,3)."""
    F_, N_ = frac.shape[:2]
    L = np.asarray(L, float)
    n = grid_n(L, a.spacing)
    rho, counts = density(frac, n, L, a)
    K = max(2, int(a.blocks))
    blocks = np.array([density(frac[idx], n, L, a)[0] for idx in np.array_split(np.arange(F_), K)], dtype=np.float32)
    return dict(rho=rho.astype(np.float32), blocks=blocks, counts=counts.astype(np.float32), L=L, n=n, F=int(F_),
                rho_uni=float(N_ / np.prod(L)), vvol=float(np.prod(L / n)))


# ---------------------------------------------------------------- lattice frame: site classes, dopant -> crystal site
def axes(n, L):
    return [(np.arange(n[i]) + 0.5) * L[i] / n[i] for i in range(3)]


def class_grid(n, L, li_frac, kind):
    """0 = nearest ideal Li site is 24d, 1 = otherwise (96h). li_frac (S,3) ideal Li sites in the MD frame (fractional)."""
    L = np.asarray(L, float)
    ax = axes(n, L)
    X = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
    p = np.mod(np.asarray(li_frac, float), 1.0) * L
    p[p >= L] = 0.0
    _, i = cKDTree(p, boxsize=L).query(X)
    return np.where(np.asarray(kind)[i] == '24d', 0, 1).astype(np.uint8).reshape(tuple(n))


def site_list(ws, spec):
    """Ideal fractional positions (n,3) of the sublattice named by spec = (element, kind|None). Same order in every system (same CIF + fold)."""
    el, kind = spec
    if el == 'Li':
        f, k = ws['frac'], np.asarray(ws['kind'])
        return f if kind is None else f[k == kind]
    if el in ws['fw']: return ws['fw'][el]
    raise ValueError(f'sublattice {spec} not available from the CIF (Li, La, Zr only)')


def map_dopants(sites, t, L, pfrac):
    """index of the ideal site (sites + t) closest to every dopant (fractional pfrac (C,3)), and the distance, A."""
    S = np.mod(np.asarray(sites, float) + t, 1.0)
    out, dist = [], []
    for pf in np.asarray(pfrac, float).reshape(-1, 3):
        d = S - pf
        d -= np.rint(d)
        r = np.linalg.norm(d * L, axis=1)
        out.append(int(r.argmin())); dist.append(float(r.min()))
    return out, dist


def lattice(cfg, traj, drift, L, cif):
    fold = tuple(int(x) for x in lio.get_fold(cfg, L))
    ws = ls.wyckoff_sites(cif, fold)
    fw = {}
    for el in ('La', 'Zr'):
        m_ = np.isin(traj.types, [t_ for t_, e_ in cfg['types'].items() if e_ == el])
        if m_.any(): fw[el] = (((traj.pos[:, m_, :] - drift[:, None, :]).mean(axis=0) - traj.lo) / L) % 1.0
    t, rms, mx, nfit = ls.fit_shift(fw, ws['fw'], L)
    if rms > 0.35 or mx > 1.0:
        warn(f"{cfg['name']}: La/Zr framework does not sit on the ideal lattice (rms {rms:.2f}, max {mx:.2f} A): site classes unreliable")
    return dict(ws=ws, t=np.asarray(t, float), rms=float(rms), fold=fold)


def dopants(cfg, traj, drift, L):
    """one dict per dopant atom: label, element, fractional position (time mean, unwrapped path), position in A, rms about it, A."""
    out, cnt = [], {}
    for t in cfg['dopant_types']:
        idx = np.where(traj.types == t)[0]
        idx = idx[np.argsort(traj.ids[idx])]
        for i in idx:
            el = cfg['types'][t]
            cnt[el] = cnt.get(el, 0) + 1
            u = (traj.pos[:, i, :] - drift - traj.lo)
            pf = np.mod((u / traj.L).mean(axis=0), 1.0)
            ser = traj.pos[:, i, :] - drift
            rms = float(np.sqrt(((ser - ser.mean(axis=0)) ** 2).sum(-1).mean()))
            out.append(dict(label=f'{el}{cnt[el]}', element=el, pf=pf, p=pf * L, rms=rms))
    return out


# ---------------------------------------------------------------- radial profiles
def dist_owner(n, L, centres):
    """distance (A, periodic minimum image) of every voxel centre to the nearest centre, and which centre (index)."""
    L = np.asarray(L, float)
    ax = axes(n, L)
    d2 = []
    for p in np.asarray(centres, float).reshape(-1, 3):
        dd = [(ax[i] - p[i]) - L[i] * np.rint((ax[i] - p[i]) / L[i]) for i in range(3)]
        d2.append((dd[0] ** 2)[:, None, None] + (dd[1] ** 2)[None, :, None] + (dd[2] ** 2)[None, None, :])
    d2 = np.stack(d2)
    return np.sqrt(d2.min(axis=0)), d2.argmin(axis=0)


def profile(rec, bins, owner, cls, who, site, nb):
    """Shell averages of one cell. who: owner index or None (pooled); site: class id or None (all). bins: int bin of every voxel (>= nb: outside)."""
    m = np.ones(bins.shape, bool)
    if who is not None: m &= owner == who
    if site is not None: m &= cls == site
    b = np.minimum(bins[m], nb)
    nvox = np.bincount(b, minlength=nb + 1)[:nb].astype(float)
    S = lambda g: np.bincount(b, weights=np.asarray(g, float)[m], minlength=nb + 1)[:nb]
    with np.errstate(all='ignore'):
        v = np.where(nvox > 0, nvox, np.nan)
        return dict(nvox=nvox, rho=S(rec['rho']) / v, rho_b=np.array([S(g) / v for g in rec['blocks']]), cnt=S(rec['counts']))


def _lnrho(p, uni):
    with np.errstate(all='ignore'):
        return np.log(p['rho_b'] / uni)


def compare(pd, pr, rd, rr, a, nb):
    """arrays (nb,) of F, Fref, dF, se, halves, flags from the doped / pure shell profiles pd, pr (cells rd, rr), all in kT units."""
    with np.errstate(all='ignore'):
        F = -np.log(pd['rho'] / rd['rho_uni'])
        Fr = -np.log(pr['rho'] / rr['rho_uni'])
        dF = F - Fr
        ld, lr = _lnrho(pd, rd['rho_uni']), _lnrho(pr, rr['rho_uni'])
        K = ld.shape[0]
        se = np.sqrt(np.var(ld, axis=0, ddof=1) / K + np.var(lr, axis=0, ddof=1) / lr.shape[0])
        h = K // 2

        def half(p, u, sl): return -np.log(p['rho_b'][sl].mean(axis=0) / u)
        h1 = half(pd, rd['rho_uni'], slice(0, h)) - half(pr, rr['rho_uni'], slice(0, h))
        h2 = half(pd, rd['rho_uni'], slice(K - h, K)) - half(pr, rr['rho_uni'], slice(K - h, K))
        hd = np.abs(h1 - h2)
        nexp = rd['rho_uni'] * pd['nvox'] * rd['vvol'] * rd['F']
        fmax = np.log(nexp / a.min_count)
        sparse = (np.minimum(pd['cnt'], pr['cnt']) < a.min_count) | (np.minimum(pd['nvox'], pr['nvox']) < a.min_vox)
        nonconv = ~(hd <= a.conv_tol)
    return dict(F=F, Fr=Fr, dF=dF, se=se, h1=h1, h2=h2, hd=hd, fmax=fmax, sparse=sparse, nonconv=nonconv)


def analyse_pair(rd, cd, labels, rr, cr, a, split=True, system='', element='', T=0):
    """rd, rr: cell records (make_rec + 'cls'); cd, cr: dopant centres (C,3), A, in the doped cell and the virtual ones in the pure cell
    (same order as labels). -> list of row dicts (see FIELDS)."""
    kT = lio.KB_EV * T
    rmax = a.rmax or 0.5 * float(min(rd['L'].min(), rr['L'].min()))
    nb = int(np.floor(rmax / a.dr + 1e-9))
    dd, od = dist_owner(rd['n'], rd['L'], cd)
    dr_, orr = dist_owner(rr['n'], rr['L'], cr)
    bd = np.floor(dd / a.dr).astype(np.int64)
    br = np.floor(dr_ / a.dr).astype(np.int64)
    sites = [('all', None)] + ([] if not split else [(s, i) for s, i in SPLIT])
    whos = [('pooled', None)] if len(labels) == 1 else [(l, i) for i, l in enumerate(labels)] + [('pooled', None)]
    edges = np.arange(nb + 1) * a.dr
    rows = []
    for who, wi in whos:
        for sname, sid in sites:
            pd_ = profile(rd, bd, od, rd['cls'], wi, sid, nb)
            pr_ = profile(rr, br, orr, rr['cls'], wi, sid, nb)
            c = compare(pd_, pr_, rd, rr, a, nb)
            for k in range(nb):
                if not (pd_['nvox'][k] > 0 and pr_['nvox'][k] > 0): continue
                rows.append(dict(system=system, element=element, n_dopant=len(labels), T=int(T), who=who, site=sname,
                                 r_lo=float(edges[k]), r_hi=float(edges[k + 1]), r_mid=float(edges[k] + 0.5 * a.dr),
                                 n_vox=int(pd_['nvox'][k]), vol_A3=float(pd_['nvox'][k] * rd['vvol']),
                                 cnt_doped=float(pd_['cnt'][k]), cnt_pure=float(pr_['cnt'][k]),
                                 F_kT=float(c['F'][k]), Fref_kT=float(c['Fr'][k]), dF_kT=float(c['dF'][k]), dF_eV=float(c['dF'][k] * kT),
                                 se_kT=float(c['se'][k]), half1_kT=float(c['h1'][k]), half2_kT=float(c['h2'][k]),
                                 half_diff_kT=float(c['hd'][k]), Fmax_kT=float(c['fmax'][k]), sparse=int(c['sparse'][k]),
                                 nonconv=int(c['nonconv'][k]), below_rmin=int(edges[k + 1] <= a.rmin + 1e-9)))
    return rows


# ---------------------------------------------------------------- summary
def summarise(rows, a):
    """one dict per (system, T, site) from the pooled curve."""
    by = {}
    for r in rows:
        if r['who'] == 'pooled': by.setdefault((r['system'], r['T'], r['site']), []).append(r)
    out = []
    for (sysn, T, site), rs in by.items():
        rs = sorted(rs, key=lambda r: r['r_mid'])
        ok = [r for r in rs if not r['below_rmin'] and np.isfinite(r['dF_kT']) and not r['sparse']]
        d = dict(system=sysn, element=rs[0]['element'], n_dopant=rs[0]['n_dopant'], T=T, site=site, n_valid=len(ok))
        if ok:
            v = np.array([r['dF_kT'] for r in ok])
            k = int(np.argmax(np.abs(v)))
            viol = [i for i, r in enumerate(ok) if abs(r['dF_kT']) >= a.thr]
            d.update(dF_ext_kT=float(v[k]), r_ext=ok[k]['r_mid'], frac_nonconv=float(np.mean([r['nonconv'] for r in ok])),
                     r_thr=(ok[viol[-1]]['r_hi'] if viol else 0.0), censored=int(bool(viol) and viol[-1] == len(ok) - 1))
        else:
            d.update(dF_ext_kT=NAN, r_ext=NAN, frac_nonconv=NAN, r_thr=NAN, censored=0)
        out.append(d)
    return sorted(out, key=lambda d: (nat(d['system']), d['T'], d['site']))


# ---------------------------------------------------------------- moving dopant centre (dopant-frame density)
def moving_dopants(cfg, traj, drift):
    """one dict per dopant atom: label, element, pf (F,3) fractional position per frame of the smoothed path (vibration out, hops kept)."""
    out, cnt = [], {}
    for t in cfg['dopant_types']:
        idx = np.where(traj.types == t)[0]
        idx = idx[np.argsort(traj.ids[idx])]
        for i in idx:
            el = cfg['types'][t]
            cnt[el] = cnt.get(el, 0) + 1
            x = lio.dopant_xyz(cfg, traj, drift, i)
            out.append(dict(label=f'{el}{cnt[el]}', element=el, pf=np.mod((x - traj.lo) / traj.L, 1.0), x=x))
    return out


def mix_sites(ps, ws):
    """ensemble mixture of shell profiles of different reference sites with visit fractions ws."""
    ws = np.asarray(ws, float) / np.sum(ws)
    z = np.nan_to_num
    nv = sum(w * p['nvox'] for w, p in zip(ws, ps))
    with np.errstate(all='ignore'):
        rho = np.where(nv > 0, sum(w * z(p['rho']) for w, p in zip(ws, ps)), np.nan)
        rb = np.where(nv > 0, sum(w * z(p['rho_b']) for w, p in zip(ws, ps)), np.nan)
    return dict(nvox=nv, rho=rho, rho_b=rb, cnt=sum(w * p['cnt'] for w, p in zip(ws, ps)))


def merge_vol(ps):
    """volume-weighted merge of the shell profiles of several dopants (same cell)."""
    z = np.nan_to_num
    nv = sum(p['nvox'] for p in ps)
    with np.errstate(all='ignore'):
        rho = np.where(nv > 0, sum(z(p['rho']) * p['nvox'] for p in ps) / nv, np.nan)
        rb = np.where(nv > 0, sum(z(p['rho_b']) * p['nvox'] for p in ps) / nv, np.nan)
    return dict(nvox=nv, rho=rho, rho_b=rb, cnt=sum(p['cnt'] for p in ps))


def _rows(c, pd_, pr_, rec, who, system, element, ndop, T, edges, nb, a):
    kT = lio.KB_EV * T
    rows = []
    for k in range(nb):
        if not (pd_['nvox'][k] > 0 and pr_['nvox'][k] > 0): continue
        rows.append(dict(system=system, element=element, n_dopant=ndop, T=int(T), who=who, site='all',
                         r_lo=float(edges[k]), r_hi=float(edges[k + 1]), r_mid=float(edges[k] + 0.5 * a.dr),
                         n_vox=int(pd_['nvox'][k]), vol_A3=float(pd_['nvox'][k] * rec['vvol']),
                         cnt_doped=float(pd_['cnt'][k]), cnt_pure=float(pr_['cnt'][k]),
                         F_kT=float(c['F'][k]), Fref_kT=float(c['Fr'][k]), dF_kT=float(c['dF'][k]), dF_eV=float(c['dF'][k] * kT),
                         se_kT=float(c['se'][k]), half1_kT=float(c['h1'][k]), half2_kT=float(c['h2'][k]),
                         half_diff_kT=float(c['hd'][k]), Fmax_kT=float(c['fmax'][k]), sparse=int(c['sparse'][k]),
                         nonconv=int(c['nonconv'][k]), below_rmin=int(edges[k + 1] <= a.rmin + 1e-9)))
    return rows


def analyse_moving(cfg, traj, drift, frac, L, lat, rr, t_ref, spec, a, T):
    """dopant-frame analysis -> (rows, info). frac (F,N,3) Li in [0,1)."""
    md = moving_dopants(cfg, traj, drift)
    if not md: raise RuntimeError('no dopant atoms in this folder (cfg dopant_types matched no atom)')
    F_ = frac.shape[0]
    Lt = traj.L
    rmax = a.rmax or 0.5 * float(min(L.min(), rr['L'].min()))
    nb = int(np.floor(rmax / a.dr + 1e-9))
    edges = np.arange(nb + 1) * a.dr
    n = grid_n(L, a.spacing)
    d0, _ = dist_owner(n, L, np.zeros((1, 3)))                    # distance of every voxel centre to the dopant (origin)
    b0 = np.floor(d0 / a.dr).astype(np.int64)
    system = cfg['name']
    element = '+'.join(sorted({d['element'] for d in md}))
    pds, prs, info, recs = [], [], [], []
    for i, d in enumerate(md):
        r_own = rmax
        for j, e in enumerate(md):
            if j == i: continue
            dd = d['pf'] - e['pf']; dd -= np.rint(dd)
            r_own = min(r_own, 0.5 * float(np.percentile(np.linalg.norm(dd * Lt, axis=1), 5)))
        nb_i = max(1, int(np.floor(r_own / a.dr + 1e-9)))
        rec = make_rec(np.mod(frac - d['pf'][:, None, :], 1.0), L, a)
        bd = np.where(b0 >= nb_i, nb, b0)
        pd_ = profile(rec, bd, None, None, None, None, nb)
        sp = spec.get(d['element'], DEFAULT_SPEC)
        sites = site_list(lat['ws'], sp)
        S = np.mod(sites + lat['t'], 1.0)
        idx_t, far = np.empty(F_, int), np.empty(F_)
        for s0 in range(0, F_, 256):
            sl = slice(s0, s0 + 256)
            dd = S[None, :, :] - d['pf'][sl, None, :]; dd -= np.rint(dd)
            r_ = np.linalg.norm(dd * Lt[sl, None, :], axis=2)
            idx_t[sl], far[sl] = r_.argmin(1), r_.min(1)
        w = np.bincount(idx_t, minlength=len(S)) / F_
        keep = np.where(w >= a.site_wmin)[0]
        if not len(keep): keep = np.array([int(w.argmax())])
        ffar = float(np.mean(far > 1.0))
        if ffar > 0.1: warn(f"{system} T={T}: {d['label']} is > 1 A from every {sp[0]}{sp[1] or ''} site in {100 * ffar:.0f}% of the frames (other sublattice / in transit): reference unreliable")
        prl = []
        for s_ in keep:
            c = np.mod(sites[s_] + t_ref, 1.0) * rr['L']
            dr_, _ = dist_owner(rr['n'], rr['L'], c[None])
            br = np.floor(dr_ / a.dr).astype(np.int64)
            prl.append(profile(rr, np.where(br >= nb_i, nb, br), None, None, None, None, nb))
        pr_ = mix_sites(prl, w[keep])
        pds.append(pd_); prs.append(pr_); recs.append(rec)
        xs = d['x'] - d['x'].mean(axis=0)
        info.append(dict(label=d['label'], element=d['element'], site=f"{sp[0]}{sp[1] or ''}", r_own_A=float(r_own),
                         rms_smoothed_A=float(np.sqrt((xs ** 2).sum(-1).mean())), frac_far=ffar, dist_to_site_A=float(far.mean()),
                         visits={int(k): float(w[k]) for k in keep}))
    rows = []
    items = [(d['label'], pd_, pr_) for d, pd_, pr_ in zip(md, pds, prs)] if len(md) > 1 else []
    items.append(('pooled', merge_vol(pds) if len(md) > 1 else pds[0], merge_vol(prs) if len(md) > 1 else prs[0]))
    for who, pd_, pr_ in items:
        c = compare(pd_, pr_, recs[0], rr, a, nb)
        rows += _rows(c, pd_, pr_, recs[0], who, system, element, len(md), T, edges, nb, a)
    return rows, info


def _doped_moving(folder, cfg, T, a, outp, key, refp, spec, traj, drift, frac, L):
    lat = lattice(cfg, traj, drift, L, a.cif)
    rr, t_ref, fold_ref = load_ref(refp)
    if fold_ref != lat['fold']: raise RuntimeError(f'fold of the pure cell {fold_ref} != {lat["fold"]} of {cfg["name"]}')
    rows, info = analyse_moving(cfg, traj, drift, frac, L, lat, rr, t_ref, spec, a, T)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, 'w') as fh: json.dump(dict(key=key, rows=rows, dopants=info, lat_rms=lat['rms']), fh)


# ---------------------------------------------------------------- cache, per (folder, T) tasks
def cache_key(a, cfg, dump, extra=''):
    return repr((VERSION, a.spacing, a.sigma, a.sigma_min, a.sigma_max, a.no_adaptive, a.dr, a.rmax, a.blocks, a.min_count, a.min_vox, a.conv_tol,
                 a.tmin, a.no_split, a.static, a.site_wmin, cfg.get('dop_smooth_ps'), tuple(sorted(cfg['types'].items())), tuple(cfg['dopant_types']), dump, os.path.getmtime(dump),
                 os.path.getmtime(a.cif), extra))


def load_cell(cfg, T, a):
    dump = lio.dump_path(cfg, T)
    traj = lio.prepare(cfg, dump, a.tmin, verbose=False)
    drift = lio.framework_drift(cfg, traj)
    li = traj.pos[:, lio.li_mask(cfg, traj), :] - drift[:, None, :]
    L = traj.L.mean(axis=0)
    return dump, traj, drift, lio.frac_wrapped(li, traj.lo, traj.L), L


def compute_ref(task):
    """pure cell -> <folder>/dfr/<name>/ref_T<T>K.npz (grids + lattice frame). Returns an error string or None."""
    folder, cfg, T, a, outp, key = task
    try:
        dump, traj, drift, frac, L = load_cell(cfg, T, a)
        rec = make_rec(frac, L, a)
        lat = lattice(cfg, traj, drift, L, a.cif)
        cls = class_grid(rec['n'], L, np.mod(lat['ws']['frac'] + lat['t'], 1.0), lat['ws']['kind'])
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        np.savez_compressed(outp, key=key, rho=rec['rho'], blocks=rec['blocks'], counts=rec['counts'], cls=cls, L=L, n=rec['n'], F=rec['F'],
                            rho_uni=rec['rho_uni'], vvol=rec['vvol'], t=lat['t'], fold=np.array(lat['fold']), lat_rms=lat['rms'])
        return None
    except Exception as e:
        import traceback
        return f'{folder} T={T} (reference): {type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}'


def load_ref(path):
    z = np.load(path, allow_pickle=False)
    rec = dict(rho=z['rho'], blocks=z['blocks'], counts=z['counts'], cls=z['cls'], L=z['L'], n=z['n'].astype(int), F=int(z['F']),
               rho_uni=float(z['rho_uni']), vvol=float(z['vvol']))
    return rec, z['t'], tuple(int(x) for x in z['fold'])


def compute_doped(task):
    """doped cell + its matched reference -> <folder>/dfr/<name>/dfr_T<T>K.json. Returns an error string or None."""
    folder, cfg, T, a, outp, key, refp, spec = task
    try:
        dump, traj, drift, frac, L = load_cell(cfg, T, a)
        if not a.static:
            _doped_moving(folder, cfg, T, a, outp, key, refp, spec, traj, drift, frac, L)
            return None
        rd = make_rec(frac, L, a)
        lat = lattice(cfg, traj, drift, L, a.cif)
        rd['cls'] = class_grid(rd['n'], L, np.mod(lat['ws']['frac'] + lat['t'], 1.0), lat['ws']['kind'])
        dops = dopants(cfg, traj, drift, L)
        if not dops: raise RuntimeError('no dopant atoms in this folder (cfg dopant_types matched no atom)')
        for d in dops:
            if d['rms'] > 1.0: warn(f"{cfg['name']} T={T}: {d['label']} moves {d['rms']:.2f} A rms about its mean position: a time-averaged centre is blurred")
        rr, t_ref, fold_ref = load_ref(refp)
        if fold_ref != lat['fold']: raise RuntimeError(f'fold of the pure cell {fold_ref} != {lat["fold"]} of {cfg["name"]}')
        info, idxs, cr = [], [], []
        for d in dops:
            sp = spec.get(d['element'], DEFAULT_SPEC)
            sites = site_list(lat['ws'], sp)
            (i,), (dist,) = map_dopants(sites, lat['t'], L, d['pf'][None])
            if dist > 1.0: warn(f"{cfg['name']} T={T}: {d['label']} is {dist:.2f} A from the nearest {sp[0]}{sp[1] or ''} site: wrong sublattice spec?")
            if i in idxs: warn(f"{cfg['name']} T={T}: {d['label']} maps to the same crystal site as another dopant")
            idxs.append(i)
            cr.append(np.mod(sites[i] + t_ref, 1.0) * rr['L'])
            info.append(dict(label=d['label'], element=d['element'], site=f'{sp[0]}{sp[1] or ""}', site_index=i, dist_to_site_A=dist, rms_A=d['rms']))
        rows = analyse_pair(rd, np.array([d['p'] for d in dops]), [d['label'] for d in dops], rr, np.array(cr), a, not a.no_split,
                            cfg['name'], '+'.join(sorted({d['element'] for d in dops})), T)
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        with open(outp, 'w') as fh: json.dump(dict(key=key, rows=rows, dopants=info, lat_rms=lat['rms']), fh)
        return None
    except Exception as e:
        import traceback
        return f'{folder} T={T}: {type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}'


def _run(fn, tasks, jobs):
    if jobs > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=jobs) as ex: return list(ex.map(fn, tasks))
    return [fn(t) for t in tasks]


def spec_for(g):
    """{element: (element, kind)} from the group's tokens like Al=Li24d (same tokens as build_doc.py)."""
    out = {}
    for el, txt in (g.get('sites') or {}).items():
        k = txt.lower().replace('-', '').replace('_', '')
        if k in SUBLAT: out[el] = SUBLAT[k]
        else: warn(f'{el}={txt}: sublattice not supported here (Li24d Li96h Li La Zr): using Li24d')
    return out


def collect(groups, a, recompute=False):
    """-> ({system: {T: {'rows', 'dopants'}}}, order). Reference cells first, then the doped cells."""
    if not os.path.exists(a.cif): sys.exit(f'CIF not found: {a.cif} (--cif)')
    refcfg, refdir, doped = None, None, []
    if a.ref:
        refdir = a.ref
        refcfg = lio.load_config(refdir, None, a)
    for g in groups:
        for f in g['folders']:
            if not os.path.isdir(f): warn(f'not a folder: {f}'); continue
            try:
                ga = argparse.Namespace(**vars(a))
                if g.get('elements'): ga.elements = g['elements']
                cfg = lio.load_config(f, None, ga)
            except SystemExit as e:
                warn(f'skipped {f}: {e}'); continue
            if cfg['dopant_types']: doped.append((f, cfg, spec_for(g)))
            elif refcfg is None: refdir, refcfg = f, cfg
            elif f != refdir and not a.ref: warn(f'{f}: second pure folder, ignored (reference = {refdir}; --ref to choose)')
    if refcfg is None: sys.exit('no pure (dopant-free) folder among the groups and no --ref: nothing to compare against')
    if not doped: sys.exit('no doped folder found')
    order = [fname(f) for f, _, _ in doped]
    temps = {}
    for f, cfg, _ in doped:
        try: temps[f] = sorted(a.T or lio.temperatures(cfg))
        except SystemExit as e: warn(f'{f}: {e}'); temps[f] = []
    allT = sorted({t for v in temps.values() for t in v})

    ref_tasks, refpath, refkey = [], {}, {}
    for T in allT:
        try: dump = lio.dump_path(refcfg, T)
        except SystemExit as e: warn(f'reference has no dump at {T} K: {e}'); continue
        p = os.path.join(refdir, 'dfr', refcfg['name'], f'ref_T{T}K.npz')
        k = cache_key(a, refcfg, dump)
        ok = False
        if os.path.exists(p) and not recompute:
            try: ok = str(np.load(p)['key']) == k
            except Exception: ok = False
        refpath[T], refkey[T] = p, k
        if not ok: ref_tasks.append((refdir, refcfg, T, a, p, k))
    print(f'[dF_radial] reference {fname(refdir)}: {len(ref_tasks)} (T) to compute, {len(refpath) - len(ref_tasks)} cached')
    for e in _run(compute_ref, ref_tasks, a.jobs):
        if e: warn(e)

    tasks, plan = [], []
    for f, cfg, spec in doped:
        for T in temps[f]:
            if T not in refpath or not os.path.exists(refpath[T]): warn(f'{f} T={T}: no reference at this T, skipped'); continue
            try: dump = lio.dump_path(cfg, T)
            except SystemExit as e: warn(f'{f} T={T}: {e}'); continue
            outp = os.path.join(f, 'dfr', cfg['name'], f'dfr_T{T}K.json')
            key = cache_key(a, cfg, dump, repr((refkey[T], os.path.getmtime(refpath[T]), sorted(spec.items()))))
            plan.append((cfg['name'], T, outp))
            ok = False
            if os.path.exists(outp) and not recompute:
                try: ok = json.load(open(outp))['key'] == key
                except Exception: ok = False
            if not ok: tasks.append((f, cfg, T, a, outp, key, refpath[T], spec))
    print(f'[dF_radial] {len(tasks)} (folder, T) to compute, {len(plan) - len(tasks)} cached')
    for e in _run(compute_doped, tasks, a.jobs):
        if e: warn(e)
    res = {}
    for name, T, p in plan:
        if os.path.exists(p): res.setdefault(name, {})[T] = json.load(open(p))
    return res, order


# ---------------------------------------------------------------- outputs
def all_rows(res, order):
    rows = []
    for s in order:
        for T in sorted(res.get(s, {})): rows += res[s][T]['rows']
    return rows


def write_csv(rows, summ, outdir):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'dF_radial_long.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, FIELDS); w.writeheader()
        for r in rows: w.writerow({k: (f'{r[k]:.6g}' if isinstance(r[k], float) else r[k]) for k in FIELDS})
    cols = ['system', 'element', 'n_dopant', 'T', 'site', 'n_valid', 'dF_ext_kT', 'r_ext', 'r_thr', 'censored', 'frac_nonconv']
    with open(os.path.join(outdir, 'dF_summary.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, cols); w.writeheader()
        for d in summ: w.writerow({k: (f'{d[k]:.6g}' if isinstance(d[k], float) else d[k]) for k in cols})


def curve(rows, who, site):
    rs = sorted((r for r in rows if r['who'] == who and r['site'] == site), key=lambda r: r['r_mid'])
    dr = rs[0]['r_hi'] - rs[0]['r_lo']
    k = np.array([int(round((r['r_mid'] - 0.5 * dr) / dr)) for r in rs])
    full = np.arange(k.min(), k.max() + 1)
    out = {'r_mid': (full + 0.5) * dr}
    for name in ('F_kT', 'Fref_kT', 'dF_kT', 'dF_eV', 'se_kT', 'sparse', 'nonconv', 'below_rmin'):
        v = np.full(len(full), 0.0 if name in ('sparse', 'nonconv', 'below_rmin') else NAN)
        v[k - k.min()] = [r[name] for r in rs]
        out[name] = v
    return out                                                    # bins without voxels are NaN: lines break there instead of joining


def _draw(ax, c, y, color, lw=1.6, label=None, band=None, marks=True):
    ok = np.isfinite(y)
    ax.plot(c['r_mid'], np.where(ok, y, np.nan), '-', color=color, lw=lw, label=label)
    if band is not None:
        m = ok & np.isfinite(band)
        ax.fill_between(c['r_mid'], np.where(m, y - band, 0), np.where(m, y + band, 0), where=m, color=color, alpha=0.2, lw=0)
    if marks:
        sp = ok & (c['sparse'] > 0)
        nc = ok & (c['nonconv'] > 0) & ~sp
        good = ok & ~sp
        ax.plot(c['r_mid'][good], y[good], 'o', ms=2.5, color=color)
        ax.plot(c['r_mid'][sp], y[sp], 'o', ms=4, mfc='none', color=color)
        ax.plot(c['r_mid'][nc], y[nc], 'x', ms=5, color='tab:red')


def _finish(ax, a, ylabel, xlabel=True):
    ax.axvspan(0, a.rmin, color='0.9', zorder=0)
    ax.axhline(0, color='0.5', lw=0.6)
    if getattr(a, 'xmax', None): ax.set_xlim(0, a.xmax)           # every plot spans 0 .. outer edge of the last bin
    ax.set_ylabel(ylabel)
    if xlabel: ax.set_xlabel('distance to dopant (A)')


def _savefig(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_system(rows, sysn, T, outdir, a):
    rs = [r for r in rows if r['system'] == sysn and r['T'] == T]
    sites = [s for s in ('all', '24d', '96h') if any(r['site'] == s for r in rs)]
    whos = [w for w in dict.fromkeys(r['who'] for r in rs) if w != 'pooled']
    cmap = plt.get_cmap('tab10')
    fig, axs = plt.subplots(2, len(sites), figsize=(4.6 * len(sites), 6.4), squeeze=False, sharex=True)
    for j, s in enumerate(sites):
        c = curve(rs, 'pooled', s)
        ax = axs[0, j]
        ax.plot(c['r_mid'], c['F_kT'], '-', color='k', label='doped')
        ax.plot(c['r_mid'], c['Fref_kT'], '--', color='0.4', label='pure (same sites)')
        _finish(ax, a, 'F / kT = -ln(rho/rho_uniform)', xlabel=False)
        ax.set_title(f'{sysn}  {T} K  site: {s}', fontsize=9)
        if j == 0: ax.legend(fontsize=7)
        ax = axs[1, j]
        for i, w in enumerate(whos):
            _draw(ax, curve(rs, w, s), curve(rs, w, s)['dF_kT'], cmap(i), lw=0.9, label=w, marks=False)
        _draw(ax, c, c['dF_kT'], 'k', lw=2.0, label='pooled', band=c['se_kT'])
        _finish(ax, a, 'dF / kT  (doped - pure)')
        if j == 0: ax.legend(fontsize=7)
    fig.text(0.01, 0.005, 'open = sparse (lower bound), red x = not converged (halves differ), grey = d < rmin', fontsize=7)
    _savefig(fig, os.path.join(outdir, f'dF_{sysn}_T{T}K.png'))


def fig_elements(rows, T, site, outdir, a):
    rs = [r for r in rows if r['T'] == T and r['site'] == site and r['who'] == 'pooled']
    els = sorted({r['element'] for r in rs})
    if not els: return
    cmap = plt.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for i, el in enumerate(els):
        sy = sorted({r['system'] for r in rs if r['element'] == el}, key=nat)
        cs = [curve([r for r in rs if r['system'] == s], 'pooled', site) for s in sy]
        for c in cs: ax.plot(c['r_mid'], c['dF_kT'], '--', color=cmap(i), lw=0.6, alpha=0.7)
        grid = np.arange(0.5 * a.dr, max(c['r_mid'].max() for c in cs) + 1e-9, a.dr)
        M = np.full((len(cs), len(grid)), np.nan)
        for k, c in enumerate(cs):
            for r_, y_, sp_ in zip(c['r_mid'], c['dF_kT'], c['sparse']):
                j = int(round((r_ - grid[0]) / a.dr))
                if 0 <= j < len(grid) and not sp_: M[k, j] = y_
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mean, lo, hi = np.nanmean(M, axis=0), np.nanmin(M, axis=0), np.nanmax(M, axis=0)
        ok = np.isfinite(mean)
        ax.fill_between(grid[ok], lo[ok], hi[ok], color=cmap(i), alpha=0.15, lw=0)
        ax.plot(grid[ok], mean[ok], '-', color=cmap(i), lw=2, label=f'{el} (n={len(cs)})')
    _finish(ax, a, 'dF / kT  (doped - pure)')
    ax.set_title(f'{T} K  site: {site}  (line = mean over systems, band = min-max, dashed = each system)', fontsize=8)
    ax.legend(fontsize=8)
    _savefig(fig, os.path.join(outdir, f'dF_elements_{site}_T{T}K.png'))


def fig_tscan(rows, sysn, outdir, a):
    rs = [r for r in rows if r['system'] == sysn and r['site'] == 'all' and r['who'] == 'pooled']
    Ts = sorted({r['T'] for r in rs})
    if len(Ts) < 2: return
    cm = plt.get_cmap('viridis')
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.0), sharex=True)
    for i, T in enumerate(Ts):
        c = curve([r for r in rs if r['T'] == T], 'pooled', 'all')
        col = cm(i / max(1, len(Ts) - 1))
        _draw(axs[0], c, c['dF_kT'], col, lw=1.4, label=f'{T} K', marks=False)
        _draw(axs[1], c, c['dF_eV'], col, lw=1.4, label=f'{T} K', marks=False)
    _finish(axs[0], a, 'dF / kT'); _finish(axs[1], a, 'dF (eV) = kT x dF/kT')
    axs[0].set_title(f'{sysn}: collapse here -> temperature independent in kT (entropic)', fontsize=8)
    axs[1].set_title('collapse here -> pure energy (scales as 1/T in kT)', fontsize=8)
    axs[0].legend(fontsize=7)
    _savefig(fig, os.path.join(outdir, f'dF_Tscan_{sysn}.png'))


def make_figs(rows, order, outdir, a):
    os.makedirs(outdir, exist_ok=True)
    out = []
    for s in order:
        for T in sorted({r['T'] for r in rows if r['system'] == s}):
            fig_system(rows, s, T, outdir, a); out.append(f'dF_{s}_T{T}K.png')
        fig_tscan(rows, s, outdir, a)
    for T in sorted({r['T'] for r in rows}):
        for site in sorted({r['site'] for r in rows}):
            fig_elements(rows, T, site, outdir, a); out.append(f'dF_elements_{site}_T{T}K.png')
    return out


# ---------------------------------------------------------------- master curves: one per dopant element
def _wls(x, y, w):
    X = np.stack([np.ones_like(x), x], 1)
    XtW = X.T * w
    cov = np.linalg.inv(XtW @ X)
    beta = cov @ (XtW @ y)
    dof = len(y) - 2
    sc = max(1.0, float((w * (y - X @ beta) ** 2).sum() / dof)) if dof > 0 else 1.0
    return beta, cov * sc


def master_points(rows, a):
    """valid points of the individual-dopant curves (site 'all'): {element: [(bin, r_mid, T, y, weight, curve id)]}."""
    out = {}
    for r in rows:
        if r['site'] != 'all' or (r['n_dopant'] > 1) == (r['who'] == 'pooled'): continue
        if r['sparse'] or r['nonconv'] or r['below_rmin'] or not (np.isfinite(r['dF_kT']) and np.isfinite(r['se_kT'])): continue
        el = r['element'] if r['who'] == 'pooled' else re.sub(r'\d+$', '', r['who'])
        dr = r['r_hi'] - r['r_lo']
        out.setdefault(el, []).append((int(round((r['r_mid'] - 0.5 * dr) / dr)), r['r_mid'], r['T'], r['dF_kT'],
                                       1.0 / (r['se_kT'] ** 2 + a.se_floor ** 2), (r['system'], r['who'])))
    return out


def master_curves(rows, a):
    """-> {element: dict(B=..., A=..., Tref=...)}; each curve = list of (r_mid, value, lo, hi, n_curves, n_T)."""
    res = {}
    for el, pts in master_points(rows, a).items():
        Ts = sorted({p[2] for p in pts})
        Tref = float(np.median(Ts))
        B, A, E, S, R = [], [], [], [], []
        for k in sorted({p[0] for p in pts}):
            ps = [p for p in pts if p[0] == k]
            r_mid = ps[0][1]
            mT = []
            for T in sorted({p[2] for p in ps}):
                q = [p for p in ps if p[2] == T]
                mT.append(sum(p[4] * p[3] for p in q) / sum(p[4] for p in q))
            B.append((r_mid, float(np.mean(mT)), float(np.min(mT)), float(np.max(mT)), len({p[5] for p in ps}), len(mT)))
            if len({p[2] for p in ps}) >= 3:
                x = np.array([1000.0 / p[2] for p in ps]); y = np.array([p[3] for p in ps]); w = np.array([p[4] for p in ps])
                beta, cov = _wls(x, y, w)
                a_, b_ = beta[0], beta[1]
                dE, edE = 1000.0 * lio.KB_EV * b_, 1000.0 * lio.KB_EV * float(np.sqrt(cov[1, 1]))
                xr = 1000.0 / Tref
                e_rec = float(np.sqrt(cov[0, 0] + xr ** 2 * cov[1, 1] + 2 * xr * cov[0, 1]))
                n_c, n_T = len({p[5] for p in ps}), len({p[2] for p in ps})
                E.append((r_mid, float(dE), float(dE - edE), float(dE + edE), n_c, n_T))
                S.append((r_mid, float(-a_), float(-a_ - np.sqrt(cov[0, 0])), float(-a_ + np.sqrt(cov[0, 0])), n_c, n_T))
                v = float(a_ + b_ * xr)
                R.append((r_mid, v, v - e_rec, v + e_rec, n_c, n_T))
        res[el] = dict(B=B, E=E, S=S, R=R, Tref=Tref, Ts=Ts)
    return res


def fig_master(rows, outdir, a):
    res = master_curves(rows, a)
    if not res: return
    cmap = plt.get_cmap('tab10')
    fig, axs = plt.subplots(1, 3, figsize=(14.0, 4.2), sharex=True)
    lines = []
    for i, (el, m) in enumerate(sorted(res.items())):
        col = cmap(i)
        for key, ax, lab in (('B', axs[0], f'{el} B: mean over T (band = T min-max)'), ('E', axs[1], el), ('S', axs[2], el)):
            c = np.array(m[key], float)
            if not len(c): continue
            ax.plot(c[:, 0], c[:, 1], '-', color=col, lw=1.8, label=lab)
            ax.fill_between(c[:, 0], c[:, 2], c[:, 3], color=col, alpha=0.2, lw=0)
        c = np.array(m['R'], float)
        if len(c): axs[0].plot(c[:, 0], c[:, 1], '--', color=col, lw=1.2, label=f"{el} A: fit at {m['Tref']:.0f} K")
        for key, name in (('B', 'B_dF_kT'), ('E', 'A_dE_eV'), ('S', 'A_dS_over_k'), ('R', 'A_dF_kT_at_Tref')):
            for r_mid, v, lo, hi, n_c, n_T in m[key]: lines.append((el, name, r_mid, v, lo, hi, n_c, n_T))
    _finish(axs[0], a, 'dF / kT  (doped - pure)')
    _finish(axs[1], a, 'dE (eV) = k_B x slope of dF/kT vs 1/T')
    _finish(axs[2], a, 'dS / k = -intercept')
    axs[0].set_title('B: T-averaged   A: fit at median T (dashed)', fontsize=8)
    axs[1].set_title('A: energetic part, T independent', fontsize=8)
    axs[2].set_title('A: entropic part, T independent', fontsize=8)
    axs[0].legend(fontsize=6)
    axs[1].legend(fontsize=7)
    _savefig(fig, os.path.join(outdir, 'dF_master.png'))
    with open(os.path.join(outdir, 'dF_master.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['element', 'curve', 'r_mid', 'value', 'lo', 'hi', 'n_curves', 'n_T'])
        for l in lines: w.writerow([l[0], l[1]] + [f'{x:.6g}' if isinstance(x, float) else x for x in l[2:]])


def print_summary(summ):
    print('\nsystem          T    site   dF_ext/kT  at_r   r_thr  nonconv  (pooled, r >= rmin; * = r_thr reaches the edge: lower bound)')
    for d in summ:
        if d['site'] != 'all': continue
        print(f"{d['system']:<14} {d['T']:>5} {d['site']:>5}  {d['dF_ext_kT']:>9.2f}  {d['r_ext']:>5.2f}  {d['r_thr']:>5.2f}{'*' if d['censored'] else ' '}  {d['frac_nonconv']:.2f}")


def main():
    try:
        import build_doc as bd                                    # group syntax + CIF default, taken from there
    except ImportError as e:
        sys.exit(f'build_doc.py (and python-docx) must be importable next to this file for the group syntax: {e}')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('items', nargs='+', help='groups: {pure} {A B C Al=Li24d elements=Li,La,Zr,Al,O}')
    ap.add_argument('--scripts', help='folder with llzo_io.py ... (default: ./with_flag next to this file)')
    ap.add_argument('--cif', default=bd.CIF_DEFAULT, help='pristine Ia-3d garnet CIF (Li, La, Zr, O)')
    ap.add_argument('--T', type=int, nargs='*', help='temperatures (default: all dumps found)')
    ap.add_argument('--tmin', type=float, help='analysis window start, ps (default: tmin_ps of the settings)')
    ap.add_argument('--jobs', type=int, default=1, help='parallel (folder, T) computations')
    ap.add_argument('--recompute', action='store_true', help='ignore the cache')
    add_args(ap)
    lio.add_config_args(ap)
    a = ap.parse_args()
    if a.blocks < 2: ap.error('--blocks must be >= 2')
    groups = bd.parse_groups(a.items)
    if not groups: sys.exit('no groups given')
    for g in groups: g['folders'] = bd.expand(g['folders'])
    res, order = collect(groups, a, a.recompute)
    rows = all_rows(res, order)
    if not rows: sys.exit('no result')
    a.xmax = max(r['r_hi'] for r in rows)
    summ = summarise(rows, a)
    write_csv(rows, summ, a.outdir)
    figs = make_figs(rows, order, a.outdir, a)
    fig_master(rows, a.outdir, a)
    print(f'[dF_radial] {a.outdir}/dF_radial_long.csv dF_summary.csv dF_master.csv  ({len(figs)} figures + dF_Tscan_* + dF_master)')
    print_summary(summ)


if __name__ == '__main__':
    main()
