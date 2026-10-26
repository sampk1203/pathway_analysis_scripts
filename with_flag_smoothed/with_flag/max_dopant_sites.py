#!/usr/bin/env python3
"""
max_dopant_sites.py - how many dopants fit on the 24d (or 96h) sites of a garnet supercell if no two may be closer than dmin?

Maximum independent set on the periodic site graph: nodes = Li 24d sites of the pristine Ia-3d CIF tiled over the supercell (taken
from llzo_sites.wyckoff_sites, so the site list, tiling and index order are exactly those of 04_sites_hops.py / build_doc.py);
two sites are joined by an edge if their minimum-image distance is < dmin. We want the largest set of nodes with no edge.

Solvers: exact (scipy MILP with edge-clique constraints, time-limited; says whether the optimum is PROVEN) and a randomised
min-degree greedy with restarts. 'both' (default) takes the better one. For big cells / small dmin use --method greedy.

dmin: pass it directly (--dmin A) or as a radius of influence (--radius R  ->  dmin = 2 R: the spheres of radius R do not overlap).
Periodic images count: a dopant also sees its own images. If min(box edge) < dmin even ONE dopant violates dmin; the script stops
and tells you (or --allow-self-image to ignore the images of the dopant itself).

--spread: after the maximum n is known, also find the LARGEST minimum distance that still allows n dopants (the most evenly
spread arrangement with n dopants), via a binary search over the distinct pair distances.

usage:
  python max_dopant_sites.py --fold 1 1 2 --dmin 5.0
  python max_dopant_sites.py --fold 2 2 2 --radius 3.0 --spread --out al24d_2x2x2
  python max_dopant_sites.py --fold 1 1 2 --dmin 0          # prints the 24d-24d distance shells, places every site
outputs: <out>_sites.csv (pick, index in the 24d list, index in the full Li site list of llzo_sites, fractional + cartesian),
         <out>.xyz (extended xyz of the dopant atoms with the lattice)
Charge compensation (Al3+ on Li+: 2 extra Li vacancies per Al) is NOT chosen here; the stoichiometry line only counts.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CIF_DEFAULT = '/media/qpivolta/8TB/archive_vm/backup-26-3-26-voltaet/2_debdipto/2x2x4-new/pathway_analysis_scripts/cubic_LLZO_wyckoff.cif'


def _scripts_dir():
    a = sys.argv
    for i, x in enumerate(a):
        if x == '--scripts' and i + 1 < len(a): return a[i + 1]
        if x.startswith('--scripts='): return x.split('=', 1)[1]
    return HERE


sys.path.insert(0, os.path.abspath(_scripts_dir()))
import llzo_sites as ls                                   # noqa: E402


# ---------------------------------------------------------------- graph
def distance_matrix(frac, L):
    d = ls._mi(frac[:, None, :] - frac[None, :, :])
    return np.linalg.norm(d * L, axis=-1)


def conflicts(D, dmin, tol):
    A = D < dmin - tol
    np.fill_diagonal(A, False)
    return A


# ---------------------------------------------------------------- solvers
def edge_clique_cover(A):
    """Greedy cover of every edge by a clique (stronger LP relaxation than one constraint per edge)."""
    N = len(A)
    covered = np.zeros((N, N), bool)
    out = []
    for i, j in np.argwhere(np.triu(A, 1)):
        if covered[i, j]: continue
        cl = [int(i), int(j)]
        for k in np.flatnonzero(A[i] & A[j]):
            if all(A[k, m] for m in cl): cl.append(int(k))
        cl_ = np.array(cl)
        covered[np.ix_(cl_, cl_)] = True
        out.append(cl)
    return out


def solve_milp(A, tl):
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError:
        print('  [warn] scipy.optimize.milp not available (scipy >= 1.9 needed): exact solver skipped')
        return None, False
    N = len(A)
    cl = edge_clique_cover(A)
    M = lil_matrix((len(cl), N))
    for r, c in enumerate(cl): M[r, c] = 1
    res = milp(c=-np.ones(N), constraints=LinearConstraint(M.tocsr(), -np.inf, 1), integrality=np.ones(N),
               bounds=Bounds(0, 1), options=dict(time_limit=float(tl), disp=False))
    if res.x is None: return None, False
    return res.x > 0.5, res.status == 0


def greedy(A, rng, restarts, deadline):
    """Randomised min-degree greedy: pick a node of smallest remaining degree (random among ties), drop it and its neighbours."""
    N = len(A)
    best = np.zeros(N, bool)
    ar = np.arange(N)
    for _ in range(restarts):
        if time.time() > deadline: break
        alive = np.ones(N, bool)
        deg = A.sum(1).astype(float)
        chosen = []
        while alive.any():
            cand = np.flatnonzero(alive)
            ties = cand[deg[cand] == deg[cand].min()]
            v = int(rng.choice(ties))
            chosen.append(v)
            rem = np.flatnonzero(alive & (A[v] | (ar == v)))
            alive[rem] = False
            deg -= A[:, rem].sum(1)
        if len(chosen) > best.sum():
            best = np.zeros(N, bool)
            best[chosen] = True
    return best


def solve(A, method, tl, restarts, seed):
    N = len(A)
    if not A.any(): return np.ones(N, bool), True
    best, proven = None, False
    if method in ('both', 'milp'):
        m, ok = solve_milp(A, tl)
        if m is not None: best, proven = m, ok
    if method in ('both', 'greedy'):
        g = greedy(A, np.random.default_rng(seed), restarts, time.time() + tl / 2)
        if best is None or g.sum() > best.sum(): best, proven = g, False
    return best, proven


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cif', default=CIF_DEFAULT, help='pristine Ia-3d garnet CIF (Li 24d + 96h)')
    ap.add_argument('--scripts', help='folder with llzo_sites.py (default: next to this file)')
    ap.add_argument('--fold', type=int, nargs=3, default=[1, 1, 2], metavar=('NX', 'NY', 'NZ'),
                    help='conventional cells along a b c (default 1 1 2 = 192 O). One conventional cell = Li56 La24 Zr16 O96')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--dmin', type=float, help='minimum dopant-dopant distance, A')
    g.add_argument('--radius', type=float, help='radius of influence, A: dmin = 2 x radius (spheres must not overlap)')
    ap.add_argument('--sublattice', choices=['24d', '96h'], default='24d')
    ap.add_argument('--dopant', default='Al', help='label written to the output files')
    ap.add_argument('--zfu', type=int, default=8, help='formula units per conventional cell (Li7La3Zr2O12: 8)')
    ap.add_argument('--tol', type=float, default=1e-3, help='A; distances within tol of dmin count as allowed')
    ap.add_argument('--method', choices=['both', 'milp', 'greedy'], default='both')
    ap.add_argument('--time', type=float, default=60.0, help='s, time limit of the exact solver (greedy gets half of it)')
    ap.add_argument('--restarts', type=int, default=500, help='greedy restarts')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--spread', action='store_true', help='also maximise the minimum distance for the maximum n')
    ap.add_argument('--allow-self-image', action='store_true', help='ignore the dopant\'s own periodic images')
    ap.add_argument('--out', default='dopant_sites')
    a = ap.parse_args()
    dmin = a.dmin if a.dmin is not None else 2.0 * a.radius

    ws = ls.wyckoff_sites(a.cif, a.fold)
    if ws['nops'] != 96: print('[WARN] not 96 symmetry operators: is this the Ia-3d garnet CIF?')
    sel = np.where(ws['kind'] == a.sublattice)[0]
    if not len(sel): sys.exit(f"no {a.sublattice} sites in the CIF (Li orbits found: {ws['orbits']})")
    frac = ws['frac'][sel]
    L = ws['a'] * np.asarray(a.fold, float)
    ncell = int(np.prod(a.fold))
    N = len(frac)
    print(f"[cell] a = {ws['a']} A, fold {tuple(a.fold)} -> box {L[0]:.3f} x {L[1]:.3f} x {L[2]:.3f} A, {ncell} conventional cell(s) = "
          f"{a.zfu * ncell} formula units (pristine Li{7 * a.zfu * ncell} La{3 * a.zfu * ncell} Zr{2 * a.zfu * ncell} O{12 * a.zfu * ncell})")
    print(f"[sites] {N} {a.sublattice} sites ({N // ncell} per cell, expected {24 if a.sublattice == '24d' else 96}); dmin = {dmin:g} A")

    D = distance_matrix(frac, L)
    iu = np.triu_indices(N, 1)
    ud = np.unique(np.round(D[iu], 2))
    row = np.round(D[0][1:], 2)
    sh = [(d, int((row == d).sum())) for d in np.unique(row) if d < min(12.0, L.min() / 2)]
    print('[geometry] distances from one site (A: number of neighbours): ' + ', '.join(f'{d:g}: {n}' for d, n in sh[:14]))

    if not a.allow_self_image and L.min() < dmin - a.tol:
        sys.exit(f"[STOP] shortest box edge {L.min():.2f} A < dmin {dmin:g} A: a dopant would sit closer than dmin to its OWN periodic image, "
                 f"so even one dopant violates the rule. Use a larger --fold (every edge >= dmin), or --allow-self-image.")

    t0 = time.time()
    A = conflicts(D, dmin, a.tol)
    print(f"[graph] {int(A.sum() // 2)} conflicting pairs (each site conflicts with {A.sum(1).min()}-{A.sum(1).max()} others)")
    mask, proven = solve(A, a.method, a.time, a.restarts, a.seed)
    n = int(mask.sum())
    print(f"[result] {n} {a.dopant} on {a.sublattice} sites  ({'PROVEN optimal' if proven else 'best found, optimality not proven (try --time longer / --method milp)'}, "
          f"{time.time() - t0:.1f} s)")

    if a.spread and n > 1:
        cand = np.unique(np.round(D[iu][D[iu] >= dmin - a.tol], 3))
        lo, hi, best = 0, len(cand) - 1, (mask, dmin)
        while lo <= hi:
            mid = (lo + hi) // 2
            m, _ = solve(conflicts(D, cand[mid], a.tol), a.method, a.time, a.restarts, a.seed)
            if m is not None and m.sum() >= n:
                best = (m, float(cand[mid])); lo = mid + 1
            else:
                hi = mid - 1
        mask, dmin_s = best
        print(f"[spread] the same n = {n} also fits with minimum distance up to {dmin_s:g} A (most evenly spread arrangement written)")

    pick = np.flatnonzero(mask)
    if n > 1:
        dd = D[np.ix_(pick, pick)] + np.where(np.eye(n, dtype=bool), np.inf, 0.0)
        print(f"[check] minimum dopant-dopant distance in the set: {dd.min():.3f} A (>= dmin {dmin:g}: {dd.min() >= dmin - a.tol})")
    nfu = a.zfu * ncell
    x = n / nfu
    vol = float(np.prod(L))
    print(f"[stoichiometry] {n} dopant atoms / {nfu} formula units -> x = {x:.4f} per formula unit; Al3+ on Li+ (2 extra vacancies per dopant): "
          f"Li{7 - 3 * x:.3f}{a.dopant}{x:.4f}La3Zr2O12  (Li {7 * nfu} -> {7 * nfu - 3 * n})")
    if n:
        print(f"[volume] {vol / n:.0f} A^3 per {a.dopant} = a sphere of radius {(3 * vol / (4 * np.pi * n)) ** (1 / 3):.2f} A")
    if n == 0:
        sys.exit('nothing placed')

    fr = frac[pick]
    cart = fr * L
    with open(a.out + '_sites.csv', 'w') as fh:
        fh.write(f'pick,index_in_{a.sublattice}_list,index_in_Li_site_list,fx,fy,fz,x_A,y_A,z_A\n')
        for k, i in enumerate(pick):
            fh.write(f'{k},{i},{sel[i]},{fr[k, 0]:.6f},{fr[k, 1]:.6f},{fr[k, 2]:.6f},{cart[k, 0]:.4f},{cart[k, 1]:.4f},{cart[k, 2]:.4f}\n')
    with open(a.out + '.xyz', 'w') as fh:
        fh.write(f'{n}\nLattice="{L[0]:.5f} 0 0 0 {L[1]:.5f} 0 0 0 {L[2]:.5f}" Properties=species:S:1:pos:R:3 pbc="T T T"\n')
        for c in cart: fh.write(f'{a.dopant} {c[0]:.5f} {c[1]:.5f} {c[2]:.5f}\n')
    print(f"[+] {a.out}_sites.csv  {a.out}.xyz   (index_in_Li_site_list = row of llzo_sites.wyckoff_sites()['frac'] for this --fold)")


if __name__ == '__main__':
    main()
