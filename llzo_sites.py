"""
llzo_sites.py - FIXED Li site list for the garnet analysis, built from the pristine Ia-3d CIF (24d + 96h).

  parse_cif      symmetry operators + atoms of a CIF
  wyckoff_sites  every symmetry-equivalent Li position (24d, 96h, ...) and La/Zr framework positions, tiled to the MD box
  fit_shift      rigid translation that puts the ideal La/Zr framework on the MD time-averaged La/Zr positions
  build          everything 04_sites_hops.py needs, with checks and warnings printed as it goes

Why a fit: the LAMMPS box origin, the framework centre-of-mass drift removal and thermal expansion move the ideal
lattice by a fraction of an Angstrom relative to the MD coordinates. The La/Zr sublattice (which barely moves) is used
as the reference, separately at every temperature.
"""
import re
import sys
from fractions import Fraction

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

KIND = {24: "24d", 96: "96h"}


def _mi(d):
    return d - np.rint(d)


def _w(x):
    """wrap fractional coordinates to [0, 1) (guards the x % 1.0 == 1.0 rounding case)."""
    x = np.asarray(x, float) % 1.0
    x[x >= 1.0] = 0.0
    return x


def _num(s):
    return float(re.sub(r"\(.*?\)", "", s))


def parse_cif(path):
    txt = open(path).read().replace("\r", "")
    ops = []
    for m in re.finditer(r"^\s*\d+\s+'([^']+)'", txt, re.M):
        R, t = np.zeros((3, 3)), np.zeros(3)
        for i, comp in enumerate(m.group(1).replace(" ", "").split(",")):
            for sgn, tok in re.findall(r"([+-]?)(\d+/\d+|\d+\.?\d*|[xyz])", comp):
                s = -1.0 if sgn == "-" else 1.0
                if tok in ("x", "y", "z"):
                    R[i, "xyz".index(tok)] += s
                else:
                    t[i] += s * float(Fraction(tok))
        ops.append((R, t))
    if not ops:
        sys.exit(f"{path}: no symmetry operators of the form  N 'x+1/2, y, z'  found")
    a = _num(re.search(r"_cell_length_a\s+(\S+)", txt).group(1))
    m = re.search(r"loop_\s*\n((?:\s*_atom_site_\S+[ \t]*\n)+)", txt)
    if not m:
        sys.exit(f"{path}: no _atom_site loop")
    cols = re.findall(r"_atom_site_(\S+)", m.group(1))
    for c in ("label", "fract_x", "fract_y", "fract_z"):
        if c not in cols:
            sys.exit(f"{path}: _atom_site_{c} missing")
    atoms = []
    for ln in txt[m.end():].splitlines():
        s = ln.split()
        if not s:
            if atoms:
                break
            continue
        if s[0].startswith(("#", "_", "loop_")):
            break
        if len(s) < len(cols):
            continue
        lab = s[cols.index("label")]
        sym = s[cols.index("type_symbol")] if "type_symbol" in cols else lab
        el = re.match(r"[A-Z][a-z]?", sym).group(0)
        atoms.append((lab, el, np.array([_num(s[cols.index(k)]) for k in ("fract_x", "fract_y", "fract_z")])))
    return a, ops, atoms


def orbit(ops, p, tol=1e-3):
    out = []
    for R, t in ops:
        q = _w(R @ p + t)
        if not any(np.abs(_mi(q - o)).max() < tol for o in out):
            out.append(q)
    return np.array(out)


def _tile(f, fold):
    fold = np.asarray(fold, int)
    return np.vstack([(f + [i, j, k]) / fold for i in range(fold[0]) for j in range(fold[1]) for k in range(fold[2])])


def wyckoff_sites(cif, fold):
    a, ops, atoms = parse_cif(cif)
    li, fw = [], {}
    for lab, el, p in atoms:
        orb = orbit(ops, p)
        if el == "Li":
            li.append((lab, orb))
        elif el in ("La", "Zr"):
            fw[el] = _tile(orb, fold) if el not in fw else np.vstack([fw[el], _tile(orb, fold)])
    # Li orbits that coincide (same Wyckoff position listed twice) are merged
    frac, kind, labs = [], [], []
    for lab, orb in li:
        t = _tile(orb, fold)
        frac.append(t)
        kind += [KIND.get(len(orb), f"{len(orb)}x")] * len(t)
        labs += [lab] * len(t)
    return dict(a=a, nops=len(ops), frac=np.vstack(frac), kind=np.array(kind), fw=fw,
                orbits=[(lab, len(o)) for lab, o in li])


def fit_shift(md, ideal, L):
    """md, ideal: {element: fractional (n,3)}. Returns (t, rms, max, n) with t a fractional translation such that
    ideal + t sits on md. Candidate translations = md atom 0 -> every ideal site of the same element, then refined."""
    els = [e for e in ideal if e in md and len(md[e])]
    if not els:
        sys.exit("no La/Zr found in both the MD and the CIF: cannot align the site list")

    def match(t):
        d, disp = [], []
        for e in els:
            tree = cKDTree(_w(ideal[e] + t) * L, boxsize=L)
            dd, i = tree.query(_w(md[e]) * L)
            d.append(dd)
            disp.append(_mi(md[e] - (ideal[e][i] + t)))
        return np.concatenate(d), np.vstack(disp)

    cands = md[els[0]][0] - ideal[els[0]]
    t = min(cands, key=lambda c: np.sqrt(np.mean(match(c)[0] ** 2)))
    for _ in range(6):
        t = t + match(t)[1].mean(axis=0)
    d, _ = match(t)
    return _w(t), float(np.sqrt(np.mean(d ** 2))), float(d.max()), len(d)


def _groups(frac, L, R):
    n = len(frac)
    d = np.linalg.norm(_mi(frac[:, None] - frac[None]) * L, axis=-1)
    adj = csr_matrix((d <= R) & (d > 0))
    nc, lab = connected_components(adj, directed=False)
    return [np.where(lab == k)[0] for k in range(nc)]


def _merge(frac, groups):
    out = np.empty((len(groups), 3))
    for k, g in enumerate(groups):
        out[k] = _w(frac[g[0]] + _mi(frac[g] - frac[g[0]]).mean(axis=0))
    return out


def build(cif, data, temps, fold, excl, merge, peaks_frac, warn_rms=0.35, warn_max=1.0):
    """data[T] needs: 'L' (3,), 'fw' {El: frac}, 'dops' [{'label','p'(A, relative to box origin)}].
    Returns dict(frac_by_T, frac_ref, kind)."""
    ws = wyckoff_sites(cif, fold)
    ncell = int(np.prod(fold))
    T0 = temps[0]
    print(f"\n[crystal sites] {cif}")
    print(f"  space-group operators: {ws['nops']} (Ia-3d has 96);  Li orbits in the CIF: "
          + ", ".join(f"{lab} -> {n} positions ({KIND.get(n, '?')})" for lab, n in ws["orbits"]))
    if ws["nops"] != 96:
        print("  [WARN] not 96 operators: is this the Ia-3d garnet CIF?")
    n_cell = sum(n for _, n in ws["orbits"])
    print(f"  tiled x{tuple(int(x) for x in fold)} = {ncell} conventional cell(s): {len(ws['frac'])} Li sites "
          f"(expected 120 x {ncell} = {120 * ncell})")
    if n_cell != 120:
        print(f"  [WARN] {n_cell} Li positions per cell, expected 24d + 96h = 120")
    for T in temps:
        rel = data[T]["L"] / (ws["a"] * np.asarray(fold)) - 1.0
        if np.abs(rel).max() > 0.10:
            sys.exit(f"  box {data[T]['L']} A at {T} K does not match CIF a = {ws['a']} A x fold {tuple(fold)} "
                     f"(off by {100 * np.abs(rel).max():.0f} %): wrong CIF or wrong fold (set 'fold' in the settings)")
        if np.abs(rel).max() > 0.04:
            print(f"  [WARN] box at {T} K differs {100 * np.abs(rel).max():.1f} % from CIF a x fold (thermal expansion is ~1-2 %)")

    # ---- per-T alignment on the La/Zr framework
    shift = {}
    for T in temps:
        L = data[T]["L"]
        md = data[T]["fw"]
        t, rms, mx, n = fit_shift(md, ws["fw"], L)
        shift[T] = t
        cnt = ", ".join(f"{e} {len(md.get(e, []))}/{len(ws['fw'][e])}" for e in ws["fw"])
        print(f"  T = {T} K: framework fit on {n} La/Zr atoms ({cnt} MD/ideal): shift {_mi(t) * L} A, "
              f"rms residual {rms:.2f} A, max {mx:.2f} A")
        if len(md.get("La", [])) != len(ws["fw"]["La"]):
            print(f"  [WARN] {T} K: La count in MD ({len(md.get('La', []))}) != ideal ({len(ws['fw']['La'])}): wrong fold or CIF")
        if rms > warn_rms or mx > warn_max:
            print(f"  [WARN] {T} K: framework does not sit on the ideal lattice (rms > {warn_rms} or max > {warn_max} A): "
                  "site assignment unreliable; check the CIF, fold, or whether the structure transformed")

    frac_T = {T: _w(ws["frac"] + shift[T]) for T in temps}

    # ---- drop sites occupied by dopants (e.g. Ga on 24d)
    keep = np.ones(len(ws["frac"]), bool)
    print(f"  sites closer than {excl:g} A to a dopant atom are removed (dopant sits on that Li site):")
    removed_by_T = {}
    for T in temps:
        L = data[T]["L"]
        rem = {}
        for dop in data[T]["dops"]:
            dist = np.linalg.norm(_mi(frac_T[T] - dop["p"] / L) * L, axis=1)
            for i in np.where(dist < excl)[0]:
                rem[i] = (dop["label"], float(dist[i]))
        removed_by_T[T] = rem
    ref = removed_by_T[T0]
    # The framework fit is only defined modulo lattice translations and the I-centring (1/2,1/2,1/2) of Ia-3d: the
    # site SET is unchanged but site INDEX i lands on a different crystal site. Pick, per T, the equivalent shift
    # for which the T0 dopant-occupied index sits on the dopant again, so index i = same crystal site at every T.
    if ref and T0 in data and data[T0]["dops"]:
        fo = np.asarray(fold, int)
        cands = [(np.array([i, j, k]) + 0.5 * h) / fo for i in range(fo[0]) for j in range(fo[1])
                 for k in range(fo[2]) for h in (0, 1)]
        ri = np.array(sorted(ref))
        for T in temps[1:]:
            L = data[T]["L"]
            def cost(c):
                f = _w(ws["frac"][ri] + shift[T] + c)
                return min(np.linalg.norm(_mi(f - dop["p"] / L) * L, axis=1).min() for dop in data[T]["dops"])
            c = min(cands, key=cost)
            if np.abs(c).max() > 0:
                print(f"  T = {T} K: shift moved by {c * L} A (lattice/I-centring equivalent) so site indices match {T0} K")
            shift[T] = _w(shift[T] + c)
            frac_T[T] = _w(ws["frac"] + shift[T])
            rem = {}
            for dop in data[T]["dops"]:
                dist = np.linalg.norm(_mi(frac_T[T] - dop["p"] / L) * L, axis=1)
                for i in np.where(dist < excl)[0]:
                    rem[i] = (dop["label"], float(dist[i]))
            removed_by_T[T] = rem
    for i, (lab, dd) in sorted(ref.items()):
        keep[i] = False
        print(f"    site {i} ({ws['kind'][i]}) removed: {dd:.2f} A from {lab}")
    if not ref:
        print("    none (dopants are not on Li sites)")
    for T in temps[1:]:
        if set(removed_by_T[T]) != set(ref):
            print(f"  [WARN] {T} K: the set of dopant-occupied sites differs from {T0} K (dopant moved?); using the {T0} K set")
    if len(ref) > 3 * max(1, len(data[T0]["dops"])):
        print("  [WARN] many sites removed: raise/lower --site-excl?")

    idx = np.where(keep)[0]
    kind = ws["kind"][idx]
    frac_T = {T: f[idx] for T, f in frac_T.items()}

    # ---- optional merging of close sites (the 96h pairs are only ~0.8 A apart)
    if merge > 0:
        gr = _groups(frac_T[T0], data[T0]["L"], merge)
        sizes = np.bincount([len(g) for g in gr])
        print(f"  --merge-pairs {merge:g} A: {len(idx)} sites -> {len(gr)} sites (group sizes: "
              + ", ".join(f"{n} x {s}" for s, n in enumerate(sizes) if n) + ")")
        kind = np.array(["/".join(sorted(set(kind[g]))) + (f" x{len(g)}" if len(g) > 1 else "") for g in gr])
        frac_T = {T: _merge(f, gr) for T, f in frac_T.items()}

    S = len(kind)
    n_li = None
    print(f"  detected sites: {S}  (expected {120 * ncell} - {int((~keep).sum())} dopant-occupied"
          + (f", then merged" if merge > 0 else "") + ")")
    if merge == 0 and S != 120 * ncell - int((~keep).sum()):
        print("  [WARN] site count differs from expected")

    # ---- cross-check against the density peaks (independent of the CIF)
    L = data[T0]["L"]
    if peaks_frac is not None and len(peaks_frac):
        dp = np.linalg.norm(_mi(frac_T[T0][:, None] - peaks_frac[None]) * L, axis=-1)
        a_ = dp.min(axis=1)
        b_ = dp.min(axis=0)
        print(f"  cross-check vs density peaks ({len(peaks_frac)}): {100 * (a_ < 0.5).mean():.0f} % of crystal sites have a peak within 0.5 A "
              f"(median {np.median(a_):.2f} A); {100 * (b_ < 0.5).mean():.0f} % of peaks have a crystal site within 0.5 A")
        if (b_ < 0.5).mean() < 0.7:
            print("  [WARN] many density peaks are far from every crystal site: Li sits at positions the CIF does not have "
                  "(dopant-distorted region? wrong alignment?)")
    return dict(frac_by_T=frac_T, frac_ref=frac_T[T0], kind=kind)
