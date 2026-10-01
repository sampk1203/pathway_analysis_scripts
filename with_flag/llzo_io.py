"""
llzo_io.py - shared helpers for the doped-LLZO MD analysis scripts.

Expects a LAMMPS dump written with:
    dump 1 all custom 100 <file> id type xu yu zu
(orthogonal box, unwrapped coordinates, all atoms).

Everything composition-specific (type -> element map, dopant types, run length,
analysis window ...) lives in a per-system config file (YAML or JSON), so the
same scripts work for every CIF/dopant set.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
import warnings
from dataclasses import dataclass

import numpy as np

KB_EV = 8.617333262e-5  # eV/K
TILT_TOL = 0.05        # A: triclinic tilt is always ignored; a warning says 'LARGE' above this

MASSES = {  # amu; extend if you use other dopants
    "Li": 6.94, "La": 138.905, "Zr": 91.224, "O": 15.999, "Ga": 69.723,
    "Ru": 101.07, "Al": 26.982, "Ta": 180.948, "Nb": 92.906, "Ca": 40.078,
    "Ba": 137.327, "Sr": 87.62, "Y": 88.906, "W": 183.84, "Mo": 95.95,
    "Sb": 121.76, "Bi": 208.98, "Ge": 72.63, "Sn": 118.71, "Ti": 47.867,
    "Hf": 178.49, "Fe": 55.845, "Sc": 44.956, "In": 114.818, "Mg": 24.305,
    "Zn": 65.38, "Te": 127.60, "Ce": 140.116, "Si": 28.085, "V": 50.942,
}

BASE_ELEMENTS = {"Li", "La", "Zr", "O"}   # anything else in element_list is treated as a dopant

DEFAULTS = {
    "name": None,             # None -> name of the folder holding the dumps
    "prefix": "lmp",
    "dump_pattern": "dump_{prefix}_{T}K.lammpstrj",
    "temperatures": None,     # None -> detected from the dump files in the folder
    "unit_cell_a": 12.9827,   # conventional cubic LLZO cell edge, Angstrom
    "fold": None,             # e.g. [1, 1, 2]; None -> derived from the dump box
    "dt_ps": 0.001,           # MD timestep in ps
    "run_steps": None,        # MD steps of the run (excludes minimisation); see prepare()
    "tmin_ps": 100.0,         # analysis window starts here (ps after run start)
    "types": None,            # None -> read from <folder>/element_list
    "dopant_types": None,     # None -> every element not in BASE_ELEMENTS
}

SETTINGS_NAME = "analysis_defaults.yaml"


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
def _read_settings(path: str) -> dict:
    with open(path) as f:
        text = f.read()
    if path.lower().endswith(".json"):
        return json.loads(text) or {}
    try:
        import yaml
    except ImportError:
        sys.exit("PyYAML is required for .yaml settings (pip install pyyaml), or use a .json file.")
    return yaml.safe_load(text) or {}


def _find_settings(folder: str | None, explicit: str | None) -> str | None:
    if explicit:
        if not os.path.exists(explicit):
            sys.exit(f"settings file not found: {explicit}")
        return explicit
    cands = []
    if folder:
        cands.append(os.path.join(folder, SETTINGS_NAME))
    cands += [os.path.join(os.getcwd(), SETTINGS_NAME),
              os.path.join(os.path.dirname(os.path.abspath(__file__)), SETTINGS_NAME)]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def read_element_list(folder: str, explicit: str | None = None, also: str | None = None) -> list[str]:
    """Elements for LAMMPS atom types 1..N, in order, e.g. 'Li La Y Zr Ta O'.
    Order of search: explicit file (--element-list), then <folder>/element_list[.txt], then <also>/ (target folder)."""
    cands = [explicit] if explicit else []
    for d in dict.fromkeys([folder, also or folder]):
        cands += [os.path.join(d, n) for n in ("element_list", "element_list.txt")]
    for p in cands:
        if os.path.exists(p):
            with open(p) as f:
                els = f.read().replace(",", " ").split()
            if not els:
                sys.exit(f"{p} is empty")
            return els
    sys.exit(f"No 'element_list' file in {folder}. Create one containing the elements in atom-type "
             f"order, e.g.  Li La Zr Ga Ru O   (or pass --elements / --element-list, or put a 'types' map in the settings file).")


def add_config_args(ap) -> None:
    """System-specific inputs as flags; each overrides the settings file / element_list / defaults."""
    g = ap.add_argument_group("system layout (override settings file)")
    g.add_argument("--dump-dir", help="folder with the dumps (absolute, or relative to the target folder; default: the target folder)")
    g.add_argument("--prefix", help="dump file prefix in dump_{prefix}_{T}K.lammpstrj")
    g.add_argument("--dump-pattern", help="dump file name pattern with {prefix} and {T}, e.g. 'dump_{prefix}_{T}K.lammpstrj'")
    g.add_argument("--elements", nargs="+", metavar="EL", help="elements for atom types 1..N in order, e.g. Li La Zr Ga Ru O (replaces element_list)")
    g.add_argument("--element-list", help="path to an element_list file (absolute, or relative to the target folder)")
    g.add_argument("--dopants", nargs="+", metavar="EL", help="dopant elements, e.g. Ga Ru (default: every element not Li La Zr O)")
    g.add_argument("--fold", type=int, nargs=3, metavar=("NX", "NY", "NZ"), help="supercell: conventional cells along a b c, e.g. 1 1 2 (default: from box / unit_cell_a)")
    g.add_argument("--name", help="system name used for results/<name> (default: target folder name)")


def load_config(target: str, settings: str | None = None, args=None) -> dict:
    """target = folder with the dumps (+ element_list)  OR  a settings .yaml/.json whose
    dump_dir points at that folder. Shared numbers (run_steps, tmin_ps, ...) come from
    analysis_defaults.yaml (found in the folder, the current directory, or next to the scripts)
    or from `settings`."""
    target = os.path.abspath(target)
    cfg = dict(DEFAULTS)
    if os.path.isdir(target):
        folder = target
        spath = _find_settings(folder, settings)
        if spath:
            cfg.update(_read_settings(spath))
        cfg["name"] = os.path.basename(folder.rstrip(os.sep))      # folder name is the system name
    else:
        spath = target
        cfg.update(_read_settings(spath))
        d = cfg.get("dump_dir") or "."
        folder = d if os.path.isabs(d) else os.path.join(os.path.dirname(spath), d)
        folder = os.path.abspath(folder)
        if not cfg.get("name"):
            cfg["name"] = os.path.basename(folder.rstrip(os.sep))
    base = folder                                                  # target folder (or the settings file's dump_dir)
    g = lambda k: getattr(args, k, None) if args is not None else None
    if g("dump_dir"):
        folder = os.path.abspath(g("dump_dir") if os.path.isabs(g("dump_dir")) else os.path.join(base, g("dump_dir")))
    for k in ("prefix", "dump_pattern", "name"):
        if g(k):
            cfg[k] = g(k)
    if g("fold"):
        cfg["fold"] = list(g("fold"))
    if g("elements"):
        cfg["types"] = {i + 1: el for i, el in enumerate(g("elements"))}
        if not g("dopants"):
            cfg["dopant_types"] = None                             # old dopant_types refer to the old type map
    cfg["dump_dir"] = folder
    cfg["_settings_file"] = spath

    if cfg.get("types"):
        cfg["types"] = {int(k): str(v) for k, v in cfg["types"].items()}
    else:
        ef = g("element_list")
        if ef and not os.path.isabs(ef):
            ef = os.path.join(base, ef)
        els = read_element_list(folder, ef, base)
        cfg["types"] = {i + 1: el for i, el in enumerate(els)}
    if g("dopants"):
        cfg["dopant_types"] = [t for t, el in cfg["types"].items() if el in g("dopants")]
        missing = [e for e in g("dopants") if e not in cfg["types"].values()]
        if missing:
            sys.exit(f"--dopants {missing} not in the element list {list(cfg['types'].values())}")
    if cfg.get("dopant_types") is None:
        cfg["dopant_types"] = [t for t, el in cfg["types"].items() if el not in BASE_ELEMENTS]
    cfg["dopant_types"] = [int(t) for t in cfg["dopant_types"]]

    for t, el in cfg["types"].items():
        if el not in MASSES:
            sys.exit(f"Element '{el}' (type {t}) missing from MASSES in llzo_io.py - add it.")
    for t in cfg["dopant_types"]:
        if t not in cfg["types"]:
            sys.exit(f"dopant type {t} is not in 'types'")
    if not li_types(cfg):
        sys.exit("no type mapped to element 'Li' (check element_list)")

    tmap = " ".join(f"{t}={el}" for t, el in cfg["types"].items())
    dop = ", ".join(f"{cfg['types'][t]}(type {t})" for t in cfg["dopant_types"]) or "none"
    print(f"[config] system '{cfg['name']}'  types: {tmap}  dopants: {dop}")
    return cfg


def li_types(cfg: dict) -> list[int]:
    return [t for t, el in cfg["types"].items() if el == "Li"]


def temperatures(cfg: dict) -> list[int]:
    """Temperatures from the config, or detected from dump_*_<T>K.lammpstrj files in the folder."""
    if cfg.get("temperatures"):
        return [int(t) for t in cfg["temperatures"]]
    found = set()
    pat = cfg["dump_pattern"]
    if "{T}" in pat:                      # temperatures from the dump pattern itself (e.g. 'MD-{T}/dump.{T}.lammpstrj')
        parts = [x.format(prefix=cfg["prefix"]) for x in pat.split("{T}")]
        rx = re.compile(re.escape(parts[0]) + "(\\d+)" + re.escape(parts[1]) + "".join("\\1" + re.escape(x) for x in parts[2:]))
        for p in glob.glob(os.path.join(cfg["dump_dir"], "*".join(parts))):
            m = rx.fullmatch(os.path.relpath(p, cfg["dump_dir"]))
            if m:
                found.add(int(m.group(1)))
    for p in glob.glob(os.path.join(cfg["dump_dir"], "*.lammpstrj")):
        m = re.search(r"_(\d+)K\.lammpstrj$", os.path.basename(p))
        if m:
            found.add(int(m.group(1)))
    if not found:
        sys.exit(f"no dumps found in {cfg['dump_dir']} (pattern {pat!r}, or *_<T>K.lammpstrj)")
    return sorted(found)


def dump_path(cfg: dict, T) -> str:
    p = os.path.join(cfg["dump_dir"], cfg["dump_pattern"].format(prefix=cfg["prefix"], T=T))
    if os.path.exists(p):
        return p
    hits = glob.glob(os.path.join(cfg["dump_dir"], f"*_{T}K.lammpstrj"))
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        sys.exit(f"several dumps match T={T} K in {cfg['dump_dir']}: {sorted(os.path.basename(h) for h in hits)}; "
                 f"set 'prefix' or 'dump_pattern' in the settings file")
    sys.exit(f"dump for T={T} K not found (looked for {p} and *_{T}K.lammpstrj)")


# --------------------------------------------------------------------------
# dump reading
# --------------------------------------------------------------------------
@dataclass
class Trajectory:
    steps: np.ndarray      # (F,)   LAMMPS timestep of each frame
    time_ps: np.ndarray    # (F,)   time since the start of the MD run
    lo: np.ndarray         # (3,)   box lower bound (frame 0)
    L: np.ndarray          # (F,3)  box lengths, Angstrom
    types: np.ndarray      # (N,)   atom types, sorted by atom id
    ids: np.ndarray        # (N,)
    pos: np.ndarray        # (F,N,3) unwrapped positions, Angstrom


def read_dump(path: str) -> Trajectory:
    steps, Ls, frames = [], [], []
    types = ids0 = lo0 = None
    tri_max = -1.0                       # largest |tilt| seen (triclinic dumps only)
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                raise ValueError(f"{path}: expected 'ITEM: TIMESTEP', got {line[:40]!r}")
            step = int(f.readline())
            f.readline()                      # ITEM: NUMBER OF ATOMS
            n = int(f.readline())
            hdr = f.readline()                # ITEM: BOX BOUNDS ...
            b = [f.readline().split() for _ in range(3)]
            lo = np.array([float(x[0]) for x in b])
            hi = np.array([float(x[1]) for x in b])
            if "xy" in hdr:                   # triclinic header: 3rd column = tilt xy xz yz. Tilt is IGNORED (box treated as orthogonal)
                xy, xz, yz = tilt = [float(x[2]) for x in b]
                tri_max = max(tri_max, max(abs(t) for t in tilt))
                # dump bounds are the bounding box of the tilted cell: remove the tilt extent to get the true box edges
                ex = [min(0.0, xy, xz, xy + xz), max(0.0, xy, xz, xy + xz)]
                ey = [min(0.0, yz), max(0.0, yz)]
                lo = lo - np.array([ex[0], ey[0], 0.0])
                hi = hi - np.array([ex[1], ey[1], 0.0])
            cols = f.readline().split()[2:]
            lines = [f.readline() for _ in range(n)]
            if not lines or not lines[-1].strip():
                warnings.warn(f"{path}: last frame (step {step}) is truncated and was dropped")
                break
            ci = {c: i for i, c in enumerate(cols)}
            for c in ("id", "type", "xu", "yu", "zu"):
                if c not in ci:
                    raise ValueError(f"{path}: dump lacks column '{c}' (need: id type xu yu zu)")
            arr = np.array(" ".join(lines).split(), dtype=float).reshape(n, len(cols))
            order = np.argsort(arr[:, ci["id"]])
            arr = arr[order]
            if types is None:
                types = arr[:, ci["type"]].astype(int)
                ids0 = arr[:, ci["id"]].astype(int)
                lo0 = lo
            elif n != len(types):
                raise ValueError(f"{path}: atom count changes between frames")
            frames.append(arr[:, [ci["xu"], ci["yu"], ci["zu"]]])
            steps.append(step)
            Ls.append(hi - lo)
    if not frames:
        raise ValueError(f"{path}: no frames read")
    if tri_max >= 0:
        warnings.warn(f"{path}: triclinic dump header, max |tilt| = {tri_max:.4f} A: tilt IGNORED, box treated as orthogonal"
                      + (f"  [LARGE: > {TILT_TOL} A, distances near the box edge are off by up to this much]" if tri_max > TILT_TOL else ""))
    steps = np.array(steps)
    return Trajectory(steps=steps, time_ps=np.zeros(len(steps)), lo=lo0, L=np.array(Ls),
                      types=types, ids=ids0, pos=np.array(frames))


def prepare(cfg: dict, dump: str, tmin_ps: float | None = None, verbose: bool = True) -> Trajectory:
    """Read a dump and keep only the analysis window (t >= tmin_ps after MD start).

    Time zero is  last_dumped_step - run_steps  when cfg['run_steps'] is set. That also
    drops any snapshots LAMMPS wrote during `minimize` (the dump is active then).
    Without run_steps, time zero is the first frame (a warning is printed).
    """
    traj = read_dump(dump)
    dt = cfg["dt_ps"]
    if cfg.get("run_steps"):
        t0 = traj.steps[-1] - int(cfg["run_steps"])
    else:
        t0 = traj.steps[0]
        if verbose:
            print("  [note] 'run_steps' not set: assuming the first frame is the start of the MD run "
                  "(set run_steps in the config if your dump also contains minimisation frames).")
    t = (traj.steps - t0) * dt
    tmin = cfg["tmin_ps"] if tmin_ps is None else tmin_ps
    keep = (t >= -1e-9) & (t >= tmin - 1e-9)
    if keep.sum() < 10:
        raise ValueError(f"{dump}: only {keep.sum()} frames in the window t >= {tmin} ps "
                         f"(run covers {t.min():.1f}-{t.max():.1f} ps)")
    n_min = int((t < -1e-9).sum())
    traj = Trajectory(steps=traj.steps[keep], time_ps=t[keep], lo=traj.lo, L=traj.L[keep],
                      types=traj.types, ids=traj.ids, pos=traj.pos[keep])
    if verbose:
        spacing = np.median(np.diff(traj.time_ps)) if len(traj.time_ps) > 1 else float("nan")
        print(f"  frames in window: {len(traj.steps)}  (t = {traj.time_ps[0]:.1f}-{traj.time_ps[-1]:.1f} ps, "
              f"spacing {spacing:.3f} ps; {n_min} pre-MD frames dropped)")
    return traj


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def masses(cfg: dict, types: np.ndarray) -> np.ndarray:
    return np.array([MASSES[cfg["types"][int(t)]] for t in types])


def li_mask(cfg: dict, traj: Trajectory) -> np.ndarray:
    return np.isin(traj.types, li_types(cfg))


def get_fold(cfg: dict, L: np.ndarray) -> tuple[int, int, int]:
    """Number of conventional cells along a, b, c."""
    if cfg.get("fold"):
        return tuple(int(x) for x in cfg["fold"])
    f = L / cfg["unit_cell_a"]
    fi = np.maximum(np.rint(f).astype(int), 1)
    if np.any(np.abs(f - fi) > 0.05 * fi):
        warnings.warn(f"box {L} is not an integer multiple of unit_cell_a={cfg['unit_cell_a']}; "
                      f"using fold={tuple(fi)} - set 'fold' in the config if wrong")
    return tuple(int(x) for x in fi)


def framework_drift(cfg: dict, traj: Trajectory) -> np.ndarray:
    """Mass-weighted displacement of the non-Li COM relative to the first frame, (F,3)."""
    fw = ~li_mask(cfg, traj)
    m = masses(cfg, traj.types)[fw]
    com = (traj.pos[:, fw, :] * m[None, :, None]).sum(axis=1) / m.sum()
    return com - com[0]


def frac_wrapped(pos: np.ndarray, lo: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Fractional coordinates in [0,1). pos (F,N,3), L (F,3)."""
    return ((pos - lo) / L[:, None, :]) % 1.0


# --------------------------------------------------------------------------
# MSD (multiple time origins) and diffusive-regime check
# --------------------------------------------------------------------------
def msd_multi_origin(pos: np.ndarray, dt_frame_ps: float, max_lag_frac: float = 0.5, n_lags: int = 150):
    """MSD(t) averaged over all atoms and all time origins in `pos` (F,N,3)."""
    F = pos.shape[0]
    lmax = max(2, int(F * max_lag_frac))
    lags = np.unique(np.linspace(1, lmax, n_lags).astype(int))
    msd = np.empty(len(lags))
    for i, l in enumerate(lags):
        d = pos[l:] - pos[:-l]
        msd[i] = np.mean(np.sum(d * d, axis=-1))
    return lags * dt_frame_ps, msd


def loglog_slope_warning(t_ps: np.ndarray, msd: np.ndarray, tol: float = 0.2, label: str = "") -> float:
    """Slope of log MSD vs log t over the upper 80% of lags. WARNS ONLY - never stops the run."""
    sel = t_ps >= t_ps[-1] * 0.2
    slope = float(np.polyfit(np.log(t_ps[sel]), np.log(msd[sel]), 1)[0])
    if abs(slope - 1.0) > tol:
        print(f"  [WARN] {label} log-log MSD slope = {slope:.2f} (diffusive regime expects ~1.0)")
    return slope


# --------------------------------------------------------------------------
# output helpers
# --------------------------------------------------------------------------
def write_dx(path: str, grid: np.ndarray, lo: np.ndarray, L: np.ndarray, title: str = "") -> None:
    """OpenDX scalar grid (x slowest, z fastest) at voxel centres; readable by VMD."""
    nx, ny, nz = grid.shape
    d = np.asarray(L, float) / np.array([nx, ny, nz])
    origin = np.asarray(lo, float) + 0.5 * d
    with open(path, "w") as f:
        f.write(f"# {title}\n")
        f.write(f"object 1 class gridpositions counts {nx} {ny} {nz}\n")
        f.write(f"origin {origin[0]:.6f} {origin[1]:.6f} {origin[2]:.6f}\n")
        f.write(f"delta {d[0]:.6f} 0 0\ndelta 0 {d[1]:.6f} 0\ndelta 0 0 {d[2]:.6f}\n")
        f.write(f"object 2 class gridconnections counts {nx} {ny} {nz}\n")
        f.write(f"object 3 class array type double rank 0 items {grid.size} data follows\n")
        flat = grid.ravel(order="C")
        for i in range(0, len(flat), 3):
            f.write(" ".join(f"{v:.6e}" for v in flat[i:i + 3]) + "\n")
        f.write('attribute "dep" string "positions"\nobject "density" class field\n'
                'component "positions" value 1\ncomponent "connections" value 2\ncomponent "data" value 3\n')


# --------------------------------------------------------------------------
# distance shells, block-jackknife errors, hop statistics
# ONE definition used by 04, 05 and 06 so their numbers can be compared.
#   shells : edges DEFAULT_SHELLS (A) -> 0-3, 3-5, 5-7, >7 ; the last one is "far"
#   far    : 'all'  far = farther than the last edge from EVERY dopant (Li next to another dopant are dropped from far)
#            'self' far = farther than the last edge from THIS dopant only
#   ref    : one dopant label (Ga1, Ru1 ...) or 'any' = distance to the nearest dopant
#   errors : delete-one-block jackknife over DEFAULT_BLOCKS contiguous time blocks (handles ratios and differences)
# --------------------------------------------------------------------------
DEFAULT_SHELLS = (3.0, 5.0, 7.0)
DEFAULT_BLOCKS = 5
MIN_N = 5            # fewer hops than this in a shell: SE reported as NaN
FEW_N = 20           # fewer than this: flagged '*' in the log (SE unreliable)

HOP_METHODS = {
    "site_left": "shell = distance of the SITE the Li leaves (fixed site list, time-mean dopant position)",
    "li_at_hop": "shell = distance of the Li ITSELF to the dopant in the hop frame (instantaneous)",
    # site-free hop detector of 06 (anchor / dwell-cluster method, no site list, no 04 output)
    "free_left": "shell = distance of the dwell position (anchor) the Li leaves, time-mean dopant position [06, site-free]",
    "free_at_hop": "shell = distance of the Li ITSELF to the dopant in the hop frame [06, site-free]",
    # O-environment hop detector of 06 (hop = the set of bonded O changes; no sites, no displacement threshold)
    "econ_left": "shell = distance of the dwell position the Li leaves, hop = change of the bonded O set [06, O-environment]",
    "econ_at_hop": "shell = distance of the Li ITSELF to the dopant in the hop frame, hop = change of the bonded O set [06]",
}
HOP_SHORT = {"site_left": "shell of the site the Li leaves", "li_at_hop": "shell of the Li in the hop frame",
             "free_left": "shell of the anchor the Li leaves", "free_at_hop": "shell of the Li in the hop frame",
             "econ_left": "shell of the dwell the Li leaves", "econ_at_hop": "shell of the Li in the hop frame"}


def shell_names(edges) -> list[str]:
    e = [0.0] + [float(x) for x in edges]
    return [f"{e[k]:g}-{e[k + 1]:g}A" for k in range(len(e) - 1)] + [f">{e[-1]:g}A"]


def dist_to(li: np.ndarray, pos: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Minimum-image distance (F,N) of every Li to ONE atom. li (F,N,3), pos (F,3), L (F,3)."""
    d = li - pos[:, None, :]
    d -= L[:, None, :] * np.rint(d / L[:, None, :])
    return np.linalg.norm(d, axis=-1)


def shell_labels(d_this, d_near, edges, far: str = "all") -> np.ndarray:
    """Shell index 0..K-1 (K-1 = far) for distances d_this (any shape). d_near = distance to the NEAREST dopant (same
    shape). far='all': a point farther than the last edge from d_this's dopant but not from every dopant gets -1
    (excluded, it is neither near nor far). far='self': no exclusion."""
    edges = np.asarray(edges, float)
    sh = np.searchsorted(edges, d_this)
    if far == "all":
        sh = np.where((sh == len(edges)) & (np.asarray(d_near) <= edges[-1]), -1, sh)
    return sh


def block_id(frames, F: int, nblocks: int) -> np.ndarray:
    """Time block (0..nblocks-1) of frame index(es). Every script uses this same split."""
    return np.minimum(np.asarray(frames) * nblocks // F, nblocks - 1)


def block_counts(labels: np.ndarray, K: int, nblocks: int, values: np.ndarray | None = None) -> np.ndarray:
    """(nblocks,K): number of Li-frames per (time block, category), or the SUM of `values` there. labels (F,N) ints,
    -1 = ignored."""
    F = labels.shape[0]
    b = block_id(np.arange(F), F, nblocks)[:, None]
    ok = labels >= 0
    idx = (b * K + labels)[ok]
    w = None if values is None else np.asarray(values, float)[ok]
    return np.bincount(idx, weights=w, minlength=nblocks * K).reshape(nblocks, K).astype(float)


def jackknife(fn, *blocks):
    """Delete-one-block jackknife. blocks: arrays (nblocks, ...) of ADDITIVE per-block sums (counts, sums of values,
    Li-time). fn(*totals) -> statistic (scalar or array): a mean, ratio, ratio of ratios, difference ...
    Returns (statistic on all blocks, jackknife standard error). SE is NaN unless >= 3 valid replicates.
    Limit: with 5 blocks the SE itself is uncertain (~35 %) and slow drifts common to all blocks are not seen."""
    B = [np.asarray(b, float) for b in blocks]
    nb = B[0].shape[0]
    tot = [b.sum(axis=0) for b in B]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = np.asarray(fn(*tot), float)
        reps = np.array([fn(*[t - b[i] for t, b in zip(tot, B)]) for i in range(nb)], float)
        reps = np.where(np.isfinite(reps), reps, np.nan)
        c = np.isfinite(reps).sum(axis=0)
        dev = reps - np.nanmean(reps, axis=0)
        se = np.sqrt((c - 1) / np.maximum(c, 1) * np.nansum(dev * dev, axis=0))
    return full, np.where(c >= 3, se, np.nan)


def jackknife_events(values: np.ndarray, blk: np.ndarray, nblocks: int, stat=np.median):
    """Jackknife SE of stat(values) when the values are single events (e.g. hop dips) tagged with a time block."""
    values, blk = np.asarray(values, float), np.asarray(blk)
    if len(values) == 0:
        return np.nan, np.nan
    full = float(stat(values))
    reps = np.array([stat(values[blk != b]) for b in range(nblocks) if (blk != b).any() and (blk == b).any()], float)
    if len(reps) < 3:
        return full, np.nan
    return full, float(np.sqrt((len(reps) - 1) / len(reps) * np.sum((reps - reps.mean()) ** 2)))


def back_flags(li, f, a, b):
    """Per hop (arrays in any order): eligible = the same Li hopped before; is_back = this hop undoes that previous hop
    (a->b right after b->a). Back-hop % is ALWAYS is_back.sum() / eligible.sum() (a Li's first hop cannot be a back-hop)."""
    li, f, a, b = (np.asarray(x) for x in (li, f, a, b))
    order = np.lexsort((f, li))
    l_, a_, b_ = li[order], a[order], b[order]
    same = np.r_[False, l_[1:] == l_[:-1]]
    back = same & np.r_[False, (a_[1:] == b_[:-1]) & (b_[1:] == a_[:-1])]
    elig, isb = np.empty(len(li), bool), np.empty(len(li), bool)
    elig[order], isb[order] = same, back
    return isb, elig


def shell_volume_fractions(dop_xyz: dict, L, edges, far: str = "all", n: int = 48) -> dict:
    """Fraction of the box volume in each shell, per reference ('any' + every dopant), on an n^3 grid (minimum image).
    Used for the 'uniform Li density' expectation. dop_xyz {label: (3,) position}, L (3,)."""
    L = np.asarray(L, float)
    g = [(np.arange(n) + 0.5) / n * L[i] for i in range(3)]
    P = np.stack(np.meshgrid(*g, indexing="ij"), axis=-1).reshape(-1, 3)
    D = {}
    for lab, p in dop_xyz.items():
        d = P - np.asarray(p, float)
        d -= L * np.rint(d / L)
        D[lab] = np.linalg.norm(d, axis=1)
    dn = np.min(list(D.values()), axis=0)
    K = len(edges) + 1
    out = {}
    for lab, d in list(D.items()) + [("any", dn)]:
        sh = shell_labels(d, dn, edges, far)
        out[lab] = np.array([(sh == k).mean() for k in range(K)])
    return out


def hop_shell_rows(T, ref, method, names, cat, f, is_back, elig, litime_blk, nmean, F, nblocks):
    """Hop statistics per distance shell for ONE (reference, method). cat (n_hops,) shell of every hop (-1 = excluded),
    f hop frame, litime_blk (nblocks,K) Li-time (ps) spent in each category per time block, nmean (K,) mean Li count.
    Every number carries a jackknife SE. 'nonback' = hops that do NOT undo the previous hop of that Li."""
    K = len(names)
    m = cat >= 0
    idx = block_id(f[m], F, nblocks) * K + cat[m]

    def cnt(w=None):
        return np.bincount(idx, weights=w, minlength=nblocks * K).reshape(nblocks, K).astype(float)

    nh, ne, nb = cnt(), cnt(elig[m].astype(float)), cnt(is_back[m].astype(float))
    t = np.asarray(litime_blk, float)
    rate, rate_se = jackknife(lambda h, tt: h / tt, nh, t)
    rr, rr_se = jackknife(lambda h, tt: (h / tt) / (h / tt)[-1], nh, t)
    bk, bk_se = jackknife(lambda b_, e_: 100 * b_ / e_, nb, ne)
    bd, bd_se = jackknife(lambda b_, e_: 100 * (b_ / e_ - (b_ / e_)[-1]), nb, ne)
    nbr, nbr_se = jackknife(lambda h, b_, tt: (h - b_) / tt, nh, nb, t)
    nr, nr_se = jackknife(lambda h, b_, tt: ((h - b_) / tt) / ((h - b_) / tt)[-1], nh, nb, t)
    # a jackknife SE from < MIN_N events is meaningless (0 events would give SE = 0): report NaN instead
    sh_, se_ = nh.sum(0) < MIN_N, ne.sum(0) < MIN_N
    rate_se, rr_se, nbr_se, nr_se = (np.where(sh_, np.nan, x) for x in (rate_se, rr_se, nbr_se, nr_se))
    bk_se, bd_se = (np.where(se_, np.nan, x) for x in (bk_se, bd_se))
    rows = []
    for k in range(K):
        rows.append(dict(T=T, ref=ref, method=method, shell=names[k], n_Li_mean=float(nmean[k]), n_hops=int(nh[:, k].sum()),
                         n_eligible=int(ne[:, k].sum()), n_back=int(nb[:, k].sum()), li_time_ps=float(t[:, k].sum()),
                         back_pct=bk[k], back_pct_se=bk_se[k], back_minus_far_pp=bd[k], back_minus_far_pp_se=bd_se[k],
                         rate_per_Li_ps=rate[k], rate_se=rate_se[k], rate_ratio_to_far=rr[k], rate_ratio_se=rr_se[k],
                         nonback_rate_per_Li_ps=nbr[k], nonback_rate_se=nbr_se[k], nonback_ratio_to_far=nr[k],
                         nonback_ratio_se=nr_se[k], blocks=nblocks))
    return rows


def hop_shell_analysis(T, dt, ev, names, edges, far, nblocks, state=None, site_d=None, li_d=None):
    """All (reference x method) hop-shell tables for one temperature.
    ev {li,f,a,b} numpy arrays; state (F,N) committed site (-1 unknown) or None; site_d {ref: (S,) site distance to that
    dopant, 'any' included} or None; li_d {ref: (F,N) Li distance, 'any' included} or None. Method 'site_left' needs
    state+site_d, method 'li_at_hop' needs li_d. Used identically by 04 and 06."""
    K = len(names)
    isb, elig = back_flags(ev["li"], ev["f"], ev["a"], ev["b"])
    rows = []
    F = state.shape[0] if state is not None else next(iter(li_d.values())).shape[0]
    refs = list(site_d) if site_d else list(li_d)
    for ref in refs:
        if state is not None and site_d is not None and ref in site_d:
            ssh = shell_labels(site_d[ref], site_d["any"], edges, far)                  # (S,)
            lab = np.where(state >= 0, ssh[np.maximum(state, 0)], -1)                   # (F,N)
            bc = block_counts(lab, K, nblocks)
            rows += hop_shell_rows(T, ref, "site_left", names, ssh[ev["a"]], ev["f"], isb, elig, bc * dt,
                                   bc.sum(0) / F, F, nblocks)
        if li_d is not None and ref in li_d:
            lab = shell_labels(li_d[ref], li_d["any"], edges, far)
            bc = block_counts(lab, K, nblocks)
            rows += hop_shell_rows(T, ref, "li_at_hop", names, lab[ev["f"], ev["li"]], ev["f"], isb, elig, bc * dt,
                                   bc.sum(0) / F, F, nblocks)
    return rows


def _pm(v, s, fmt):
    return (f"{v:{fmt}}" if np.isfinite(v) else "   nan") + (f" +/- {s:{fmt.replace('+', '')}}" if np.isfinite(s) else " +/-  nan")


def print_hop_shell(rows, T, ind="  "):
    """Log table of hop_shell_analysis rows of one temperature. Error = jackknife SE over time blocks."""
    rr = [r for r in rows if r["T"] == T]
    for ref in dict.fromkeys(r["ref"] for r in rr):
        for meth, desc in HOP_METHODS.items():
            sel = [r for r in rr if r["ref"] == ref and r["method"] == meth]
            if not sel:
                continue
            print(f"{ind}[{ref}] {desc}")
            print(f"{ind}    shell      <Li>   hops(elig)   back-hop %          back - far (pp)      hop rate /Li/ps           "
                  f"rate / far            non-back rate / far")
            for r in sel:
                far = r is sel[-1]
                star = "*" if r["n_hops"] < FEW_N else " "
                print(f"{ind}  {r['shell']:>8} {r['n_Li_mean']:7.2f} {r['n_hops']:6d}({r['n_eligible']:5d}){star} "
                      f"{_pm(r['back_pct'], r['back_pct_se'], '5.1f'):>17}   "
                      f"{'(reference)' if far else _pm(r['back_minus_far_pp'], r['back_minus_far_pp_se'], '+5.1f'):>17}   "
                      f"{_pm(r['rate_per_Li_ps'], r['rate_se'], '.4f'):>21}   "
                      f"{'1' if far else _pm(r['rate_ratio_to_far'], r['rate_ratio_se'], '.2f'):>17}   "
                      f"{'1' if far else _pm(r['nonback_ratio_to_far'], r['nonback_ratio_se'], '.2f'):>17}")
    print(f"{ind}* = fewer than {FEW_N} hops (SE unreliable; NaN below {MIN_N}).")
    print(f"{ind}error bars = delete-one-block jackknife over {rr[0]['blocks'] if rr else '?'} time blocks (hops of one Li are "
          "correlated and the far shell is not independent of the others: treat differences < 2 SE as noise). "
          "back-hop % = hops that undo the Li's previous hop / hops that have a previous hop.")


def plot_hop_shell(rows, names, key, ylabel, path, title, hline=None, skip_far=False, ylim=None):
    """One figure, panels = reference (rows) x method (columns), x = T, one line per shell, jackknife error bars.
    ALL panels share ONE y axis so they can be compared by eye. skip_far: for ratios to far / differences from far."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    refs = list(dict.fromkeys(r["ref"] for r in rows))
    meths = [m for m in HOP_METHODS if any(r["method"] == m for r in rows)]
    if not refs or not meths:
        return False
    K = len(names)
    cols = [plt.cm.viridis(k / max(K - 2, 1)) for k in range(K - 1)] + ["k"]
    fig, axs = plt.subplots(len(refs), len(meths), figsize=(4.8 * len(meths), 3.1 * len(refs) + 1.3), dpi=150,
                            squeeze=False, sharex=True, sharey=True)
    for ri, ref in enumerate(refs):
        for ci, meth in enumerate(meths):
            a = axs[ri][ci]
            for k in range(K - (1 if skip_far else 0)):
                sel = sorted([r for r in rows if r["ref"] == ref and r["method"] == meth and r["shell"] == names[k]],
                             key=lambda r: r["T"])
                if not sel:
                    continue
                a.errorbar([r["T"] for r in sel], [r[key] for r in sel], [r[_se_key(key)] for r in sel], marker="o", ms=4,
                           capsize=2, color=cols[k], ls="--" if k == K - 1 else "-")
                lo_ = [r for r in sel if r["n_hops"] < FEW_N]                 # too few hops: open symbol = do not trust
                if lo_:
                    a.plot([r["T"] for r in lo_], [r[key] for r in lo_], ls="", marker="o", ms=4.5, mfc="white", mec=cols[k], zorder=5)
            if hline is not None:
                a.axhline(hline, color="0.5", lw=0.8)
            nn = ", ".join(f"{names[k]}: {sum(r['n_hops'] for r in rows if r['ref'] == ref and r['method'] == meth and r['shell'] == names[k])}"
                           for k in range(K))
            a.set_title(f"{ref} - {meth}", fontsize=10)
            a.text(0.02, 0.02, f"hops (all T): {nn}", transform=a.transAxes, fontsize=6, color="0.3")
            if ri == len(refs) - 1:
                a.set_xlabel("T (K)")
            if ci == 0:
                a.set_ylabel(ylabel, fontsize=8)
    if ylim:
        axs[0][0].set_ylim(*ylim)
    h = [Line2D([], [], color=cols[k], marker="o", lw=2, ls="--" if k == K - 1 else "-",
                label=f"{names[k]} from dopant" + (" (far, reference)" if k == K - 1 else "")) for k in range(K - (1 if skip_far else 0))]
    h.append(Line2D([], [], color="0.4", marker="o", mfc="white", ls="", label=f"open symbol: < {FEW_N} hops (SE unreliable / none)"))
    fig.legend(handles=h, loc="lower center", ncol=min(len(h), 4), fontsize=8, frameon=False)
    fig.suptitle(title + "\n" + " | ".join(f"{m}: {HOP_SHORT[m]}" for m in meths) + " | error bars: block jackknife", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(path)
    plt.close(fig)
    return True


def _se_key(key):
    return {"back_pct": "back_pct_se", "back_minus_far_pp": "back_minus_far_pp_se", "rate_per_Li_ps": "rate_se",
            "rate_ratio_to_far": "rate_ratio_se", "nonback_rate_per_Li_ps": "nonback_rate_se",
            "nonback_ratio_to_far": "nonback_ratio_se"}[key]


def write_hop_shell_outputs(rows, names, outdir, name, prefix=""):
    """hop_shell_stats.csv + the shared figures (same files from 04 and 06)."""
    import csv as _csv
    if not rows:
        return []
    keys = list(rows[0].keys())
    with open(os.path.join(outdir, prefix + "hop_shell_stats.csv"), "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    out = [prefix + "hop_shell_stats.csv"]
    for fn, key, yl, hl, sk, ttl in (
            (prefix + "back_hop_shell.png", "back_pct", "back-hops (% of hops with a previous hop)", None, False,
             f"{name}: back-hop fraction by distance to the dopant"),
            (prefix + "hop_rate_ratio_shell.png", "rate_ratio_to_far", "hop rate(shell) / hop rate(far)", 1.0, True,
             f"{name}: hop rate relative to far Li (<1 = slower near the dopant)"),
            (prefix + "nonback_ratio_shell.png", "nonback_ratio_to_far", "non-back hop rate(shell) / (far)", 1.0, True,
             f"{name}: hops that do not undo the previous hop, relative to far Li")):
        if plot_hop_shell(rows, names, key, yl, os.path.join(outdir, fn), ttl, hline=hl, skip_far=sk):
            out.append(fn)
    return out
