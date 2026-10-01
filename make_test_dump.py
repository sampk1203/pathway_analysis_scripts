#!/usr/bin/env python3
"""
make_test_dump.py - build a SYNTHETIC lammpstrj (id type xu yu zu) from a CIF, only for testing
the analysis scripts. Framework atoms vibrate about their CIF sites, Li hop between ideal
24d/96h sites (needs pymatgen), the framework COM drifts slowly, and 3 fake 'minimisation'
frames are written first. The physics is NOT realistic.

usage: python make_test_dump.py LiRemoved-3_10.cif out.lammpstrj [--frames 600] [--seed 1]
"""
import argparse

import numpy as np
from scipy.spatial import cKDTree

TYPE_OF = {"Li+": 1, "La3+": 2, "Zr4+": 3, "Ga": 4, "Ru": 5, "O2-": 6}


def read_cif(path):
    a = b = c = None
    atoms = []
    for line in open(path):
        p = line.split()
        if line.startswith("_cell_length_a"):
            a = float(p[1])
        elif line.startswith("_cell_length_b"):
            b = float(p[1])
        elif line.startswith("_cell_length_c"):
            c = float(p[1])
        elif len(p) == 7 and p[2] == "1" and p[0] in TYPE_OF:
            atoms.append((TYPE_OF[p[0]], np.array([float(p[3]), float(p[4]), float(p[5])])))
    return np.array([a, b, c]), atoms


def ideal_sites(a_cell, fold):
    from pymatgen.symmetry.groups import SpaceGroup
    sg = SpaceGroup("Ia-3d")
    def gen(ref):
        s = np.round([op.operate(ref) % 1 for op in sg.symmetry_ops], 6) % 1
        return np.unique(s, axis=0)
    cell = np.vstack([gen([0.375, 0, 0.25]), gen([0.0959, 0.6922, 0.5731])])
    out = []
    for i in range(fold[0]):
        for j in range(fold[1]):
            for k in range(fold[2]):
                out.append((cell + [i, j, k]) / np.array(fold))
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cif")
    ap.add_argument("out")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--hop", type=float, default=0.03, help="hop attempt probability per frame")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    L, atoms = read_cif(args.cif)
    types = np.array([t for t, _ in atoms])
    frac0 = np.array([f for _, f in atoms]) % 1.0
    cart0 = frac0 * L
    fold = tuple(int(x) for x in np.rint(L / 12.9827))
    sites = ideal_sites(12.9827, fold) * L                          # (S,3) Cartesian
    tree = cKDTree(sites % L, boxsize=L)

    li = np.where(types == 1)[0]
    _, occ = tree.query(cart0[li] % L)
    occ = occ.copy()
    occupied = np.zeros(len(sites), bool)
    occupied[occ] = True
    pairs = tree.query_pairs(2.3, output_type="ndarray")
    nb = [[] for _ in range(len(sites))]
    for i, j in pairs:
        nb[i].append(j)
        nb[j].append(i)

    li_pos = cart0[li].copy()                                       # unwrapped Li
    site_xyz = sites.copy()
    fw = np.where(types != 1)[0]
    frames = []
    drift = np.zeros(3)
    for f in range(args.frames):
        for n in range(len(li)):                                    # Li hops
            if rng.random() < args.hop:
                s = occ[n]
                if nb[s]:
                    s2 = rng.choice(nb[s])
                    if not occupied[s2]:
                        d = site_xyz[s2] - site_xyz[s]
                        d -= L * np.rint(d / L)                     # minimum image
                        li_pos[n] += d
                        occupied[s] = False
                        occupied[s2] = True
                        occ[n] = s2
        pos = np.empty_like(cart0)
        pos[li] = li_pos + rng.normal(0, 0.18, li_pos.shape)
        pos[fw] = cart0[fw] + rng.normal(0, 0.10, (len(fw), 3))
        drift += rng.normal(0, 0.004, 3)
        pos += drift
        frames.append(pos)

    perm = rng.permutation(len(types))                              # shuffle atom order in the file
    ids = perm + 1
    steps = [0, 100, 200] + [300 + 100 * i for i in range(args.frames)]
    pre = [cart0.copy() for _ in range(3)]                          # fake minimisation frames
    with open(args.out, "w") as fh:
        for st, pos in zip(steps, pre + frames):
            fh.write(f"ITEM: TIMESTEP\n{st}\nITEM: NUMBER OF ATOMS\n{len(types)}\n")
            fh.write("ITEM: BOX BOUNDS pp pp pp\n")
            for d in range(3):
                fh.write(f"0.0 {L[d]:.6f}\n")
            fh.write("ITEM: ATOMS id type xu yu zu\n")
            for k in perm:
                fh.write(f"{k + 1} {types[k]} {pos[k, 0]:.5f} {pos[k, 1]:.5f} {pos[k, 2]:.5f}\n")
    print(f"wrote {args.out}: {len(steps)} frames ({args.frames} 'MD' + 3 'minimisation'), "
          f"{len(types)} atoms, box {L}, last step {steps[-1]}")
    print(f"use run_steps = {steps[-1] - 300} for this file")


if __name__ == "__main__":
    main()
