#!/usr/bin/env python3
"""collect_pngs.py - copy every .png under a folder (recursively) into ONE flat folder.

python collect_pngs.py <main_folder> [-o OUT] [--individual]

  -o OUT         destination (default: <main_folder>/all_pngs). It is skipped while scanning, so re-running is safe.
  --individual   only the per-panel files (those inside an 'individual/' folder)

Names: the plain file name when it is unique; when the same name occurs in several folders, ALL of those copies get their
relative path as name (sweep/01_s04/individual/a.png -> sweep__01_s04__individual__a.png), so nothing is overwritten.
"""
import argparse
import os
import shutil
import sys
from collections import Counter


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--individual", action="store_true")
    a = ap.parse_args()

    src = os.path.abspath(a.src)
    out = os.path.abspath(a.out or os.path.join(src, "all_pngs"))
    if not os.path.isdir(src):
        sys.exit(f"not a folder: {src}")

    files = []
    for root, dirs, names in os.walk(src):
        dirs[:] = sorted(d for d in dirs if os.path.abspath(os.path.join(root, d)) != out)
        if a.individual and "individual" not in os.path.relpath(root, src).split(os.sep):
            continue
        files += [os.path.join(root, n) for n in sorted(names) if n.lower().endswith(".png")]
    if not files:
        sys.exit("no .png files found")

    cnt = Counter(os.path.basename(f) for f in files)
    os.makedirs(out, exist_ok=True)
    for f in files:
        b = os.path.basename(f)
        name = b if cnt[b] == 1 else os.path.relpath(f, src).replace(os.sep, "__")
        shutil.copy2(f, os.path.join(out, name))
    print(f"copied {len(files)} png -> {out}  ({sum(1 for v in cnt.values() if v > 1)} duplicate names renamed by path)")


if __name__ == "__main__":
    main()
