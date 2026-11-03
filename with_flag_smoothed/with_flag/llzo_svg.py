"""
llzo_svg.py - global figure output for the whole analysis (name kept so no import has to change).

Every figure the analysis saves as .png is now written
  1. at 1200 dpi (forced: any dpi= given at the savefig call is overridden; savefig.dpi is also set to 1200), and
  2. as ONE .png PER PANEL in a sub-folder  <outdir>/individual/<figure>_pNN_<title>.png  (1200 dpi, tight crop,
     colourbar kept with its panel, twin axes kept together). Single-panel figures are not split.
  3. as the editable .svg next to the .png (unchanged; Inkscape: File > Open, text stays text).

It works by wrapping Figure.savefig once, so no plotting code has to change: importing this module is enough
(llzo_io, llzo_maps and doping_model import it, so every script is covered).
Switches (environment):
  LLZO_DPI=300      png dpi (default 1200)        LLZO_PANELS=0   no per-panel files      LLZO_SVG=0   no .svg copies
1200 dpi = big files and slow saves (a 15 x 4.6 in figure is 18000 x 5520 px). For quick runs:  LLZO_DPI=200 LLZO_PANELS=0 ...
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402,F401
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

DPI = int(os.environ.get("LLZO_DPI", "1200"))
PANELS = os.environ.get("LLZO_PANELS", "1") != "0"
SVG = os.environ.get("LLZO_SVG", "1") != "0"
PAD_IN = 0.08                                    # white margin around a cropped panel, inches
_DROP = ("format", "dpi", "bbox_inches", "pad_inches")


def _panel_groups(fig):
    """List of axes-lists, one per panel. Twin axes (same position) share a panel; a colourbar joins its parent's panel."""
    axs = [a for a in fig.axes if a.get_visible()]
    cbs = [a for a in axs if hasattr(a, "_colorbar_info") or a.get_label() == "<colorbar>"]
    main = [a for a in axs if a not in cbs
            and (a.axison or a.lines or a.images or a.collections or a.patches or a.texts or a.get_legend() is not None)]
    groups = []                                  # [(position, [axes])]
    for a in main:
        pos = tuple(round(v, 4) for v in a.get_position().bounds)
        for g in groups:
            if g[0] == pos:
                g[1].append(a)
                break
        else:
            groups.append((pos, [a]))
    out = [g[1] for g in groups]
    for c in cbs:
        parents = (getattr(c, "_colorbar_info", None) or {}).get("parents") or []
        tgt = next((g for g in out if any(p in g for p in parents)), None)
        if tgt is None and out:                  # no parent info: nearest panel centre
            cx, cy = [(v[0] + v[2] / 2, v[1] + v[3] / 2) for v in [c.get_position().bounds]][0]
            tgt = min(out, key=lambda g: (g[0].get_position().bounds[0] + g[0].get_position().bounds[2] / 2 - cx) ** 2
                      + (g[0].get_position().bounds[1] + g[0].get_position().bounds[3] / 2 - cy) ** 2)
        if tgt is not None:
            tgt.append(c)
    return out


def _slug(ax):
    t = ax.get_title() or ax.get_title(loc="left") or ""
    t = re.sub(r"\$[^$]*\$", "", t)
    return re.sub(r"[^A-Za-z0-9]+", "-", t).strip("-")[:30]


def _save_panels(orig, fig, fname, kw):
    groups = _panel_groups(fig)
    if len(groups) < 2:
        return
    d, base = os.path.split(os.fspath(fname))
    stem = os.path.splitext(base)[0]
    odir = os.path.join(d, "individual")
    os.makedirs(odir, exist_ok=True)
    allax = list(fig.axes)
    vis = [a.get_visible() for a in allax]
    renderer = fig.canvas.get_renderer()
    inv = fig.dpi_scale_trans.inverted()
    boxes = []                                   # measured first: a hidden axes has no tight bbox
    for g in groups:
        bb = Bbox.union([a.get_tightbbox(renderer) for a in g]).transformed(inv)
        boxes.append(Bbox.from_extents(bb.x0 - PAD_IN, bb.y0 - PAD_IN, bb.x1 + PAD_IN, bb.y1 + PAD_IN))
    try:
        for i, (g, bb) in enumerate(zip(groups, boxes), 1):
            for a in allax:
                a.set_visible(a in g)
            slug = _slug(g[0])
            orig(fig, os.path.join(odir, f"{stem}_p{i:02d}" + (f"_{slug}" if slug else "") + ".png"),
                 bbox_inches=bb, dpi=DPI, **kw)
    finally:
        for a, v in zip(allax, vis):
            a.set_visible(v)


if not getattr(Figure.savefig, "_llzo_svg", False):
    matplotlib.rcParams["svg.fonttype"] = "none"     # keep text as text (editable in Inkscape)
    matplotlib.rcParams["savefig.dpi"] = DPI
    _orig = Figure.savefig

    def _savefig(self, fname, *args, **kwargs):
        is_png = isinstance(fname, (str, os.PathLike)) and str(fname).lower().endswith(".png")
        kw = {k: v for k, v in kwargs.items() if k not in _DROP}
        if is_png:
            kwargs["dpi"] = DPI                          # global 1200 dpi, overrides any dpi= at the call site
        _orig(self, fname, *args, **kwargs)
        if not is_png:
            return
        if SVG:
            try:
                _orig(self, os.fspath(fname)[:-4] + ".svg", format="svg", **kw)
            except Exception as e:                      # never let the extra copy break an analysis run
                print(f"  [warn] could not write the .svg copy of {fname}: {e}")
        if PANELS:
            try:
                _save_panels(_orig, self, fname, kw)
            except Exception as e:
                print(f"  [warn] could not write the per-panel .png files of {fname}: {e}")

    _savefig._llzo_svg = True
    Figure.savefig = _savefig
