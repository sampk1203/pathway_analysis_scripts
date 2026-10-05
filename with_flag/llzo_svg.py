"""
llzo_svg.py - every figure the analysis saves as .png is ALSO saved as an editable .svg next to it.

Open the .svg in Inkscape (File > Open) to move, restyle or delete anything: axes, lines, markers, legends and
all text are separate objects, and the text stays text (svg.fonttype = 'none'; if Inkscape shows another font, the
font named in the file is not installed - pick any font, the layout does not depend on it).
Image-type panels (density / free-energy maps, drawn with imshow / pcolormesh) are embedded as ONE raster picture
inside the .svg; their axes, labels and colourbars are still editable. GIMP opens the .svg too (it rasterises it),
or just open the .png there.

It works by wrapping Figure.savefig once, so no plotting code has to change: importing this module is enough
(llzo_io and llzo_maps import it, so all scripts 01-09 are covered).
Switch off:  LLZO_SVG=0 bash run_pipeline.sh ...      (halves the plotting time of the big sweeps)
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402,F401
from matplotlib.figure import Figure  # noqa: E402

if os.environ.get("LLZO_SVG", "1") != "0" and not getattr(Figure.savefig, "_llzo_svg", False):
    matplotlib.rcParams["svg.fonttype"] = "none"     # keep text as text (editable in Inkscape)
    _orig = Figure.savefig

    def _savefig(self, fname, *args, **kwargs):
        _orig(self, fname, *args, **kwargs)
        if isinstance(fname, (str, os.PathLike)) and str(fname).lower().endswith(".png"):
            kw = {k: v for k, v in kwargs.items() if k not in ("format", "dpi")}
            try:
                _orig(self, os.fspath(fname)[:-4] + ".svg", format="svg", **kw)
            except Exception as e:                      # never let the extra copy break an analysis run
                print(f"  [warn] could not write the .svg copy of {fname}: {e}")

    _savefig._llzo_svg = True
    Figure.savefig = _savefig
