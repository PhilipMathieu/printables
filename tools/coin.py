"""Make a coin from the command line, and look at it without a slicer.

    python -m tools.coin --obverse 26 --legend "MONT-TREMBLANT" -o out/coin

Anything a flag can carry -- text, an SVG, the dimensions -- goes here. A mark
that has to be drawn rather than typed goes in a module under parts/ that
exposes a ``PARAMS``, and this builds that instead:

    python -m tools.coin --part parts.tremblant_coin

The preview shades both faces by height rather than by a light, because a
light tells you nothing about a coin: every raised surface is flat and faces
straight up, so a lit plan view of a struck coin is a plain disc. Height
shading shows the strike itself, which is the thing worth checking before
committing half an hour of printer time.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build123d import export_stl  # noqa: E402

from geom.motif import BOTTOM, TOP, Legend, Motif, Svg, Text  # noqa: E402
from p2s import profiles  # noqa: E402
from parts.coin import Params, build  # noqa: E402
from tools.render import INK, mesh, section  # noqa: E402

METAL = ["#2b2f36", "#6c7482", "#aeb6c2", "#e3e8ef"]


def _face_view(ax, points, tris, from_above: bool, title: str) -> None:
    """Plan view shaded by height, painted back to front.

    Only the triangles facing the viewer are drawn, so the far side of the coin
    cannot show through the near one; sorting by height then makes the relief
    stack correctly without any depth buffer.

    Height alone gives a flat, muddy picture, because a struck coin is a few
    plateaus a fraction of a millimetre apart. What makes it legible is the
    walls: they are kept (their normals lie in the plane, so they face neither
    way) and painted dark, which outlines every raised letter the way a shadow
    would. Without them the relief has no edges at all.
    """
    tri = points[tris]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    unit = normals / np.where(lengths == 0, 1, lengths)
    facing = unit[:, 2] >= 0 if from_above else unit[:, 2] <= 0
    # Keeping the walls means keeping the *other* face's walls too, and those
    # have no flat surface in this view to hide behind -- the far side's letters
    # ghost through. Nothing on the far half of the coin belongs in this view,
    # so cut at the mid-plane and the question does not arise.
    mid = points[:, 2].min() + (points[:, 2].max() - points[:, 2].min()) / 2
    on_side = tri[:, :, 2].mean(axis=1) >= mid
    facing &= on_side if from_above else ~on_side
    tri, tilt = tri[facing], np.abs(unit[facing, 2])

    depth = tri[:, :, 2].mean(axis=1)
    order = np.argsort(depth if from_above else -depth)
    tri, depth, tilt = tri[order], depth[order], tilt[order]

    span = depth.max() - depth.min()
    shade = (depth - depth.min()) / (span if span else 1)
    if not from_above:
        shade = 1 - shade  # deeper cuts read darker, as a shadow would
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("metal", METAL)
    rgb = cmap(shade)[:, :3] * (0.45 + 0.55 * tilt)[:, None]
    # Mirrored in x, because the underside is seen from the other side.
    xy = tri[:, :, :2] * ((1, 1) if from_above else (-1, 1))
    # Stroking each facet in its own colour closes the hairline seams that
    # antialiasing leaves between abutting triangles.
    ax.add_collection(
        PolyCollection(xy, facecolors=rgb, edgecolors=rgb, linewidths=0.3)
    )
    lim = np.abs(points[:, :2]).max() * 1.04
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, color=INK, fontsize=11)


def preview(params: Params, dest: Path, part=None) -> Path:
    """Obverse, reverse and a section through the rim, as a PNG."""
    part = build(params) if part is None else part
    points, tris = mesh(part, tol=0.02)

    fig = plt.figure(figsize=(12, 5.0), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.04,
                          left=0.01, right=0.99, top=0.88, bottom=0.06)
    _face_view(fig.add_subplot(gs[0, 0]), points, tris, True,
               f"obverse · {params.relief} mm relief")
    _face_view(fig.add_subplot(gs[0, 1]), points, tris, False,
               f"reverse · {params.engrave} mm engraved")

    # --- section: the whole coin, then the rim close up --------------------
    # A coin is 3mm in 38, so a section drawn to scale shows the silhouette and
    # nothing about the profile. The detail view is where the rim, the ramp and
    # the height of the relief are actually checkable.
    inner = gs[0, 2].subgridspec(2, 1, height_ratios=[1, 1.6], hspace=0.0)
    segs = section(points, tris, 1, 0.0)
    lines = (
        np.array([[s[:, 0], s[:, 2]] for s in segs]).transpose(1, 2, 0)
        if len(segs) else None
    )

    ax = fig.add_subplot(inner[0, 0])
    if lines is not None:
        ax.plot(*lines, color=INK, lw=1.2)
    ax.set_xlim(-params.radius * 1.08, params.radius * 1.08)
    ax.set_ylim(-params.thickness, params.thickness * 2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"section · {params.diameter} × {params.thickness} mm",
        color=INK, fontsize=11,
    )

    ax = fig.add_subplot(inner[1, 0])
    if lines is not None:
        ax.plot(*lines, color=INK, lw=1.4)
    detail = max(params.rim, params.chamfer) * 4 + params.relief
    ax.set_xlim(params.radius - detail, params.radius + detail * 0.12)
    ax.set_ylim(-params.thickness * 0.35, params.thickness * 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"rim detail · {params.rim} mm wide, "
        f"field {params.field_depth} mm down",
        color=INK, fontsize=10, pad=2,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest


def _mark(spec: str | None, font: str) -> Motif | None:
    """A flag's worth of motif: an SVG if it looks like a path, else text."""
    if not spec:
        return None
    if spec.lower().endswith(".svg"):
        return Svg(path=Path(spec))
    return Text(text=spec, font=font)


def _face(spec: str | None, legend: str | None, exergue: str | None,
          font: str) -> tuple[Motif, ...]:
    marks = [_mark(spec, font)]
    if legend:
        marks.append(Legend(text=legend, font=font, at=TOP))
    if exergue:
        marks.append(Legend(text=exergue, font=font, at=BOTTOM))
    # A central mark shares the field with a legend, so pull it in to make room.
    if marks[0] is not None and len(marks) > 1:
        marks[0] = dataclasses.replace(marks[0], fill=0.52)
    return tuple(m for m in marks if m is not None)


def params_from(args: argparse.Namespace) -> Params:
    if args.part:
        module = importlib.import_module(args.part)
        if not hasattr(module, "PARAMS"):
            raise SystemExit(f"{args.part} defines no PARAMS to build")
        return module.PARAMS
    return Params(
        obverse=_face(args.obverse, args.legend, args.exergue, args.font),
        reverse=_face(args.reverse, args.reverse_legend, args.reverse_exergue,
                      args.font),
        diameter=args.diameter,
        thickness=args.thickness,
        rim=args.rim,
        relief=args.relief,
        engrave=args.engrave,
        reeds=args.reeds,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", help="module exposing PARAMS, for drawn marks")
    ap.add_argument("--obverse", help="text, or an .svg, for the top face")
    ap.add_argument("--reverse", help="text, or an .svg, for the underside")
    ap.add_argument("--legend", help="text arched over the obverse")
    ap.add_argument("--exergue", help="text along the bottom of the obverse")
    ap.add_argument("--reverse-legend", help="text arched over the reverse")
    ap.add_argument("--reverse-exergue", help="text along the bottom of the reverse")
    ap.add_argument("--font", default="Helvetica")
    ap.add_argument("--diameter", type=float, default=Params.diameter)
    ap.add_argument("--thickness", type=float, default=Params.thickness)
    ap.add_argument("--rim", type=float, default=Params.rim)
    ap.add_argument("--relief", type=float, default=Params.relief)
    ap.add_argument("--engrave", type=float, default=Params.engrave)
    ap.add_argument("--reeds", type=int, default=Params.reeds)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="shrink or grow the whole coin, e.g. 0.8 for a test print")
    ap.add_argument("--nozzle", type=float, default=profiles.DEFAULT_NOZZLE,
                    help="nozzle the 3mf is configured for; a finer one holds "
                         "finer lettering")
    ap.add_argument("-o", "--out", default="out/coin",
                    help="output basename; .stl, .3mf and .png are written")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args(argv)

    params = params_from(args)
    if args.scale != 1.0:
        params = params.scaled(args.scale)
    part = build(params)
    base = Path(args.out)
    base.parent.mkdir(parents=True, exist_ok=True)
    export_stl(part, str(base.with_suffix(".stl")))
    print(f"wrote {base.with_suffix('.stl')}")

    # The 3mf is the useful one, but it needs Bambu Studio's profiles on disk;
    # the STL is already written either way, so a missing slicer is a note and
    # not a failure.
    try:
        from p2s.slicer import write_project

        print(f"wrote {write_project(part, base.with_suffix('.3mf'), nozzle=args.nozzle)}")
    except Exception as exc:  # noqa: BLE001 - profiles missing is not fatal here
        print(f"no 3mf: {exc}")

    if not args.no_preview:
        print(f"wrote {preview(params, base.with_suffix('.png'), part)}")
    bb = part.bounding_box()
    print(f"valid={part.is_valid} {bb.size.X:.1f} × {bb.size.Y:.1f} × "
          f"{bb.size.Z:.1f} mm, {part.volume / 1000:.2f} cm^3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
