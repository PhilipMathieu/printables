"""Make bag holders from the command line, and look at them without a slicer.

    python -m tools.bag_holder
    python -m tools.bag_holder --slot 4 --belly 42 --copies 2

Almost nobody needs to touch any of it. ``--slot`` is the one worth trying a
second value of, because it is what decides how hard the handles are pinched,
and how tightly a knotted bag ties is a thing about the bags you buy. After
that, ``--gap`` if the swivel comes off the plate fused or turns stiffly, and
``--strap`` if the lead is not a 3/4 inch one.

It is written in ASA unless told otherwise. This is an outdoor part on a lead:
UV, cold mornings, and the pavement every time it is dropped. Hang it on a ball
chain, a split ring or a small carabiner -- the chain is not decoration, it is
what lets the funnel stay pointing up while the lead does whatever it likes.

The preview is three views: what it looks like printed, the profile that is
actually the design, and what the funnel does with bundles of handles of
different sizes.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.patches import Circle as CirclePatch
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import Polygon as MplPolygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build123d import Part, export_stl  # noqa: E402

from p2s import inventory, profiles  # noqa: E402
from geom import webbing  # noqa: E402
from parts.bag_holder import Params, build, profile  # noqa: E402
from tools.render import ASA, INK, iso, mesh, plate  # noqa: E402

BAG = "#2f8f5b"
COLLAR = "#3b6ea5"

BUNDLES = (26.0, 17.0, 9.0)
"""Bundles of tied handles to draw, in mm across. A single knot in a thin bag
is the small one; a doubled bag tied twice is the big one."""


def _loop(wire, samples: int = 700):
    return [(p.X, p.Y) for p in (wire.position_at(i / samples) for i in range(samples))]


def _outline(ax, face, fill: str = ASA, lw: float = 1.3) -> None:
    """The profile, painted outside in: body, then its two holes back in white.

    Same trick as the fidget's plan -- an even-odd polygon carrying every loop
    at once leaves a seam where two of them join.
    """
    ax.add_patch(MplPolygon(_loop(face.outer_wire()), closed=True, facecolor=fill,
                            edgecolor=INK, lw=lw, zorder=1))
    for hole in face.inner_wires():
        ax.add_patch(MplPolygon(_loop(hole), closed=True, facecolor="white",
                                edgecolor=INK, lw=lw, zorder=2))


def _span(ax, y: float, half: float, label: str, colour: str):
    """A width, measured across the axis and labelled just inside the opening."""
    ax.annotate("", xy=(-half, y), xytext=(half, y),
                arrowprops=dict(arrowstyle="<->", color=colour, lw=1.1))
    ax.text(0, y + 1.5, label, fontsize=8.5, color=colour,
            ha="center", va="bottom", zorder=5)


def _collar(ax, params: Params) -> None:
    """The turning body, in its socket, with the lead's slot through it."""
    ax.add_patch(CirclePatch((0, 0), params.socket_radius, facecolor="white",
                             edgecolor=INK, lw=0.9, zorder=3))
    ax.add_patch(CirclePatch((0, 0), params.collar_radius, facecolor=COLLAR,
                             alpha=0.28, edgecolor=COLLAR, lw=1.3, zorder=4))
    ax.add_patch(FancyBboxPatch(
        (-params.slot_width / 2 + params.slot_corner,
         -params.slot_height / 2 + params.slot_corner),
        params.slot_width - 2 * params.slot_corner,
        params.slot_height - 2 * params.slot_corner,
        boxstyle=f"round,pad={params.slot_corner}", facecolor="white",
        edgecolor=COLLAR, lw=1.1, zorder=5))


def _design_view(ax, params: Params, face) -> None:
    """The profile, with the three numbers that are the design on it."""
    _outline(ax, face)
    _collar(ax, params)

    ax.annotate(
        f"the collar turns in its socket — printed\nthere, {params.gap:.2f} mm "
        f"clear at every height,\n{params.engagement:.2f} mm of catch holding it in",
        xy=(params.collar_radius * 0.74, params.collar_radius * 0.68),
        xytext=(params.width / 2 + 2, params.head_radius + 7),
        fontsize=8.5, color=COLLAR, ha="left", va="bottom",
        arrowprops=dict(arrowstyle="->", color=COLLAR, lw=1,
                        connectionstyle="arc3,rad=-0.3"), zorder=6)
    ax.annotate(
        f"the handle, folded,\nthrough {params.slot_width:.0f} × "
        f"{params.slot_height:.0f} mm",
        xy=(-params.slot_width / 4, 0),
        xytext=(-params.width / 2 - 2, params.head_radius + 5),
        fontsize=8.5, color=COLLAR, ha="right", va="bottom",
        arrowprops=dict(arrowstyle="->", color=COLLAR, lw=1,
                        connectionstyle="arc3,rad=0.3"), zorder=6)

    _span(ax, params.belly_y, params.belly_radius, f"belly {params.belly:.0f} mm", BAG)
    ax.annotate(
        "the biggest bundle\nthat goes in at all",
        xy=(-params.belly_radius * 0.8, params.belly_y),
        xytext=(-params.width / 2 - 3, params.belly_y - 6),
        fontsize=8.5, color=BAG, ha="right", va="center",
        arrowprops=dict(arrowstyle="->", color=BAG, lw=0.9), zorder=5)

    # The funnel's angle, drawn on the wall it actually describes.
    lean = params.funnel
    top, run = params.belly_y - params.belly_radius * math.sin(math.radians(lean)), 19.0
    x0 = (params.belly_radius - (params.belly_y - top) * math.sin(math.radians(lean))) \
        / math.cos(math.radians(lean))
    ax.plot([x0, x0 - run * math.tan(math.radians(lean))], [top, top - run],
            color=BAG, lw=0.8, ls=(0, (5, 3)), zorder=4)
    ax.plot([x0, x0], [top, top - run], color=INK, lw=0.7, ls=(0, (2, 3)), zorder=4)
    ax.add_patch(Arc((x0, top), 17, 17, angle=0, theta1=180 + lean, theta2=270,
                     color=BAG, lw=1.0, zorder=4))
    ax.text(x0 + 1.5, top - run / 2, f"{params.funnel:.0f}° funnel",
            fontsize=8.5, color=BAG, ha="left", va="center", zorder=5)

    mid = (params.throat_top_y + params.throat_bottom_y) / 2
    ax.annotate(
        f"throat {params.slot:.0f} mm —\nwhat pinches",
        xy=(0, mid), xytext=(params.width / 2 + 3, mid),
        fontsize=8.5, color=BAG, ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color=BAG, lw=1), zorder=5)
    ax.set_xlim(-params.width / 2 - 30, params.width / 2 + 32)
    ax.set_ylim(params.aperture_bottom - params.wall - 4, params.head_radius + 26)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("the design", color=INK, fontsize=11)


def _holds_view(ax, params: Params, face) -> None:
    """The funnel doing its one job: stopping each bundle at its own size."""
    _outline(ax, face, lw=1.1)
    _collar(ax, params)
    for bundle in BUNDLES:
        y = params.seats(bundle)
        ax.add_patch(CirclePatch((0, y), bundle / 2, facecolor=BAG, alpha=0.45,
                                 edgecolor=BAG, lw=1.2, zorder=3))
        ax.annotate(
            f"{bundle:.0f} mm",
            xy=(bundle / 2 * 0.7, y), xytext=(params.width / 2 + 4, y),
            fontsize=8.5, color=BAG, ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color=BAG, lw=0.9), zorder=5)

    lo, hi = params.grip
    ax.text(0, params.aperture_bottom - params.wall - 8,
            f"between {lo:.0f} and {hi:.0f} mm across, a bundle wedges\n"
            f"in the funnel and never reaches the throat's floor.\n"
            f"Nothing of any size reaches the outside.",
            fontsize=8.5, color=INK, ha="center", va="top", zorder=5)

    ax.set_xlim(-params.width / 2 - 8, params.width / 2 + 24)
    ax.set_ylim(params.aperture_bottom - params.wall - 26, params.head_radius + 22)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("each bundle stops at its own size", color=INK, fontsize=11)


def preview(params: Params, dest: Path, part: Part | None = None) -> Path:
    """One holder, three ways: as printed, as designed, and holding a bag."""
    part = build(params) if part is None else part
    points, tris = mesh(part, tol=0.02)
    face = profile(params).faces()[0]

    fig = plt.figure(figsize=(12.5, 7.4), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.25, 1], wspace=0.02,
                          left=0.01, right=0.99, top=0.94, bottom=0.02)
    bb = part.bounding_box()
    # Steeper than the default: this part is its profile, and a low view of an
    # 8mm prism shows the wall it was extruded into and none of the design.
    iso(fig.add_subplot(gs[0, 0], projection="3d"), points, tris,
        f"as printed · {bb.size.X:.0f} × {bb.size.Y:.0f} × {bb.size.Z:.0f} mm",
        elev=58, azim=-90)
    _design_view(fig.add_subplot(gs[0, 1]), params, face)
    _holds_view(fig.add_subplot(gs[0, 2]), params, face)

    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest


def draw(dest: Path | None = None) -> Path:
    """The house entry point: redraw this part's figure from the model.

    Every load-bearing part has one -- see CLAUDE.md. It takes no arguments so
    the drawing can always be regenerated, and so a test can prove it still
    can.
    """
    return preview(Params(), dest or Path("docs/bag_holder.png"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--strap", default="standard",
                    help=f"lead the collar is cut for: "
                         f"{', '.join(webbing.CATALOGUE)}")
    ap.add_argument("--gap", type=float, default=Params.gap,
                    help="clearance between the collar and its socket")
    ap.add_argument("--interlock", type=float, default=Params.interlock,
                    help="how much both swell at mid height, to hold the collar in")
    ap.add_argument("--belly", type=float, default=Params.belly,
                    help="widest part of the opening; the biggest bundle that "
                         "can be pushed in")
    ap.add_argument("--slot", type=float, default=Params.slot,
                    help="width of the throat, which is what pinches the handles")
    ap.add_argument("--funnel", type=float, default=Params.funnel,
                    help="degrees off the axis for the taper into the throat")
    ap.add_argument("--throat", type=float, default=Params.throat,
                    help="parallel length of the throat")
    ap.add_argument("--cheek", type=float, default=Params.cheek,
                    help="how far the shoulders bow out past a straight flare")
    ap.add_argument("--slot-ease", type=float, default=Params.slot_ease,
                    help="multiplies the collar's slot, so a folded handle has "
                         "room to be worked through")
    ap.add_argument("--edge-break", type=float, default=Params.edge_break,
                    help="chamfer along the top edges")
    ap.add_argument("--base-break", type=float, default=Params.base_break,
                    help="chamfer along the plate edges; together with "
                         "--edge-break this makes the section an octagon. Costs "
                         "first-layer width, so 0 goes back to a flat foot")
    ap.add_argument("--band", type=float, default=Params.band,
                    help="height of the extrusion, in mm")
    ap.add_argument("--wall", type=float, default=Params.wall,
                    help="width of the ribbon, everywhere")
    ap.add_argument("--copies", type=int, default=1,
                    help="how many to lay out on the plate")
    ap.add_argument("--material", default="ASA",
                    help="what to slice it for; ASA because this lives outdoors")
    ap.add_argument("--nozzle", type=float, default=profiles.DEFAULT_NOZZLE)
    ap.add_argument("-o", "--out", default="out/bag_holder",
                    help="output basename; .stl, .3mf and .png are written")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args(argv)

    params = Params(
        strap=webbing.named(args.strap), gap=args.gap, interlock=args.interlock,
        belly=args.belly, slot=args.slot, funnel=args.funnel, throat=args.throat,
        cheek=args.cheek, band=args.band, wall=args.wall,
        slot_ease=args.slot_ease, edge_break=args.edge_break,
        base_break=args.base_break,
    )
    holders = [build(params) for _ in range(args.copies)]
    part = holders[0] if len(holders) == 1 else plate(holders, nozzle=args.nozzle)

    base = Path(args.out)
    base.parent.mkdir(parents=True, exist_ok=True)
    export_stl(part, str(base.with_suffix(".stl")))
    print(f"wrote {base.with_suffix('.stl')}")

    # As with the clip: the 3mf needs Bambu Studio's profiles on disk, and the
    # STL is already written, so a missing slicer is a note and not a failure.
    try:
        from p2s.slicer import write_project

        filament = inventory.default(args.material).preset_for(args.nozzle)
        print(f"wrote {write_project(part, base.with_suffix('.3mf'), nozzle=args.nozzle, filament=filament)}"
              f" [{filament}]")
    except Exception as exc:  # noqa: BLE001 - profiles missing is not fatal here
        print(f"no 3mf: {exc}")

    if not args.no_preview:
        print(f"wrote {preview(params, base.with_suffix('.png'), holders[0])}")

    lo, hi = params.grip
    for bundle in BUNDLES:
        below = params.belly_y - params.seats(bundle)
        print(f"a {bundle:.0f} mm bundle wedges {below:.0f} mm below the belly")
    print(f"{args.copies} x {params.length:.0f} x {params.width:.0f} mm, wedging "
          f"{lo:.0f}-{hi:.0f} mm bundles")
    print(f"collar {2 * params.collar_radius:.1f} mm for a {params.strap.nominal} "
          f"lead, {params.engagement:.2f} mm engagement, leaning "
          f"{params.lean:.0f} deg")
    bb = part.bounding_box()
    print(f"valid={part.is_valid} {bb.size.X:.1f} × {bb.size.Y:.1f} × "
          f"{bb.size.Z:.1f} mm, {part.volume / 1000:.2f} cm^3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
