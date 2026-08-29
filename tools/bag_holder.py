"""Make bag holders from the command line, and look at them without a slicer.

    python -m tools.bag_holder --strap standard
    python -m tools.bag_holder --set -o out/bag_holders

``--strap`` is the only number most people need: measure the lead, pick the
width it was sold as. ``--set`` builds one for every width in the catalogue,
which is the thing to print if there is more than one lead by the door -- the
horn is identical on all four, so a set costs nothing but the heads.

Everything is written in ASA unless told otherwise. This is an outdoor part on
a lead: UV, cold mornings, and the pavement every time it is dropped.

The preview is three views: what it looks like printed, the profile that is
actually the design, with the climb the bag would have to make to get out of
it, and the slot with a folded handle drawn in it.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as CirclePatch
from matplotlib.patches import FancyBboxPatch, Polygon as MplPolygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build123d import Part, export_stl  # noqa: E402

from geom import webbing  # noqa: E402
from p2s import inventory, profiles  # noqa: E402
from parts.bag_holder import Params, build, profile  # noqa: E402
from tools.render import ASA, INK, iso, mesh, plate  # noqa: E402

BAG = "#2f8f5b"
WEBBING = "#3b6ea5"

KNOT = 8.0
"""Diameter of a tied bag's handles, drawn for scale. Soft: a knot squeezes,
which is why the gate is not sized off it and the geometry does the holding."""


def _loop(wire, samples: int = 500):
    return [(p.X, p.Y) for p in (wire.position_at(i / samples) for i in range(samples))]


def _outline(ax, params: Params, face, fill: str = ASA, lw: float = 1.4) -> None:
    """The profile, painted outside in: body, then the slot back in white.

    Same trick as the fidget's plan -- an even-odd polygon with both loops in it
    leaves a seam where the two joins, and here that seam runs right across the
    face the strap bears on.
    """
    ax.add_patch(MplPolygon(_loop(face.outer_wire()), closed=True, facecolor=fill,
                            edgecolor=INK, lw=lw, zorder=1))
    for hole in face.inner_wires():
        ax.add_patch(MplPolygon(_loop(hole), closed=True, facecolor="white",
                                edgecolor=INK, lw=lw, zorder=2))


def _at(params: Params, bearing: float, radius: float) -> tuple[float, float]:
    """A point on the horn, by bearing off straight up and radius from its centre."""
    a = math.radians(bearing)
    return radius * math.sin(a), params.horn_y + radius * math.cos(a)


def _design_view(ax, params: Params, face) -> None:
    """The profile, with the bag in it and the only way out marked."""
    _outline(ax, params, face)

    seat_bottom = params.horn_y - params.seat_radius
    ax.add_patch(CirclePatch((0, params.horn_y), params.seat_radius, facecolor="none",
                             edgecolor=BAG, lw=1.0, ls=(0, (4, 3)), zorder=3))
    ax.add_patch(CirclePatch((0, seat_bottom + KNOT / 2), KNOT / 2, facecolor=BAG,
                             alpha=0.55, edgecolor=BAG, lw=1.2, zorder=3))

    # The gate, drawn where it is measured: across the seat, not across the tips,
    # which are further apart and are not what holds anything in.
    lo = _at(params, params.gate_bearing - params.gate_half, params.seat_radius)
    hi = _at(params, params.escape_bearing, params.seat_radius)
    ax.plot(*zip(lo, hi), color=BAG, lw=1.8, solid_capstyle="butt", zorder=4)
    ghost = _at(params, params.gate_bearing, params.seat_radius)
    ax.add_patch(CirclePatch(ghost, KNOT / 2, facecolor="none", edgecolor=BAG,
                             lw=1.0, ls=(0, (2, 2)), zorder=3))
    ax.annotate(f"gate {params.gate:.0f} mm", xy=ghost,
                xytext=(params.outer_radius + 2,
                        params.horn_y + params.seat_radius + 10),
                fontsize=9, color=BAG, ha="left",
                arrowprops=dict(arrowstyle="->", color=BAG, lw=1,
                                connectionstyle="arc3,rad=0.25"))

    # The centreline, and the climb from the seat up to the lowest way out.
    ax.plot([0, params.outer_radius + 14], [params.horn_y] * 2,
            color=INK, lw=0.8, ls=(0, (6, 4)), zorder=0)
    ax.text(params.outer_radius + 2, params.horn_y - 1.0,
            "the horn's centreline —\nthe whole gate is above it",
            fontsize=8.5, color=INK, ha="left", va="top")
    rail = -params.outer_radius - 3
    ax.annotate("", xy=(rail, seat_bottom), xytext=(rail, hi[1]),
                arrowprops=dict(arrowstyle="<->", color=BAG, lw=1.2))
    ax.text(rail - 1.2, (seat_bottom + hi[1]) / 2, f"lift {params.lift:.1f} mm",
            fontsize=9, color=BAG, ha="right", va="center", rotation=90)

    ax.set_xlim(-params.outer_radius - 14, params.outer_radius + 22)
    ax.set_ylim(params.horn_y - params.outer_radius - 3, params.head_height / 2 + 3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"the design · {params.seat:.0f} mm seat, the bag climbs to leave",
        color=INK, fontsize=11,
    )


def _slot_view(ax, params: Params, face) -> None:
    """The head, with the lead's folded handle drawn where it actually sits."""
    _outline(ax, params, face, lw=1.6)

    # The strap runs through the slot at right angles to this page, so what is
    # drawn is its section: each ply as wide as the webbing and as thick as one
    # layer of it, stacked against the top of the slot, which is where the
    # holder's whole weight pushes them.
    top = params.slot_height / 2
    for i in range(params.plies):
        y = top - (i + 1) * params.strap.thickness
        ax.add_patch(FancyBboxPatch(
            (-params.strap.width / 2 + 0.4, y + 0.2),
            params.strap.width - 0.8, params.strap.thickness - 0.4,
            boxstyle="round,pad=0.2", facecolor=WEBBING,
            alpha=0.45 if i % 2 else 0.22,  # so the plies read as two, not one
            edgecolor=WEBBING, lw=1.1, zorder=3))
    ax.annotate(
        f"{params.plies} plies of folded handle,\nrunning out of the page\n"
        f"through {params.band:.0f} mm of slot",
        xy=(0, top - params.strap.stack(params.plies) / 2),
        xytext=(params.head_width / 2 + 3, -params.head_height / 2 - 1),
        fontsize=8.5, color=WEBBING, ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color=WEBBING, lw=1,
                        connectionstyle="arc3,rad=-0.3"), zorder=4)

    flat = params.slot_width / 2 - params.slot_corner
    ax.plot([-flat, flat], [top] * 2, color=WEBBING, lw=2.6,
            solid_capstyle="butt", zorder=4)
    ax.annotate(
        f"bears on {params.bearing_area:.0f} mm² of wall",
        xy=(flat * 0.5, top),
        xytext=(params.head_width / 2 + 3, params.head_height / 2 + 6),
        fontsize=9, color=WEBBING, ha="left",
        arrowprops=dict(arrowstyle="->", color=WEBBING, lw=1,
                        connectionstyle="arc3,rad=-0.25"))
    ax.set_xlim(-params.head_width / 2 - 5, params.head_width / 2 + 33)
    ax.set_ylim(-params.head_height / 2 - params.neck - 3, params.head_height / 2 + 9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"the slot · {params.slot_width:.1f} × {params.slot_height:.1f} mm, cut to "
        f"pass a folded {params.strap.nominal} handle",
        color=INK, fontsize=11,
    )


def preview(params: Params, dest: Path, part: Part | None = None) -> Path:
    """One holder, three ways: as printed, as designed, and on a lead."""
    part = build(params) if part is None else part
    points, tris = mesh(part, tol=0.02)
    face = profile(params).faces()[0]

    fig = plt.figure(figsize=(13.5, 5.2), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.05, 1.2], wspace=0.02,
                          left=0.01, right=0.99, top=0.92, bottom=0.03)
    bb = part.bounding_box()
    # Steeper than the default: this part is its profile, and a low view of a
    # 12mm prism shows the wall it was extruded into and none of the design.
    iso(fig.add_subplot(gs[0, 0], projection="3d"), points, tris,
        f"as printed · {bb.size.X:.0f} × {bb.size.Y:.0f} × {bb.size.Z:.0f} mm",
        elev=58, azim=-90)
    _design_view(fig.add_subplot(gs[0, 1]), params, face)
    _slot_view(fig.add_subplot(gs[0, 2]), params, face)

    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest


def params_from(args: argparse.Namespace) -> list[Params]:
    """One Params per holder on the plate, widths first and copies after."""
    base = Params(
        plies=args.plies,
        band=args.band,
        wall=args.wall,
        seat=args.seat,
        gate=args.gate,
        gate_bearing=args.gate_bearing,
    )
    straps = tuple(webbing.CATALOGUE) if args.set else (args.strap,)
    return [base.for_strap(s) for s in straps for _ in range(args.copies)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--strap", default="standard",
                    help=f"lead the slot is cut for: {', '.join(webbing.CATALOGUE)}")
    ap.add_argument("--set", action="store_true",
                    help="one holder per width in the catalogue, instead of --strap")
    ap.add_argument("--copies", type=int, default=1,
                    help="how many of each to lay out on the plate")
    ap.add_argument("--plies", type=int, default=Params.plies,
                    help="layers of webbing the slot has to pass; 2 is the handle")
    ap.add_argument("--band", type=float, default=Params.band,
                    help="height of the extrusion, in mm")
    ap.add_argument("--wall", type=float, default=Params.wall,
                    help="width of the ribbon, everywhere")
    ap.add_argument("--seat", type=float, default=Params.seat,
                    help="bore of the horn the bag hangs in")
    ap.add_argument("--gate", type=float, default=Params.gate,
                    help="narrowest part of the way in, measured across the seat")
    ap.add_argument("--gate-bearing", type=float, default=Params.gate_bearing,
                    help="degrees from straight up to the middle of the gate; "
                         "smaller holds harder")
    ap.add_argument("--material", default="ASA",
                    help="what to slice it for; ASA because this lives outdoors")
    ap.add_argument("--nozzle", type=float, default=profiles.DEFAULT_NOZZLE)
    ap.add_argument("-o", "--out", default="out/bag_holder",
                    help="output basename; .stl, .3mf and .png are written")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args(argv)

    every = params_from(args)
    holders = [build(p) for p in every]
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
        print(f"wrote {preview(every[0], base.with_suffix('.png'), holders[0])}")

    tally: dict[Params, int] = {}
    for p in every:
        tally[p] = tally.get(p, 0) + 1
    for p, n in tally.items():
        print(f"{n} x {p.strap.name} ({p.strap.nominal}): slot "
              f"{p.slot_width:.1f} x {p.slot_height:.1f} mm, {p.lift:.1f} mm lift")
    bb = part.bounding_box()
    print(f"valid={part.is_valid} {bb.size.X:.1f} × {bb.size.Y:.1f} × "
          f"{bb.size.Z:.1f} mm, {part.volume / 1000:.2f} cm^3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
