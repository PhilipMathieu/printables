"""Make the desk switch box from the command line, and look at it without a slicer.

    python -m tools.desk_switch
    python -m tools.desk_switch --finger 25 --jack-hole 11.5 -o out/desk_switch
    python -m tools.desk_switch --mirror            # jack on the left, lead on the right

Most of it nobody needs to touch. ``--finger`` is the one worth a thought: it is
the gap between the bat at the top of its throw and the desk, for the finger
that pushes it down, and the box is exactly as tall as that gap plus the
switch. ``--jack-hole`` is the one worth a measurement, because the threaded
5.5 x 2.1mm jacks are sold at anything from 11 to 12.3mm across.

Both parts land on one plate: the shell rim-down, the lid face-down. It is
written in ASA unless told otherwise, because the switch nut is torqued onto
the front wall and PLA creeps out from under a nut over a summer.

WIRING. The GSW-117 is SPDT, (on)-off-(on): one common and two momentary
outputs. On its own it can switch a motor on in one direction from either
throw, but it cannot reverse one -- that takes both poles of a DPDT, or a
pair of relays. For desk-up / desk-down from one bat, use the GSW-123, which
is the same body in the same hole (``--switch gsw-123``), and wire it as the
classic reversing X: the two centre terminals to the motor, the supply to one
end pair, and the other end pair cross-wired to the first. With the SPDT,
wire the supply's positive to the common, one throw to the motor, and treat
the other throw as spare. Either way the jack's negative goes straight to the
motor's other lead and the switch only ever breaks the positive.

The preview is three views: both parts as printed, the side section that is
the design -- the desk, the bat at rest and thrown, and the finger in the
gap -- and a plan at the switch axis with the jack, the lead and the screws.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle as CirclePatch
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import replace  # noqa: E402

from build123d import Part, export_stl  # noqa: E402

from p2s import inventory, profiles  # noqa: E402
from parts.desk_switch import (  # noqa: E402
    GSW_117,
    GSW_123,
    SLICE,
    Params,
    assembly,
    lid,
    shell,
)
from tools.render import INK, iso, mesh, plate, section  # noqa: E402

DESK = "#b08a5a"
HARDWARE = "#4a6fa5"
FINGER = "#c8792b"
SWITCHES = {"gsw-117": GSW_117, "gsw-123": GSW_123}


def _side_view(ax, points, tris, params: Params) -> None:
    """The design in one picture: section at the switch axis, desk up."""
    p, sw = params, params.switch
    segs = section(points, tris, 0, 0.0)
    # Installed: Z grows downward in the model, so flip it for the drawing.
    yz = segs[:, :, 1:3].copy()
    yz[:, :, 1] *= -1
    ax.add_collection(LineCollection(yz, colors=[INK], lw=1.6))

    # The desk, a band above the rim.
    w, d = p.footprint
    ax.add_patch(Rectangle((-30, 0), d + 40, 18, facecolor=DESK, alpha=0.25,
                           edgecolor=DESK, lw=1.0, hatch="///"))
    ax.text(-26, 9, "desktop", fontsize=9, color=DESK, ha="left", va="center")

    # The switch: body behind the wall, nut on it, bat at rest and thrown.
    z = -p.axis_depth
    ax.add_patch(Rectangle((p.panel, z - sw.body_length / 2), sw.body_depth, sw.body_length,
                           facecolor=HARDWARE, alpha=0.18, edgecolor=HARDWARE, lw=1.0))
    ax.add_patch(Rectangle((-(sw.washer + sw.nut_thickness), z - sw.nut_corners / 2),
                           sw.washer + sw.nut_thickness, sw.nut_corners,
                           facecolor=HARDWARE, alpha=0.5, edgecolor=HARDWARE, lw=0.8))
    for angle, alpha, lw in ((0, 0.35, 3.2), (-sw.throw, 1.0, 3.2), (sw.throw, 0.55, 3.2)):
        a = math.radians(angle)
        ax.plot([0, -sw.toggle * math.cos(a)], [z, z + sw.toggle * math.sin(a)],
                color=HARDWARE, lw=lw, alpha=alpha, solid_capstyle="round")
    ax.text(p.panel + sw.body_depth + 1.5, z, sw.name, fontsize=9, color=HARDWARE,
            ha="left", va="center")

    # The finger, where it goes to push the bat down.
    ax.add_patch(CirclePatch((-p.finger / 2, -p.finger / 2), p.finger / 2,
                             facecolor=FINGER, alpha=0.18, edgecolor=FINGER, lw=1.2))
    ax.annotate(
        f"{p.finger:.0f} mm for a finger",
        xy=(-p.finger / 2, -p.finger / 2), xytext=(-p.finger - 24, -p.finger - 14),
        fontsize=9, color=FINGER, ha="left",
        arrowprops=dict(arrowstyle="->", color=FINGER, lw=1, connectionstyle="arc3,rad=0.3"),
    )
    ax.annotate(
        "", xy=(-p.finger - 6, 0), xytext=(-p.finger - 6, z),
        arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9),
    )
    ax.text(-p.finger - 7.5, z / 2, f"axis {p.axis_depth:.1f} mm\nbelow the desk",
            fontsize=8.5, color=INK, ha="right", va="center")

    # A wood screw in the back flange, dashed, for the record.
    sy = p.outer_depth + p.flange / 2
    ax.plot([sy, sy], [-p.flange_thickness - 1, 14], color=INK, lw=1.1, ls=(0, (3, 2)))
    ax.text(sy + 2, 4, "#8 wood screws,\n4 of them", fontsize=8.5, color=INK, ha="left", va="center")
    ax.text(p.outer_depth / 2, -p.overall_height - 3, "lid comes off for wiring",
            fontsize=8.5, color=INK, ha="center", va="top")

    ax.set_xlim(-p.finger - 40, d + 26)
    ax.set_ylim(-p.overall_height - 9, 20)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("section at the toggle · bat thrown up, finger above it", color=INK, fontsize=11)


def _plan_view(ax, points, tris, params: Params) -> None:
    """Plan at the switch axis: walls, posts, and everything that goes through them."""
    p, sw, jk = params, params.switch, params.jack
    cut = section(points, tris, 2, p.axis_depth)
    ax.add_collection(LineCollection(cut[:, :, :2], colors=[INK], lw=1.6))

    # The flanges are above this cut; draw the footprint and the screws.
    w, d = p.footprint
    ax.add_patch(FancyBboxPatch((-w / 2, 0), w, d, boxstyle="round,pad=0,rounding_size=4",
                                facecolor="none", edgecolor=INK, lw=0.9, ls=(0, (3, 2))))
    for x, y in p.screw_positions:
        ax.add_patch(CirclePatch((x, y), p.screw_head / 2, facecolor="none",
                                 edgecolor=INK, lw=0.9, ls=(0, (3, 2))))
    for x, y in p.boss_positions:
        ax.add_patch(CirclePatch((x, y), p.boss_hole / 2, facecolor="white", edgecolor=INK, lw=0.8))

    # The switch, the jack with a plug in it, the lead through its notch.
    ax.add_patch(Rectangle((-sw.body_width / 2, p.panel), sw.body_width, sw.body_depth,
                           facecolor=HARDWARE, alpha=0.18, edgecolor=HARDWARE, lw=1.0))
    ax.plot([0, 0], [0, -sw.toggle], color=HARDWARE, lw=3.2, solid_capstyle="round")
    ax.add_patch(Rectangle((p.jack_x - jk.body / 2, p.back_inner - jk.body_depth), jk.body,
                           jk.body_depth, facecolor=HARDWARE, alpha=0.18, edgecolor=HARDWARE, lw=1.0))
    ax.add_patch(Rectangle((p.jack_x - 5.5, p.outer_depth), 11, jk.plug,
                           facecolor=HARDWARE, alpha=0.35, edgecolor=HARDWARE, lw=1.0))
    ax.text(p.jack_x, p.outer_depth + jk.plug + 2, "12 V in", fontsize=9, color=HARDWARE,
            ha="center", va="bottom")
    ax.plot([p.notch_x, p.notch_x], [p.tie_y - 12, p.outer_depth + jk.plug + 2],
            color=INK, lw=4.5, alpha=0.35, solid_capstyle="round")
    sx, sy = p.tie_slot
    for x in (p.notch_x - p.tie_spread / 2, p.notch_x + p.tie_spread / 2):
        ax.add_patch(Rectangle((x - sx / 2, p.tie_y - sy / 2), sx, sy, facecolor=INK, edgecolor="none"))
    ax.text(p.notch_x, p.outer_depth + jk.plug + 2, "to the motor", fontsize=9, color=INK,
            ha="center", va="bottom")
    ax.text(0, -sw.toggle - 2, "front, flush with the desk's edge", fontsize=8.5, color=INK,
            ha="center", va="top")

    ax.set_xlim(-w / 2 - 4, w / 2 + 4)
    ax.set_ylim(-sw.toggle - 9, p.outer_depth + jk.plug + 9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"plan at the axis · {p.width:.0f} × {p.depth:.0f} mm inside, {w:.0f} mm over the flanges",
        color=INK, fontsize=11,
    )


def preview(params: Params, dest: Path, parts: tuple[Part, Part] | None = None) -> Path:
    """Both parts as printed, the section that is the design, and the plan."""
    box, cover = parts if parts is not None else (shell(params), lid(params))
    laid = plate([box, cover], check=False)
    lp, lt = mesh(laid, tol=0.05)
    ap, at = mesh(assembly(params), tol=0.02)

    fig = plt.figure(figsize=(15, 5.2), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.25, 1.0], wspace=0.02,
                          left=0.01, right=0.99, top=0.90, bottom=0.04)
    bb = box.bounding_box()
    iso(fig.add_subplot(gs[0, 0], projection="3d"), lp, lt,
        f"as printed · shell {bb.size.X:.0f} × {bb.size.Y:.0f} × {bb.size.Z:.0f} mm, and its lid",
        elev=32, azim=-50)
    _side_view(fig.add_subplot(gs[0, 1]), ap, at, params)
    _plan_view(fig.add_subplot(gs[0, 2]), ap, at, params)

    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest


def _plated(parts: list[Part], nozzle: float) -> Part:
    """Both parts on one plate, checked against the bed where there is one."""
    try:
        return plate(parts, nozzle=nozzle)
    except profiles.ProfileError as exc:
        print(f"no bed check: {exc}")
        return plate(parts, nozzle=nozzle, check=False)


def params_from(args: argparse.Namespace) -> Params:
    switch = SWITCHES[args.switch.lower()]
    if args.no_tab_hole:
        switch = replace(switch, tab_offset=0.0)
    jack = replace(Params.jack, hole=args.jack_hole)
    jack_x, notch_x = args.jack_x, args.notch_x
    if args.mirror:
        jack_x, notch_x = -jack_x, -notch_x
    return Params(
        switch=switch, jack=jack, finger=args.finger, width=args.width, depth=args.depth,
        jack_x=jack_x, notch_x=notch_x, jack_depth=args.jack_depth,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--finger", type=float, default=Params.finger,
                    help="gap between the thrown-up bat and the desk, mm")
    ap.add_argument("--switch", default="gsw-117", choices=sorted(SWITCHES),
                    help="gsw-117 (SPDT, as asked) or gsw-123 (DPDT, reverses a motor)")
    ap.add_argument("--no-tab-hole", action="store_true",
                    help="leave out the lock ring's anti-rotation hole")
    ap.add_argument("--jack-hole", type=float, default=Params.jack.hole,
                    help="panel hole for the DC jack; measure the thread and add 0.5")
    ap.add_argument("--jack-x", type=float, default=Params.jack_x)
    ap.add_argument("--jack-depth", type=float, default=Params.jack_depth)
    ap.add_argument("--notch-x", type=float, default=Params.notch_x)
    ap.add_argument("--mirror", action="store_true",
                    help="swap the jack and the lead exit to the other sides")
    ap.add_argument("--width", type=float, default=Params.width, help="inside, mm")
    ap.add_argument("--depth", type=float, default=Params.depth, help="inside, mm")
    ap.add_argument("--part", default="both", choices=("both", "shell", "lid"))
    ap.add_argument("--material", default="ASA",
                    help="what to slice it for; ASA because a nut on PLA loosens")
    ap.add_argument("--nozzle", type=float, default=profiles.DEFAULT_NOZZLE)
    ap.add_argument("--slice", action="store_true",
                    help="also slice the 3mf with this part's wall count, for a time")
    ap.add_argument("-o", "--out", default="out/desk_switch",
                    help="output basename; .stl, .3mf and .png are written")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args(argv)

    params = params_from(args)
    box, cover = shell(params), lid(params)
    chosen = {"both": [box, cover], "shell": [box], "lid": [cover]}[args.part]
    part = chosen[0] if len(chosen) == 1 else _plated(chosen, args.nozzle)

    base = Path(args.out)
    base.parent.mkdir(parents=True, exist_ok=True)
    export_stl(part, str(base.with_suffix(".stl")))
    print(f"wrote {base.with_suffix('.stl')}")

    # The 3mf needs Bambu Studio's profiles on disk, and the STL is already
    # written, so a missing slicer is a note and not a failure.
    try:
        from p2s import slicer

        filament = inventory.default(args.material).preset_for(args.nozzle)
        project = slicer.write_project(part, base.with_suffix(".3mf"), nozzle=args.nozzle, filament=filament)
        print(f"wrote {project} [{filament}]")
        if args.slice:
            result = slicer.slice_project(project, nozzle=args.nozzle, filament=filament, overrides=SLICE)
            print(result)
    except Exception as exc:  # noqa: BLE001 - profiles missing is not fatal here
        print(f"no 3mf: {exc}")

    if not args.no_preview:
        print(f"wrote {preview(params, base.with_suffix('.png'), (box, cover))}")

    sw = params.switch
    print(f"{sw.name}: axis {params.axis_depth:.1f} mm below the desk, "
          f"{params.finger:.0f} mm finger gap, {params.panel} mm panel under the nut")
    w, d = params.footprint
    print(f"footprint on the desk {w:.0f} × {d:.0f} mm, {params.overall_height:.0f} mm tall with the lid")
    print(f"slice with wall_loops={SLICE['wall_loops']} so the wall round the bushing is solid")
    bb = part.bounding_box()
    print(f"valid={part.is_valid} {bb.size.X:.1f} × {bb.size.Y:.1f} × "
          f"{bb.size.Z:.1f} mm, {part.volume / 1000:.1f} cm^3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
