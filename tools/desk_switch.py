"""Make the desk switch box from the command line, and look at it without a slicer.

    python -m tools.desk_switch
    python -m tools.desk_switch --coupon                # just the front, to test-fit the switch
    python -m tools.desk_switch --cutout 22.2 26.2 --jack-hole 11.5 -o out/desk_switch
    python -m tools.desk_switch --mirror                # jack on the left, lead on the right

Two numbers are worth a measurement before printing. ``--cutout`` is the
rocker's panel hole, width by length: KCD4 is a family name, and the housing
behind the bezel is 22 x 30mm on some and 22 x 26mm on others, so measure it
and add 0.2. ``--jack-hole`` is the threaded 5.5 x 2.1mm jack's, which are
sold at anything from 11 to 12.3mm across. ``--coupon`` prints just the front
of the box, the same way up, so the switch can be snapped into the real
cutout and web in twenty minutes before committing two hours to the shell.

Both parts land on one plate: the shell rim-down, the lid face-down. It is
written in ASA unless told otherwise, because the clips hold the switch by
pressing on the web, and PLA creeps out from under a constant load.

WIRING. The switch is DPDT (on)-off-(on): two poles side by side, each a
column of three tabs with its common in the middle. Wired as the reversing
cross it drives the motor one way while the top is held, the other way while
the bottom is held, and stops when let go:

    the two middle tabs           -> the motor's two leads
    top-left and bottom-right     -> jack +   (a jumper across the back)
    top-right and bottom-left     -> jack -   (the other jumper)

Which end pair closes on which press is up to the maker -- on many rockers
pressing the top closes the bottom pair -- and it does not matter: the cross
puts the supply across the motor one way for one press and the other way
for the other, and the centre off shorts nothing. If the desk goes the wrong
way, swap the motor leads. Check the cross with a meter before the jack sees
power: a jumper on the wrong pair is a dead short on one press.

The preview is three views: both parts as printed, the side section that is
the design -- the desk, the rocker, a fingertip on its top half, and the web
the clips grab -- and a plan at the switch with the jack, the lead and the
screws.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle as CirclePatch
from matplotlib.patches import Polygon as PolygonPatch
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import replace  # noqa: E402

from build123d import Part, export_stl  # noqa: E402

from p2s import inventory, profiles  # noqa: E402
from parts.desk_switch import (  # noqa: E402
    KCD4,
    SLICE,
    Params,
    assembly,
    coupon,
    lid,
    shell,
)
from tools.render import INK, iso, mesh, plate, section  # noqa: E402

DESK = "#b08a5a"
HARDWARE = "#4a6fa5"
FINGER = "#c8792b"


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

    # The switch: body behind the wall, bezel on the face, the rocker tipped
    # with its top end out, which is the end the finger presses.
    z = -p.axis_depth
    bl, bt = sw.bezel[1], sw.bezel_thickness
    cl = sw.cutout[1]
    ax.add_patch(Rectangle((0, z - sw.body[1] / 2), sw.body_depth, sw.body[1],
                           facecolor=HARDWARE, alpha=0.18, edgecolor=HARDWARE, lw=1.0))
    ax.add_patch(Rectangle((-bt, z - bl / 2), bt, bl,
                           facecolor=HARDWARE, alpha=0.5, edgecolor=HARDWARE, lw=0.8))
    ax.add_patch(PolygonPatch(
        [(-bt, z + cl / 2 - 1.5), (-bt - sw.rocker_rise, z + cl / 2 - 3.0),
         (-bt - 1.5, z - cl / 2 + 1.5), (-bt, z - cl / 2 + 1.5)],
        closed=True, facecolor=HARDWARE, alpha=0.8, edgecolor=HARDWARE, lw=0.8))
    ax.text(sw.body_depth + 1.5, z, sw.name, fontsize=9, color=HARDWARE,
            ha="left", va="center")
    ax.annotate(
        f"{p.clip_panel:.0f} mm web\nfor the clips", xy=(p.clip_panel / 2, z - cl / 2 - 1.5),
        xytext=(-bt - sw.rocker_rise - 3, z - cl / 2 - 6), fontsize=8.5, color=INK, ha="right",
        arrowprops=dict(arrowstyle="->", color=INK, lw=0.8),
    )

    # The fingertip, on the top half, with the desk above it.
    front = bt + sw.rocker_rise
    pz = -p.press_depth
    ax.add_patch(Rectangle((-front - 34, pz - p.finger / 2), 34, p.finger,
                           facecolor=FINGER, alpha=0.18, edgecolor=FINGER, lw=1.2))
    ax.text(-front - 17, pz, "press for up", fontsize=8.5, color=FINGER, ha="center", va="center")
    ax.annotate(
        "", xy=(-front - 38, 0), xytext=(-front - 38, z + bl / 2),
        arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9),
    )
    ax.text(-front - 39.5, (z + bl / 2) / 2, f"{p.headroom:.0f} mm", fontsize=8.5, color=INK,
            ha="right", va="center")

    # A wood screw in the back flange, dashed, for the record.
    sy = p.outer_depth + p.flange / 2
    ax.plot([sy, sy], [-p.flange_thickness - 1, 14], color=INK, lw=1.1, ls=(0, (3, 2)))
    ax.text(sy + 2, 4, "#8 wood screws,\n4 of them", fontsize=8.5, color=INK, ha="left", va="center")
    ax.text(p.outer_depth / 2, -p.overall_height - 3, "lid comes off for wiring",
            fontsize=8.5, color=INK, ha="center", va="top")

    ax.set_xlim(-front - 52, d + 26)
    ax.set_ylim(-p.overall_height - 9, 20)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("section at the rocker · a fingertip on its top half", color=INK, fontsize=11)


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
    front = sw.bezel_thickness + sw.rocker_rise
    ax.add_patch(Rectangle((-sw.body[0] / 2, 0), sw.body[0], sw.body_depth,
                           facecolor=HARDWARE, alpha=0.18, edgecolor=HARDWARE, lw=1.0))
    ax.add_patch(Rectangle((-sw.bezel[0] / 2, -sw.bezel_thickness), sw.bezel[0], sw.bezel_thickness,
                           facecolor=HARDWARE, alpha=0.5, edgecolor=HARDWARE, lw=0.8))
    ax.add_patch(Rectangle((-sw.cutout[0] / 2 + 1.5, -front), sw.cutout[0] - 3, sw.rocker_rise,
                           facecolor=HARDWARE, alpha=0.8, edgecolor=HARDWARE, lw=0.8))
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
    ax.text(0, -front - 2, "front, flush with the desk's edge", fontsize=8.5, color=INK,
            ha="center", va="top")

    ax.set_xlim(-w / 2 - 4, w / 2 + 4)
    ax.set_ylim(-front - 9, p.outer_depth + jk.plug + 9)
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
    w, length = args.cutout
    grow_w, grow_l = w - KCD4.cutout[0], length - KCD4.cutout[1]
    switch = replace(
        KCD4,
        cutout=(w, length),
        bezel=(KCD4.bezel[0] + grow_w, KCD4.bezel[1] + grow_l),
        body=(KCD4.body[0] + grow_w, KCD4.body[1] + grow_l),
    )
    jack = replace(Params.jack, hole=args.jack_hole)
    jack_x, notch_x = args.jack_x, args.notch_x
    if args.mirror:
        jack_x, notch_x = -jack_x, -notch_x
    return Params(
        switch=switch, jack=jack, headroom=args.headroom, clip_panel=args.clip_panel,
        width=args.width, depth=args.depth, jack_x=jack_x, notch_x=notch_x,
        jack_depth=args.jack_depth,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cutout", type=float, nargs=2, metavar=("W", "L"), default=KCD4.cutout,
                    help="rocker panel hole, width by length; measure the housing and add 0.2")
    ap.add_argument("--clip-panel", type=float, default=Params.clip_panel,
                    help="web thickness the clips grab, mm (1 to 3)")
    ap.add_argument("--headroom", type=float, default=Params.headroom,
                    help="desk to the top of the bezel, mm")
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
    ap.add_argument("--coupon", action="store_true",
                    help="only the front of the shell, to test-fit the switch")
    ap.add_argument("--material", default="ASA",
                    help="what to slice it for; ASA because the clips' grip creeps out of PLA")
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
    if args.coupon:
        chosen = [coupon(params)]
        if args.out == "out/desk_switch":
            args.out = "out/desk_switch_coupon"
        args.no_preview = True
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
    print(f"{sw.name}: {sw.cutout[0]} x {sw.cutout[1]} mm cutout in a {params.clip_panel} mm web, "
          f"bezel {params.headroom:.0f} mm below the desk")
    w, d = params.footprint
    print(f"footprint on the desk {w:.0f} × {d:.0f} mm, {params.overall_height:.0f} mm tall with the lid")
    print(f"slice with wall_loops={SLICE['wall_loops']} so the web and the wall round it are solid")
    bb = part.bounding_box()
    print(f"valid={part.is_valid} {bb.size.X:.1f} × {bb.size.Y:.1f} × "
          f"{bb.size.Z:.1f} mm, {part.volume / 1000:.1f} cm^3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
