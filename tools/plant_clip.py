"""Make plant clips from the command line, and look at them without a slicer.

    python -m tools.plant_clip --stem 8 --copies 6
    python -m tools.plant_clip --set --copies 3 -o out/vine_clips

``--set`` builds one clip per size in ``parts.plant_clip.STEMS``, which is the
useful thing to print first: a clip holds a band of stem diameters a couple of
millimetres wide, so the way to find out what fits a given pothos is to stick a
few sizes on the wall and see. ``--copies`` repeats whatever was asked for and
lays it out on the plate, because one clip is a four-minute print and nobody
wants one clip.

Everything is written in ASA unless told otherwise. The clip is a small part
under a small load that never lets up, which is exactly the case PLA creeps in.

The preview is three views: what it looks like printed, the section that is
actually the design, and a plan with the Command strip drawn on it so it is
obvious that the pad stops short of the pull tab.
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
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build123d import Part, export_stl  # noqa: E402

from geom import strips  # noqa: E402
from p2s import inventory, profiles  # noqa: E402
from parts.plant_clip import STEMS, Params, build  # noqa: E402
from tools.render import INK, iso, mesh, plate, section  # noqa: E402

STEM = "#4f7a3a"
ADHESIVE = "#c8792b"


def _section_view(ax, points, tris, params: Params) -> None:
    """The design, in one picture: the C, the stem it is cut for, the mouth."""
    segs = section(points, tris, 0, params.positions[0])
    ax.add_collection(LineCollection(segs[:, :, 1:3], colors=[INK], lw=1.8))

    zc = params.axis_z
    ax.add_patch(CirclePatch((0, zc), params.stem / 2, facecolor=STEM, alpha=0.5,
                             edgecolor=STEM, lw=1.2, zorder=0))
    ax.add_patch(CirclePatch((0, zc), params.bore / 2, facecolor="none",
                             edgecolor=STEM, lw=1.0, ls=(0, (4, 3)), zorder=0))

    # The mouth, drawn where it is actually measured: across the bore, not
    # across the tips, which are further apart and not what holds anything.
    half = math.radians(params.mouth_angle)
    y, z = params.bore_radius * math.sin(half), params.bore_radius * math.cos(half)
    ax.plot([-y, y], [zc + z, zc + z], color=ADHESIVE, lw=1.6, solid_capstyle="butt")
    ax.annotate(
        f"mouth {params.mouth:.1f} mm",
        xy=(0, zc + z), xytext=(params.outer_radius + 3, zc + params.outer_radius + 2),
        fontsize=9, color=ADHESIVE, ha="left",
        arrowprops=dict(arrowstyle="->", color=ADHESIVE, lw=1,
                        connectionstyle="arc3,rad=0.25"),
    )
    ax.plot([-params.pad_width / 2, params.pad_width / 2], [0, 0],
            color=ADHESIVE, lw=2.5, solid_capstyle="butt")
    ax.text(params.pad_width / 2, -1.4, "adhesive face, on the plate",
            fontsize=8.5, color=ADHESIVE, ha="right", va="top")

    lo, hi = params.grip
    ax.set_xlim(-params.pad_width / 2 - 2, params.pad_width / 2 + 8)
    ax.set_ylim(-4, zc + params.outer_radius + 6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"section · {params.wrap:.0f}° wrap holds {lo:.1f}–{hi:.1f} mm stems",
        color=INK, fontsize=11,
    )


def _plan_view(ax, points, tris, params: Params) -> None:
    """Plan of the pad with the strip on it, tab hanging off the end."""
    strip = params.strip
    x0 = -params.pad_length / 2
    ax.add_patch(Rectangle((x0, -strip.width / 2), strip.bond, strip.width,
                           facecolor=ADHESIVE, alpha=0.18, edgecolor=ADHESIVE,
                           lw=1.0, ls=(0, (4, 3))))
    ax.add_patch(Rectangle((x0 + strip.bond, -strip.width / 2), strip.tab,
                           strip.width, facecolor="white", edgecolor=ADHESIVE,
                           lw=1.0, hatch="////", alpha=0.9))
    # Two cuts: the pad at mid thickness, and the arms at the stem's height.
    # The pad alone would not say where along it the clips actually are, since
    # their bases are inside its outline and the union leaves no edge there.
    for z, lw, colour in ((params.pad_thickness / 2, 1.8, INK),
                          (params.axis_z, 1.1, STEM)):
        cut = section(points, tris, 2, z)
        if len(cut):
            ax.add_collection(LineCollection(cut[:, :, :2], colors=[colour], lw=lw))
    ax.text(x0 + strip.bond + strip.tab / 2, strip.width / 2 + 1.2,
            "pull tab, clear of the pad", fontsize=8.5, color=ADHESIVE,
            ha="center", va="bottom")
    ax.annotate(
        "", xy=(x0, -params.pad_width / 2 - 1.5),
        xytext=(x0 + params.pad_length, -params.pad_width / 2 - 1.5),
        arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9),
    )
    ax.text(0, -params.pad_width / 2 - 2.2,
            f"pad {params.pad_length:.1f} mm = {strip.name} strip less its tab",
            fontsize=8.5, color=INK, ha="center", va="top")

    ax.set_xlim(x0 - 3, x0 + strip.length + 3)
    ax.set_ylim(-params.pad_width / 2 - 8, params.pad_width / 2 + 5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"plan · {params.count} clip{'s' if params.count > 1 else ''} on a "
        f"{strip.name} strip",
        color=INK, fontsize=11,
    )


def preview(params: Params, dest: Path, part: Part | None = None) -> Path:
    """One clip, three ways: as printed, in section, and in plan."""
    part = build(params) if part is None else part
    points, tris = mesh(part, tol=0.02)

    fig = plt.figure(figsize=(13.5, 4.6), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1, 1.25], wspace=0.02,
                          left=0.01, right=0.99, top=0.90, bottom=0.04)
    bb = part.bounding_box()
    iso(fig.add_subplot(gs[0, 0], projection="3d"), points, tris,
         f"as printed · {bb.size.X:.0f} × {bb.size.Y:.0f} × {bb.size.Z:.0f} mm")
    _section_view(fig.add_subplot(gs[0, 1]), points, tris, params)
    _plan_view(fig.add_subplot(gs[0, 2]), points, tris, params)

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
    return preview(Params(), dest or Path("docs/plant_clip.png"))


def params_from(args: argparse.Namespace) -> list[Params]:
    """One Params per clip on the plate, sizes first and copies after."""
    base = Params(
        strip=strips.named(args.strip),
        clearance=args.clearance,
        wrap=args.wrap,
        wall=args.wall,
        clip_width=args.clip_width,
        count=args.count,
    )
    stems = STEMS if args.set else (args.stem,)
    return [base.for_stem(s) for s in stems for _ in range(args.copies)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stem", type=float, default=Params.stem,
                    help="stem diameter the clip is cut for, in mm")
    ap.add_argument("--set", action="store_true",
                    help=f"one clip of each size in {STEMS}, instead of --stem")
    ap.add_argument("--strip", default="small",
                    help=f"command strip to build the pad around: "
                         f"{', '.join(strips.CATALOGUE)}")
    ap.add_argument("--count", type=int, default=Params.count,
                    help="clips on one pad")
    ap.add_argument("--copies", type=int, default=1,
                    help="how many of each to lay out on the plate")
    ap.add_argument("--wrap", type=float, default=Params.wrap,
                    help="degrees of stem the C wraps; more grips harder and "
                         "prints worse")
    ap.add_argument("--wall", type=float, default=Params.wall)
    ap.add_argument("--clearance", type=float, default=Params.clearance)
    ap.add_argument("--clip-width", type=float, default=Params.clip_width)
    ap.add_argument("--material", default="ASA",
                    help="what to slice it for; ASA because PLA creeps")
    ap.add_argument("--nozzle", type=float, default=profiles.DEFAULT_NOZZLE)
    ap.add_argument("-o", "--out", default="out/plant_clip",
                    help="output basename; .stl, .3mf and .png are written")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args(argv)

    every = params_from(args)
    clips = [build(p) for p in every]
    part = clips[0] if len(clips) == 1 else plate(clips, nozzle=args.nozzle)

    base = Path(args.out)
    base.parent.mkdir(parents=True, exist_ok=True)
    export_stl(part, str(base.with_suffix(".stl")))
    print(f"wrote {base.with_suffix('.stl')}")

    # As with the coin: the 3mf needs Bambu Studio's profiles on disk, and the
    # STL is already written, so a missing slicer is a note and not a failure.
    try:
        from p2s.slicer import write_project

        filament = inventory.default(args.material).preset_for(args.nozzle)
        print(f"wrote {write_project(part, base.with_suffix('.3mf'), nozzle=args.nozzle, filament=filament)}"
              f" [{filament}]")
    except Exception as exc:  # noqa: BLE001 - profiles missing is not fatal here
        print(f"no 3mf: {exc}")

    if not args.no_preview:
        print(f"wrote {preview(every[0], base.with_suffix('.png'), clips[0])}")

    tally: dict[Params, int] = {}
    for p in every:
        tally[p] = tally.get(p, 0) + 1
    for p, n in sorted(tally.items(), key=lambda kv: kv[0].stem):
        lo, hi = p.grip
        print(f"{n} x {p.stem:.0f} mm clip: holds {lo:.1f}-{hi:.1f} mm stems")
    bb = part.bounding_box()
    print(f"valid={part.is_valid} {bb.size.X:.1f} × {bb.size.Y:.1f} × "
          f"{bb.size.Z:.1f} mm, {part.volume / 1000:.2f} cm^3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
