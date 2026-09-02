"""Make the drill press dog set from the command line, and see how it works.

    python -m tools.bench_dogs --part ladder          # print this first
    python -m tools.bench_dogs --part deck
    python -m tools.bench_dogs --set --copies 1       # a starter plate
    python -m tools.bench_dogs --part fence --span 6

``--part ladder`` is the one to run first: four stops at four shank clearances,
about twenty minutes of printing, and the tightest one that still drops into the
deck under its own weight tells you what ``--fit`` should be for everything
else. Nothing in the set is worth printing at scale before that is known.

Then the deck, then a starter plate. The fence and the deck are their own
prints -- at 130 and 184mm they do not lay out usefully beside a 16mm stop.

The preview is three views: the part as it comes off the plate, a plan of the
whole set holding a pedal enclosure so it is obvious what plugs in where, and a
section through a stop sitting in its hole, which is where all the fits are.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as CirclePatch
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build123d import Part, export_stl  # noqa: E402

from geom import dog_grid  # noqa: E402
from p2s import inventory, profiles  # noqa: E402
from parts import bench_dogs, dog_deck  # noqa: E402
from parts.bench_dogs import BUILDERS  # noqa: E402
from tools.render import INK, iso, mesh, plate  # noqa: E402

STEEL = "#8d949c"
WORK = "#3c6ea5"
DECK = "#b9bfc6"
HOT = "#c8792b"

EXAMPLE = (112.0, 60.0)
"""A 1590B pedal enclosure, as the workpiece in the plan view. Nothing in the
set is cut for it -- it is there for scale, because "112 x 60" means less than a
picture of one sitting between a fence and a clamp."""


def starter(params: bench_dogs.Params) -> list[Part]:
    """The plate to print once the fit is known: enough to hold something.

    Four stops is two reference points and two to push against; one clamp and
    its pad; four backers because they are consumables and the first one gets
    drilled through on day one.
    """
    return (
        [bench_dogs.stop(params) for _ in range(4)]
        + [bench_dogs.clamp(params), bench_dogs.pad(params)]
        + [bench_dogs.puck(params) for _ in range(4)]
    )


def _plan_view(ax, params: bench_dogs.Params, deck: dog_deck.Params) -> None:
    """The set in use, drawn from the numbers rather than from a mesh.

    Fence along one row, work against it, clamp pushing from the other side,
    spindle on the centre station. Everything here is where the arithmetic says
    it is, which is the point of the grid.
    """
    ax.add_patch(
        Rectangle(
            (-deck.width / 2, -deck.depth / 2), deck.width, deck.depth,
            facecolor=DECK, alpha=0.25, edgecolor=DECK, lw=1.2,
        )
    )
    for x, y in deck.stations:
        ax.add_patch(
            CirclePatch((x, y), params.grid.hole / 2, facecolor="white",
                        edgecolor=DECK, lw=1.0)
        )

    pitch = params.grid.pitch
    w, d = EXAMPLE
    cols = sorted({x for x, _ in deck.stations})
    rows = sorted({y for _, y in deck.stations})

    # Fence on the back row, which is where the deepest workpiece needs it.
    fence_y = rows[-1]
    face = fence_y - params.fence_reach
    ax.add_patch(
        Rectangle(
            (-params.fence_length / 2, fence_y - params.fence_thickness / 2),
            params.fence_length, params.fence_thickness,
            facecolor=HOT, alpha=0.30, edgecolor=HOT, lw=1.4,
        )
    )
    reach = params.grid.span(params.fence_span + 1) / 2
    for x in (-reach, reach):
        ax.add_patch(CirclePatch((x, fence_y), params.shank / 2,
                                 facecolor=HOT, edgecolor="none", alpha=0.8))

    # Two stops in one column fix the work sideways. The work's position
    # follows from the stations they sit in, not the other way round: that is
    # what it means for the grid to be the coordinate system.
    stop_x = cols[1]
    left = stop_x + params.reach
    for y in (face - pitch, face - 2 * pitch):
        ax.add_patch(CirclePatch((stop_x, y), params.head / 2,
                                 facecolor=HOT, alpha=0.55, edgecolor=HOT, lw=1.2))

    ax.add_patch(
        Rectangle((left, face - d), w, d, facecolor=WORK, alpha=0.28,
                  edgecolor=WORK, lw=1.6)
    )

    # Clamp on the nearest row that clears the work, bolt bridging the rest.
    clamp_y = max(y for y in rows if y + params.clamp_depth / 2 <= face - d)
    front = clamp_y + params.clamp_depth / 2
    # Its two shanks are a pitch apart, so the body straddles a pair of columns
    # rather than centring on one. Pick the pair nearest the middle of the work.
    middle = left + w / 2
    cx = min(((a + b) / 2 for a, b in zip(cols, cols[1:])), key=lambda m: abs(m - middle))
    ax.add_patch(
        Rectangle((cx - params.clamp_length / 2, clamp_y - params.clamp_depth / 2),
                  params.clamp_length, params.clamp_depth,
                  facecolor=HOT, alpha=0.30, edgecolor=HOT, lw=1.4)
    )
    for x in (cx - pitch / 2, cx + pitch / 2):
        ax.add_patch(CirclePatch((x, clamp_y), params.shank / 2,
                                 facecolor=HOT, edgecolor="none", alpha=0.8))
    gap = (face - d) - front
    ax.plot([cx, cx], [front, front + gap], color=STEEL, lw=3.5, solid_capstyle="butt")
    ax.add_patch(CirclePatch((cx, front + gap - params.pad_thickness / 2),
                             params.pad / 2, facecolor=STEEL, alpha=0.7,
                             edgecolor="none"))
    ax.annotate(
        f"M6 bridges the {gap:.0f} mm the grid left over\n"
        f"(it closes {params.reach_window[0]:.0f}-{params.reach_window[1]:.0f}, "
        f"and the pitch is {pitch:.1f})",
        xy=(cx, front + gap / 2), xytext=(deck.width / 2 - 4, clamp_y - 6),
        fontsize=8.5, color=INK, ha="right", va="top",
        arrowprops=dict(arrowstyle="->", color=INK, lw=0.9,
                        connectionstyle="arc3,rad=0.3"),
    )

    # The spindle, on the centre station.
    ax.plot([0], [0], marker="+", color=INK, ms=14, mew=1.6)
    ax.annotate(
        "spindle on the centre station:\nthe datum, and where the backer goes",
        xy=(0, 0), xytext=(-deck.width / 2 + 4, -deck.depth / 2 + 4),
        fontsize=8.5, color=INK, ha="left", va="bottom",
        arrowprops=dict(arrowstyle="->", color=INK, lw=0.9,
                        connectionstyle="arc3,rad=-0.25"),
    )

    ax.set_xlim(-deck.width / 2 - 6, deck.width / 2 + 6)
    ax.set_ylim(-deck.depth / 2 - 6, deck.depth / 2 + 6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"in use · {deck.cols} × {deck.rows} stations on a {pitch:.1f} mm pitch, "
        f"holding a {w:.0f} × {d:.0f} mm enclosure "
        f"(up to {bench_dogs.capacity(params, deck.rows):.0f} mm deep)",
        color=INK, fontsize=11,
    )


def _fit_view(ax, params: bench_dogs.Params, deck: dog_deck.Params) -> None:
    """A stop in its hole, the way up it is used: where every fit lives."""
    t = deck.deck
    hole, c = params.grid.hole, deck.hole_chamfer
    span = params.grid.pitch * 1.6

    # Deck in section, hole and chamfer cut out of it.
    for sign in (-1, 1):
        edge = sign * hole / 2
        ax.add_patch(
            MplPolygon(
                [(sign * span, 0), (edge, 0), (edge, t - c),
                 (sign * (hole / 2 + c), t), (sign * span, t)],
                closed=True, facecolor=DECK, alpha=0.45, edgecolor=DECK, lw=1.2,
            )
        )

    # The dog, turned over from how it prints: head down on the deck, shank up
    # into it, tip stopping short of the underside.
    rh, rs = params.head / 2, params.shank / 2
    lead, brk = params.lead, params.break_edge
    top = t + params.rise
    right = [
        (0, t - params.shank_length),
        (rs - lead, t - params.shank_length),
        (rs, t - params.shank_length + lead),
        (rs, t - params.root),
        (rs + params.root, t),
        (rh, t),
        (rh, top - brk),
        (rh - brk, top),
        (0, top),
    ]
    outline = [(-x, y) for x, y in reversed(right)] + right
    ax.add_patch(MplPolygon(outline, closed=True, facecolor=HOT, alpha=0.55,
                            edgecolor=HOT, lw=1.6))

    ax.annotate(
        f"{params.relief:.1f} mm of relief: the shank never reaches\n"
        f"the underside, so it cannot stand the deck off the casting",
        xy=(0, t - params.shank_length),
        xytext=(-span, -4), fontsize=8.5, color=INK, ha="left", va="top",
        arrowprops=dict(arrowstyle="->", color=INK, lw=0.9,
                        connectionstyle="arc3,rad=0.2"),
    )
    ax.annotate(
        f"{params.root:.1f} mm root cone, riding in\nthe hole's own chamfer",
        xy=(rs + params.root / 2, t - params.root / 2),
        xytext=(-span, top + 1), fontsize=8.5, color=INK, ha="left", va="bottom",
        arrowprops=dict(arrowstyle="->", color=INK, lw=0.9,
                        connectionstyle="arc3,rad=0.2"),
    )
    ax.annotate(
        f"reach {params.reach:.1f} mm, in every direction",
        xy=(rh, t + params.rise / 2), xytext=(span, top + 13),
        fontsize=8.5, color=INK, ha="right", va="bottom",
        arrowprops=dict(arrowstyle="->", color=INK, lw=0.9,
                        connectionstyle="arc3,rad=-0.25"),
    )
    ax.plot([-span, span], [t, t], color=INK, lw=0.8, ls=(0, (5, 4)))

    ax.set_xlim(-span, span)
    ax.set_ylim(-12, top + 22)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"a stop in its hole · {params.fit:.2f} mm clearance on "
        f"{params.grid.hole:.2f} mm",
        color=INK, fontsize=11,
    )


def preview(params: bench_dogs.Params, deck: dog_deck.Params, part: Part,
            label: str, dest: Path) -> Path:
    points, tris = mesh(part, tol=0.05)
    bb = part.bounding_box()

    fig = plt.figure(figsize=(14.5, 5.0), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 1.0], wspace=0.02,
                          left=0.01, right=0.99, top=0.90, bottom=0.04)
    iso(fig.add_subplot(gs[0, 0], projection="3d"), points, tris,
        f"{label} as printed · {bb.size.X:.0f} × {bb.size.Y:.0f} × "
        f"{bb.size.Z:.0f} mm")
    _plan_view(fig.add_subplot(gs[0, 1]), params, deck)
    _fit_view(fig.add_subplot(gs[0, 2]), params, deck)

    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", default="stop",
                    choices=[*BUILDERS, "deck", "ladder"],
                    help="what to build")
    ap.add_argument("--set", action="store_true",
                    help="a starter plate instead of one part")
    ap.add_argument("--grid", default="half-inch",
                    help=f"hole pattern: {', '.join(dog_grid.CATALOGUE)}")
    ap.add_argument("--fit", default="slip",
                    help=f"shank clearance: {', '.join(dog_grid.FITS)}")
    ap.add_argument("--span", type=int, default=bench_dogs.Params.fence_span,
                    help="pitches between the fence's shanks")
    ap.add_argument("--cols", type=int, default=dog_deck.Params.cols)
    ap.add_argument("--rows", type=int, default=dog_deck.Params.rows)
    ap.add_argument("--copies", type=int, default=1)
    ap.add_argument("--material", default="ASA",
                    help="what to slice it for; not PLA, which is soft by 55C")
    ap.add_argument("--nozzle", type=float, default=profiles.DEFAULT_NOZZLE)
    ap.add_argument("-o", "--out", default=None,
                    help="output basename; .stl, .3mf and .png are written")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args(argv)

    grid = dog_grid.named(args.grid)
    params = bench_dogs.Params(
        grid=grid, fit=dog_grid.fit(args.fit), fence_span=args.span
    )
    deck = dog_deck.Params(grid=grid, cols=args.cols, rows=args.rows)

    if args.set:
        label, built = "starter set", starter(params) * args.copies
    elif args.part == "ladder":
        label, built = "fit ladder", bench_dogs.ladder(params) * args.copies
    elif args.part == "deck":
        label, built = "deck", [dog_deck.build(deck)] * args.copies
    else:
        label, built = args.part, [BUILDERS[args.part](params)] * args.copies

    part = built[0] if len(built) == 1 else plate(built, nozzle=args.nozzle)

    stem = args.out or f"out/bench_{'set' if args.set else args.part}"
    base = Path(stem)
    base.parent.mkdir(parents=True, exist_ok=True)
    export_stl(part, str(base.with_suffix(".stl")))
    print(f"wrote {base.with_suffix('.stl')}")

    # As elsewhere: the STL is already written and the 3mf needs Bambu Studio's
    # profiles on disk, so a missing slicer is a note and not a failure.
    try:
        from p2s.slicer import write_project

        filament = inventory.default(args.material).preset_for(args.nozzle)
        written = write_project(part, base.with_suffix(".3mf"),
                                nozzle=args.nozzle, filament=filament)
        print(f"wrote {written} [{filament}]")
    except Exception as exc:  # noqa: BLE001 - profiles missing is not fatal here
        print(f"no 3mf: {exc}")

    if not args.no_preview:
        print(f"wrote {preview(params, deck, part, label, base.with_suffix('.png'))}")

    if args.part == "ladder" and not args.set:
        print("fits, loosest first: " + ", ".join(
            f"{n} {dog_grid.FITS[n]:.2f}mm" for n in dog_grid.LADDER))
    bb = part.bounding_box()
    print(f"{label}: {len(built)} part{'s' if len(built) > 1 else ''}, "
          f"valid={part.is_valid}, {bb.size.X:.1f} × {bb.size.Y:.1f} × "
          f"{bb.size.Z:.1f} mm, {part.volume / 1000:.1f} cm^3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
