"""Make the hole saw case from the command line, and look at it without a slicer.

    python -m tools.hole_saw_case
    python -m tools.hole_saw_case --nest 4      # two stacks of four, 5mm lower
    python -m tools.hole_saw_case --material ASA --pin steel

The set in ``geom.hole_saws`` was measured with calipers. For another set of
the same kind, ``--saw-height``, ``--rise`` and ``--mandrel`` take its numbers
straight from the set in hand. ``--clearance`` is the next thing to touch if a
saw drops in stiffly.

Two halves, one plate each: the tray prints floor down and the lid prints on
its face, so its lip and hinge knuckles point up. ASA for a case that rides in
a vehicle, where PLA would soften and creep in a hot cab; PLA if it lives in a
drawer. After printing, cut a piece of 1.75mm filament to the length the
command prints, straighten it, and push it through the knuckles from one end;
it grips in the tray's and turns in the lid's. ``--pin steel`` bores the
knuckles for 2mm steel rod instead -- 5/64in music wire or a drill blank, cut
to length, deburred, and tapped home -- and ``--pin 2.4`` for any other rod
you have, with the steel fits.

The whole set nests, so by default it goes in as one stack in one pocket;
``--nest`` splits it into shorter stacks, down to 1 for every saw in its own
pocket, trading floor for height.

``--coupon`` prints three knuckles of the hinge, cut from the real tray and
lid, for checking the pin's fits in a quarter of an hour before committing to
the case: the pin should need pressing into the outer two and turn freely in
the middle one. If it is loose in the outer two, or tight in the middle,
change ``Pin.press`` or ``Pin.play`` to suit your printer and print the
coupon again.

The preview is three views: the case open with the lid swung back, the layout
that is actually the design, and the hinge in section with the lid drawn at
each stage of its swing.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle as CirclePatch
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import Polygon as MplPolygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build123d import Part, export_stl  # noqa: E402

from geom import hole_saws  # noqa: E402
from p2s import inventory, profiles  # noqa: E402
from parts.hole_saw_case import (  # noqa: E402
    PINS,
    STEEL_PIN,
    Params,
    Pin,
    coupon,
    coupon_for_print,
    coupon_pin_length,
    hinge_axis,
    layout,
    lid,
    lid_for_print,
    opened,
    pin_length,
    tray,
)
from tools.render import ASA, INK, iso, mesh, section  # noqa: E402

STEEL = "#8a6d3b"
MANDREL = "#3b6ea5"


def _fine(points, tris, longest: float = 6.0):
    """Split every triangle until no edge is longer than ``longest``.

    A tray is big flat faces with holes in them, and OCCT tessellates those as
    slivers the width of the case. Matplotlib sorts each triangle by its centre,
    so a sliver sorts wrong against everything it spans and paints over it.
    Short triangles sort right.
    """
    tri = points[tris]
    out = []
    while len(tri):
        edge = np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2).max(axis=1)
        done, tri = tri[edge <= longest], tri[edge > longest]
        out.append(done)
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        tri = np.concatenate([np.stack(t, axis=1) for t in
                              ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])
    tri = np.concatenate(out)
    return tri.reshape(-1, 3), np.arange(len(tri) * 3).reshape(-1, 3)


def _outline(ax, geom, **kw) -> None:
    for poly in getattr(geom, "geoms", [geom]):
        ax.add_patch(MplPolygon(list(poly.exterior.coords), closed=True, **kw))


def _layout_view(ax, params: Params) -> None:
    """The plan: every pocket, the cradle and the key's slot, and the case
    round them."""
    lay = layout(params)
    w, g = params.wall, params.corner
    ax.add_patch(FancyBboxPatch((-w + g, -w + g), lay.width + 2 * w - 2 * g,
                                lay.depth + 2 * w - 2 * g,
                                boxstyle=f"round,pad={g}", facecolor=ASA,
                                edgecolor=INK, lw=1.2, alpha=0.35, zorder=1))
    _outline(ax, lay.reserved, facecolor="white", edgecolor=MANDREL, lw=1.1, zorder=2)
    for p in lay.stacks:
        ax.add_patch(CirclePatch((p.x, p.y), params.pocket(p.stack) / 2,
                                 facecolor="white", edgecolor=INK, lw=1.0, zorder=2))
        # Every saw in the stack, one inside the next, as they sit.
        for saw in p.stack.saws:
            ax.add_patch(CirclePatch((p.x, p.y), saw.diameter / 2, facecolor=STEEL,
                                     alpha=0.12, edgecolor=STEEL, lw=0.8, zorder=3))
        ax.text(p.x, p.y, p.stack.label, ha="center", va="center", fontsize=8.5,
                color=INK, zorder=4)
    x, y = lay.mandrel
    ax.text(x + 6, y, "mandrel", fontsize=8, color=MANDREL, va="center", zorder=4)
    (a, b, c, d), _ = lay.key_bars
    ax.text((a + c) / 2, (b + d) / 2, "key", fontsize=8, color=MANDREL,
            ha="center", va="center", zorder=4,
            rotation=90 if d - b > c - a else 0)
    ax.annotate(
        f"{lay.width + 2 * w:.0f} × {lay.depth + 2 * w:.0f} mm, "
        f"{params.height:.0f} mm tall closed",
        xy=(lay.width / 2, -w - 4), ha="center", va="top", fontsize=9, color=INK,
    )
    ax.set_xlim(-w - 4, lay.width + w + 4)
    ax.set_ylim(-w - 14, lay.depth + w + 14)
    ax.set_aspect("equal")
    ax.axis("off")
    nested = max(len(p.stack.saws) for p in lay.stacks)
    ax.set_title(f"the layout: {len(params.kit.saws)} saws, nested {nested} deep",
                 color=INK, fontsize=11)


def _hinge_view(ax, params: Params, bottom: Part, top: Part) -> None:
    """The back of the case in section, with the lid at every stage of its
    swing -- which is what the lipless back edge is for."""
    lay = layout(params)
    ya, za = hinge_axis(params)
    xs = [(a + b) / 2 for a, b, own in _knuckles(params)]
    cut_tray, cut_lid = xs[0], xs[1]
    pts, tris = mesh(bottom, tol=0.05)
    segs = section(pts, tris, 0, cut_tray)
    ax.add_collection(LineCollection(segs[:, :, 1:], colors=[INK], lw=1.4, zorder=3))
    pts, tris = mesh(top, tol=0.05)
    for degrees in (0, 45, 90, 135, 180):
        ghost = opened(params, degrees, top) if degrees else top
        gp, gt = mesh(ghost, tol=0.05)
        segs = section(gp, gt, 0, cut_lid)
        ax.add_collection(LineCollection(
            segs[:, :, 1:], colors=[MANDREL], lw=1.3 if degrees in (0, 180) else 0.7,
            alpha=1 if degrees in (0, 180) else 0.55, zorder=4))
    ax.plot([ya], [za], "o", color=INK, ms=3, zorder=5)
    ax.annotate(f"{params.pin.name} pin", xy=(ya, za), xytext=(ya + 8, za - 16),
                fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax.set_xlim(lay.depth - 40, ya + 45)
    ax.set_ylim(-4, params.height + 44)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("the hinge: tray in black, lid through 180°", color=INK, fontsize=11)


def _knuckles(params: Params):
    from parts.hole_saw_case import _hinge_xs

    return _hinge_xs(params)


def preview(params: Params, dest: Path, bottom: Part | None = None,
            top: Part | None = None) -> Path:
    bottom = tray(params) if bottom is None else bottom
    top = lid(params) if top is None else top
    fig = plt.figure(figsize=(16, 6.2), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.25, 0.8], wspace=0.03,
                          left=0.01, right=0.99, top=0.92, bottom=0.02)
    shown = bottom + opened(params, 105, top)
    points, tris = _fine(*mesh(shown, tol=0.08))
    iso(fig.add_subplot(gs[0, 0], projection="3d"), points, tris,
        f"open · {params.kit.pieces} pieces", elev=38, azim=-62)
    _layout_view(fig.add_subplot(gs[0, 1]), params)
    _hinge_view(fig.add_subplot(gs[0, 2]), params, bottom, top)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest


def coupon_preview(params: Params, dest: Path) -> Path:
    """The coupon as it comes off the plate, and put together, half open."""
    bottom, top = coupon(params)
    fig = plt.figure(figsize=(10, 5), facecolor="white")
    gs = fig.add_gridspec(1, 2, wspace=0.02, left=0.01, right=0.99, top=0.92,
                          bottom=0.02)
    points, tris = _fine(*mesh(coupon_for_print(params), tol=0.03), longest=3.0)
    iso(fig.add_subplot(gs[0, 0], projection="3d"), points, tris,
        "on the plate", elev=30, azim=-50)
    points, tris = _fine(*mesh(bottom + opened(params, 90, top), tol=0.03), longest=3.0)
    iso(fig.add_subplot(gs[0, 1], projection="3d"), points, tris,
        f"pinned, open 90° · {coupon_pin_length(params):.0f} mm of {params.pin.name}",
        elev=22, azim=-35)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest


def _mandrel(spec: str) -> tuple[hole_saws.Segment, ...]:
    """'36x11.5,16x32,...': length x diameter, chuck end first."""
    names = [s.name for s in hole_saws.WARRIOR_MANDREL]
    out = []
    for i, part in enumerate(spec.split(",")):
        length, diameter = (float(v) for v in part.lower().split("x"))
        out.append(hole_saws.Segment(names[i] if i < len(names) else f"part {i}",
                                     length, diameter))
    return tuple(out)


def _pin(spec: str) -> Pin:
    """'filament', 'steel', or a steel rod's diameter in mm."""
    if spec in PINS:
        return PINS[spec]
    try:
        diameter = float(spec)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{spec!r}: {', '.join(PINS)} or a rod's diameter in mm") from None
    return dataclasses.replace(STEEL_PIN, name=f"{diameter:g}mm rod", diameter=diameter)


def params_from(args: argparse.Namespace) -> Params:
    kit = hole_saws.named(args.kit)
    if args.saw_height is not None:
        kit = dataclasses.replace(kit, saws=tuple(
            dataclasses.replace(s, height=args.saw_height) for s in kit.saws))
    if args.mandrel:
        kit = dataclasses.replace(kit, mandrel=_mandrel(args.mandrel))
    if args.rise is not None:
        kit = dataclasses.replace(kit, rise=args.rise)
    return Params(kit=kit, nest=args.nest, clearance=args.clearance, web=args.web,
                  wall=args.wall, snap=args.snap, label=args.label, pin=args.pin)


def _write(part: Part, base: Path, args) -> None:
    export_stl(part, str(base.with_suffix(".stl")))
    print(f"wrote {base.with_suffix('.stl')}")
    # The 3mf needs Bambu Studio's profiles on disk, and the STL is already
    # written, so a missing slicer is a note and not a failure.
    try:
        from p2s.slicer import write_project

        filament = inventory.default(args.material).preset_for(args.nozzle)
        print(f"wrote {write_project(part, base.with_suffix('.3mf'), nozzle=args.nozzle, filament=filament)}"
              f" [{filament}]")
    except Exception as exc:  # noqa: BLE001 - profiles missing is not fatal here
        print(f"no 3mf: {exc}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kit", default="warrior-57523",
                    help=f"which set: {', '.join(hole_saws.CATALOGUE)}")
    ap.add_argument("--saw-height", type=float, default=None,
                    help="hub to tooth tips, in mm, for a set other than the catalogue's")
    ap.add_argument("--mandrel", default=None,
                    help="the mandrel as LENGTHxDIAMETER segments, chuck end first, "
                         "hexes across the corners")
    ap.add_argument("--nest", type=int, default=Params.nest,
                    help="most saws nested in one stack; 1 lays them all out flat")
    ap.add_argument("--rise", type=float, default=None,
                    help="how much prouder each nested saw stands than the one it "
                         "sits in, in mm")
    ap.add_argument("--clearance", type=float, default=Params.clearance,
                    help="added to each saw's diameter for its pocket")
    ap.add_argument("--web", type=float, default=Params.web,
                    help="least plastic between pockets")
    ap.add_argument("--wall", type=float, default=Params.wall)
    ap.add_argument("--snap", type=float, default=Params.snap,
                    help="how far the lid's bead reaches into the front wall")
    ap.add_argument("--pin", type=_pin, default=Params.pin,
                    help="the hinge pin: filament (1.75mm), steel (2mm rod), or "
                         "any rod's diameter in mm")
    ap.add_argument("--label", type=float, default=Params.label,
                    help="depth sizes are engraved into the pocket floors; 0 for none")
    ap.add_argument("--material", default="PLA")
    ap.add_argument("--nozzle", type=float, default=profiles.DEFAULT_NOZZLE)
    ap.add_argument("-o", "--out", default="out/hole_saw_case",
                    help="output basename; _tray and _lid .stl/.3mf and a .png")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--coupon", action="store_true",
                    help="print only three knuckles of the hinge, to check the pin "
                         "fits before printing the case")
    args = ap.parse_args(argv)

    params = params_from(args)
    if args.coupon:
        base = Path(args.out)
        base = base.with_name(base.name + "_coupon")
        base.parent.mkdir(parents=True, exist_ok=True)
        _write(coupon_for_print(params), base, args)
        if not args.no_preview:
            print(f"wrote {coupon_preview(params, base.with_suffix('.png'))}")
        print(f"coupon pin: {coupon_pin_length(params):.0f} mm of {params.pin.name}; "
              f"it should press into the outer two knuckles and turn in the middle one")
        return 0
    bottom, top = tray(params), lid(params)
    base = Path(args.out)
    base.parent.mkdir(parents=True, exist_ok=True)
    _write(bottom, base.with_name(base.name + "_tray"), args)
    _write(lid_for_print(params, top), base.with_name(base.name + "_lid"), args)
    if not args.no_preview:
        print(f"wrote {preview(params, base.with_suffix('.png'), bottom, top)}")

    lay = layout(params)
    print(f"{lay.width + 2 * params.wall:.0f} × {lay.depth + 2 * params.wall:.0f} × "
          f"{params.height:.1f} mm closed, {params.kit.pieces} pieces")
    for p in lay.stacks:
        print(f"  {p.stack.label:>10}: {len(p.stack.saws)} nested, "
              f"{p.stack.height:.1f} mm tall, {params.pocket(p.stack):.1f} mm pocket")
    print(f"hinge pin: {pin_length(params):.0f} mm of {params.pin.name}")
    for name, part in (("tray", bottom), ("lid", top)):
        print(f"{name}: valid={part.is_valid}, {part.volume / 1000:.0f} cm^3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
