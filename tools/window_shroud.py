"""Make a window shroud from the command line, and look at it without a slicer.

    python -m tools.window_shroud
    python -m tools.window_shroud --standoff 6 --fit 0.4
    python -m tools.window_shroud --coupon -o out/shroud_fit

Every estimated camera dimension in geom.cameras has a flag here, so a set of
caliper readings goes straight in without editing anything. Print ``--coupon``
first: it is the sleeve and a sliver of the lip, and it answers whether the
head fits in ten minutes rather than an hour.

The preview is three views: the part as printed, a section through the lens
with the picture's edge rays drawn in, and the glass face with the face's
sensors marked, so it is obvious which parts of the face see out past the hood.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle as CirclePatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build123d import Part, export_stl  # noqa: E402

from geom import cameras  # noqa: E402
from p2s import inventory, profiles  # noqa: E402
from parts.window_shroud import Params, build, coupon  # noqa: E402
from tools.render import INK, iso, mesh, section  # noqa: E402

LIGHT = "#c8792b"
GLASS = "#5b8fb9"


def _section_view(ax, points, tris, params: Params) -> None:
    """Through the lens, left to right: the hood, the bezel, the picture."""
    lx, ly = params.lens
    segs = section(points, tris, 1, ly)
    ax.add_collection(LineCollection(segs[:, :, [0, 2]], colors=[INK], lw=1.6))

    c = params.camera
    ax.axhline(0, color=GLASS, lw=2.2)
    ax.text(params.outer_size[0] / 2 + params.flange, -0.6, "glass, on the plate",
            color=GLASS, fontsize=8.5, ha="right", va="top")
    # The head, as a ghost: where it sits on the lip, and its bezel.
    w = c.head_w
    ax.add_patch(plt.Rectangle((-w / 2, params.standoff), w,
                               params.sleeve + 4, facecolor="#dddddd",
                               edgecolor="none", zorder=0))
    ax.add_patch(plt.Rectangle((lx - c.bezel_d / 2, params.standoff - c.bezel_proud),
                               c.bezel_d, c.bezel_proud, facecolor="#bbbbbb",
                               edgecolor="none", zorder=0))
    # The picture's widest rays in this plane: horizontal, at the margin.
    d = c.direction(np.array([-1.0, 1.0]), np.array([0.0, 0.0]), params.margin)
    z = params.pupil_z
    for ray in d:
        t = z / -ray[2]
        ax.plot([lx, lx + ray[0] * t], [z, 0], color=LIGHT, lw=1.1)
    ax.text(lx, z * 0.45, f"{c.fov_h + 2 * params.margin:.0f}°", color=LIGHT,
            fontsize=8.5, ha="center", va="center")

    half = params.outer_size[0] / 2 + params.flange
    ax.set_xlim(-half - 2, half + 2)
    ax.set_ylim(-3, params.height + 6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"section · lens {params.pupil_z:.1f} mm off the glass",
                 color=INK, fontsize=11)


def _plan_view(ax, points, tris, params: Params) -> None:
    """The glass face, as seen from outside: what touches the pane, and what
    looks out through it."""
    cut = section(points, tris, 2, 0.5)
    # Seen from outside, the camera's right is on the left, so flip x.
    if len(cut):
        xy = cut[:, :, :2] * np.array([-1, 1])
        ax.add_collection(LineCollection(xy, colors=[INK], lw=1.2))
    rib_cut = section(points, tris, 2, params.standoff - params.rib_gap - 0.3)
    if len(rib_cut):
        ax.add_collection(LineCollection(rib_cut[:, :, :2] * np.array([-1, 1]),
                                         colors=[INK], lw=0.6, alpha=0.5))

    lx, ly = params.lens
    c = params.camera
    u, v = c.border(64)
    d = c.direction(u, v)
    t = params.pupil_z / -d[:, 2]
    ax.fill(-(lx + d[:, 0] * t), ly + d[:, 1] * t, color=LIGHT, alpha=0.25, lw=0)
    ax.text(-lx, ly, "picture", color=LIGHT, fontsize=8, ha="center", va="center")
    for name, x, y, dia in params.features:
        ax.add_patch(CirclePatch((-x, y), dia / 2, color=GLASS))
        ax.text(-x + 2, y, name, color=GLASS, fontsize=7.5, va="center")

    half = params.outer_size[0] / 2 + params.flange
    ax.set_xlim(-half - 2, half + 2)
    ax.set_ylim(-half - 2, half + 2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("glass face, from outside", color=INK, fontsize=11)


def preview(params: Params, dest: Path, part: Part | None = None) -> Path:
    part = build(params) if part is None else part
    points, tris = mesh(part, tol=0.05)

    fig = plt.figure(figsize=(13.5, 4.6), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.2, 1], wspace=0.02,
                          left=0.01, right=0.99, top=0.90, bottom=0.04)
    bb = part.bounding_box()
    iso(fig.add_subplot(gs[0, 0], projection="3d"), points, tris,
        f"as printed · {bb.size.X:.0f} × {bb.size.Y:.0f} × {bb.size.Z:.0f} mm",
        elev=38)
    _section_view(fig.add_subplot(gs[0, 1]), points, tris, params)
    _plan_view(fig.add_subplot(gs[0, 2]), points, tris, params)

    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest


CAMERA_FLAGS = ("head_w", "head_h", "head_corner", "head_depth", "panel_w",
                "panel_h", "panel_corner", "lens_x", "lens_y", "bezel_d",
                "bezel_proud", "pupil_depth")


def params_from(args: argparse.Namespace) -> Params:
    camera = cameras.named(args.camera)
    measured = {k: getattr(args, k) for k in CAMERA_FLAGS if getattr(args, k) is not None}
    camera = replace(camera, **measured)
    return Params(
        camera=camera,
        standoff=args.standoff,
        margin=args.margin,
        wall=args.wall,
        fit=args.fit,
        sleeve=args.sleeve,
        hood_clearance=args.hood_clearance,
        ribs=tuple(args.ribs),
        flange=args.flange,
        stalk_notch=args.stalk_notch,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--camera", default="c100",
                    help=f"which head: {', '.join(cameras.CATALOGUE)}")
    ap.add_argument("--standoff", type=float, default=Params.standoff,
                    help="face panel to glass, mm; smaller keeps the hood small")
    ap.add_argument("--margin", type=float, default=Params.margin,
                    help="degrees added round the field of view")
    ap.add_argument("--wall", type=float, default=Params.wall)
    ap.add_argument("--fit", type=float, default=Params.fit,
                    help="gap each side between head and sleeve")
    ap.add_argument("--sleeve", type=float, default=Params.sleeve,
                    help="how far back over the head the sleeve reaches")
    ap.add_argument("--hood-clearance", type=float, default=Params.hood_clearance)
    ap.add_argument("--ribs", type=float, nargs="+", default=list(Params().ribs),
                    help="rib angles in degrees, camera's right = 0, going up")
    ap.add_argument("--flange", type=float, default=Params.flange,
                    help="adhesive flange width; 0 for none")
    ap.add_argument("--stalk-notch", type=float, default=Params.stalk_notch,
                    help="width of the slot for the stand's stalk; 0 for none")
    for name in CAMERA_FLAGS:
        ap.add_argument(f"--{name.replace('_', '-')}", type=float, default=None,
                        help=f"measured {name.replace('_', ' ')}, mm")
    ap.add_argument("--coupon", action="store_true",
                    help="just the sleeve band, to check the fit")
    ap.add_argument("--material", default="ASA",
                    help="what to slice it for; not PLA, which softens in a "
                         "sunny window")
    ap.add_argument("--nozzle", type=float, default=profiles.DEFAULT_NOZZLE)
    ap.add_argument("-o", "--out", default=None,
                    help="output basename; .stl, .3mf and .png are written")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args(argv)

    params = params_from(args)
    part = coupon(params) if args.coupon else build(params)
    out = args.out or ("out/window_shroud_coupon" if args.coupon else "out/window_shroud")

    base = Path(out)
    base.parent.mkdir(parents=True, exist_ok=True)
    export_stl(part, str(base.with_suffix(".stl")))
    print(f"wrote {base.with_suffix('.stl')}")

    try:
        from p2s.slicer import write_project

        filament = inventory.default(args.material).preset_for(args.nozzle)
        print(f"wrote {write_project(part, base.with_suffix('.3mf'), nozzle=args.nozzle, filament=filament)}"
              f" [{filament}]")
    except Exception as exc:  # noqa: BLE001 - profiles missing is not fatal here
        print(f"no 3mf: {exc}")

    if not args.no_preview and not args.coupon:
        print(f"wrote {preview(params, base.with_suffix('.png'), part)}")

    hx, hy = params.hood_mouth
    bb = part.bounding_box()
    print(f"{params.camera.name}: lens {params.pupil_z:.1f} mm off the glass, "
          f"hood mouth {2 * hx:.1f} × {2 * hy:.1f} mm")
    print(f"valid={part.is_valid} {bb.size.X:.1f} × {bb.size.Y:.1f} × "
          f"{bb.size.Z:.1f} mm, {part.volume / 1000:.2f} cm^3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
