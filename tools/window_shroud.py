"""Make a window shroud from the command line, and look at it without a slicer.

    python -m tools.window_shroud
    python -m tools.window_shroud --tilt 15 --fit 0.4
    python -m tools.window_shroud --coupon -o out/shroud_fit

Every estimated camera dimension in geom.cameras has a flag here, so a set of
caliper readings goes straight in without editing anything. Print ``--coupon``
first: it is the sleeve and a sliver of the lip, and it answers whether the
head fits in ten minutes rather than an hour.

The preview is three views: the part as printed; a section up through the lens
with the tilted head, the top and bottom of the picture and an IR LED's axis
drawn in; and the glass face with the picture's footprint and the face's
sensors marked, so it is obvious which chamber each one looks out through.
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
    """Through the lens, bottom to top of the window: the wedge, the tilted
    head, the partition, and the top and bottom of the picture."""
    lx, _ = params.lens2d
    segs = section(points, tris, 0, lx)
    ax.add_collection(LineCollection(segs[:, :, [1, 2]], colors=[INK], lw=1.6))

    c = params.camera
    ax.axhline(0, color=GLASS, lw=2.2)
    lo = params.footprint.bounds[1] - params.wall - params.flange
    ax.text(lo, -0.8, "glass, on the plate", color=GLASS, fontsize=8.5,
            ha="left", va="top")
    # The head, as a ghost, sat on the lip at the tilt.
    corners = params.on_face([[0, -c.head_h / 2], [0, c.head_h / 2]])
    back = corners + params.normal * (params.sleeve + 6)
    quad = np.array([corners[0], corners[1], back[1], back[0]])[:, 1:]
    ax.add_patch(plt.Polygon(quad, facecolor="#dddddd", edgecolor="none", zorder=0))
    # Top and bottom of the picture, at the margin.
    d = params.to_world(c.direction(np.array([0.0, 0.0]), np.array([1.0, -1.0]),
                                    params.margin))
    o = params.pupil
    for ray in d:
        t = o[2] / -ray[2]
        ax.plot([o[1], o[1] + ray[1] * t], [o[2], 0], color=LIGHT, lw=1.1)
    # The IR LEDs' axes.
    for name, x, y, _ in params.features:
        if "IR LED" in name:
            q = params.on_face([[x, y]])[0]
            t = q[2] / -params.forward[2]
            ax.plot([q[1], q[1] + params.forward[1] * t], [q[2], 0], color="#b03a2e",
                    lw=0.9, ls="--")
            ax.text(q[1] + 1.5, q[2] + 1, "IR LED axis", color="#b03a2e",
                    fontsize=7.5, ha="left", va="bottom")
            break
    ax.text(lo + 30, -5, "← room floor", color=INK, fontsize=7.5,
            ha="left", va="bottom")
    ax.text(o[1] - 3, o[2] * 0.5, f"{c.fov_v + 2 * params.margin:.0f}°",
            color=LIGHT, fontsize=8.5, ha="center", va="center")

    bb_y = (lo - 2, quad[:, 0].max() + 2)
    ax.set_xlim(*bb_y)
    ax.set_ylim(-4, quad[:, 1].max() + 2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"section · tilted {params.tilt:.0f}° down · lens "
                 f"{params.pupil_height:.1f} mm off the glass",
                 color=INK, fontsize=11)


def _plan_view(ax, points, tris, params: Params) -> None:
    """The glass face, as seen from outside: what touches the pane, and what
    looks out through it."""
    cut = section(points, tris, 2, 0.3)
    # Seen from outside, the camera's right is on the left, so flip x.
    if len(cut):
        ax.add_collection(LineCollection(cut[:, :, :2] * np.array([-1, 1]),
                                         colors=[INK], lw=1.2))
    fp = np.asarray(params.footprint.exterior.coords)
    ax.fill(-fp[:, 0], fp[:, 1], color=LIGHT, alpha=0.25, lw=0)
    ax.text(0, params.footprint.bounds[1] + 5, "picture", color=LIGHT, fontsize=8,
            ha="center", va="center")
    for name, x, y, dia in params.features:
        q = params.on_face([[x, y]])[0]
        ax.add_patch(CirclePatch((-q[0], q[1]), dia / 2, color=GLASS))
        left = -q[0] < -1 or name == "status light"
        ax.text(-q[0] + (-2.5 if left else 2.5), q[1], name, color=GLASS,
                fontsize=7, va="center", ha="right" if left else "left")

    x0, y0, x1, y1 = params.footprint.buffer(params.wall + params.flange + 2).bounds
    _, oh = params.outer_size
    y1 = max(y1, params.on_face([[0, oh / 2]])[0][1] + params.flange + 2)
    ax.set_xlim(-x1, -x0)
    ax.set_ylim(y0, y1)
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


CAMERA_FLAGS = ("head_w", "head_h", "head_n", "head_depth", "panel_w",
                "panel_h", "panel_n", "lens_x", "lens_y", "bezel_d",
                "bezel_proud", "pupil_depth")


def params_from(args: argparse.Namespace) -> Params:
    camera = cameras.named(args.camera)
    measured = {k: getattr(args, k) for k in CAMERA_FLAGS if getattr(args, k) is not None}
    camera = replace(camera, **measured)
    return Params(
        camera=camera,
        tilt=args.tilt,
        standoff=args.standoff,
        margin=args.margin,
        wall=args.wall,
        fit=args.fit,
        sleeve=args.sleeve,
        hood_clearance=args.hood_clearance,
        partition_drop=args.partition_drop,
        flange=args.flange,
        stalk_notch=args.stalk_notch,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--camera", default="c100",
                    help=f"which head: {', '.join(cameras.CATALOGUE)}")
    ap.add_argument("--tilt", type=float, default=Params.tilt,
                    help="degrees the camera looks down through the glass")
    ap.add_argument("--standoff", type=float, default=Params.standoff,
                    help="face centre to glass, mm; leave out for as close as "
                         "the head allows")
    ap.add_argument("--margin", type=float, default=Params.margin,
                    help="degrees added round the field of view")
    ap.add_argument("--wall", type=float, default=Params.wall)
    ap.add_argument("--fit", type=float, default=Params.fit,
                    help="gap each side between head and sleeve")
    ap.add_argument("--sleeve", type=float, default=Params.sleeve,
                    help="how far back over the head the sleeve reaches")
    ap.add_argument("--hood-clearance", type=float, default=Params.hood_clearance)
    ap.add_argument("--partition-drop", type=float, default=Params.partition_drop,
                    help="how far below the lens the partition meets the face")
    ap.add_argument("--flange", type=float, default=Params.flange,
                    help="adhesive flange width; 0 for none")
    ap.add_argument("--stalk-notch", type=float, default=Params.stalk_notch,
                    help="width of the slot for the stand's stalk; 0 for none")
    for name in CAMERA_FLAGS:
        ap.add_argument(f"--{name.replace('_', '-')}", type=float, default=None,
                        help=(f"squareness of the {name.split('_')[0]} outline, "
                              f"as a superellipse exponent" if name.endswith("_n")
                              else f"measured {name.replace('_', ' ')}, mm"))
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

    x0, y0, x1, y1 = params.footprint.bounds
    bb = part.bounding_box()
    print(f"{params.camera.name} tilted {params.tilt:.0f}°: lens "
          f"{params.pupil_height:.1f} mm off the glass, picture spans "
          f"{x1 - x0:.1f} × {y1 - y0:.1f} mm of it")
    print(f"valid={part.is_valid} {bb.size.X:.1f} × {bb.size.Y:.1f} × "
          f"{bb.size.Z:.1f} mm, {part.volume / 1000:.2f} cm^3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
