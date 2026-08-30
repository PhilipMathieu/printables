"""Preview and export the Lamborghini script badge.

Three panels, and the third is the one that earns its place: a zoom on an i-dot
and its stem. Everything else about this part is easy to judge from the face-on
view, but whether the dots are convincingly attached is a detail you cannot see
at 150mm and cannot afford to get wrong -- an unattached dot is a loose disc.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p2s import inventory, profiles  # noqa: E402
from parts.lamborghini_badge import Params, artwork, plated, report  # noqa: E402

ASA = "#22262b"
PANEL = "#b9c0c6"
TAPE = "#c8792b"
INK = "#20252b"
RULE = "#7c858d"

# PRINTED 2026-08-17 and these are the settings that did it, not a guess.
#
# The badge printed almost perfectly from a GUI slice of Studio's stock
# 0.20mm Standard with ironing switched on, so these now match that rather than
# the heavier defaults carried over from the MUSTANG badge. What changed, and
# why the earlier values are gone:
#
#   - NO BRIM. The MUSTANG settings asked for 12mm fused at zero gap, which was
#     never printed and never needed: both badges held to the textured plate
#     with no brim at all, four prints between them. It cost ten minutes a
#     print to insure against something that has not happened.
#   - 2 walls and 15% infill, not 3 and 25%. This is a 4mm slab; the extra
#     material bought nothing visible.
#   - monotonicline, not monotonic. Studio's current default, and better tuned.
#   - reduce_crossing_wall off. It was added for the MUSTANG's island-hopping
#     and this part is one island.
#
# The script is ONE island from the plate to the top face, so the clumping
# detector that stopped the MUSTANG twice has far less to misread here. Turn it
# off on the PRINTER regardless: a GUI-sliced job embeds the routine even with
# enable_wrapping_detection = 0, so the file's setting does not decide.
SLICE = {
    "enable_support": "0",
    "ironing_type": "top",
    # 0.21mm is the stock inset and it leaves a slight ridge where ironing meets
    # the perimeter: the pass runs just inside the wall at 10% flow and pushes
    # melt outward against it. Holding off further keeps the same flat faces
    # without piling material on the boundary. Reported as "a tiny bit rough on
    # the edges where the ironing meets the wall" on the first print.
    "ironing_inset": "0.4",
}

COOLING = {
    "fan_min_speed": "0",
    "fan_max_speed": "0",
    "close_fan_the_first_x_layers": "5",
    "overhang_fan_speed": "0",
}


def mesh(solid, tol: float = 0.05):
    verts, tris = solid.tessellate(tol)
    return np.array([(v.X, v.Y, v.Z) for v in verts]), np.array(tris)


def section(points, tris, axis: int, value: float):
    segs = []
    for tri in tris:
        p = points[tri]
        d = p[:, axis] - value
        hits = []
        for i in range(3):
            a, b = d[i], d[(i + 1) % 3]
            if (a > 0) != (b > 0):
                t = a / (a - b)
                hits.append(p[i] + t * (p[(i + 1) % 3] - p[i]))
        if len(hits) == 2:
            segs.append(hits)
    return np.array(segs) if segs else np.empty((0, 2, 3))


def _face_on(ax, params, part, info, pts, tris) -> None:
    box = part.bounding_box()
    pad_x, pad_y = box.size.X * 0.06, box.size.Y * 0.30
    ax.add_collection(
        PolyCollection(
            [np.array([
                (box.min.X - pad_x, box.min.Y - pad_y),
                (box.max.X + pad_x, box.min.Y - pad_y),
                (box.max.X + pad_x, box.max.Y + pad_y),
                (box.min.X - pad_x, box.max.Y + pad_y),
            ])],
            facecolors=PANEL, edgecolors="none", zorder=0,
        )
    )
    base = section(pts, tris, 2, 0.15)
    ax.add_collection(LineCollection(base[:, :, :2], colors=[ASA], lw=0.8, zorder=2))
    top = section(pts, tris, 2, box.max.Z - 0.05)
    ax.add_collection(LineCollection(top[:, :, :2], colors=[ASA], lw=2.0, zorder=3))

    ax.plot([box.min.X, box.max.X], [box.min.Y - pad_y * 0.55] * 2,
            color=RULE, lw=0.9, zorder=4)
    ax.annotate(f"{info['length']:.0f} mm overall",
                xy=(box.center().X, box.min.Y - pad_y * 0.9),
                ha="center", fontsize=9, color=INK)
    ax.set_xlim(box.min.X - pad_x, box.max.X + pad_x)
    ax.set_ylim(box.min.Y - pad_y, box.max.Y + pad_y)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("on the panel  ·  traced wordmark, not a font",
                 color=INK, fontsize=11)


def _iso(ax, part, pts, tris) -> None:
    tri_pts = pts[tris]
    light = np.array([0.35, -0.65, 0.68])
    light /= np.linalg.norm(light)
    n = np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(norm == 0, 1, norm)
    shade = 0.42 + 0.72 * np.clip(n @ light, 0, 1)
    base = np.array(matplotlib.colors.to_rgb(ASA))
    colors = np.clip(base * shade[:, None] + 0.10 * shade[:, None], 0, 1)
    ax.add_collection3d(Poly3DCollection(tri_pts, facecolors=colors, edgecolors="none"))
    box = part.bounding_box()
    ax.set_xlim(box.min.X, box.max.X)
    ax.set_ylim(box.min.Y - box.size.X * 0.15, box.max.Y + box.size.X * 0.15)
    ax.set_zlim(0, box.size.X * 0.30)
    ax.set_box_aspect((box.size.X, box.size.X * 0.5, box.size.X * 0.30))
    ax.view_init(elev=54, azim=-72)
    ax.set_axis_off()
    ax.set_title("as printed  ·  bond face on the plate", color=INK,
                 fontsize=11, y=0.94)


def _dot_detail(ax, params, part, pts, tris) -> None:
    """The join that decides whether this is one badge or three pieces."""
    art = artwork(params)
    faces = art.faces()
    body = max(faces, key=lambda f: f.area)
    dots = [f for f in faces if f is not body]
    if not dots:
        ax.axis("off")
        return
    dot = dots[0]
    c = dot.bounding_box().center()
    r = 9.0

    base = section(pts, tris, 2, 0.15)
    top = section(pts, tris, 2, part.bounding_box().max.Z - 0.05)
    ax.add_collection(LineCollection(base[:, :, :2], colors=["#b03a2e"], lw=1.2,
                                     label="base (widest)"))
    ax.add_collection(LineCollection(top[:, :, :2], colors=[ASA], lw=2.0,
                                     label="letter face"))
    ax.set_xlim(c.X - r, c.X + r)
    ax.set_ylim(c.Y - r * 1.4, c.Y + r * 0.7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.set_title(f"i-dot join  ·  {params.dot_neck} mm stem", color=INK, fontsize=11)


def render(params: Params, dest: Path) -> Path:
    part = plated(params)
    info = report(params, part)
    pts, tris = mesh(part)

    fig = plt.figure(figsize=(16, 4.6), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.1, 0.8], wspace=0.05,
                          left=0.01, right=0.99, top=0.86, bottom=0.04)
    _face_on(fig.add_subplot(gs[0, 0]), params, part, info, pts, tris)
    _iso(fig.add_subplot(gs[0, 1], projection="3d"), part, pts, tris)
    _dot_detail(fig.add_subplot(gs[0, 2]), params, part, pts, tris)

    fig.suptitle(
        f"Lamborghini script badge  ·  {info['length']:.0f} × {info['height']:.1f} × "
        f"{info['thickness']:.1f} mm  ·  {info['mass_g']:.1f} g black ASA  ·  "
        f"{info['bond_area_mm2']:.0f} mm² bonded  ·  {info['pieces']} piece",
        color=INK, fontsize=12, y=0.97,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest


def export(params: Params, out: Path, nozzle: float = 0.4) -> dict:
    from build123d import Mesher

    from p2s import slicer

    out.mkdir(parents=True, exist_ok=True)
    part = plated(params)
    stl = out / "lamborghini_badge.stl"
    mesher = Mesher()
    mesher.add_shape(part)
    mesher.write(str(stl))
    asa = inventory.default("ASA").preset_for(nozzle)
    project = slicer.write_project(
        part, out / "lamborghini_badge.3mf", nozzle=nozzle, filament=asa
    )
    return {"stl": stl, "project": project, "filament": asa}


def slice_it(params: Params, out: Path, nozzle: float = 0.4):
    from p2s import slicer

    asa = inventory.default("ASA").preset_for(nozzle)
    return slicer.slice_project(
        out / "lamborghini_badge.3mf", out / "sliced",
        nozzle=nozzle, filament=asa,
        overrides=SLICE, filament_overrides=COOLING,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--length", type=float, default=Params.length)
    ap.add_argument("--thickness", type=float, default=Params.thickness)
    ap.add_argument("--bevel", type=float, default=Params.bevel)
    ap.add_argument("--dot-neck", type=float, default=Params.dot_neck,
                    help="0 leaves the i-dots as separate pieces")
    ap.add_argument("--sag", type=float, default=Params.sag,
                    help="gap under a straightedge across the badge; 0 = flat")
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--slice", action="store_true")
    args = ap.parse_args()

    params = Params(length=args.length, thickness=args.thickness, bevel=args.bevel,
                    dot_neck=args.dot_neck, sag=args.sag)
    part = plated(params)
    info = report(params, part)
    for key, value in info.items():
        print(f"  {key:16s} {value if value is None else f'{value:.2f}'}")

    png = render(params, args.out / "lamborghini_badge_preview.png")
    print("wrote", png)
    if args.export or args.slice:
        made = export(params, args.out)
        print("wrote", made["stl"])
        print("wrote", made["project"], f"({made['filament']})")
        machine = profiles.machine(0.4)
        box = part.bounding_box()
        print(f"  fits the {machine.bed_x:.0f}×{machine.bed_y:.0f} bed: "
              f"{machine.fits((box.size.X, box.size.Y, box.size.Z))}")
    if args.slice:
        result = slice_it(params, args.out)
        print("sliced", result.output)
        print(f"  {result.summary()}")


if __name__ == "__main__":
    main()
