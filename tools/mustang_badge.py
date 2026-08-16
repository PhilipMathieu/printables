"""Preview and export the badge.

The face-on panel is the one that matters. Everything else about this part is
easy -- it is a flat thing with a flat back -- and the only question worth
asking of a render is whether the lettering looks right, which you can only
judge square on at true proportions.
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
from parts.mustang_badge import (  # noqa: E402
    Params,
    build,
    plated,
    proportion_note,
    report,
)

ASA = "#22262b"     # black ASA, lifted enough to read against the panel
PANEL = "#b9c0c6"   # unpainted-looking tailgate, so the badge is what you see
TAPE = "#c8792b"
INK = "#20252b"
RULE = "#7c858d"

# Strings, not numbers: Bambu stores its whole config as strings. Kept next to
# the part it belongs to so the settings travel with the design.
SLICE = {
    "brim_type": "outer_only",
    "brim_width": "6",
    "enable_support": "0",
    "wall_loops": "3",
    "sparse_infill_density": "25%",
    "top_surface_pattern": "monotonic",
    "ironing_type": "top",
}


def mesh(solid, tol: float = 0.04):
    verts, tris = solid.tessellate(tol)
    return np.array([(v.X, v.Y, v.Z) for v in verts]), np.array(tris)


def section(points, tris, axis: int, value: float):
    """Segments where the mesh crosses ``axis == value``."""
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


def _face_on(ax, params: Params, part, info: dict) -> None:
    """The badge as it will look on the car, square on and to scale."""
    box = part.bounding_box()
    pad_x, pad_y = box.size.X * 0.10, box.size.Y * 0.75
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
    # Silhouette straight off the solid, so what is drawn is what is modelled.
    pts, tris = mesh(part)
    outline = section(pts, tris, 2, params.rail / 2)
    ax.add_collection(
        LineCollection(outline[:, :, :2], colors=[ASA], lw=1.0, zorder=2)
    )
    faces = section(pts, tris, 2, box.max.Z - 0.05)
    ax.add_collection(
        LineCollection(faces[:, :, :2], colors=[ASA], lw=2.4, zorder=3)
    )

    ax.plot([box.min.X, box.max.X], [box.min.Y - pad_y * 0.55] * 2,
            color=RULE, lw=0.9, zorder=4)
    ax.annotate(
        f"{info['length']:.0f} mm overall",
        xy=(0, box.min.Y - pad_y * 0.55), xytext=(0, box.min.Y - pad_y * 0.85),
        ha="center", fontsize=9, color=INK,
    )
    ax.annotate(
        f"{info['cap_height']:.1f} mm caps  ·  {info['aspect']:.1f}:1",
        xy=(box.min.X, box.max.Y + pad_y * 0.30),
        ha="left", va="center", fontsize=9, color=INK,
    )
    ax.set_xlim(box.min.X - pad_x, box.max.X + pad_x)
    ax.set_ylim(box.min.Y - pad_y, box.max.Y + pad_y)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"on the panel  ·  {params.font_path or params.font}",
        color=INK, fontsize=11,
    )


def _iso(ax, part) -> None:
    pts, tris = mesh(part)
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
    ax.set_ylim(box.min.Y - box.size.X * 0.2, box.max.Y + box.size.X * 0.2)
    ax.set_zlim(0, box.size.X * 0.35)
    ax.set_box_aspect((box.size.X, box.size.X * 0.55, box.size.X * 0.35))
    ax.view_init(elev=52, azim=-70)
    ax.set_axis_off()
    ax.set_title("as printed  ·  bond face on the plate", color=INK,
                 fontsize=11, y=0.94)


def _section(ax, params: Params, part, info: dict) -> None:
    """Vertical cut through a letter: thickness, rail, and the back profile."""
    pts, tris = mesh(part)
    box = part.bounding_box()
    # Cut where the most material is, so the rail and a letter both appear.
    best_x, best = 0.0, np.empty((0, 2, 3))
    for cand in np.linspace(box.min.X, box.max.X, 160)[4:-4]:
        segs = section(pts, tris, 0, cand)
        if len(segs) > len(best):
            best_x, best = cand, segs
    ax.add_collection(
        LineCollection(best[:, :, [1, 2]], colors=[ASA], lw=2.0, zorder=3)
    )

    ax.add_collection(
        PolyCollection(
            [np.array([(box.min.Y - 8, -params.tape), (box.max.Y + 8, -params.tape),
                       (box.max.Y + 8, 0.0), (box.min.Y - 8, 0.0)])],
            facecolors=TAPE, alpha=0.55, edgecolors="none", zorder=1,
        )
    )
    ax.plot([box.min.Y - 12, box.max.Y + 12], [-params.tape] * 2,
            color=RULE, lw=1.6, zorder=2)
    ax.annotate(
        f"tape {params.tape} mm", xy=(box.min.Y - 11, -params.tape - 2.2),
        fontsize=8, color=INK, va="center",
    )
    ax.annotate("panel", xy=(box.max.Y + 3, -params.tape - 2.2),
                fontsize=8, color=INK, va="center")
    ax.annotate(
        f"letters {params.thickness} mm · rail {params.rail} mm",
        xy=(box.min.Y, params.thickness + 1.2), fontsize=9, color=INK,
    )
    ax.set_xlim(box.min.Y - 12, box.max.Y + 12)
    ax.set_ylim(-params.tape - 4, params.thickness + 5)
    ax.set_aspect("equal")
    ax.axis("off")
    crown = (
        f"flat back" if params.sag <= 0
        else f"back hollowed {params.sag} mm  ·  panel R {info['panel_radius']:.0f} mm"
    )
    ax.set_title(f"section at x={best_x:.0f} mm  ·  {crown}", color=INK, fontsize=11)


def render(params: Params, dest: Path) -> Path:
    part = plated(params)
    info = report(params, part)

    fig = plt.figure(figsize=(15, 5.2), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.0], wspace=0.06,
                          left=0.01, right=0.99, top=0.88, bottom=0.04)
    _face_on(fig.add_subplot(gs[0, 0]), params, part, info)
    _iso(fig.add_subplot(gs[0, 1], projection="3d"), part)
    _section(fig.add_subplot(gs[0, 2]), params, part, info)

    fig.suptitle(
        f"MUSTANG badge  ·  {info['length']:.0f} × {info['overall_height']:.1f} × "
        f"{info['thickness']:.1f} mm  ·  {info['mass_g']:.1f} g black ASA  ·  "
        f"{info['bond_area_mm2']:.0f} mm² bonded",
        color=INK, fontsize=12, y=0.97,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest


def export(params: Params, out: Path, nozzle: float = 0.4) -> dict:
    """STL always; a project 3mf too, carrying the ASA preset and SLICE."""
    from build123d import Mesher

    from p2s import slicer

    out.mkdir(parents=True, exist_ok=True)
    part = plated(params)

    stl = out / "mustang_badge.stl"
    mesher = Mesher()
    mesher.add_shape(part)
    mesher.write(str(stl))

    asa = inventory.default("ASA").preset_for(nozzle)
    project = slicer.write_project(
        part, out / "mustang_badge.3mf", nozzle=nozzle, filament=asa
    )
    return {"stl": stl, "project": project, "filament": asa}


def slice_it(params: Params, out: Path, nozzle: float = 0.4):
    """Slice with the ASA presets and this part's settings actually applied."""
    from p2s import slicer

    asa = inventory.default("ASA").preset_for(nozzle)
    return slicer.slice_project(
        out / "mustang_badge.3mf",
        out / "sliced",
        nozzle=nozzle,
        filament=asa,
        overrides=SLICE,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--font", help="installed family name")
    ap.add_argument("--font-path", help="path to a .ttf/.otf -- preferred")
    ap.add_argument("--text", default=Params.text)
    ap.add_argument("--length", type=float, default=Params.length)
    ap.add_argument("--stretch", type=float, default=Params.stretch)
    ap.add_argument("--thickness", type=float, default=Params.thickness)
    ap.add_argument("--sag", type=float, default=Params.sag,
                    help="gap under a straightedge across the badge; 0 = flat")
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--export", action="store_true", help="also write STL + 3mf")
    ap.add_argument("--slice", action="store_true", help="export, then slice it")
    args = ap.parse_args()

    params = Params(
        text=args.text, length=args.length, stretch=args.stretch,
        thickness=args.thickness, sag=args.sag,
        **({"font_path": args.font_path} if args.font_path else {}),
        **({"font": args.font} if args.font else {}),
    )
    part = plated(params)
    info = report(params, part)
    for key, value in info.items():
        print(f"  {key:16s} {value if value is None else f'{value:.2f}'}")
    if note := proportion_note(info):
        print(f"\n  NOTE: {note}\n")

    png = render(params, args.out / "mustang_badge_preview.png")
    print("wrote", png)
    if args.export or args.slice:
        made = export(params, args.out)
        print("wrote", made["stl"])
        print("wrote", made["project"], f"({made['filament']})")
        machine = profiles.machine(0.4)
        box = part.bounding_box()
        fits = machine.fits((box.size.X, box.size.Y, box.size.Z))
        print(f"  fits the {machine.bed_x:.0f}×{machine.bed_y:.0f} bed: {fits}")
    if args.slice:
        result = slice_it(params, args.out)
        print("sliced", result.output)
        print(f"  {result.summary()}")


if __name__ == "__main__":
    main()
