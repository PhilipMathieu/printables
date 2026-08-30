"""Software preview of a part -- no GPU, so it works where Bambu Studio can't.

Three views: a shaded isometric, a plan section showing how the rings nest,
and an elevation section through the ring walls.

The mesh, section, iso and plate helpers are shared: every part's own preview draws
its sections off the same triangles, and every command line that can print more
than one of something lays them out with the same grid.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build123d import Part, Pos  # noqa: E402

from p2s import profiles  # noqa: E402
from parts.einstein_fidget import Params, build  # noqa: E402

FILAMENT = "#1d7a4c"  # fenway green
ASA = "#79808a"  # black ASA as mid grey: shaded true black, every facet reads the same
RING_STOPS = ["#0f5c38", "#1d7a4c", "#3ba873", "#7fc9a2", "#cfeadd"]
INK = "#20252b"


def ring_colors(n: int):
    """Continuous ramp outer-to-inner; a short cycling palette reads as noise
    once there are twenty-odd rings."""
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("rings", RING_STOPS)
    return [cmap(i / max(1, n - 1)) for i in range(n)]


def plate(parts: list[Part], gap: float = 6.0, nozzle: float = profiles.DEFAULT_NOZZLE):
    """Lay parts out in a roughly square grid, centred on the origin.

    Sized off the largest one so a mixed set -- clips cut for different stems,
    say -- still lands on a regular grid; the slicer centres the whole thing on
    the bed afterwards.
    """
    sizes = [p.bounding_box().size for p in parts]
    pitch_x = max(s.X for s in sizes) + gap
    pitch_y = max(s.Y for s in sizes) + gap
    cols = max(1, math.ceil(math.sqrt(len(parts) * pitch_y / pitch_x)))
    rows = math.ceil(len(parts) / cols)

    laid = Part()
    for i, part in enumerate(parts):
        col, row = i % cols, i // cols
        laid += Pos(
            (col - (cols - 1) / 2) * pitch_x, (row - (rows - 1) / 2) * pitch_y, 0
        ) * part
    size = laid.bounding_box().size
    mach = profiles.machine(nozzle)
    if not mach.fits((size.X, size.Y, size.Z)):
        raise SystemExit(
            f"{len(parts)} parts lay out {size.X:.0f} x {size.Y:.0f} mm, past the "
            f"{mach.bed_x:.0f} x {mach.bed_y:.0f} mm bed. Print fewer at a time."
        )
    return laid


def mesh(solid, tol: float = 0.05):
    verts, tris = solid.tessellate(tol)
    return np.array([(v.X, v.Y, v.Z) for v in verts]), np.array(tris)


def section(points, tris, axis: int, value: float):
    """Segments where the mesh crosses the plane ``axis == value``."""
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


def iso(ax, points, tris, title: str, colour: str = ASA,
        elev: float = 26, azim: float = -58) -> None:
    """Flat-shaded three-quarter view: what comes off the plate.

    One light, one colour, per-facet shading and no edges -- enough to read the
    form of a part in a README and cheap enough to run in a test suite. The box
    is forced cubic so a tall part is not stretched to fill the axes.
    """
    tri = points[tris]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(length == 0, 1, length)
    light = np.array([0.35, -0.75, 0.56])
    light /= np.linalg.norm(light)
    shade = 0.45 + 0.65 * np.clip(n @ light, 0, 1)
    base = np.array(matplotlib.colors.to_rgb(colour))
    ax.add_collection3d(
        Poly3DCollection(
            tri, facecolors=np.clip(base * shade[:, None], 0, 1), edgecolors="none"
        )
    )
    lo, hi = points.min(axis=0), points.max(axis=0)
    span = max(hi - lo)
    mid = (lo + hi) / 2
    for setter, i in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
        setter(mid[i] - span / 2, mid[i] + span / 2)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, color=INK, fontsize=11, y=0.92)


def render(params: Params, dest: Path) -> Path:
    part = build(params)
    solids = sorted(part.solids(), key=lambda s: -s.bounding_box().size.X)
    meshes = [mesh(s) for s in solids]
    colors_by_ring = ring_colors(len(solids))
    bb = part.bounding_box()
    cx, cy = (bb.min.X + bb.max.X) / 2, (bb.min.Y + bb.max.Y) / 2

    fig = plt.figure(figsize=(14, 5.0), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.05,
                          left=0.01, right=0.99, top=0.90, bottom=0.02)

    # --- isometric, flat-shaded -------------------------------------------
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    light = np.array([0.4, -0.7, 0.6])
    light /= np.linalg.norm(light)
    for pts, tris in meshes:
        tri_pts = pts[tris]
        n = np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0])
        norm = np.linalg.norm(n, axis=1, keepdims=True)
        n = n / np.where(norm == 0, 1, norm)
        shade = 0.48 + 0.62 * np.clip(n @ light, 0, 1)
        base = np.array(matplotlib.colors.to_rgb(FILAMENT))
        colors = np.clip(base * shade[:, None], 0, 1)
        ax.add_collection3d(
            Poly3DCollection(tri_pts, facecolors=colors, edgecolors="none")
        )
    pad = 0.06 * bb.size.X
    ax.set_xlim(bb.min.X - pad, bb.max.X + pad)
    ax.set_ylim(bb.min.Y - pad, bb.max.Y + pad)
    ax.set_zlim(bb.min.Z - bb.size.X * 0.30, bb.max.Z + bb.size.X * 0.30)
    ax.set_box_aspect((bb.size.X, bb.size.Y, bb.size.X * 0.80))
    ax.view_init(elev=34, azim=-62)
    ax.set_axis_off()
    ax.set_title(
        f"as printed  ·  {bb.size.X:.0f} × {bb.size.Y:.0f} × {bb.size.Z:.0f} mm",
        color=INK, fontsize=11, y=0.90,
    )

    # --- plan section at mid height ---------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    zmid = (bb.min.Z + bb.max.Z) / 2
    for i, (pts, tris) in enumerate(meshes):
        segs = section(pts, tris, 2, zmid)
        ax.add_collection(
            LineCollection(segs[:, :, :2], colors=[colors_by_ring[i]], lw=1.6)
        )
    ax.set_xlim(bb.min.X - 2, bb.max.X + 2)
    ax.set_ylim(bb.min.Y - 2, bb.max.Y + 2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"plan at mid height  ·  {len(solids)} free bodies, {params.gap} mm gaps",
        color=INK, fontsize=11,
    )

    # --- elevation section through the walls ------------------------------
    # Pick the cut that crosses the most walls; an arbitrary one can slice
    # through a lobe where only a single ring is present and show nothing.
    ax = fig.add_subplot(gs[0, 2])
    best_y, best_cuts = cy, []
    for cand in np.linspace(bb.min.Y, bb.max.Y, 120)[5:-5]:
        cuts = [section(pts, tris, 1, cand) for pts, tris in meshes]
        if sum(len(c) for c in cuts) > sum(len(c) for c in best_cuts):
            best_y, best_cuts = cand, cuts

    xs = np.concatenate([c[:, :, 0].ravel() for c in best_cuts if len(c)])
    right = xs.max()
    span = params.rings * params.pitch + params.wall + 3
    for i, segs in enumerate(best_cuts):
        if len(segs):
            ax.add_collection(
                LineCollection(segs[:, :, [0, 2]],
                               colors=[colors_by_ring[i]], lw=2.2)
            )
    ax.set_xlim(right - span, right + 1.5)
    ax.set_ylim(-2, params.thickness + 7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"wall profile in section  ·  {params.interlock} mm interlock",
        color=INK, fontsize=11, y=0.98,
    )
    ax.annotate(
        f"walls bulge {params.interlock} mm at mid height and the gap stays\n"
        f"exactly {params.gap} mm, so rings travel ±{params.travel:.1f} mm then wedge",
        xy=(right - params.pitch * 1.5, params.thickness / 2),
        xytext=(right - span + 0.5, params.thickness + 3.4),
        fontsize=9, color=INK,
        arrowprops=dict(arrowstyle="->", color=INK, lw=1,
                        connectionstyle="arc3,rad=-0.25"),
    )

    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest


if __name__ == "__main__":
    out = render(Params(), Path("out/einstein_fidget_preview.png"))
    print("wrote", out)
