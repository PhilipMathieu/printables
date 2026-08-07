"""Plan view straight from the ring outlines.

The solid build takes tens of minutes once there are twenty-odd rings, so this
draws the same geometry from the offsets themselves -- exact, and instant,
because the outlines are what the solids are made of.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parts.einstein_fidget import Params, _bulge, _outline  # noqa: E402
from tools.render import INK, ring_colors  # noqa: E402


def _loop(sketch, samples: int = 600):
    wire = sketch.wires()[0]
    return [(p.X, p.Y) for p in (wire.position_at(i / samples) for i in range(samples))]


def plan(params: Params, dest: Path, at_mid: bool = True) -> Path:
    z = params.thickness / 2 if at_mid else 0.0
    shift = _bulge(params, z)
    colors = ring_colors(params.total_rings + 1)

    fig, ax = plt.subplots(figsize=(9, 8), facecolor="white")
    # Painted outside in: each ring's outer loop in its colour, then its inner
    # loop in the background, which is exactly the empty gap. Cutting the hole
    # with a single even-odd polygon instead leaves a seam line where the two
    # loops join.
    for n, i in enumerate(range(-params.outer_rings, params.rings)):
        outer = _loop(_outline(params, round(i * params.pitch - shift, 6)))
        inner = _loop(_outline(params, round(i * params.pitch + params.wall - shift, 6)))
        ax.add_patch(MplPolygon(outer, closed=True, facecolor=colors[n],
                                edgecolor="none", zorder=n * 2))
        ax.add_patch(MplPolygon(inner, closed=True, facecolor="white",
                                edgecolor="none", zorder=n * 2 + 1))
    core = _loop(_outline(params, round(params.deepest_inset - shift, 6)))
    ax.add_patch(MplPolygon(core, closed=True, facecolor=colors[-1],
                            edgecolor="none", zorder=params.total_rings * 2))

    w, h = params.size
    ax.set_xlim(-params.outer_rings * params.pitch - params.unit * 1.6, w - 30)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"{w:.0f} × {h:.0f} mm  ·  {params.total_rings} rings + core  ·  "
        f"{params.gap} mm gaps, {params.wall} mm walls",
        color=INK, fontsize=12, pad=14,
    )
    ax.text(
        0.5, -0.02,
        f"interlock {params.interlock} mm — {params.engagement:.1f} mm engagement, "
        f"±{params.travel:.2f} mm travel per ring",
        transform=ax.transAxes, ha="center", va="top", fontsize=10, color=INK,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest


if __name__ == "__main__":
    print("wrote", plan(Params(), Path("out/einstein_coaster_plan.png")))
