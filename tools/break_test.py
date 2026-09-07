"""Draw the break test, because prose kept getting the orientation wrong.

    python -m tools.break_test

Every part in the set is modelled as it prints, which is upside down from how it
is used: on a stop, z=0 is the *head's outer face* and the shank runs up from
the seat. The break arm inherits that, so it comes off the plate with its shank
pointing at the ceiling and has to be turned over before it means anything. Two
sentences of text never once made that clear, hence a picture.

Drawing it is also what caught the flaw the arm was first built with. The lever
was coplanar with the head's seat, so it lay flat on the deck -- and a lever
lying on the surface its own fulcrum is in hands the moment straight to the
surface. On the real plate the root would have seen a twelfth of the intended
load and read four times too strong, with nothing in the result to say so. The
lever now stands off, and this drawing exists partly so that the next thing of
that kind gets seen before it is printed rather than after.

Everything in it is read out of ``bench_dogs.Params`` and ``dog_rig``, so the
dimensions on the drawing cannot drift from the ones in the STL, and the
thresholds cannot drift from ``bench_dogs.KNOB``. If a number here looks wrong,
the model is wrong.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parts import bench_dogs, dog_rig  # noqa: E402

INK = "#2b2f36"
PLASTIC = "#c8792b"
WOOD = "#b08b57"
BENCH = "#d8d2c6"
STEEL = "#8d949c"
BAD = "#b4413c"
OK = "#4a8a53"
WARN = "#d9a441"

EYE = 8.0
WIDTH = 20.0
STAND = dog_rig.STAND
"""Both read from the model rather than set here, because the drawing has to be
wrong when the part is, not agreeable. ``BLOCK`` is a function of the measured
fit, so it moved when the ladder was printed -- see ``dog_rig.block_reach``."""


def _outline(p, flipped: bool, upto: float):
    """(x, z) polygon of the arm in side elevation, origin on the shank axis.

    ``upto`` cuts the lever short so the near end can be drawn at true scale;
    the far end is shown with a break line rather than squashed, because a
    squashed shank is exactly the thing this drawing exists to show honestly.
    """
    rh, rs = p.head / 2, p.shank / 2
    top, deep = p.rise + p.shank_length, p.rise - STAND
    pts = [
        (-rh, 0.0), (upto, 0.0), (upto, deep), (rh, deep), (rh, p.rise),
        (rs, p.rise), (rs, top), (-rs, top), (-rs, p.rise), (-rh, p.rise),
    ]
    return pts if not flipped else [(x, p.rise - z) for x, z in pts]


def _break_line(ax, x, z0, z1):
    ax.plot([x - 2, x + 2, x - 2, x + 2], [z0, z0 + (z1 - z0) / 3,
            z0 + 2 * (z1 - z0) / 3, z1], color=INK, lw=1.0, zorder=8)


def _flip_panel(ax, p):
    """The same object twice, at true scale. The only point of the panel."""
    ax.set_title("modelled as it prints — turn it over to use it",
                 color=INK, fontsize=11)
    cut = 46.0
    for row, (flipped, caption) in enumerate(
        ((False, "off the plate:\nshank UP"), (True, "in the test:\nshank DOWN"))
    ):
        dy = -row * 40.0
        pts = [(x, z + dy) for x, z in _outline(p, flipped, cut)]
        ax.fill(*zip(*pts), color=PLASTIC, alpha=0.85, ec=INK, lw=1.1, zorder=3)
        _break_line(ax, cut, dy, dy + (p.rise - STAND))
        if flipped:
            ax.add_patch(Rectangle((-30, dy - 16), 76, 16, fc=WOOD, ec=INK,
                                   lw=1.1, zorder=1))
            ax.add_patch(Rectangle((-p.shank / 2, dy - 16), p.shank, 16,
                                   fc="white", ec=INK, lw=0.8, zorder=2))
            ax.annotate(f"{STAND:.0f} mm clear — the deck must\nnever touch "
                        "the lever", xy=(34, dy + 1.4), xytext=(30, dy + 24),
                        color=BAD, fontsize=8, ha="center",
                        arrowprops=dict(arrowstyle="-|>", color=BAD, lw=1.0))
            ax.annotate("only the head bears,\nand it tips onto this edge",
                        xy=(p.head / 2, dy), xytext=(-30, dy - 26), color=BAD,
                        fontsize=8, arrowprops=dict(arrowstyle="-|>", color=BAD,
                        lw=1.0, connectionstyle="arc3,rad=0.3"))
        else:
            ax.plot([-30, 46], [dy, dy], color=INK, lw=2.2, zorder=1)
            ax.text(52, dy - 5, "build plate", color=INK, fontsize=8)
        ax.text(-34, dy + p.rise / 2, caption, color=INK, fontsize=9,
                ha="right", va="center", fontweight="bold")

    ax.add_patch(FancyArrowPatch((56, -8), (56, -30), connectionstyle="arc3,rad=0.8",
                                 arrowstyle="-|>", mutation_scale=13, color=INK, lw=1.4))
    ax.set_xlim(-72, 76)
    ax.set_ylim(-74, 34)
    ax.set_aspect("equal")
    ax.axis("off")


def _rig_panel(ax, p):
    """Side elevation of the setup, to scale, with the bench edge in it."""
    at = dog_rig.eye_at(p)
    ax.set_title(f"the setup, in section · eye at {at:.0f} mm, "
                 f"effective arm {dog_rig.LEVER:.0f} mm", color=INK, fontsize=11)
    rs, rh = p.shank / 2, p.head / 2
    block_t, bench_t, edge = 19.0, 30.0, round(dog_rig.block_reach(p), -1)

    ax.add_patch(Rectangle((-190, -block_t - bench_t), 190 + edge, bench_t,
                           fc=BENCH, ec=INK, lw=1.0, zorder=1))
    ax.text(-140, -block_t - bench_t / 2, "bench", color=INK, fontsize=9,
            ha="center", va="center")
    ax.add_patch(Rectangle((-110, -block_t), 110 + edge, block_t, fc=WOOD,
                           ec=INK, lw=1.2, zorder=2))
    ax.add_patch(Rectangle((-rs, -block_t), 2 * rs, block_t, fc="white",
                           ec=INK, lw=1.0, zorder=3))
    ax.text(-62, -block_t / 2, f"{p.grid.hole:.0f} mm hole in scrap,\n"
            "drilled on the stand", color=INK, fontsize=8.5, ha="center",
            va="center", zorder=4)

    for dx in (-2.5, 2.5):
        ax.plot([-96 + dx, -96 + dx], [-block_t - bench_t - 20, 6], color=STEEL,
                lw=2.5, solid_capstyle="butt", zorder=5)
    for z in (-block_t - bench_t - 20, 6):
        ax.plot([-104, -88], [z, z], color=STEEL, lw=5, solid_capstyle="round",
                zorder=5)
    ax.text(-96, 14, "clamp the block down", color=INK, fontsize=8.5, ha="center")

    pts = _outline(p, flipped=True, upto=at + WIDTH)
    ax.fill(*zip(*pts), color=PLASTIC, alpha=0.9, ec=INK, lw=1.2, zorder=6)
    ax.add_patch(Circle((at, (p.rise - STAND) / 2), EYE / 2, fc="white", ec=INK,
                        lw=1.0, zorder=7))

    ax.annotate(f"the block must stop within {edge:.0f} mm of the hole, or the\n"
                "lever touches down and quietly takes the load back",
                xy=(edge, -2.0), xytext=(edge - 4, -52), color=BAD, fontsize=8.5, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=BAD, lw=1.1,
                                connectionstyle="arc3,rad=0.25"))
    ax.plot([edge, edge], [-block_t - 4, 6], color=BAD, lw=1.0,
            ls=(0, (4, 3)), zorder=8)

    ax.plot([at, at], [(p.rise - STAND) / 2 - 3, -56], color=STEEL, lw=1.6, zorder=6)
    ax.add_patch(Rectangle((at - 16, -84), 32, 28, fc=STEEL, ec=INK, lw=1.0,
                           alpha=0.75, zorder=6))
    ax.text(at, -70, "weight", color="white", fontsize=8.5, ha="center",
            va="center", fontweight="bold")

    for x in (rh, at):
        ax.plot([x, x], [p.rise, 50], color=INK, lw=0.6, ls=(0, (2, 2)), zorder=0)
    ax.annotate("", xy=(rh, 50), xytext=(at, 50),
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1.0))
    ax.text((rh + at) / 2, 53, f"{dog_rig.LEVER:.0f} mm — the moment arm, measured "
            f"from the head's edge\n(the eye itself is {at:.0f} from the axis)",
            color=INK, fontsize=9, ha="center")

    ax.annotate(f"it should break HERE\n(root fillet, {p.root:.1f} mm cone)",
                xy=(rs, -1), xytext=(-96, 26), color=BAD, fontsize=8.5,
                arrowprops=dict(arrowstyle="-|>", color=BAD, lw=1.2,
                                connectionstyle="arc3,rad=-0.3"))
    ax.set_xlim(-196, 176)
    ax.set_ylim(-96, 74)
    ax.set_aspect("equal")
    ax.axis("off")


def _scale_panel(ax, p):
    """What the number off the scale means, without arithmetic at the bench."""
    ax.set_title("read it off the scale", color=INK, fontsize=11)
    per_kg = 9.81 * dog_rig.LEVER / 1000
    light, firm = bench_dogs.KNOB
    lo = light * p.screw_height / 1000 / per_kg
    hi = firm * p.screw_height / 1000 / per_kg
    z = math.pi * p.shank**3 / 32
    pred = [z * mpa / 1000 / per_kg for mpa in (20, 35)]
    top = 7.6

    for a, b, colour, label in (
        (0, lo, BAD, "a LIGHT hand on the knob\nbreaks a shank —\nunusable as drawn"),
        (lo, hi, WARN, "usable, but a firm hand\ncan break it — fit a\nsmaller knob"),
        (hi, top, OK, "a firm hand is inside\nthe margin — change\nnothing"),
    ):
        ax.add_patch(Rectangle((0, a), 1.0, b - a, fc=colour, alpha=0.30, ec="none"))
        ax.text(1.14, (a + b) / 2, label, color=INK, fontsize=8.5, va="center")

    for y in (lo, hi):
        ax.plot([0, 1.0], [y, y], color=INK, lw=1.2)
        ax.text(-0.08, y, f"{y:.1f} kg", color=INK, fontsize=9.5, ha="right",
                va="center", fontweight="bold")

    ax.plot([0.5, 0.5], pred, color=INK, lw=3.0, solid_capstyle="butt", zorder=5)
    for y in pred:
        ax.plot([0.38, 0.62], [y, y], color=INK, lw=1.6, zorder=5)
    ax.text(0.5, pred[1] + 0.2, f"arithmetic says\n{pred[0]:.1f}–{pred[1]:.1f} kg",
            color=INK, fontsize=8.5, ha="center", va="bottom")

    ax.set_xlim(-0.8, 2.6)
    ax.set_ylim(0, top + 1.1)
    ax.axis("off")


def draw(dest: Path | None = None) -> Path:
    p = bench_dogs.Params()
    dest = dest or Path("docs/break_test.png")
    fig = plt.figure(figsize=(16.0, 5.2), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[0.62, 1.70, 0.58], wspace=0.03,
                          left=0.01, right=0.99, top=0.91, bottom=0.03)
    _flip_panel(fig.add_subplot(gs[0, 0]), p)
    _rig_panel(fig.add_subplot(gs[0, 1]), p)
    _scale_panel(fig.add_subplot(gs[0, 2]), p)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest


if __name__ == "__main__":
    print(f"wrote {draw()}")
