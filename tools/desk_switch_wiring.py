"""The desk switch's wiring: one netlist, a schematic and a picture of the tabs.

    python -m tools.desk_switch_wiring
    python -m tools.desk_switch_wiring -o docs/desk_switch_wiring.png

The KCD4 is DPDT (on)-off-(on): two poles side by side, each a column of
three 6.3mm tabs with its common in the middle. ``WIRING`` says which wire
goes on which tab, and everything else here is drawn from it -- the
schematic, the view of the back of the switch, and the checks in
``tests/test_desk_switch.py`` that each press runs the motor one way, the
other press the other way, and that no position of the switch ever puts the
supply's two sides on one tab.

The cross: the jack's positive on one diagonal pair of end tabs, its
negative on the other, the motor on the two middles. Which end pair closes
on which press is up to the maker -- on many rockers pressing the top closes
the bottom pair -- and the cross does not care: one press puts the supply
across the motor one way and the other press the other way. It does not
care about left and right either: the mirror image of the cross is the same
circuit, so the picture of the tabs wires the same read from either side.
If the desk goes the wrong way, swap the two motor leads.
"""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import schemdraw  # noqa: E402
import schemdraw.elements as elm  # noqa: E402

from tools.render import INK  # noqa: E402

ROWS = ("top", "middle", "bottom")
"""Tab rows along the rocker, desk side first. The middle row is the commons."""
COLS = ("left", "right")
"""The two poles, as seen from behind the switch with the desk up."""

WIRING: dict[tuple[str, str], str] = {
    ("top", "left"): "+",
    ("bottom", "right"): "+",
    ("top", "right"): "-",
    ("bottom", "left"): "-",
    ("middle", "left"): "M1",
    ("middle", "right"): "M2",
}
"""Which wire lands on each tab. ``+`` and ``-`` are the jack's pigtail and
the jumper that carries it to the opposite corner."""

NETS = {
    "+": ("12 V +", "#c0392b"),
    "-": ("12 V −", "#20252b"),
    "M1": ("motor lead 1", "#d68910"),
    "M2": ("motor lead 2", "#2874a6"),
}
"""What each net is called on the drawing, and the colour its wire is drawn."""


# --- the switch as logic --------------------------------------------------------


def closed(position: str) -> dict[str, str | None]:
    """Which end row each pole's common touches. ``position`` is the row
    the press closes -- ``top``, ``bottom`` -- or ``off``."""
    if position == "off":
        return {col: None for col in COLS}
    if position not in ("top", "bottom"):
        raise ValueError(f"a rocker is top, bottom or off, not {position!r}")
    return {col: position for col in COLS}


def motor(position: str, wiring: dict[tuple[str, str], str] = WIRING) -> tuple[str, str] | None:
    """What the motor's two leads see with the switch in ``position``:
    ``(net on M1, net on M2)``, or None when it is not driven.

    Raises if any position joins ``+`` to ``-``, which is a dead short.
    """
    nodes: dict[str, set[str]] = {}
    for col in COLS:
        common = wiring[("middle", col)]
        node = {common}
        row = closed(position)[col]
        if row is not None:
            node.add(wiring[(row, col)])
        for net in node:
            nodes.setdefault(net, set()).update(node)
    # Nets that share a tab-pair are one node; walk them to closure.
    for _ in range(len(nodes)):
        for net, joined in nodes.items():
            for other in list(joined):
                joined |= nodes.get(other, set())
    for joined in nodes.values():
        if {"+", "-"} <= joined:
            raise ValueError(f"switch {position}: + and - meet, a dead short")
    leads = []
    for lead in ("M1", "M2"):
        supply = nodes.get(lead, set()) & {"+", "-"}
        leads.append(next(iter(supply)) if supply else None)
    if None in leads:
        return None
    return leads[0], leads[1]


def check(wiring: dict[tuple[str, str], str] = WIRING) -> None:
    """Every tab has a wire, the presses run the motor opposite ways, and
    the centre leaves it alone."""
    missing = [t for t in product(ROWS, COLS) if t not in wiring]
    if missing:
        raise ValueError(f"tabs with nothing on them: {missing}")
    up, down, off = motor("top", wiring), motor("bottom", wiring), motor("off", wiring)
    if off is not None:
        raise ValueError("the motor runs with the switch in the middle")
    if up is None or down is None:
        raise ValueError("a press leaves the motor without power")
    if up == down or set(up) != {"+", "-"}:
        raise ValueError(f"the presses give {up} and {down}: they have to be opposite")


# --- the drawings -----------------------------------------------------------------


def _schematic(ax) -> None:
    """The circuit: the jack, the cross, the motor."""
    pos, neg = NETS["+"][1], NETS["-"][1]
    # Built against the axes and drawn once at the end: schemdraw's context
    # manager does not draw onto a canvas it was handed.
    d = schemdraw.Drawing(canvas=ax, show=False)
    if True:
        d.config(unit=2.5, fontsize=11, color=INK, lw=1.6)
        sw = d.add(elm.SwitchDpdt().scale(2.2))
        # Motor across the two commons.
        d.add(elm.Line().at(sw.p1).left(1.6))
        d.add(elm.Line().down(1.0))
        d.add(elm.Motor().down().toy(sw.p2[1] + 1.0).label("motor", loc="bottom", ofst=0.3))
        d.add(elm.Line().down(1.0).toy(sw.p2[1]))
        d.add(elm.Line().right().tox(sw.p2[0]))

        # The two buses, crossing on their way to the far ends of the poles.
        xp, xn = sw.t1[0] + 2.0, sw.t1[0] + 3.4
        top = sw.t1[1] + 1.2
        bottom = sw.t4[1] - 1.2
        d.add(elm.Line(color=pos).at(sw.t1).right().tox(xp))
        d.add(elm.Line(color=pos).at(sw.t4).right().tox(xp))
        d.add(elm.Line(color=pos).at((xp, sw.t4[1])).up().toy(top))
        d.add(elm.Dot(color=pos).at((xp, sw.t1[1])))
        d.add(elm.Line(color=neg).at(sw.t2).right().tox(xn))
        d.add(elm.Line(color=neg).at(sw.t3).right().tox(xn))
        d.add(elm.Line(color=neg).at((xn, sw.t2[1])).down().toy(bottom))
        d.add(elm.Dot(color=neg).at((xn, sw.t3[1])))

        # The jack, its centre pin positive.
        xs = xn + 2.6
        d.add(elm.Line(color=pos).at((xp, top)).right().tox(xs))
        d.add(elm.Line(color=neg).at((xn, bottom)).right().tox(xs))
        d.add(elm.BatteryCell().at((xs, top)).down().toy(bottom)
              .label("12 V in\n5.5 × 2.1 jack\ncentre +", loc="bottom", ofst=0.9))

        # One press closes both a throws, the other both b throws.
        for t, name in ((sw.t1, "a"), (sw.t2, "b"), (sw.t3, "a"), (sw.t4, "b")):
            d.add(elm.Label().at((t[0] + 0.45, t[1] + 0.32)).label(name, fontsize=10))
        d.add(elm.Label().at((sw.p1[0] + 1.0, sw.t4[1] - 1.5))
              .label("KCD4 (on)-off-(on)\none press closes both a, the other both b", fontsize=10))
    d.draw(show=False)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_title("the circuit · two poles crossed, so the presses swap the motor's polarity",
                 color=INK, fontsize=11)


def _tabs(ax) -> None:
    """The back of the switch with its six tabs, and what goes on each."""
    pitch_x, pitch_y = 11.0, 9.0
    body_w, body_h = 22.0, 30.0
    ax.add_patch(FancyBboxPatch((-body_w / 2, -body_h / 2), body_w, body_h,
                                boxstyle="round,pad=0,rounding_size=2",
                                facecolor="#e6e8eb", edgecolor=INK, lw=1.2))
    where: dict[tuple[str, str], tuple[float, float]] = {}
    for (i, row), (j, col) in product(enumerate(ROWS), enumerate(COLS)):
        x = (j - 0.5) * pitch_x
        y = (1 - i) * pitch_y
        where[(row, col)] = (x, y)
        ax.add_patch(Rectangle((x - 3.15, y - 0.5), 6.3, 1.0, facecolor="#b9bec5",
                               edgecolor=INK, lw=0.9, zorder=3))

    # Wires in from outside: the pigtail to the top corners, the motor to
    # the middles, then the two jumpers corner to corner.
    def wire(a, b, net, ls="-", z=4):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=NETS[net][1], lw=3.2, ls=ls,
                solid_capstyle="round", zorder=z)

    for (row, col), net in WIRING.items():
        x, y = where[(row, col)]
        side = -1 if col == "left" else 1
        if row == "middle":
            wire((x, y), (x + side * 16, y), net)
            ax.text(x + side * 17, y, NETS[net][0], color=NETS[net][1], fontsize=9.5,
                    ha="left" if side > 0 else "right", va="center")
        elif row == "top":
            wire((x, y), (x + side * 16, y + 8), net)
            ax.text(x + side * 17, y + 8, f"{NETS[net][0]} from the jack", color=NETS[net][1],
                    fontsize=9.5, ha="left" if side > 0 else "right", va="center")
    for net in ("+", "-"):
        (a, b) = [where[t] for t, n in WIRING.items() if n == net]
        wire(a, b, net, ls=(0, (4, 2)), z=5 if net == "+" else 4)
    for (row, col), (x, y) in where.items():
        ax.plot(x, y, "o", color=INK, ms=4, zorder=6)

    ax.text(0, body_h / 2 + 3, "desk side", fontsize=9, color=INK, ha="center", va="bottom")
    ax.text(0, -body_h / 2 - 3,
            "seen from inside the box · dashed: the two jumpers, insulated, crossing\n"
            "meter it before power: each end tab goes to the opposite corner's net",
            fontsize=8.5, color=INK, ha="center", va="top")
    ax.set_xlim(-46, 46)
    ax.set_ylim(-body_h / 2 - 14, body_h / 2 + 14)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("the tabs · what goes on each", color=INK, fontsize=11)


def diagram(dest: Path) -> Path:
    """The schematic beside the tabs, checked before it is drawn."""
    check()
    fig = plt.figure(figsize=(15, 5.6), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.22,
                          left=0.02, right=0.98, top=0.90, bottom=0.04)
    _schematic(fig.add_subplot(gs[0, 0]))
    _tabs(fig.add_subplot(gs[0, 1]))
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out", default="out/desk_switch_wiring.png")
    args = ap.parse_args(argv)
    print(f"wrote {diagram(Path(args.out))}")
    for position in ("top", "off", "bottom"):
        print(f"  press {position:>6}: motor sees {motor(position)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
