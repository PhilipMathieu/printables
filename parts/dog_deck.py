"""The deck: a flat plate of dog holes that bolts onto the drill stand's base.

The Craftsman stand's cast base has four T-bolt slots in an X and nothing else.
As a coordinate system that is close to useless -- two diagonals through one
point -- and every awkward thing about fixturing on it comes from trying to say
"an inch and a half in from that edge" in a language that only has "somewhere
along a diagonal". The deck exists to change the coordinate system. Bolt it down
once and the slots are never touched again.

WHY IT IS PRINTED AND NOT CUT FROM PLYWOOD. The accuracy of the whole set lives
in this one part. A stop's position is known because the hole it sits in is
known, and a hole bored by hand in ply is a millimetre from where you meant it.
Printed, the grid is exact and free, and the parts that plug into it were cut
from the same numbers. Ply is the better material in every other respect --
stiffer, cheaper, and it does not mind being drilled into -- so if this gets
made in wood one day, bore it through a printed copy of this plate rather than
off a rule.

IT DOES NOT COST YOU ANY DEPTH. 10mm under the work sounds like 10mm off a
70mm stroke, but the drill carriage clamps at whatever height you set on the
column, so the deck moves the whole stroke up rather than eating into it. What
it costs is 10mm of the column's height, which nothing here is close to using.

THE CENTRE HOLE IS THE DATUM. Odd counts of columns and rows put a station
exactly on the origin, and the origin is meant to be the spindle axis. Chuck a
centre finder, drop it into that hole, and mark through the casting's slots from
underneath: the four mounting holes come out where the casting actually wants
them and no measurement of the casting is needed to get there. That hole then
holds the sacrificial backer for every through-hole you drill, and lines up over
the casting's own bit-clearance hole beneath.

MOUNTING HOLES ARE DELIBERATELY NOT MODELLED. They depend on measurements of a
particular casting rather than on anything in this repository, and marking them
through the slots is both easier and more accurate than transferring them. Drill
them 7mm for 1/4-20, and counterbore the top so a washer and nut sit below the
surface -- a nut standing proud of the deck is a nut the work rocks on.
"""

from __future__ import annotations

from dataclasses import dataclass

from build123d import Part, Plane, Polyline, Pos, RectangleRounded, extrude, make_face, revolve, Axis

from geom import dog_grid
from geom.dog_grid import Grid


@dataclass(frozen=True)
class Params:
    """Everything the GUI would expose as a slider."""

    grid: Grid = dog_grid.HALF_INCH
    cols: int = 7
    rows: int = 5
    """Stations across and along. Both odd, so one lands on the spindle axis --
    see the module note. 7 x 5 on a one inch pitch spans 152 x 102mm, which
    covers the casting's field and leaves a row of stations hanging off each end
    for work that is longer than the machine."""

    margin: float = 16.0
    """Plate beyond the outermost hole centres.

    Two things set it. It has to be at least a full web, so an outer station is
    backed by no less material than an inner one and the plate's edge is not the
    first thing a stop breaks out of. And it has to be at least half the clamp's
    depth, or a clamp in the outermost row hangs its back end off the plate --
    which is where a clamp holding the deepest workpiece the deck can take
    necessarily goes. 16mm clears both: the web is 12.7 and the clamp is 30 deep.
    """
    thickness: float = 0.0
    """Plate thickness. Zero means take the grid's own deck figure, which is
    what the dogs were cut to; setting it to anything else and not recutting the
    dogs gives you shanks that are the wrong length."""
    corner: float = 6.0
    """Plan radius on the plate's corners."""
    hole_chamfer: float = 0.8
    """Lead-in at the mouth of every hole. Small: it has to stay well inside the
    head of a stop, or the stop seats on a slope instead of on the deck."""

    @property
    def deck(self) -> float:
        return self.thickness or self.grid.deck

    @property
    def width(self) -> float:
        return self.grid.span(self.cols) + 2 * self.margin

    @property
    def depth(self) -> float:
        return self.grid.span(self.rows) + 2 * self.margin

    @property
    def stations(self) -> tuple[tuple[float, float], ...]:
        return self.grid.stations(self.cols, self.rows)

    def validate(self) -> None:
        if self.cols % 2 == 0 or self.rows % 2 == 0:
            raise ValueError(
                f"a {self.cols} x {self.rows} grid puts the spindle between four "
                f"holes instead of in one. Both counts have to be odd: that centre "
                f"station is the datum the deck is aligned by, the hole the "
                f"sacrificial backer lives in, and where the bit goes when it comes "
                f"through the work."
            )
        if self.cols < 3 or self.rows < 3:
            raise ValueError("a grid narrower than three stations cannot hold a fence")
        if self.grid.web <= 0:
            raise ValueError(
                f"{self.grid.hole}mm holes on a {self.grid.pitch}mm pitch overlap"
            )
        if self.deck <= 0:
            raise ValueError("a deck with no thickness is a sticker")
        if self.margin < self.grid.web:
            raise ValueError(
                f"a {self.margin}mm margin is thinner than the {self.grid.web:.1f}mm "
                f"web between two holes, so the edge of the plate is weaker than its "
                f"middle and an outer station is the first thing to break out"
            )
        if 2 * self.hole_chamfer >= self.grid.web:
            raise ValueError(
                f"a {self.hole_chamfer}mm chamfer on both mouths of neighbouring "
                f"holes eats the {self.grid.web:.1f}mm web between them"
            )
        if self.corner > min(self.width, self.depth) / 2:
            raise ValueError("the corner radius leaves no straight edge")


def _hole(params: Params) -> Part:
    """One hole cutter: a bore with a chamfered mouth, long at both ends.

    Run past the plate top and bottom so the boolean has no coincident faces to
    resolve -- two solids sharing a face is the usual way a subtraction leaves a
    zero-thickness sliver behind and the result stops being valid.
    """
    r, t, c = params.grid.hole / 2, params.deck, params.hole_chamfer
    over = 1.0
    profile = [
        (0.0, -over),
        (r, -over),
        (r, t - c),
        (r + c, t),
        (r + c, t + over),
        (0.0, t + over),
    ]
    return revolve(
        Plane.XZ * make_face(Polyline(*profile, close=True).wire()), Axis.Z
    )


def build(params: Params) -> Part:
    """The plate, holes and all."""
    params.validate()
    plate = extrude(
        Plane.XY * RectangleRounded(params.width, params.depth, params.corner),
        amount=params.deck,
    )
    # Summed into one cutter and subtracted once. Thirty-five sequential
    # booleans against a plate this size takes minutes; this takes seconds.
    cutter = _hole(params)
    holes = Part()
    for x, y in params.stations:
        holes += Pos(x, y, 0) * cutter
    return plate - holes


if __name__ == "__main__":
    from pathlib import Path

    from build123d import export_stl

    p = Params()
    part = build(p)
    out = Path("out")
    out.mkdir(exist_ok=True)
    export_stl(part, str(out / "dog_deck.stl"))
    bb = part.bounding_box()
    print(f"valid={part.is_valid} solids={len(part.solids())}")
    print(f"bbox={bb.size.X:.1f} x {bb.size.Y:.1f} x {bb.size.Z:.1f} mm")
    print(f"{len(p.stations)} stations, {part.volume / 1000:.1f} cm^3")
