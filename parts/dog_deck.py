"""The deck: a flat plate of dog holes that bolts onto the drill stand's base.

The Craftsman stand's cast base has four T-bolt slots in an X and nothing else.
As a coordinate system that is close to useless -- two diagonals through one
point -- and every awkward thing about fixturing on it comes from trying to say
"an inch and a half in from that edge" in a language that only has "somewhere
along a diagonal". The deck exists to change the coordinate system. Bolt it down
once and the slots are never touched again.

PRINTED OR WOODEN, AND THE ARGUMENT IS CLOSER THAN IT LOOKS. The accuracy of the
whole set lives in this one part: a stop's position is known because the hole it
sits in is known, and a hole laid out by hand in ply is a millimetre from where
you meant it. Printed, the grid is exact and free.

But wood is the better material in every other respect. It is stiffer, it does
not creep under a clamp left tight for a week, it does not care about being
drilled into, it costs nothing, and it will not warp on the plate -- which is
the real risk in a printed plate this size, and the one thing about this design
that a print either survives or does not. MDF over ply, for hole quality and
because it moves less with the weather.

So the accuracy argument is not really an argument for plastic; it is an
argument against *marking out by hand*. ``template`` settles it: a printed
lattice that pilots every station off the same numbers this plate is built
from, so a wooden deck bored through it has the grid exactly and the material
advantages as well. Print the plate to get going, and cut the wooden one when
the numbers have stopped moving.

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

MOUNTING HOLES ARE DELIBERATELY NOT MODELLED. They depend on where a particular
casting's slots are rather than on anything in this repository, and marking them
through the slots with the deck sitting on the casting is both easier than
transferring a measurement and incapable of being off.

The hardware comes out of ``geom.drill_stand``, and it is 5/16-18, not the
1/4-20 this first said -- an 8.6mm slot with 13.4mm pockets takes a half-inch
hex head through the pocket and will not let it back out through the slot, and
there is 22.3mm of air under the casting for the head to live in. M8 also fits,
with 0.4mm at the pocket against 5/16's 0.7.

Drill 8.5mm through and counterbore the top 15mm wide and 6mm deep, so the nut
finishes below the surface: a nut standing proud is a nut the work rocks on. Use
a jam nut rather than a full one -- 4.4mm against 6.75 -- because the counterbore
comes out of the ten millimetres the deck has, and four left under a full nut is
less than is comfortable to tighten against. Snug, not cranked; that four
millimetres is in punching shear around the hole. A 5/16-18 x 1in reaches.
"""

from __future__ import annotations

from dataclasses import dataclass

from build123d import (
    Align,
    Axis,
    Box,
    Cylinder,
    Part,
    Plane,
    Polyline,
    Pos,
    RectangleRounded,
    extrude,
    make_face,
    revolve,
)

from geom import dog_grid
from geom.dog_grid import Grid


@dataclass(frozen=True)
class Params:
    """Everything the GUI would expose as a slider."""

    grid: Grid = dog_grid.DEFAULT
    cols: int = 7
    rows: int = 5
    """Stations across and along. Both odd, so one lands on the spindle axis --
    see the module note. 7 x 5 on a 25mm pitch spans 150 x 100mm, which covers
    the casting's 152mm field to within a millimetre at the outermost column and
    leaves a row of stations hanging off each end for work that is longer than
    the machine."""

    margin: float = 16.0
    """Plate beyond the outermost hole centres.

    Two things set it. It has to be at least a full web, so an outer station is
    backed by no less material than an inner one and the plate's edge is not the
    first thing a stop breaks out of. And it has to be at least half the clamp's
    depth, or a clamp in the outermost row hangs its back end off the plate --
    which is where a clamp holding the deepest workpiece the deck can take
    necessarily goes. 16mm clears both: the web is 13 and the clamp is 30 deep.
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


def hole(params: Params) -> Part:
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
    cutter = hole(params)
    holes = Part()
    for x, y in params.stations:
        holes += Pos(x, y, 0) * cutter
    return plate - holes


def suits_board(thickness: float, grid: Grid = dog_grid.DEFAULT) -> bool:
    """Whether a wooden deck of this thickness takes the dogs already printed.

    One-sided, and the side matters. Thicker is fine: the shank engages the top
    of the hole either way and the extra is material around it. Thinner is not,
    because shanks are cut to ``grid.deck`` less their relief and a board under
    that has them standing proud underneath, holding the deck off the casting --
    the failure the deck exists to prevent, arriving by way of the deck.

    So the rule is one comparison, and it is here rather than in a sentence in
    the README because the nearest common board is on the wrong side of it: 9mm
    MDF fails a 10mm grid, 12mm passes with two to spare. The upper end is set
    by the mounting hardware rather than the dogs, and lives in the stand's
    tests where the bolt length is.
    """
    return thickness >= grid.deck


def template(
    params: Params, thickness: float = 2.5, rib: float = 14.0, pilot: float = 3.2
) -> Part:
    """A drilling template: the grid as pilot holes, for boring a wooden deck.

    Clamp it to the board, run a 3mm bit through every hole, take it off, and
    bore the 12mm holes on the stand with a brad point or a Forstner dropped
    into each pilot. The grid then comes off the same arithmetic as the printed
    plate rather than off a rule and a square, which is the only reason the
    printed plate was preferred in the first place.

    THE BOARD HAS TO BE THICK ENOUGH FOR THE DOGS ALREADY CUT, and the trap is
    that the nearest common size is on the wrong side of it. Shanks are cut to
    the grid's deck less its relief, so a board thinner than ``grid.deck`` has
    every shank standing proud of its underside -- which is the one thing that
    stands the whole deck off the casting and undoes the flatness the deck
    exists to provide. At 10mm that rules out 9mm MDF, which is what a merchant
    hands you if you ask for "about ten". See ``suits_board``.

    A lattice rather than a sheet, because a solid template of this footprint is
    half the material of the deck it exists to avoid printing. Ribs down every
    row and column carry a pilot at each crossing and nothing anywhere else.

    Bore the pilots with a hand drill, not on the stand: the board is wider than
    this machine's throat, so the far row cannot be reached without turning the
    board around, and pilots are the step where that would cost you the grid.

    It aligns off its own outer ribs, whose centres sit ``margin`` in from where
    the deck's edges would be -- so cut the board to ``width`` x ``depth``, set
    the template back that far on every side, and clamp.
    """
    params.validate()
    across, along = params.grid.span(params.cols), params.grid.span(params.rows)
    lattice = Part()
    for _, y in {(0, y) for _, y in params.stations}:
        lattice += Pos(0, y, 0) * Box(
            across + rib, rib, thickness, align=(Align.CENTER, Align.CENTER, Align.MIN)
        )
    for x, _ in {(x, 0) for x, _ in params.stations}:
        lattice += Pos(x, 0, 0) * Box(
            rib, along + rib, thickness, align=(Align.CENTER, Align.CENTER, Align.MIN)
        )
    pilots = Part()
    for x, y in params.stations:
        pilots += Pos(x, y, thickness / 2) * Cylinder(pilot / 2, thickness * 3)
    return lattice - pilots


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
