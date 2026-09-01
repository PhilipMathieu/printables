"""Bench dogs for a small drill press deck: stops, a fence, a screw clamp.

Everything here plugs into one hole pattern -- see ``geom.dog_grid`` -- and the
whole set exists to answer one question: what stops the work turning when the
bit grabs. On a drill stand made for a portable drill that is the only force
worth designing against. The thrust goes straight down into the deck and needs
nothing; the torque wants to spin the workpiece out of your fingers, and it has
to be routed into the casting through something that is not your fingers.

THE LOAD PATH, WHICH IS THE WHOLE DESIGN. Work bears on a printed face; the
printed face bears on a shank; the shank bears on the wall of a hole in the
deck; the deck is bolted to the casting. Every step in that chain is plastic in
compression or bearing, which is the direction FDM is good in. Nothing in the
set is a printed thread, nothing is glued, and nothing carries a load in tension
across its layers. The one fastener is a steel M6 bolt in the clamp, and it is
steel precisely because it is the one part that has to pull.

WHY THE STOPS ARE ROUND. A round dog touches a straight edge at one point, and
that point is exactly ``head/2`` from the hole's centre whichever way the dog
happens to be facing. So a stop never has to be oriented, never has to be keyed,
and its position is known from the grid alone -- you can work out where a fence
lands before you print anything. A square-faced stop would locate better against
a straight edge and would have to be indexed to do it, and indexing a dog is a
mechanism where a circle is a fact. Two round dogs in adjacent holes make a
straight reference between them, which is what the fence is for the long case.

EVERYTHING PRINTS THE SAME WAY UP: seating face towards the sky, shanks
pointing up, working faces vertical. That is upside down from how the parts are
used, and it is the orientation in which not one facet in the set overhangs past
45 degrees. It falls out of a single rule -- a shank is always narrower than
what it grows out of, so putting the shank last means every diameter change is a
step inward. The alternative, printing them the way up they are used, hangs the
head out over the shank on every part and puts a chamfer under a face whose job
is to sit flat on a deck.

WHY A SCREW AND NOT A CAM OR A WEDGE. The gap between a workpiece and the
nearest dog hole is anything from nothing to a full pitch, so a clamp has to
cover 25.4mm of travel. An eccentric that threw that far would not be
self-locking, and a folding wedge pair that threw that far would be 28mm thick
at the fat end. An M6 x 60 bolt covers it, costs pennies, and puts the plastic
back into compression where it belongs.

Print in ASA, or PETG. Not PLA: this sits on a machine, gets left clamped, and
takes warm swarf off an aluminium enclosure, and PLA is soft by 55 degrees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from build123d import (
    Align,
    Axis,
    Box,
    Part,
    Plane,
    Polygon,
    Polyline,
    Pos,
    extrude,
    make_face,
    revolve,
)

from geom import dog_grid
from geom.dog_grid import Grid

MU = 0.35
"""Coefficient of friction taken for printed plastic on printed plastic. Only
used to say whether something would hold, never to size anything."""


def _revolved(points: list[tuple[float, float]]) -> Part:
    """A solid of revolution from a (radius, height) profile about Z."""
    return revolve(Plane.XZ * make_face(Polyline(*points, close=True).wire()), Axis.Z)


@dataclass(frozen=True)
class Params:
    """Everything the GUI would expose as a slider."""

    grid: Grid = dog_grid.HALF_INCH
    """The hole pattern every part here plugs into."""
    fit: float = dog_grid.FITS["slip"]
    """Diametral clearance between a shank and its hole. Print the ladder before
    trusting this one -- see ``geom.dog_grid.FITS``."""

    # --- the stop ---------------------------------------------------------
    head: float = 16.0
    """Diameter of the part of a stop that stands above the deck, and so twice
    the distance from a hole's centre to the face a workpiece rests on."""
    rise: float = 12.0
    """How far a stop stands above the deck. Tall enough to catch the side of a
    pedal enclosure, short enough that the bit's flutes clear it."""
    relief: float = 1.0
    """How far short of the deck's underside a shank stops. Without it a shank
    printed a hair long stands the deck off the casting and every other number
    here is measured from a surface that rocks."""
    lead: float = 0.8
    """Chamfer on the tip of a shank, so it finds the hole rather than the deck."""
    break_edge: float = 0.6
    """Chamfer on the exposed edges of a head."""

    # --- the fence --------------------------------------------------------
    fence_span: int = 4
    """Pitches between the fence's two shanks. Four puts them 101.6mm apart,
    which spans the long side of a 1590B with a station to spare.

    Even, deliberately. The shanks sit half the span either side of the bar's
    middle, so an even span puts the middle on a station too and the fence
    centres on the grid; an odd one puts it half a pitch over. Both work -- an
    odd fence is simply offset -- but only the even one lines up with a stop
    dropped in the column the bar is centred on."""
    fence_height: float = 18.0
    fence_thickness: float = 16.0
    """The same as ``head``, and not by coincidence. At exactly that thickness a
    fence face and a stop's face stand the same distance from their hole
    centres, so the two are coplanar and interchangeable: a fence over four
    pitches and a stop six pitches out are one straight reference, and a
    workpiece bridging them touches both. Thicker and the stops are behind the
    fence, doing nothing; thinner and they stand proud of it, and the work rests
    on two points instead of the face."""
    fence_end: float = 14.0
    """How far the bar runs past each shank."""

    # --- the screw clamp --------------------------------------------------
    clamp_depth: float = 30.0
    """Front to back. This is the base that resists the clamp tipping forward,
    so it is deliberately wider than it looks like it needs to be."""
    clamp_height: float = 20.0
    clamp_end: float = 12.0
    screw_height: float = 9.0
    """Height of the bolt's axis above the deck.

    Low, because everything about the clamp tipping forward scales with it, and
    bounded below by the nut trap: the gable over the trap stands the across-
    corners radius plus half the across-flats above the bolt's axis, and all of
    that has to stay inside the body."""
    bore: float = 6.6
    """Clearance hole for an M6 bolt."""
    screw_length: float = 60.0
    """Length under the head of the bolt this is cut for. It has to be longer
    than the body by more than one pitch -- see ``travel``."""
    nut_af: float = 10.3
    """Across the flats of an M6 nut, plus enough not to have to force it."""
    nut_deep: float = 5.6
    pad: float = 14.0
    """Diameter of the floating pressure pad the bolt pushes."""
    pad_thickness: float = 4.0

    # --- the sacrificial backer -------------------------------------------
    puck_grip: float = 0.2
    """How much wider a backing puck is at the top than the bottom. It is a
    taper, not a press fit: the bit pushes down on it, and a taper that way up
    tightens under exactly that load."""

    # --- what follows from those ------------------------------------------

    @property
    def shank(self) -> float:
        return self.grid.shank(self.fit)

    @property
    def shank_length(self) -> float:
        return self.grid.deck - self.relief

    @property
    def reach(self) -> float:
        """Distance from a hole's centre to a stop's working face."""
        return self.head / 2

    @property
    def fence_length(self) -> float:
        return self.grid.span(self.fence_span + 1) + 2 * self.fence_end

    @property
    def fence_reach(self) -> float:
        """Distance from the shank centres to either face of the fence.

        Either: the bar is symmetric about its shanks, so it has two working
        faces and no wrong way round.
        """
        return self.fence_thickness / 2

    @property
    def clamp_length(self) -> float:
        return self.grid.pitch + 2 * self.clamp_end

    @property
    def travel(self) -> float:
        """How far the bolt tip can stand out from the front face.

        With the head wound back against the rear face the tip reaches this far,
        and that distance has to beat one pitch. Otherwise there is a band of
        workpiece sizes that falls between two dog holes: too big for the clamp
        in one hole, too small to reach from the next.
        """
        return self.screw_length - self.clamp_depth

    def at(self, fit_name: str) -> Params:
        """The same set cut to a different shank clearance."""
        return replace(self, fit=dog_grid.fit(fit_name))

    def validate(self) -> None:
        g = self.grid
        if self.shank <= 0:
            raise ValueError(
                f"a {self.fit}mm clearance leaves nothing of a {g.hole}mm hole"
            )
        if self.fit <= 0:
            raise ValueError(
                f"a {self.fit}mm clearance is an interference fit; a dog has to "
                f"drop into its hole under its own weight, not be driven in"
            )
        if self.head <= g.hole:
            raise ValueError(
                f"a {self.head}mm head drops straight through a {g.hole}mm hole. "
                f"The head is what a stop stands on."
            )
        if self.head >= g.pitch:
            raise ValueError(
                f"a {self.head}mm head is wider than the {g.pitch}mm pitch, so two "
                f"stops in neighbouring holes foul each other and the grid loses "
                f"every station next to an occupied one"
            )
        if self.relief >= g.deck:
            raise ValueError(
                f"{self.relief}mm of relief on a {g.deck}mm deck leaves no shank"
            )
        if self.relief <= 0:
            raise ValueError(
                "a shank flush with the deck's underside is a shank that stands "
                "the deck off the casting if it prints a hair long"
            )
        if self.lead >= self.shank / 2:
            raise ValueError(f"a {self.lead}mm lead chamfer eats the whole shank")
        if self.fence_span < 1:
            raise ValueError("a fence on one shank is a stop that can swing")
        if self.fence_thickness < self.head:
            raise ValueError(
                f"a {self.fence_thickness}mm fence is thinner than the {self.head}mm "
                f"head of a stop, so a stop in the next hole along stands "
                f"{(self.head - self.fence_thickness) / 2:.1f}mm proud of the fence "
                f"face and the work rests on the stop instead of on the face. Make "
                f"it {self.head} and the two are coplanar."
            )
        if self.screw_height <= self.bore / 2 + 1:
            raise ValueError(
                f"a bore {self.screw_height}mm up breaks out of the bottom of the "
                f"clamp"
            )
        if self.screw_height + self.bore / 2 >= self.clamp_height:
            raise ValueError("the bore breaks out of the top of the clamp")
        nut_across_corners = self.nut_af * 2 / math.sqrt(3)
        if nut_across_corners >= self.clamp_height:
            raise ValueError(
                f"an {self.nut_af}mm nut measures {nut_across_corners:.1f}mm across "
                f"its corners and the clamp is only {self.clamp_height}mm tall"
            )
        gable = self.nut_af / 2 + nut_across_corners / 4
        if self.screw_height < gable:
            raise ValueError(
                f"a bolt axis {self.screw_height}mm up puts the peak of the nut "
                f"trap's gable {gable - self.screw_height:.2f}mm out through the "
                f"top of the clamp. Raise it to {gable:.2f} or use a smaller nut."
            )
        if self.screw_height + nut_across_corners / 2 >= self.clamp_height:
            raise ValueError(
                f"the nut trap's bottom corner comes out through the underside of "
                f"the clamp: {nut_across_corners / 2:.1f}mm below a bolt axis that "
                f"is {self.clamp_height - self.screw_height:.1f}mm off the plate"
            )
        if self.nut_deep >= self.clamp_depth / 2:
            raise ValueError("the nut trap runs more than halfway through the clamp")
        if self.travel < g.pitch:
            raise ValueError(
                f"{self.travel:.0f}mm of screw travel does not cover the {g.pitch}mm "
                f"pitch, so there are workpiece sizes that fall between two holes "
                f"and cannot be clamped at all"
            )
        if self.puck_grip <= 0:
            raise ValueError("a puck with no taper is a puck that falls out")


def _shank(params: Params) -> Part:
    """One shank, base at z=0, growing upward with a lead chamfer on its tip.

    Upward is how they print. In use the part is turned over and this is the bit
    in the hole.
    """
    r, ln, c = params.shank / 2, params.shank_length, params.lead
    return _revolved([(0.0, 0.0), (r, 0.0), (r, ln - c), (r - c, ln), (0.0, ln)])


def stop(params: Params) -> Part:
    """A plain round dog: the locating element the rest of the set is built on.

    Modelled as printed, which is upside down: z=0 is the head's outer face, the
    seat that lands on the deck is at ``rise``, and the shank runs up from
    there. No facet on it overhangs at all -- every change of diameter is a step
    inward on the way up.
    """
    params.validate()
    rh, rs = params.head / 2, params.shank / 2
    b, c = params.break_edge, params.lead
    top = params.rise + params.shank_length
    return _revolved(
        [
            (0.0, 0.0),
            (rh - b, 0.0),
            (rh, b),
            (rh, params.rise),
            (rs, params.rise),
            (rs, top - c),
            (rs - c, top),
            (0.0, top),
        ]
    )


def puck(params: Params) -> Part:
    """A sacrificial backer that sits flush in a hole and gets drilled into.

    The reason holes come out of thin aluminium round instead of triangular, and
    out of plastic without a crack running away from them: the bit breaks
    through into something solid rather than into air. It is a consumable.
    Print a dozen; each one is a couple of minutes and a gram.

    Tapered, wider at the top, so the drill's own thrust wedges it home. Push a
    spent one out from underneath with the next one.
    """
    params.validate()
    rb = params.shank / 2
    rt = rb + params.puck_grip
    b = params.break_edge
    return _revolved(
        [(0.0, 0.0), (rb - b, 0.0), (rb, b), (rt, params.grid.deck), (0.0, params.grid.deck)]
    )


def fence(params: Params) -> Part:
    """A straight reference edge on two shanks, a pitch multiple apart.

    Symmetric about its shanks, so both long faces work and there is no wrong
    way to put it down. The corner a workpiece is pushed into is broken on both
    sides for the same reason a fence is: an edge that catches is an edge that
    holds the work off the face it is supposed to be against.

    Printed on its top face -- the two broken edges are the ones on the plate,
    where a first layer's squash-out would otherwise leave the lip that catches.
    """
    params.validate()
    t2, h, b = params.fence_thickness / 2, params.fence_height, params.break_edge
    profile = Polygon(
        (-t2 + b, 0.0),
        (t2 - b, 0.0),
        (t2, b),
        (t2, h),
        (-t2, h),
        (-t2, b),
        align=None,
    )
    bar = extrude(Plane.YZ * profile, amount=params.fence_length / 2, both=True)
    reach = params.grid.span(params.fence_span + 1) / 2
    for x in (-reach, reach):
        bar += Pos(x, 0, h) * _shank(params)
    return bar


def _teardrop(radius: float, at: tuple[float, float]) -> Polygon:
    """A bore that prints lying down: a circle with a roof on it.

    A round horizontal hole sags at the top because there is nothing under the
    last of it. Capping it with two facets at 45 degrees costs a sliver of
    clearance at the very top of the bore, where a bolt does not touch, and
    means the clamp needs no support anywhere.
    """
    x0, z0 = at
    n = 48
    # Walked as one continuous sweep from the 45 degree point clockwise round
    # the bottom to 135, then up to the apex. Collecting the same points by
    # filtering a full circle and sorting them by angle splits the arc where
    # atan2 wraps, and closes the polygon through the middle of the bore.
    start, sweep = math.pi / 4, 3 * math.pi / 2
    arc = [
        (x0 + radius * math.cos(start - sweep * i / n),
         z0 + radius * math.sin(start - sweep * i / n))
        for i in range(n + 1)
    ]
    return Polygon(*arc, (x0, z0 + radius * math.sqrt(2.0)), align=None)


def _hex(across_flats: float, at: tuple[float, float]) -> Polygon:
    """A nut trap lying down, with a gable over it so nothing prints over air.

    Neither orientation of a bare hexagon is printable on its side. Point up,
    the two edges closing over the top run 30 degrees off horizontal -- 60
    degrees off vertical, which is past what FDM holds, and the received wisdom
    that point-up nut traps are the printable ones is about traps whose axis is
    vertical, not ones bored sideways like this. Flat up gives 30 degree side
    walls and a short flat roof, which bridges, but a bridge over the pocket a
    bolt has to be threaded through is where a sagging strand ends up.

    So: flat up for the walls, and the same 45 degree gable over it that the
    bore gets. The void above the nut is wasted space and nothing else.
    """
    x0, z0 = at
    r = across_flats / math.sqrt(3)  # across corners / 2
    corners = [
        (x0 + r * math.cos(i * math.pi / 3), z0 + r * math.sin(i * math.pi / 3))
        for i in range(6)
    ]
    # The apex goes in between the two corners the flat top spans, replacing
    # that one horizontal edge with two at 45 degrees. Half the top edge is
    # r/2 wide, so a 45 degree gable stands exactly that far above it.
    apex = (x0, z0 + across_flats / 2 + r / 2)
    return Polygon(*corners[:2], apex, *corners[2:], align=None)


def clamp(params: Params) -> Part:
    """The screw dog: two shanks, a captive nut, and an M6 bolt that pushes.

    Takes an M6 x 60 bolt and one M6 nut, and nothing else. The nut drops into
    the trap in the back face point-first, the bolt goes in behind it, and the
    tip comes out of the front face and pushes the pad against the work. The
    plastic never sees the thread: it holds a hex in a pocket and takes the
    reaction in compression across the body.

    Sits on two shanks a pitch apart rather than one, because a clamp on a
    single shank turns about it under exactly the load it is there to apply.

    Printed on its top face, shanks up, same as the fence.
    """
    params.validate()
    d, h = params.clamp_depth, params.clamp_height
    body = Box(
        params.clamp_length, d, h, align=(Align.CENTER, Align.CENTER, Align.MIN)
    )
    # Print-space height of the bolt axis: the part is modelled upside down, so
    # a bore 8mm above the deck in use is 8mm below the top face here.
    z = h - params.screw_height
    body -= extrude(Plane.XZ * _teardrop(params.bore / 2, (0.0, z)), amount=d, both=True)
    # The trap opens on the back face, which in use faces away from the work.
    body -= Pos(0, -d / 2 + params.nut_deep, 0) * extrude(
        Plane.XZ * _hex(params.nut_af, (0.0, z)), amount=params.nut_deep, both=False
    )
    for x in (-params.grid.pitch / 2, params.grid.pitch / 2):
        body += Pos(x, 0, h) * _shank(params)
    return body


def pad(params: Params) -> Part:
    """A floating disc between the bolt tip and the work.

    A bolt tightened straight onto a workpiece turns against it as it goes, and
    on an anodised or powder-coated enclosure that leaves a spiral scar. The pad
    is not fixed to anything: it sits on the tip, the tip turns inside its
    dimple, and the face against the work stays still.
    """
    params.validate()
    r, t = params.pad / 2, params.pad_thickness
    b = params.break_edge
    dimple = params.bore / 2
    return _revolved(
        [
            (0.0, 0.0),
            (r - b, 0.0),
            (r, b),
            (r, t - b),
            (r - b, t),
            (dimple, t),
            (0.0, t - dimple),
        ]
    )


def capacity(params: Params, rows: int) -> float:
    """The deepest workpiece a deck this many rows deep can trap, in mm.

    Fence on the back row, clamp on the front one, bolt wound right out. The
    shallow end needs no arithmetic: the clamp steps forward a row at a time and
    the bolt reaches further than a pitch, so every size below this is reachable
    from some row without a gap. That is the whole reason ``travel`` is checked
    against the pitch.
    """
    back = params.grid.span(rows) / 2
    return (back - params.reach) - (-back + params.clamp_depth / 2)


BUILDERS = {
    "stop": stop,
    "fence": fence,
    "clamp": clamp,
    "pad": pad,
    "puck": puck,
}


def ladder(params: Params) -> list[Part]:
    """One stop at each fit in the catalogue, loosest first.

    The first thing to print. Shank clearance is the only number in this set
    that cannot be reasoned out -- it depends on the deck, and a printed deck
    and a plywood one are not the same hole. Drop all four in and keep the
    tightest that still falls in under its own weight.
    """
    return [stop(params.at(name)) for name in dog_grid.LADDER]


if __name__ == "__main__":
    from pathlib import Path

    from build123d import export_stl

    p = Params()
    out = Path("out")
    out.mkdir(exist_ok=True)
    for name, builder in BUILDERS.items():
        part = builder(p)
        export_stl(part, str(out / f"bench_{name}.stl"))
        bb = part.bounding_box()
        print(
            f"{name:6s} valid={part.is_valid} "
            f"{bb.size.X:6.1f} x {bb.size.Y:6.1f} x {bb.size.Z:6.1f} mm "
            f"{part.volume / 1000:6.2f} cm^3"
        )
