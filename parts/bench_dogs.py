"""Bench dogs for a small drill press deck: stops, a fence, a screw clamp.

Everything here plugs into one hole pattern -- see ``geom.dog_grid`` -- and the
whole set exists to answer one question: what stops the work turning when the
bit grabs. On a drill stand made for a portable drill that is the only force
worth designing against. The thrust goes straight down into the deck and needs
nothing; the torque wants to spin the workpiece out of your fingers, and it has
to be routed into the casting through something that is not your fingers.

THE LOAD PATH, WHICH IS THE WHOLE DESIGN. Work bears on a printed face; the
printed face bears on a shank; the shank bears on the wall of a hole in the
deck; the deck is bolted to the casting. Almost all of that is plastic in
compression or bearing, which is the direction FDM is good in, and nothing in
the set is a printed thread.

The exception, and it is worth naming rather than glossing: every shank prints
with its layers across its axis, so a side load bends it at the root and that
root is in layer-line tension -- FDM's worst direction. For a stop taking drill
torque the numbers are small. For the clamp they need not be, because the
largest force in this whole assembly is not the drill, it is the clamp's own
screw: an M6 wound up with a tool reaches thousands of newtons, well past what
an unfilleted printed root holds. Hence two things. Every shank root gets a 45
degree cone, which is a fillet in the only direction that still prints. And the
bolt wants a knob or a wing head rather than a hex or a socket, so the torque a
hand can apply is the limit rather than the plastic being it.

WHAT THE FOUR MOUNTING BOLTS ACTUALLY SEE, which is almost nothing. A fence and
a clamp holding one workpiece push against each other, and both reactions land
in the same plate: the loop closes inside the deck and never reaches the
casting. The four 5/16-18s hold the deck flat and locate it, and that is all
they do.

The one steel part is the M6 in the clamp. Between its nut and its tip it is a
strut in compression, not something that pulls -- which is why the nut needs
body behind it and not in front of it. See ``clamp``.

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

WHICH PLASTIC, AND IT IS PLA -- which reverses what this said first. Two things
decide it and both point the same way. The deck is 184mm of flat plate, which is
precisely what ASA warps and PLA does not, and it is also the longest print
here: the material that removes that risk is worth more than the one that makes
it something to measure. And the failure mode that is actually marginal is a
shank root in layer-line tension, which is ASA's weak axis and PLA's strong one.

What PLA is bad at is heat and sustained load, and neither is load-bearing here
-- work is clamped for minutes, not weeks. Two parts still want ASA once the
numbers stop moving: the backers, which catch warm swarf off aluminium and will
emboss where PLA goes soft around 55 degrees, and the clamp, if you are the sort
who leaves things clamped for a week.
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
    mirror,
    revolve,
)

from geom import dog_grid
from geom.dog_grid import Grid

NUT_COVER = 2.0
"""Millimetres of material required between the peak of the nut trap's gable and
the face the clamp seats on. The gable is dead space above the nut, so it is
free to make it thinner by lowering the bolt -- right up until the roof over it
is two layers, on the face the whole clamp tips against."""


def _revolved(points: list[tuple[float, float]]) -> Part:
    """A solid of revolution from a (radius, height) profile about Z."""
    return revolve(Plane.XZ * make_face(Polyline(*points, close=True).wire()), Axis.Z)


@dataclass(frozen=True)
class Params:
    """Everything the GUI would expose as a slider."""

    grid: Grid = dog_grid.DEFAULT
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
    root: float = 0.6
    """Radius of the cone at the base of every shank, where it meets whatever it
    grew out of.

    A square root is a stress raiser in layer-line tension, which is where a
    shank fails; this is the fillet, cut as a 45 degree cone because a cone
    still prints and a fillet's underside would not. It costs nothing, because
    it lives inside the lead-in chamfer at the mouth of the hole -- so it takes
    no width off the seat, and being a cone in a cone it centres the dog and
    takes out most of the half-a-clearance the reach would otherwise wander by.
    It has to stay inside that chamfer, which is what ``dog_deck`` is checked
    against."""
    break_edge: float = 0.6
    """Chamfer on the exposed edges of a head."""

    mark: str = ""
    """A word cut into the face that lands on the build plate. Empty by default.

    Only the ladders use it, and they need it: four stops that differ by 0.15mm
    of shank are the same object to look at, so a fit test you cannot read after
    the fact is a fit test you have to run again with calipers -- and not having
    calipers is half of why the ladder exists.

    It is engraved rather than raised because the face it goes on is the one
    against the plate, and nothing can be raised off that. On a stop that face
    is the head's top, so the mark is the side you look at with the dog in the
    deck; on a backer it is the end that goes into the hole first, which is
    exactly when you want to read it."""
    mark_depth: float = 0.5
    mark_font: str = "Helvetica"
    """On the machine this is written for. Anywhere without it the toolkit
    substitutes a sans face and says so, which for a shop label is fine."""

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
    screw_height: float = 10.5
    """Height of the bolt's axis above the deck.

    Wants to be low, because everything about the clamp tipping forward scales
    with it, and is bounded below by the nut trap: the gable stands ``gable``
    above the axis and there has to be real material left between its peak and
    the face the clamp seats on. 10.5 leaves 2.4mm there.

    The cost is that thin work sits under the screw. The bore's underside is
    ``screw_height - bore/2`` off the deck, so a bare PCB or an enclosure lid is
    below the bolt and cannot be clamped by it -- for those, two stops and a
    third pressing down is the answer, not this."""
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
    pad_dimple: float = 1.5
    """Depth of the cone the bolt tip turns in. Shallow: at the bore's own
    radius it only has to centre a tip, and sunk the full radius it would leave
    less than a millimetre of pad under it."""

    # --- the sacrificial backer -------------------------------------------
    puck_entry: float = 0.35
    """Diametral clearance at the bottom of a backing puck, so it starts into
    the hole by hand."""
    puck_grip: float = 0.25
    """Diametral *interference* at the top of a backing puck.

    Measured against the hole, not against a dog's shank. Taking it off the
    shank instead -- which is what this did first -- makes the grip depend on
    the fit the dogs happen to be cut to: at a slip fit it came out at 0.05mm,
    which is printer noise, and at a loose fit the puck was smaller than the
    hole and fell straight through. Through the centre station, which is the
    only one it is ever used at and the one with the casting's open clearance
    hole underneath it."""
    puck_band: float = 2.5
    """How much of a puck's height is the tapered grip, the rest being parallel.

    Spreading the taper over the whole 10mm made seating height forty-odd times
    more sensitive to radial print error than the error itself. Confining it to
    a short band near the top keeps the wedge steep enough that where it stops
    is where it is meant to stop."""

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
    def nut_across_corners(self) -> float:
        return self.nut_af * 2 / math.sqrt(3)

    @property
    def gable(self) -> float:
        """How far the nut trap's peak stands above the bolt's axis.

        Flat up, so the nut's own half-height is the across-flats radius, and
        the 45 degree gable adds half the top edge on top of that. The vertical
        extent is the across-flats one; across corners is the *horizontal*
        measurement on a flat-up hex, which is the trap's width, not its height.
        """
        return self.nut_af / 2 + self.nut_across_corners / 4

    @property
    def reach_window(self) -> tuple[float, float]:
        """Gaps the clamp can close, measured from its front face to the work.

        The pad does not eat into this, which is worth being explicit about
        because it reads as though it should. The pad sits beyond the bolt tip,
        so it shifts the window out by its own thickness rather than shortening
        it: the shortest gap the clamp can close is a pad thickness (tip flush)
        and the longest is that plus the whole travel. The window's *width* is
        the travel, and that is what has to beat the pitch.
        """
        return self.pad_thickness, self.pad_thickness + self.travel

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
        if self.screw_height < self.gable + NUT_COVER:
            raise ValueError(
                f"a bolt axis {self.screw_height}mm up leaves only "
                f"{self.screw_height - self.gable:.2f}mm of material between the peak "
                f"of the nut trap's gable and the face the clamp seats and tips on. "
                f"Raise it to {self.gable + NUT_COVER:.2f} or use a smaller nut."
            )
        if self.screw_height + self.nut_af / 2 >= self.clamp_height:
            raise ValueError(
                f"the nut trap breaks out of the clamp's top face: a flat-up hex "
                f"stands {self.nut_af / 2:.1f}mm above an axis that is only "
                f"{self.clamp_height - self.screw_height:.1f}mm below it"
            )
        if self.nut_deep >= self.clamp_depth / 2:
            raise ValueError("the nut trap runs more than halfway through the clamp")
        near, far = self.reach_window
        if far - near < g.pitch:
            raise ValueError(
                f"the clamp closes gaps from {near:.0f} to {far:.0f}mm, a window "
                f"{far - near:.0f}mm wide, and the rows it can sit in are {g.pitch}mm "
                f"apart. A window narrower than the pitch leaves workpiece sizes that "
                f"fall between two rows: too big for the clamp in one, too small to "
                f"reach from the next. Use a longer bolt or a shallower body."
            )
        if self.root >= self.head / 2 - self.shank / 2:
            raise ValueError(
                f"a {self.root}mm root cone is wider than the {self.head / 2 - self.shank / 2:.2f}mm "
                f"step from shank to head, so it eats the seat it is supposed to sit inside of"
            )
        if self.root <= 0:
            raise ValueError(
                "a square shank root is where a shank breaks, in layer-line tension"
            )
        if self.puck_grip <= 0:
            raise ValueError(
                f"a puck {g.hole - self.puck_grip:.2f}mm across at the top of a "
                f"{g.hole}mm hole does not grip it, it falls through it"
            )
        if self.puck_entry <= 0:
            raise ValueError("a puck has to start into the hole before it wedges")
        if self.mark.strip():
            if self.mark_depth <= 0:
                raise ValueError("a mark with no depth is not a mark")
            if self.mark_depth >= min(self.rise, g.deck) / 2:
                raise ValueError(
                    f"a {self.mark_depth}mm mark is more than half way through the "
                    f"thinnest thing it is cut into"
                )
        if self.puck_band >= g.deck:
            raise ValueError(
                f"a {self.puck_band}mm taper over a {g.deck}mm puck is the whole "
                f"puck, and a taper that shallow makes where it seats a lottery"
            )


def _engrave(body: Part, params: Params, field: float) -> Part:
    """Cut ``params.mark`` into whatever face of ``body`` is on the plate.

    Mirrored, and that is not a detail. The engraved face is the one at z=0,
    which is against the build plate, and every part here is turned over to be
    used -- so a mark that reads the right way round in the model reads
    backwards in the hand. Which horizontal axis it is turned about only sets
    where the word ends up pointing, and on a round dog that is free.
    """
    if not params.mark.strip():
        return body
    from geom.motif import Text

    word = Text(text=params.mark, font=params.mark_font).sketch(field)
    return body - extrude(
        Plane.XY * mirror(word, about=Plane.XZ), amount=params.mark_depth
    )


def _shank(params: Params) -> Part:
    """One shank, base at z=0, growing upward with a lead chamfer on its tip.

    Upward is how they print. In use the part is turned over and this is the bit
    in the hole.

    The cone at the base is the fillet. It flares outward at the very bottom,
    which going up the print is a step inward like everything else here, and it
    sits in the lead-in chamfer at the mouth of the hole rather than on the
    face -- so it buys a filleted root for nothing.
    """
    r, ln, c, k = params.shank / 2, params.shank_length, params.lead, params.root
    return _revolved(
        [(0.0, 0.0), (r + k, 0.0), (r, k), (r, ln - c), (r - c, ln), (0.0, ln)]
    )


def stop(params: Params) -> Part:
    """A plain round dog: the locating element the rest of the set is built on.

    Modelled as printed, which is upside down: z=0 is the head's outer face, the
    seat that lands on the deck is at ``rise``, and the shank runs up from
    there. No facet on it overhangs at all -- every change of diameter is a step
    inward on the way up.
    """
    params.validate()
    rh, rs = params.head / 2, params.shank / 2
    b, c, k = params.break_edge, params.lead, params.root
    top = params.rise + params.shank_length
    body = _revolved(
        [
            (0.0, 0.0),
            (rh - b, 0.0),
            (rh, b),
            (rh, params.rise),
            (rs + k, params.rise),
            (rs, params.rise + k),
            (rs, top - c),
            (rs - c, top),
            (0.0, top),
        ]
    )
    return _engrave(body, params, params.head - 2 * params.break_edge)


def puck(params: Params) -> Part:
    """A sacrificial backer that sits flush in a hole and gets drilled into.

    The reason holes come out of thin aluminium round instead of triangular, and
    out of plastic without a crack running away from them: the bit breaks
    through into something solid rather than into air. It is a consumable.
    Print a dozen; each one is a couple of minutes and a gram.

    Parallel most of the way up and then flared over a short band at the top, so
    it drops in by hand and wedges only over the last couple of millimetres.
    Wider at the top means the drill's own thrust tightens it rather than
    driving it out. Push a spent one through from underneath with the next one
    -- which works because the only station a puck is ever used at is the
    centre one, and that is the only station with the casting's clearance hole
    open beneath it.

    Both diameters come off the hole, never off a dog's shank: a puck sized
    relative to the shank inherits whatever fit the dogs were cut to, and at a
    loose fit that makes it smaller than the hole it is supposed to grip.
    """
    params.validate()
    t, band = params.grid.deck, params.puck_band
    rb = (params.grid.hole - params.puck_entry) / 2
    rt = (params.grid.hole + params.puck_grip) / 2
    b = params.break_edge
    body = _revolved(
        [(0.0, 0.0), (rb - b, 0.0), (rb, b), (rb, t - band), (rt, t), (0.0, t)]
    )
    return _engrave(body, params, 2 * rb - 2 * b)


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

    Takes an M6 x 60 bolt -- a knob or wing head, not a hex, see the module note
    -- and one M6 nut, and nothing else.

    WHICH FACE THE NUT TRAP OPENS ON IS THE WHOLE THING, and it is the front:
    the face towards the work. Trace it. The tip pushes the work forward, so the
    work pushes back on the bolt, so the bolt pulls rearward on the nut. The nut
    therefore needs plastic *behind* it, and gets 24mm of it between the pocket
    and the back face, loaded in compression. Cut into the back face instead --
    which is where this was first drawn, on the reasoning that the nut goes in
    at the end the bolt does -- and the pocket opens in the direction the load
    pushes: the nut walks straight out of it under the first turn of the screw
    and there is nothing left holding the thread. It is also the reason the bolt
    is fitted from the back rather than the front, and a happy side effect that
    a 10mm hex head no longer parks itself inside a 10.3mm pocket.

    So: nut in through the front, bolt in through the back, tip out of the
    front, pad between tip and work.

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
    # an axis screw_height above the deck in use is that far below the top face
    # here, and the gable over the nut points towards the seating face.
    z = h - params.screw_height
    body -= extrude(Plane.XZ * _teardrop(params.bore / 2, (0.0, z)), amount=d, both=True)
    # The trap opens on the front face, and is run 1mm past it so the cut has no
    # face coincident with the body's -- the same reason dog_deck's hole cutter
    # overshoots, and the same sliver it avoids.
    over = 1.0
    body -= Pos(0, d / 2 + over, 0) * extrude(
        Plane.XZ * _hex(params.nut_af, (0.0, z)),
        amount=params.nut_deep + over,
        both=False,
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
    return _revolved(
        [
            (0.0, 0.0),
            (r - b, 0.0),
            (r, b),
            (r, t - b),
            (r - b, t),
            (params.bore / 2, t),
            (0.0, t - params.pad_dimple),
        ]
    )


def capacity(params: Params, rows: int) -> float:
    """The deepest workpiece a deck this many rows deep can actually clamp, mm.

    Fence on the back row, clamp on the front one, bolt wound right out -- less
    the pad, which stands between the tip and the work and so is part of what
    fills the gap. Leaving the pad out of this overstates the capacity by its
    own thickness, which is small and would still be wrong.

    The shallow end needs no arithmetic: the clamp steps back a row at a time
    and its reach window is wider than a pitch, so every size below this one is
    reachable from some row without a gap. That is what the window check in
    ``validate`` is for.
    """
    back = params.grid.span(rows) / 2
    span = (back - params.reach) - (-back + params.clamp_depth / 2)
    return span - params.pad_thickness


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
    return [
        stop(replace(params.at(name), mark=name.upper())) for name in dog_grid.LADDER
    ]


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
