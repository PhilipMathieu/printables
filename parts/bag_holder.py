"""Doo-loop-style bag holder: a slot for the lead and an open horn for the bag.

You tie the bag off one-handed, drop the knot into the horn, and walk on. That
is the whole brief, and everything below is either the slot, the horn, or the
ribbon joining them.

WHY THE WHOLE PART IS ONE PROFILE. Draw it flat, extrude it once, print it
lying in the plane it was drawn in. Nothing in the part then overhangs at all
-- every face is either a vertical wall or the plate and the top -- and the
lead slot, which would be a bridge in any other orientation, is a plain
vertical hole. It also puts the load in the right place: the bag hangs in the
profile's own plane, so the pull runs along the extrusions rather than across
them, and the layers are never asked to hold the weight apart. The one force
that does try to split them is the bag swinging sideways, and that is a
fraction of it.

That orientation is also why the horn can wrap as far as it likes. The plant
clip's C is capped at 270 degrees because it prints standing up and the tips
lean as they curl; this one is printed lying down, so how far the horn comes
round is a question about holding a bag in and nothing else.

WHY THE GATE IS ABOVE THE HORN'S CENTRE. The bag sits at the bottom of the
seat and gravity keeps it there, so the only opening that cannot let it out is
one the bag would have to climb to reach. Put the whole gate above the seat's
centreline and the bag has to be lifted ``lift`` before there is any way out
at all -- which a walking swing does not do, and a hand does without thinking.
That is retention with no moving part, no catch, and nothing to fatigue: the
gate is a gap that happens to be in the wrong place to be useful to gravity.

WHY THE SLOT IS TWICE THE WEBBING'S THICKNESS. It has to swallow the folded
handle, because that is the only way onto a lead that does not involve getting
past the snap hook. Thread the handle through the slot and the holder rides
free on the lead; pass the rest of the lead back through the handle first and
the same slot becomes a girth hitch that stays put at your hand. Either way it
is the fold, two plies, that decides the slot.

Which leaves the holder loose on a single thickness of webbing, and that is
wanted rather than tolerated. It has to swing to hang plumb and it has to
slide to be pushed out of the way; a slot that gripped the strap would hold
the holder cocked at whatever angle the lead happened to be.

WHY THE STRAP BEARS ON A WALL. The lead's whole weight goes through the top of
the slot, and there that is a flat face as wide as the webbing and as long as
the part is tall -- a couple of hundred square millimetres of printed wall
against a flat strap, not a strap sawing over an edge. Webbing wears against
hard edges; a slot is the shape that gives it none.

Print it in ASA. This lives outdoors on a lead: UV, sub-zero mornings, and the
pavement every time it is dropped. PLA is the one that goes brittle in the
cold, soft in a car in July, and chalky in a year of sun.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from build123d import (
    Circle,
    Part,
    Plane,
    Polygon,
    Pos,
    RectangleRounded,
    Sketch,
    extrude,
)

from geom import webbing
from geom.webbing import Webbing

MIN_WALL = 0.8
"""Two extrusions on the 0.4mm nozzle. A ribbon thinner than that is a single
bead with no wall either side of it, and it is the only thing holding the bag."""

MIN_SEAT = 14.0
"""Below this a tied bag's handles bunch in the seat and there is no room left
to get a finger in, which is what hooking it on one-handed actually takes."""

MIN_GATE = 4.0
"""Narrower than this and the knot has to be threaded rather than pushed, which
is a two-handed job and defeats the point."""

MAX_ESCAPE_BEARING = 90.0
"""Degrees off straight up. The lowest edge of the gate has to stay above the
horn's centreline or the bag can leave without climbing. See the module note."""


@dataclass(frozen=True)
class Params:
    """Everything the GUI would expose as a slider."""

    strap: Webbing = webbing.STANDARD
    """Lead the slot is cut for. Only the head changes with it; the horn holds
    a bag, and a bag is the same size whatever the dog is on."""
    plies: int = 2
    """Layers of webbing the slot has to pass. Two, because that is the folded
    handle, and the handle is how the holder gets onto the lead."""

    band: float = 12.0
    """Height of the extrusion: how much of the lead's length the slot spans,
    and equally how thick the horn is where the bag pulls on it."""
    wall: float = 3.0
    """Width of the ribbon, everywhere. One number, because it is the same
    ribbon all the way round and it sets the strength at every point of it."""

    seat: float = 22.0
    """Diameter of the horn's bore -- where the bag hangs, and where a finger
    has to fit alongside it."""
    gate: float = 9.0
    """Narrowest part of the way in, measured across the seat. The cut faces
    are radial, so the opening widens outward from there and leads the knot in."""
    gate_bearing: float = 55.0
    """Degrees from straight up to the middle of the gate. Retention is this
    number: the whole gate has to stay above the horn's centreline."""

    slot_clearance: float = 1.2
    """Added to the webbing's own size to get the slot. Swallows the stitching
    down a handle's fold, which is thicker than the webbing it is made of."""
    slot_corner: float = 1.5
    """Plan radius in the slot's corners. Webbing has rounded edges and a sharp
    internal corner is where a printed part cracks from."""
    corner: float = 2.5
    """Plan radius on the head's outside corners. Kept modest because the neck
    has to land on the flat between them, and on the narrowest lead in the
    catalogue there is not much flat to land on."""
    neck: float = 2.5
    """Straight run between the head and the horn. Reads as two loops rather
    than one blob, and costs nothing but its own length."""
    neck_width: float = 7.0
    """Width of that run. The bag's whole weight crosses it, in tension, over
    the band -- which at the default is 84mm^2 for a load measured in newtons,
    so this is set by what fits the narrowest head rather than by strength."""
    flare: float = 1.5
    """How much wider the neck's foot is where it lands on the head, each side.
    A taper, not a fillet: it gussets the joint and stays a straight wall."""

    # --- what follows from those ------------------------------------------

    @property
    def slot_width(self) -> float:
        return self.strap.width + self.slot_clearance

    @property
    def slot_height(self) -> float:
        """Across the folded strap: what decides whether it goes on at all."""
        return self.strap.stack(self.plies) + self.slot_clearance

    @property
    def head_width(self) -> float:
        return self.slot_width + 2 * self.wall

    @property
    def head_height(self) -> float:
        return self.slot_height + 2 * self.wall

    @property
    def bearing_area(self) -> float:
        """Flat wall the strap's weight lands on, in mm^2.

        The straight part of the slot's top face, less the corner radii, times
        the band. Webbing wears on edges and not on faces, so this is the
        number that says the slot is not one.
        """
        return (self.slot_width - 2 * self.slot_corner) * self.band

    @property
    def seat_radius(self) -> float:
        return self.seat / 2

    @property
    def outer_radius(self) -> float:
        return self.seat_radius + self.wall

    @property
    def horn_y(self) -> float:
        """Centre of the horn, below the slot's centre at the origin."""
        return -(self.head_height / 2 + self.neck + self.outer_radius)

    @property
    def gate_half(self) -> float:
        """Half the gate, in degrees, subtended at the seat radius."""
        return math.degrees(math.asin(min(1.0, self.gate / (2 * self.seat_radius))))

    @property
    def escape_bearing(self) -> float:
        """Bearing of the lowest point of the gate: the one way the bag leaves."""
        return self.gate_bearing + self.gate_half

    @property
    def lift(self) -> float:
        """How far the bag has to climb out of the seat to reach that point.

        Measured up from the bottom of the seat, where the bag hangs. This is
        the retention, in millimetres.
        """
        return self.seat_radius * (1 + math.cos(math.radians(self.escape_bearing)))

    @property
    def neck_bearing(self) -> float:
        """Angle the neck itself occupies, measured where it is widest.

        The neck reaches into the horn's ring, and inside the ring its edges
        subtend more angle the further in they go, so the seat radius is where
        it takes up the most and the gate has to clear that.
        """
        return math.degrees(
            math.asin(min(1.0, (self.neck_width / 2) / self.seat_radius))
        )

    def for_strap(self, name: str) -> Params:
        """The same holder cut for a different lead."""
        return replace(self, strap=webbing.named(name))

    def validate(self) -> None:
        if self.plies < 1:
            raise ValueError("a slot that passes no webbing is a hole in a hook")
        if self.wall < MIN_WALL:
            raise ValueError(
                f"a {self.wall}mm ribbon is under {MIN_WALL}mm, which is two "
                f"extrusions on the 0.4mm nozzle. Below that it prints as a "
                f"single bead with no wall either side, and it is the only "
                f"thing holding the bag."
            )
        if self.band < self.slot_height:
            raise ValueError(
                f"a {self.band}mm band on a {self.slot_height:.1f}mm slot is a "
                f"bearing taller than it is long, so the holder cocks across the "
                f"strap and jams instead of sliding. Give it at least "
                f"{self.slot_height:.1f}mm."
            )
        if self.seat < MIN_SEAT:
            raise ValueError(
                f"a {self.seat}mm seat is under {MIN_SEAT}mm: the bag's handles "
                f"fill it and leave nothing to get a finger into, which is what "
                f"hooking it on one-handed takes"
            )
        if self.gate < MIN_GATE:
            raise ValueError(
                f"a {self.gate}mm gate has to be threaded rather than pushed "
                f"through, which is a two-handed job on a lead that is already "
                f"in one of them"
            )
        if self.gate >= self.seat:
            raise ValueError(
                f"a {self.gate}mm gate across a {self.seat}mm seat is not a gate, "
                f"it is the open side of a hook"
            )
        if self.escape_bearing >= MAX_ESCAPE_BEARING:
            raise ValueError(
                f"the gate's lower edge sits {self.escape_bearing:.0f} degrees off "
                f"straight up, at or below the horn's centreline, so the bag can "
                f"swing out without ever rising. Bring gate_bearing under "
                f"{MAX_ESCAPE_BEARING - self.gate_half:.0f} degrees, or narrow the "
                f"gate."
            )
        if self.gate_bearing - self.gate_half <= self.neck_bearing:
            raise ValueError(
                f"a gate starting {self.gate_bearing - self.gate_half:.0f} degrees "
                f"off straight up opens into the neck, which reaches "
                f"{self.neck_bearing:.0f} degrees. Swing the gate round, narrow it, "
                f"or thin the neck."
            )
        if self.slot_corner > min(self.slot_width, self.slot_height) / 2:
            raise ValueError(
                f"a {self.slot_corner}mm radius in a "
                f"{self.slot_width:.1f} x {self.slot_height:.1f}mm slot leaves no "
                f"flat for the strap to bear on"
            )
        if self.corner > self.head_height / 2:
            raise ValueError(
                f"a {self.corner}mm corner on a {self.head_height:.1f}mm head "
                f"leaves no straight edge between the corners"
            )
        if self.neck_width / 2 + self.flare > self.head_width / 2 - self.corner:
            raise ValueError(
                f"the neck's foot is {self.neck_width + 2 * self.flare:.1f}mm "
                f"across and lands on the head's rounded corners rather than its "
                f"flat. Narrow the neck, drop the flare, or use a wider lead."
            )


def _gate_cutter(params: Params) -> Sketch:
    """The wedge taken out of the ring, in the horn's own frame.

    Radial faces, so the opening is narrowest at the seat and flares outward
    from there, which is what leads a knot in. Its apex is at the horn's centre,
    inside the seat that has already been removed, so the cut leaves no knife
    edge behind. The outer points are pushed out far enough that the chord
    between them still clears the ring -- a triangle whose corners just reach
    the outer radius would slice a flat across it.
    """
    half = math.radians(params.gate_half)
    reach = 1.25 * params.outer_radius / math.cos(half)
    lo = math.radians(params.gate_bearing) - half
    hi = math.radians(params.gate_bearing) + half
    return Polygon(
        (0.0, 0.0),
        (reach * math.sin(hi), reach * math.cos(hi)),
        (reach * math.sin(lo), reach * math.cos(lo)),
        align=None,
    )


def _head(params: Params) -> Sketch:
    """The slotted end, centred on the origin: the part that is on the lead."""
    outer = RectangleRounded(params.head_width, params.head_height, params.corner)
    slot = RectangleRounded(params.slot_width, params.slot_height, params.slot_corner)
    return outer - slot


def _neck(params: Params) -> Sketch:
    """The trapezoid joining the two, overlapping both so the union is sound.

    It reaches a little way inside the head at the top and a little way into
    the ring's wall at the bottom; neither end is a face of the finished part.
    """
    top = -params.head_height / 2 + params.corner
    bottom = params.horn_y + params.seat_radius + params.wall / 2
    foot = params.neck_width / 2 + params.flare
    return Polygon(
        (-foot, top),
        (-params.neck_width / 2, bottom),
        (params.neck_width / 2, bottom),
        (foot, top),
        align=None,
    )


def _horn(params: Params) -> Sketch:
    """The open ring the bag hangs in, in place under the head."""
    ring = Circle(params.outer_radius) - Circle(params.seat_radius)
    return Pos(0, params.horn_y) * (ring - _gate_cutter(params))


def profile(params: Params) -> Sketch:
    """The design, as the one face the whole part is extruded from.

    Every polygon in this module is wound anticlockwise, which is not a style
    choice: a clockwise face carries a normal pointing into the bed, and it
    both extrudes the wrong way and refuses to fuse with the faces beside it,
    leaving three loose prisms where the part should be. ``clean`` then drops
    the seams the fuse leaves behind, so what comes back is a single face with
    the slot as its only hole -- which is the check that the head, the neck and
    the horn really did land on one another.
    """
    params.validate()
    return (_head(params) + _neck(params) + _horn(params)).clean()


def build(params: Params) -> Part:
    """One profile, extruded ``band`` up off the plate. That is the entire part."""
    return extrude(Plane.XY * profile(params), amount=params.band)


if __name__ == "__main__":
    from pathlib import Path

    from build123d import export_stl

    p = Params()
    part = build(p)
    out = Path("out")
    out.mkdir(exist_ok=True)
    export_stl(part, str(out / "bag_holder.stl"))
    bb = part.bounding_box()
    print(f"valid={part.is_valid} solids={len(part.solids())}")
    print(f"bbox={bb.size.X:.1f} x {bb.size.Y:.1f} x {bb.size.Z:.1f} mm")
    print(f"slot {p.slot_width:.1f} x {p.slot_height:.1f} mm for {p.strap.nominal} "
          f"webbing, {p.plies} plies")
    print(f"seat {p.seat:.0f} mm, gate {p.gate:.0f} mm, {p.lift:.1f} mm lift to escape")
