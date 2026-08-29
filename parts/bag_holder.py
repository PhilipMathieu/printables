"""Doo loop: a closed wire form that wedges a tied bag by its handles.

An eye at the top for a chain, a wide opening under it, and that opening
funnelled down into a narrow throat. You push the tied handles through the
opening, pull down, and they wedge where the funnel gets to their size. There
is no hook, no gate and no catch anywhere in it.

WHY THE APERTURE IS CLOSED. That is the whole retention argument. The hole the
bag goes into has no way out of it -- it is a hole, not a hook -- so nothing
that is in there can fall out however hard the lead is swung, and the bag comes
off the way it went on, lifted back up the funnel. A hook has to be either easy
to load or hard to unload; a closed hole is both, because the direction that
gets a bag in is a direction gravity never pushes it.

WHY IT HANGS ON A CHAIN AND NOT ON THE LEAD ITSELF. The funnel only works
pointing up. Anything that clamps to the webbing holds the holder square to the
lead, and a lead is at whatever angle the dog has put it -- so the funnel would
spend the walk on its side, where a bag slides across the taper instead of down
it. A ball chain, a split ring or a small carabiner through the eye lets it
hang plumb whatever the lead is doing, and lets it swing out of the way of a
knee. Which is why the top of this part is an eye and not a slot.

WHY IT IS ONE PROFILE, EXTRUDED. Drawn flat, extruded once, printed lying in
the plane it was drawn in: nothing in the part overhangs at all, and the bag
hangs in the profile's own plane, so its weight runs along the extrusions
rather than across them. The moulded original is round rod, which is the one
section that cannot be printed this way -- lying down its whole underside is an
overhang, and standing up the part is a tower of air. The flat ribbon is also
the better wedge: the handles are pinched between two walls the depth of the
band rather than caught between two lines.

WHY THE RIBBON IS THE SAME WIDTH EVERYWHERE. It is the aperture offset outward
by ``wall`` and nothing else, which is how a wire form is made and what makes
this one read as bent wire rather than as a shape with a hole in it. It also
means there is nowhere in the part with less section than anywhere else, so
there is nowhere in particular for it to break.

WHY THE APERTURE IS TANGENT-CONTINUOUS. It is a chain of four circles and the
hulls between them -- crown, belly, throat top, throat bottom -- so every join
in the hole is smooth. A corner in there would be two things at once: somewhere
for a thin plastic handle to snag on the way down, and a notch for the part to
crack from, since the inside of the aperture is where the ribbon is in tension.

One join does not come out smooth on its own, and it is the one that matters
most: where the funnel's straight taper runs into the throat's parallel wall.
A tangent hull cannot fix it, because a hull only ever bulges outward and this
corner turns inward -- the taper crosses the wall at the full funnel angle and
leaves a notch exactly where every bundle is dragged past. So it is filleted,
by ``blend``, which is the bend radius a wire form would have had anyway.

WHY THE FUNNEL IS CAPPED. The taper is there so that loading the thing does not
involve aiming: anything landed anywhere in the opening is walked down to the
throat by pulling. Past about 55 degrees off the axis the wall a bundle meets
is more across than down, it stops walking anything anywhere, and you are back
to threading a bag into a slot one-handed while a dog pulls.

Print it in ASA. This lives outdoors on a lead: UV, cold mornings, and the
pavement every time it is dropped. PLA is the one that goes brittle in the
cold, soft in a car in July, and chalky in a year of sun.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from build123d import (
    Circle,
    Kind,
    Part,
    Plane,
    Pos,
    Sketch,
    extrude,
    fillet,
    make_face,
    make_hull,
    offset,
)

MIN_WALL = 0.8
"""Two extrusions on the 0.4mm nozzle. Thinner than that the ribbon is a single
bead with no wall either side, and the ribbon is the entire part."""

MIN_EYE = 6.0
"""Below this nothing anyone would actually hang it by goes through: a split
ring, a ball chain connector, the gate of a small carabiner."""

MIN_SLOT = 3.0
"""Narrower than this the handles have to be threaded into the throat rather
than pulled down it, and not having to aim is the point of a funnel."""

MAX_FUNNEL = 55.0
"""Degrees off the axis. Past this the taper stops being a lead-in. See the
module note."""


@dataclass(frozen=True)
class Params:
    """Everything the GUI would expose as a slider."""

    eye: float = 8.0
    """Bore of the ring at the top, in mm. Sized for whatever is to hand -- a
    ball chain connector, a split ring, a small carabiner -- because what hangs
    it is not printed and the chain is what lets it hang plumb."""

    band: float = 8.0
    """Height of the extrusion: the depth over which the throat pinches the
    handles, and the thickness of everything else."""
    wall: float = 3.5
    """Width of the ribbon, everywhere. One number, because the ribbon is an
    offset of the aperture and there is no other section in the part."""

    crown: float = 9.0
    """Radius of the top of the opening, where it tucks under the eye. Sets how
    much of a dome the opening has rather than a point."""
    belly: float = 38.0
    """Widest part of the opening. The biggest bundle of handles that can be
    pushed in, and how much room there is for two fingers behind it."""
    shoulder: float = 25.0
    """Degrees off the axis for the flare from crown to belly. Cosmetic more
    than anything -- it is what makes the opening a teardrop and not a circle
    -- but it also sets how much of the part is opening rather than funnel."""

    slot: float = 5.0
    """Width of the throat: what actually pinches the handles. Anything looser
    than its own bundle simply travels to the bottom of the throat, which is
    closed, so a slot that is too wide costs grip and never the bag."""
    funnel: float = 42.0
    """Degrees off the axis for the taper from belly to throat. Steep keeps the
    part short and rounds the bottom of the opening off, because the belly's own
    arc stays exposed further round before the straight run starts; shallow
    guides better and draws the whole thing out. Capped -- see the module note."""
    throat: float = 24.0
    """Parallel length of the throat below the taper."""

    blend: float = 6.0
    """Radius of the fillet where the funnel meets the throat. The one join in
    the aperture that is not tangent by construction. See the module note."""
    tail: float = 14.0
    """Stem below the throat. What two fingers hold while the other hand pushes
    a bag in -- holding the loop itself closes a hand over the opening -- and
    the ballast that keeps the thing hanging plumb on its chain."""
    tail_width: float = 5.0
    joint: float = 2.0
    """How far the body's top reaches into the eye's ring. Enough to fuse, and
    less than the wall so it never reaches the bore."""

    # --- what follows from those ------------------------------------------

    @property
    def eye_radius(self) -> float:
        return self.eye / 2

    @property
    def eye_outer(self) -> float:
        return self.eye_radius + self.wall

    @property
    def belly_radius(self) -> float:
        return self.belly / 2

    @property
    def slot_radius(self) -> float:
        return self.slot / 2

    @property
    def ribbon_top(self) -> float:
        """Top of the body, which sits ``joint`` inside the eye's ring."""
        return -(self.eye_outer - self.joint)

    @property
    def crown_y(self) -> float:
        return self.ribbon_top - self.wall - self.crown

    @property
    def belly_y(self) -> float:
        """Where the opening is widest.

        Set by the flare angle rather than given: on a tangent hull of two
        circles the angle off the axis has ``sin`` equal to the difference in
        radii over the distance between centres, so the distance follows from
        the angle and the two radii.
        """
        return self.crown_y - self._span(self.belly_radius, self.crown, self.shoulder)

    @property
    def throat_top_y(self) -> float:
        return self.belly_y - self._span(
            self.belly_radius, self.slot_radius, self.funnel
        )

    @property
    def throat_bottom_y(self) -> float:
        return self.throat_top_y - self.throat

    @property
    def aperture_bottom(self) -> float:
        return self.throat_bottom_y - self.slot_radius

    @property
    def corner_y(self) -> float:
        """Where the funnel's taper crosses the throat's wall, before blending.

        Below the throat's own top, because the taper is tangent to the throat's
        top circle further round than its equator and so is still outside the
        wall when it gets there.
        """
        lean = math.radians(self.funnel)
        return self.belly_y - (
            self.belly_radius - self.slot_radius * math.cos(lean)
        ) / math.sin(lean)

    @property
    def tail_top(self) -> float:
        """Centre of the tail's top, set so its crown lands mid-way through the
        ribbon's cap: far enough in to fuse, and never poking into the hole."""
        return self.aperture_bottom - self.wall / 2 - self.tail_width / 2

    @property
    def tail_end(self) -> float:
        return self.aperture_bottom - self.wall - self.tail

    @property
    def length(self) -> float:
        """Eye to tail, in mm."""
        return self.eye_outer - self.tail_end

    @property
    def width(self) -> float:
        return self.belly + 2 * self.wall

    @property
    def grip(self) -> tuple[float, float]:
        """Bundles of handles this holds, smallest to largest.

        The largest is the belly, because that is the biggest thing that can be
        pushed into the opening at all. The smallest is the throat: below that
        a bundle stops being pinched and simply lies in the bottom of the
        throat, which is closed, so it is held either way.
        """
        return self.slot, self.belly

    @property
    def blend_start(self) -> float:
        """Height at which the funnel's straight taper gives way to the blend."""
        lean = math.radians(self.funnel)
        return self.corner_y + self.blend * math.tan(lean / 2) * math.cos(lean)

    def touches(self, bundle: float) -> float:
        """Height at which a wedged bundle meets the wall.

        Below its own centre, because the wall leans: the contact is where the
        wall's normal through the centre lands on it. Which wall a bundle is
        actually on is the difference between ``seats`` being exact and being a
        bound -- see there.
        """
        return self.seats(bundle) - bundle / 2 * math.sin(math.radians(self.funnel))

    def seats(self, bundle: float) -> float:
        """Height at which a bundle of handles that many mm across comes to rest.

        In the funnel a bundle stops where it touches both walls at once, and
        the walls there are the straight tangent between the belly and the
        throat, so the answer is exact: a disc of radius p tangent to two lines
        that lean ``funnel`` off the axis sits ``(R - p) / sin(funnel)`` below
        the belly's centre. Below the throat's width there is nothing left to
        wedge against and the bundle lies in the bottom of the throat instead,
        which is closed, so it is held there rather than lost.

        Exact wherever ``touches`` is above ``blend_start``, which is every
        bundle but the smallest few millimetres' worth. Below that the bundle
        is riding the blend rather than the taper, and the blend is material
        added into the corner, so it comes to rest a fraction of a millimetre
        higher than this says. The error is in the direction of less travel,
        never of a bundle going deeper than the arithmetic expects.
        """
        if bundle > self.belly:
            raise ValueError(
                f"a {bundle}mm bundle does not go through a {self.belly}mm belly"
            )
        if bundle <= self.slot:
            return self.aperture_bottom + bundle / 2
        return self.belly_y - self._span(self.belly_radius, bundle / 2, self.funnel)

    @staticmethod
    def _span(big: float, small: float, angle: float) -> float:
        """Distance between two circles' centres for a tangent line at ``angle``."""
        return (big - small) / math.sin(math.radians(angle))

    def validate(self) -> None:
        if self.wall < MIN_WALL:
            raise ValueError(
                f"a {self.wall}mm ribbon is under {MIN_WALL}mm, which is two "
                f"extrusions on the 0.4mm nozzle. Below that it prints as a "
                f"single bead with no wall either side, and the ribbon is the "
                f"whole part."
            )
        if self.eye < MIN_EYE:
            raise ValueError(
                f"a {self.eye}mm eye takes no split ring, no ball chain and no "
                f"carabiner gate worth the name, and this hangs on one of those "
                f"or on nothing"
            )
        if self.joint >= self.wall:
            raise ValueError(
                f"a {self.joint}mm joint on a {self.wall}mm wall pushes the body "
                f"through the eye's ring and into its bore, leaving nothing to "
                f"thread a chain through"
            )
        if self.slot < MIN_SLOT:
            raise ValueError(
                f"a {self.slot}mm throat has to be threaded rather than pulled "
                f"into, which is a two-handed job on a lead that is already in "
                f"one of them"
            )
        if self.slot >= self.belly:
            raise ValueError(
                f"a {self.slot}mm throat under a {self.belly}mm belly is not a "
                f"funnel, it is a slot with a bulge in it"
            )
        if self.crown >= self.belly_radius:
            raise ValueError(
                f"a {self.crown}mm crown radius in a {self.belly}mm belly makes "
                f"the opening flare inward on the way down. Shrink the crown, or "
                f"widen the belly."
            )
        if not 0 < self.funnel <= MAX_FUNNEL:
            raise ValueError(
                f"a {self.funnel} degree funnel is outside the 0 to "
                f"{MAX_FUNNEL} degrees that guides a bundle down rather than "
                f"letting it bridge across"
            )
        if not 0 < self.shoulder < 90:
            raise ValueError(f"a {self.shoulder} degree shoulder is not a flare")
        if self.band < self.slot:
            raise ValueError(
                f"a {self.band}mm band on a {self.slot}mm throat is a pinch "
                f"wider than it is deep, so the bundle rolls out of the side of "
                f"it instead of being held. Give it at least {self.slot}mm."
            )
        if not 0 < self.blend <= self.belly_radius:
            raise ValueError(
                f"a {self.blend}mm blend is not a bend radius a wire would take "
                f"between a {self.funnel} degree taper and a {self.slot}mm throat"
            )
        if self.tail <= self.tail_width - self.wall / 2:
            raise ValueError(
                f"a {self.tail}mm tail is shorter than the {self.tail_width}mm it "
                f"is wide, so it is a bump on the bottom of the throat rather "
                f"than something to hold"
            )
        if self.tail_width > 2 * (self.slot_radius + self.wall):
            raise ValueError(
                f"a {self.tail_width}mm tail is wider than the "
                f"{2 * (self.slot_radius + self.wall):.1f}mm throat it hangs off, "
                f"so it stands proud of the outline instead of continuing it"
            )
        if self.throat <= 0 or self.tail <= 0 or self.crown <= 0:
            raise ValueError("the crown, the throat and the tail all have length")


def _stadium(r1: float, y1: float, r2: float, y2: float) -> Sketch:
    """Tangent hull of two circles on the axis: the shape a wire makes.

    Every straight run in this part is one of these, so every join between a
    run and the circle it came from is tangent rather than a corner.
    """
    discs = (Pos(0, y1) * Circle(r1), Pos(0, y2) * Circle(r2))
    return make_face(make_hull(list(discs[0].edges()) + list(discs[1].edges())))


def aperture(params: Params) -> Sketch:
    """The hole the bag goes into: crown, belly, funnel, throat.

    A chain of four circles hulled in pairs. The first pair flares out to the
    belly, the second tapers in to the throat, and the third -- two circles of
    the same radius -- is the parallel throat itself.
    """
    params.validate()
    chain = (
        (params.crown, params.crown_y),
        (params.belly_radius, params.belly_y),
        (params.slot_radius, params.throat_top_y),
        (params.slot_radius, params.throat_bottom_y),
    )
    hole = Sketch()
    for (r1, y1), (r2, y2) in zip(chain, chain[1:]):
        hole += _stadium(r1, y1, r2, y2)
    hole = hole.clean()

    # The one corner a hull cannot round: pick it out by where it has to be
    # rather than by walking the wire, because the booleans above leave a few
    # sub-tenth-of-a-millimetre seams that are vertices too, and every one of
    # them is tangent and wants leaving alone.
    corners = [
        v
        for v in hole.faces()[0].vertices()
        if abs(abs(v.X) - params.slot_radius) < 1e-3
        and abs(v.Y - params.corner_y) < 0.5
    ]
    if len(corners) != 2:
        raise RuntimeError(
            f"expected the funnel to meet the throat at two corners near "
            f"y={params.corner_y:.2f}, found {len(corners)}"
        )
    return fillet(corners, radius=params.blend)


def profile(params: Params) -> Sketch:
    """The design, as the one face the whole part is extruded from.

    The body is the aperture grown outward by one wall and the aperture taken
    back out of it, which is a constant-width ribbon by construction. The eye
    lands on top of it and the tail hangs off the bottom; ``clean`` drops the
    seams both leave, so what comes back is a single face with exactly two
    holes in it -- the eye's bore and the aperture -- which is the check that
    all three landed on one another.
    """
    hole = aperture(params)
    body = offset(hole.faces()[0], params.wall, kind=Kind.ARC) - hole

    eye = Circle(params.eye_outer) - Circle(params.eye_radius)
    tail = _stadium(
        params.tail_width / 2,
        params.tail_top,
        params.tail_width / 2,
        params.tail_end + params.tail_width / 2,
    )
    return (body + eye + tail).clean()


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
    lo, hi = p.grip
    print(f"valid={part.is_valid} solids={len(part.solids())}")
    print(f"bbox={bb.size.X:.1f} x {bb.size.Y:.1f} x {bb.size.Z:.1f} mm")
    print(f"{p.eye:.0f} mm eye, {p.belly:.0f} mm belly, {p.slot:.0f} mm throat")
    print(f"wedges bundles from {lo:.0f} to {hi:.0f} mm across")
