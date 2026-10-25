"""Doo loop: a closed wire form that wedges a tied bag by its handles.

A small loop at the top for a mini carabiner, a wide opening under it, and that
opening funnelled down into a narrow throat. You push the tied handles through
the opening, pull down, and they wedge where the funnel gets to their size.
There is no hook, no gate and no catch anywhere in it, and nothing moves.

WHY THE APERTURE IS CLOSED. That is the whole retention argument. The hole the
bag goes into has no way out of it -- it is a hole, not a hook -- so nothing
that is in there can fall out however hard the lead is swung, and the bag comes
off the way it went on, lifted back up the funnel. A hook has to be either easy
to load or hard to unload; a closed hole is both, because the direction that
gets a bag in is a direction gravity never pushes it.

WHY IT HANGS FROM A LOOP. The funnel only works pointing up, and anything
clamped round webbing points wherever the webbing does. The last version fixed
that with a print-in-place swivel between a collar on the handle and the body,
and in use it turned out not to be needed: a mini carabiner through a loop does
the same job with nothing to print-in-place. The holder pivots on the
carabiner's rod in its own plane, the carabiner turns on whatever it is clipped
to -- the handle, a D-ring, a bag strap -- and between the two the funnel hangs
plumb. It also comes off the lead with one thumb rather than being threaded
onto the handle for good.

WHY THE LOOP IS SMALL. It only has to pass a mini carabiner's rod, which is
four or five millimetres on the keychain sizes. It is sized a few
millimetres over that to be safe -- carabiners vary and the bore must never be
the thing that does not fit -- and no bigger, because past that it is just
more of the part swinging about. ``eye`` is the bore; the ribbon round it is
the same ``wall`` as everywhere else, and that section -- ``wall`` by ``band``
-- is what has to go through the carabiner's gate, so it is checked against
``MINI_GATE``.

WHY THE RIBBON IS THE SAME WIDTH EVERYWHERE. It is the aperture offset outward
by ``wall`` and nothing else, which is how a wire form is made and what makes
this one read as bent wire rather than as a shape with a hole in it. The loop
is the same ribbon bent round its own bore. It also means there is nowhere in
the part with less section than anywhere else, so there is nowhere in
particular for it to break.

WHY THE APERTURE IS TANGENT-CONTINUOUS. It is a chain of circles and the hulls
between them, so every join in the hole is smooth. A corner in there would be
two things at once: somewhere for a thin plastic handle to snag on the way
down, and a notch for the part to crack from, since the inside of the aperture
is where the ribbon is in tension.

Two joins do not come out smooth on their own. Where the funnel's straight
taper runs into the throat's parallel wall, the taper crosses at the full
funnel angle and leaves a notch exactly where every bundle is dragged past;
and where the loop's ring runs into the crown, two circles cross. A tangent
hull cannot fix either, because a hull only ever bulges outward and both of
these turn inward. So both are filleted -- ``blend`` and ``neck`` -- which is
the bend radius a wire form would have had anyway.

WHY THE FUNNEL IS CAPPED. The taper is there so that loading the thing does not
involve aiming: anything landed anywhere in the opening is walked down to the
throat by pulling. Past about 55 degrees off the axis the wall a bundle meets
is more across than down, it stops walking anything anywhere, and you are back
to threading a bag into a slot one-handed while a dog pulls.

WHY IT IS ONE PROFILE, EXTRUDED. Drawn flat, extruded once, printed lying in
the plane it was drawn in, so nothing in it overhangs but the plate break --
and the bag hangs in the profile's own plane, so its weight runs along the
extrusions rather than across them. Losing the swivel gave this back: the
swell that held the collar in its socket was the only thing that leaned.

THE BREAKS. Both ends, so the ribbon's section is an octagon rather than a
rectangle: the first print was sharp in the hand, and the second, broken on the
top alone, still was -- one chamfer on one side does not read as rounded and
eight short faces very nearly do. The plate break is the smaller of the two,
because it comes straight off the first layer's width and this is a hundred
millimetres of thin ribbon with very little footprint holding it down. The
loop's bore is broken too, which is what the carabiner's rod bears on.

Both are cut as a stack of insets one printed layer high rather than as
chamfers, because the unions that build the aperture leave seams a few
thousandths of a millimetre long and no chamfer will run across one -- see
``_break``. That leaves the plate break's risers as flat downward ledges, one
layer's inset each, which is exactly what a slicer makes of any chamfer.

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

MIN_SLOT = 3.0
"""Narrower than this the handles have to be threaded into the throat rather
than pulled down it, and not having to aim is the point of a funnel."""

MAX_FUNNEL = 55.0
"""Degrees off the axis. Past this the taper stops being a lead-in."""

MINI_ROD = 5.0
"""Rod of the stoutest keychain-size carabiner, in mm. The loop's bore has to
take it with a millimetre to spare, or the holder stops hanging plumb and
starts hanging however it was last pushed."""

MINI_GATE = 6.0
"""Gate opening of the smallest carabiner worth calling one, in mm. The loop's
ribbon has to go through it sideways to be clipped on at all."""


@dataclass(frozen=True)
class Params:
    """Everything the GUI would expose as a slider."""

    eye: float = 9.0
    """Bore of the loop at the top, in mm. A mini carabiner's rod is four or
    five millimetres; this is twice the millimetre ``MINI_ROD`` asks for over
    the stoutest of them, so it swings freely on whatever is to hand and still
    takes a rod a little fatter than the keychain sizes."""

    band: float = 4.0
    """Height of the extrusion: the depth over which the throat pinches the
    handles, and the thickness of the rest.

    Halved off the first print, which came out about twice as thick as it
    needed to be. It costs out-of-plane stiffness, which goes as the cube of
    this and is not the direction anything is loaded in.
    """
    wall: float = 3.5
    """Width of the ribbon, everywhere -- round the loop included. One number,
    because the ribbon is an offset of the aperture and there is no other
    section in the part."""

    # --- the opening ------------------------------------------------------

    crown: float = 9.0
    """Radius of the top of the opening, where it tucks under the loop."""
    belly: float = 36.0
    """Widest part of the opening. The biggest bundle of handles that can be
    pushed in, and how much room there is for two fingers behind it."""
    shoulder: float = 25.0
    """Degrees off the axis for the flare from crown to belly."""
    cheek: float = 4.0
    """How much wider than the straight flare the shoulders bow out, each side.
    Nothing structural: it is the difference between a kite and something with
    a bit of life in it, and it costs a millimetre or two of width."""

    slot: float = 5.0
    """Width of the throat: what actually pinches the handles. Anything looser
    than its own bundle simply travels to the bottom of the throat, which is
    closed, so a slot that is too wide costs grip and never the bag."""
    funnel: float = 42.0
    """Degrees off the axis for the taper from belly to throat."""
    throat: float = 22.0
    """Parallel length of the throat below the taper."""

    blend: float = 6.0
    """Fillet where the funnel meets the throat."""
    neck: float = 3.0
    """Fillet where the loop's ring meets the crown."""
    joint: float = 2.5
    """How far the body's top reaches into the loop's ring. Enough to fuse, and
    less than the wall so it never reaches the bore."""
    slice_height: float = 0.2
    """Step the breaks are cut in. One printed layer, so it never shows -- the
    same argument, and the same number, as the fidget's stacked profile."""
    edge_break: float = 0.9
    """Chamfer along the top edges."""
    base_break: float = 0.6
    """Chamfer along the plate edges. Together with ``edge_break`` this makes
    the ribbon's section an octagon rather than a rectangle with its top two
    corners knocked off, which is what a print said it wanted: one chamfer on
    one side does not read as rounded in the hand, and eight short faces very
    nearly do.

    Smaller than the top, and that is the one real cost of breaking this edge
    at all: it comes straight off the first layer's width, and this part is a
    hundred millimetres of thin ribbon with very little footprint holding it to
    the plate. Set them equal for a properly regular section if the plate is
    behaving; set this to nought to go back to a flat foot."""

    # --- the loop ---------------------------------------------------------

    @property
    def eye_radius(self) -> float:
        return self.eye / 2

    @property
    def eye_outer(self) -> float:
        return self.eye_radius + self.wall

    # --- the opening ------------------------------------------------------

    @property
    def belly_radius(self) -> float:
        return self.belly / 2

    @property
    def slot_radius(self) -> float:
        return self.slot / 2

    @property
    def ribbon_top(self) -> float:
        """Top of the body, which sits ``joint`` inside the loop's ring."""
        return -(self.eye_outer - self.joint)

    @property
    def crown_y(self) -> float:
        return self.ribbon_top - self.wall - self.crown

    @property
    def cheek_radius(self) -> float:
        return (self.crown + self.belly_radius) / 2 + self.cheek

    @property
    def cheek_y(self) -> float:
        """Half way down the flare, where the bow is widest."""
        return self.crown_y - self._span(self.cheek_radius, self.crown, self.shoulder)

    @property
    def belly_y(self) -> float:
        """Where the opening is widest.

        Set by the flare angle rather than given: on a tangent hull of two
        circles the angle off the axis has ``sin`` equal to the difference in
        radii over the distance between centres, so the distance follows from
        the angle and the two radii.
        """
        return self.cheek_y - self._span(
            self.belly_radius, self.cheek_radius, self.shoulder
        )

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
        """Where the funnel's taper crosses the throat's wall, before blending."""
        lean = math.radians(self.funnel)
        return self.belly_y - (
            self.belly_radius - self.slot_radius * math.cos(lean)
        ) / math.sin(lean)

    @property
    def neck_corner(self) -> tuple[float, float]:
        """Where the loop's outside crosses the crown's, before filleting.

        Two circles of known centres and radii, so solved rather than searched
        for: the chord between their crossings sits this far below the loop's
        centre.
        """
        d = -self.crown_y
        outer = self.crown + self.wall
        y = (d**2 + self.eye_outer**2 - outer**2) / (2 * d)
        return math.sqrt(max(0.0, self.eye_outer**2 - y**2)), -y

    @property
    def length(self) -> float:
        return self.eye_outer - (self.aperture_bottom - self.wall)

    @property
    def width(self) -> float:
        return max(self.belly + 2 * self.wall, 2 * self.eye_outer)

    @property
    def grip(self) -> tuple[float, float]:
        """Bundles of handles this wedges, smallest to largest."""
        return self.slot, self.belly

    @property
    def blend_start(self) -> float:
        lean = math.radians(self.funnel)
        return self.corner_y + self.blend * math.tan(lean / 2) * math.cos(lean)

    def touches(self, bundle: float) -> float:
        """Height at which a wedged bundle meets the wall."""
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

        Exact wherever ``touches`` is above ``blend_start``. Below that the
        bundle rides the blend rather than the taper and comes to rest a
        fraction higher than this says -- an error in the direction of less
        travel, never of a bundle going deeper than the arithmetic expects.
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
                f"single bead with no wall either side."
            )
        if self.eye < MINI_ROD + 1:
            raise ValueError(
                f"a {self.eye}mm loop leaves a {MINI_ROD}mm carabiner rod no "
                f"room to swing in, so it hangs however it was last pushed"
            )
        if min(self.wall, self.band) >= MINI_GATE:
            raise ValueError(
                f"a {self.wall} x {self.band}mm ribbon does not go through a "
                f"{MINI_GATE}mm carabiner gate either way up, so there is "
                f"nothing to clip it on with"
            )
        if self.joint >= self.wall:
            raise ValueError(
                f"a {self.joint}mm joint on a {self.wall}mm wall pushes the body "
                f"through the loop's ring and into its bore"
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
        if self.crown >= self.cheek_radius or self.cheek_radius >= self.belly_radius:
            raise ValueError(
                f"a {self.crown}mm crown, a {self.cheek_radius:.1f}mm cheek and a "
                f"{self.belly_radius}mm belly do not widen in that order, so the "
                f"opening flares inward somewhere on the way down"
            )
        if not 0 < self.funnel <= MAX_FUNNEL:
            raise ValueError(
                f"a {self.funnel} degree funnel is outside the 0 to "
                f"{MAX_FUNNEL} degrees that guides a bundle down rather than "
                f"letting it bridge across"
            )
        if not 0 < self.shoulder < 90:
            raise ValueError(f"a {self.shoulder} degree shoulder is not a flare")
        if min(self.edge_break, self.base_break) < 0:
            raise ValueError("a negative chamfer is a burr")
        if self.edge_break + self.base_break >= self.wall:
            raise ValueError(
                f"breaks of {self.edge_break} and {self.base_break}mm take the "
                f"whole of a {self.wall}mm ribbon from both sides at once, "
                f"leaving no flat on top of it and no wall to speak of"
            )
        if self.edge_break + self.base_break >= self.band:
            raise ValueError(
                f"breaks of {self.edge_break} and {self.base_break}mm meet in the "
                f"middle of a {self.band}mm band, leaving it no side wall"
            )
        if not 0 < self.blend <= self.belly_radius:
            raise ValueError(f"a {self.blend}mm blend is not a bend radius")
        if not 0 < self.neck <= self.eye_outer:
            raise ValueError(f"a {self.neck}mm neck fillet is not a bend radius")


def _stadium(r1: float, y1: float, r2: float, y2: float) -> Sketch:
    """Tangent hull of two circles on the axis: the shape a wire makes."""
    discs = (Pos(0, y1) * Circle(r1), Pos(0, y2) * Circle(r2))
    return make_face(make_hull(list(discs[0].edges()) + list(discs[1].edges())))


def _pick(sketch: Sketch, x: float, y: float, near: float = 0.5):
    """Corners at a computed place, both sides of the axis.

    By where they have to be rather than by walking the wire: the booleans
    leave a few sub-tenth-of-a-millimetre seams that are vertices too, and
    every one of those is tangent and wants leaving alone.
    """
    found = [
        v
        for v in sketch.faces()[0].vertices()
        if abs(abs(v.X) - x) < 1e-3 and abs(v.Y - y) < near
    ]
    if len(found) != 2:
        raise RuntimeError(
            f"expected two corners at x=±{x:.2f}, y={y:.2f}; found {len(found)}"
        )
    return found


def aperture(params: Params) -> Sketch:
    """The hole the bag goes into: crown, cheeks, belly, funnel, throat.

    A chain of five circles hulled in pairs. The cheek is only there to bow the
    flare out; take it away and the two straight runs become one and the
    opening is a kite.
    """
    params.validate()
    chain = (
        (params.crown, params.crown_y),
        (params.cheek_radius, params.cheek_y),
        (params.belly_radius, params.belly_y),
        (params.slot_radius, params.throat_top_y),
        (params.slot_radius, params.throat_bottom_y),
    )
    hole = Sketch()
    for (r1, y1), (r2, y2) in zip(chain, chain[1:]):
        hole += _stadium(r1, y1, r2, y2)
    hole = hole.clean()
    return fillet(_pick(hole, params.slot_radius, params.corner_y), radius=params.blend)


def profile(params: Params) -> Sketch:
    """The whole outline: the loop, the crown, the shoulders, the throat.

    The body is the aperture grown outward by one wall and the aperture taken
    back out of it, which is a constant-width ribbon by construction. The loop
    is a disc on top of it, its two corners with the crown filleted into a
    neck, and its bore taken out last so the fillet never reaches it.
    """
    hole = aperture(params)
    body = offset(hole.faces()[0], params.wall, kind=Kind.ARC) - hole
    joined = (body + Circle(params.eye_outer)).clean()
    joined = fillet(_pick(joined, *params.neck_corner), radius=params.neck)
    return joined - Circle(params.eye_radius)


def _break(solid: Part, outline: Sketch, params: Params) -> Part:
    """Chamfer the top and plate edges of ``outline``, as a stack of insets.

    OCCT will not chamfer the body's top wire, and it is worth saying why
    rather than reaching for a smaller number: the unions that build the
    aperture leave a dozen seams a few thousandths of a millimetre long, and a
    chamfer cannot be run across one. Dropping them from the selection does not
    help either, because the chain then has gaps in it.

    So the break is cut the way the fidget builds its bulge -- as thin slabs of
    a true offset, one printed layer each. Every slab is an exact offset of the
    face above it, the staircase is at the layer height the slicer would have
    discretised to anyway, and none of it depends on an operation that has an
    opinion about how short an edge is allowed to be. Insetting a face lets
    its holes out by the same amount, so the aperture and the loop's bore are
    broken in the same cut as the outside.
    """
    for depth, at_top in ((params.edge_break, True), (params.base_break, False)):
        if depth <= 0:
            continue
        steps = max(1, round(depth / params.slice_height))
        cut = depth / steps
        for k in range(steps):
            back = depth - k * cut if not at_top else (k + 1) * cut
            moved = offset(outline, -back, kind=Kind.ARC, min_edge_length=0.05)
            z = params.band - depth + k * cut if at_top else k * cut
            solid -= extrude(Plane.XY.offset(z) * (outline - moved), amount=cut)
    return solid


def build(params: Params) -> Part:
    """The profile extruded and broken at both ends. One body, nothing moving."""
    face = profile(params).faces()[0]
    return _break(extrude(Plane.XY * face, amount=params.band), face, params)


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
    print(f"valid={part.is_valid} bodies={len(part.solids())}")
    print(f"bbox={bb.size.X:.1f} x {bb.size.Y:.1f} x {bb.size.Z:.1f} mm")
    print(f"{p.eye:.1f} mm loop for a mini carabiner")
    print(f"wedges bundles from {lo:.0f} to {hi:.0f} mm across")
