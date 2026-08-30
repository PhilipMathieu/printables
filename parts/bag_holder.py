"""Doo loop: a closed wire form that wedges a tied bag by its handles.

A collar at the top that the lead's handle threads through, a wide opening
under it, and that opening funnelled down into a narrow throat. You push the
tied handles through the opening, pull down, and they wedge where the funnel
gets to their size. There is no hook, no gate and no catch anywhere in it, and
nothing to buy: the collar is a separate body printed inside its own socket in
one go, and it turns.

WHY THE APERTURE IS CLOSED. That is the whole retention argument. The hole the
bag goes into has no way out of it -- it is a hole, not a hook -- so nothing
that is in there can fall out however hard the lead is swung, and the bag comes
off the way it went on, lifted back up the funnel. A hook has to be either easy
to load or hard to unload; a closed hole is both, because the direction that
gets a bag in is a direction gravity never pushes it.

WHY THERE IS A SWIVEL. The funnel only works pointing up, and a collar clamped
round webbing points wherever the webbing does. Webbing twists, but not
locally and not willingly, so a rigid collar spends the walk holding the funnel
over on its side. One turning joint between the collar and the body fixes it:
the collar goes where the lead puts it and the body hangs off it plumb. It buys
one axis rather than the two a ball chain would, so the funnel stays within the
lead's own inclination of upright rather than dead upright -- and near a hand,
which is where this rides, a lead is not far off level.

HOW THE SWIVEL IS PRINTED. The same way the fidget's rings are: both the
collar's rim and the socket it sits in swell by ``interlock`` at mid height and
come back to nominal at the plate and the top, so the collar's widest is wider
than either end of its own socket and it cannot be lifted out. Because both
surfaces are the same profile shifted radially, the gap between them is exactly
``gap`` at every single height, which is the one thing that decides whether a
print-in-place joint comes off the plate turning or comes off fused.

WHAT THE SWIVEL COSTS. The zero-overhang claim, and only that. Everything else
here is a profile extruded straight up, but the swell has to lean: at the
default it leans about 17 degrees off vertical, which is the steepest thing
anywhere in the part and less than half of what FDM bridges unsupported.

WHY THE COLLAR TAKES THE HANDLE FOLDED. Two plies, because that is the only way
onto a lead that does not involve getting past the snap hook. Thread the handle
through and the holder rides free on the lead; pass the rest of the lead back
through the handle first and the same slot is a girth hitch that stays at your
hand.

WHY THE RIBBON IS THE SAME WIDTH EVERYWHERE. It is the aperture offset outward
by ``wall`` and nothing else, which is how a wire form is made and what makes
this one read as bent wire rather than as a shape with a hole in it. It also
means there is nowhere in the part with less section than anywhere else, so
there is nowhere in particular for it to break.

WHY THE APERTURE IS TANGENT-CONTINUOUS. It is a chain of circles and the hulls
between them, so every join in the hole is smooth. A corner in there would be
two things at once: somewhere for a thin plastic handle to snag on the way
down, and a notch for the part to crack from, since the inside of the aperture
is where the ribbon is in tension.

Two joins do not come out smooth on their own. Where the funnel's straight
taper runs into the throat's parallel wall, the taper crosses at the full
funnel angle and leaves a notch exactly where every bundle is dragged past;
and where the collar's ring runs into the shoulders, two circles cross. A
tangent hull cannot fix either, because a hull only ever bulges outward and
both of these turn inward. So both are filleted -- ``blend`` and ``neck`` --
which is the bend radius a wire form would have had anyway.

WHY THE FUNNEL IS CAPPED. The taper is there so that loading the thing does not
involve aiming: anything landed anywhere in the opening is walked down to the
throat by pulling. Past about 55 degrees off the axis the wall a bundle meets
is more across than down, it stops walking anything anywhere, and you are back
to threading a bag into a slot one-handed while a dog pulls.

THE BREAKS. Both ends, so the ribbon's section is an octagon rather than a
rectangle: the first print was sharp in the hand, and the second, broken on the
top alone, still was -- one chamfer on one side does not read as rounded and
eight short faces very nearly do. The plate break is the smaller of the two,
because it comes straight off the first layer's width and this is a hundred and
twenty millimetres of thin ribbon with very little footprint holding it down.

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
from dataclasses import dataclass, replace

from build123d import (
    Axis,
    Circle,
    Kind,
    Part,
    Plane,
    Polygon,
    Pos,
    RectangleRounded,
    Sketch,
    extrude,
    fillet,
    make_face,
    make_hull,
    offset,
    revolve,
)

from geom import webbing
from geom.webbing import Webbing

MIN_WALL = 0.8
"""Two extrusions on the 0.4mm nozzle. Thinner than that the ribbon is a single
bead with no wall either side, and the ribbon is the entire part."""

MIN_SLOT = 3.0
"""Narrower than this the handles have to be threaded into the throat rather
than pulled down it, and not having to aim is the point of a funnel."""

MAX_FUNNEL = 55.0
"""Degrees off the axis. Past this the taper stops being a lead-in."""

MAX_LEAN = 45.0
"""Degrees off vertical for the swivel's swell, which is the only thing in the
part that leans at all. FDM stops holding past this."""


@dataclass(frozen=True)
class Params:
    """Everything the GUI would expose as a slider."""

    strap: Webbing = webbing.STANDARD
    """Lead the collar is cut for. It sets the collar, and the collar sets the
    head; nothing below the neck knows about it."""
    plies: int = 2
    """Layers of webbing the collar's slot has to pass. Two: the folded handle."""

    band: float = 4.0
    """Height of the extrusion: the depth over which the throat pinches the
    handles, the length of the swivel's bearing, and the thickness of the rest.

    Halved off the first print, which came out about twice as thick as it
    needed to be. It buys a steeper swell on the swivel for the same catch --
    see ``lean`` -- and costs out-of-plane stiffness, which goes as the cube of
    this and is not the direction anything is loaded in.
    """
    wall: float = 3.5
    """Width of the ribbon, everywhere. One number, because the ribbon is an
    offset of the aperture and there is no other section in the part."""

    # --- the swivel -------------------------------------------------------

    gap: float = 0.35
    """Clearance between the collar and its socket, radially, at every height.
    The one number that decides whether it comes off the plate turning."""
    interlock: float = 0.7
    """How much both surfaces swell at mid height.

    Down from 1.2 once the plate break started carrying part of the catch. The
    swell's underside is an overhang and the roughest face in the part, and it
    is also the face the joint slides on, so the shallower it can be made the
    better it turns -- which is the complaint this is answering. Engagement
    still comes out ahead of where it was, because the break at each end draws
    the socket's mouth in by more than this gave up."""
    collar_wall: float = 2.6
    """Ribbon around the collar's own slot. Thinner than ``wall`` because it is
    a hoop in compression rather than a link in tension, and because every
    millimetre of it is a millimetre on the head's diameter."""
    slot_clearance: float = 1.1
    """Added to the folded webbing before ``slot_ease``. Swallows the stitching
    down a handle's fold, which is thicker than the webbing it is made of."""
    slot_ease: float = 1.5
    """Multiplies the slot after the clearance. A print says the additive
    figure alone is not enough: working a folded handle through a hole needs
    room in proportion to the strap, not a fixed margin, because it is the
    fold's own bending radius that has to fit and that scales with the webbing.
    Half again in each direction is what a 3/4 in lead actually wanted."""
    slot_corner: float = 1.5
    """Plan radius in the slot's corners. Webbing has rounded edges and a sharp
    internal corner is where a printed part cracks from."""

    # --- the opening ------------------------------------------------------

    crown: float = 9.0
    """Radius of the top of the opening, where it tucks under the collar."""
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
    neck: float = 7.0
    """Fillet where the collar's head meets the shoulders."""
    joint: float = 2.5
    """How far the body's top reaches into the head's ring. Enough to fuse, and
    less than the wall so it never reaches the socket."""
    slice_height: float = 0.2
    """Step the top break is cut in. One printed layer, so it never shows --
    the same argument, and the same number, as the fidget's stacked profile."""
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
    hundred and twenty millimetres of thin ribbon with very little footprint
    holding it to the plate. Set them equal for a properly regular section if
    the plate is behaving; set this to nought to go back to a flat foot.

    It also does something the top break cannot: it relieves the squash at the
    bottom of the swivel. Elephant's foot spreads the collar and its socket
    towards each other across the one gap that has to stay open, and on a 4mm
    part that first layer is a quarter of the whole joint."""

    # --- the collar and its socket ----------------------------------------

    @property
    def slot_width(self) -> float:
        return (self.strap.width + self.slot_clearance) * self.slot_ease

    @property
    def slot_height(self) -> float:
        return (self.strap.stack(self.plies) + self.slot_clearance) * self.slot_ease

    @property
    def slot_diagonal(self) -> float:
        """Corner to corner. What the collar has to be round enough to contain."""
        return math.hypot(self.slot_width, self.slot_height)

    @property
    def collar_radius(self) -> float:
        return self.slot_diagonal / 2 + self.collar_wall

    @property
    def socket_radius(self) -> float:
        return self.collar_radius + self.gap

    @property
    def head_radius(self) -> float:
        return self.socket_radius + self.wall

    @property
    def engagement(self) -> float:
        """How much wider the collar's waist is than the socket's mouth.

        The swell past the clearance, plus whichever break is the shallower:
        the collar leaves by the easier of the two ends, and both ends of the
        socket are drawn in by their own break. Which is why breaking the plate
        edge buys catch rather than costing it.
        """
        return self.interlock - self.gap + min(self.edge_break, self.base_break)

    @property
    def lean(self) -> float:
        """Degrees off vertical of the swell's underside on the collar.

        The face the joint slides on, and an overhang, so also the roughest
        face in the part -- which is why it is worth keeping shallow. Measured
        from where the swell actually starts, which is the top of the plate
        break rather than the plate: breaking that edge shortens the run the
        swell rises over and steepens it.
        """
        return math.degrees(
            math.atan(self.interlock / (self.band / 2 - self.base_break))
        )

    @property
    def ceiling(self) -> float:
        """Degrees off vertical of the socket's roof, above the swell.

        The same cone seen from the other side and from the other end: where
        the collar's upper half faces the sky and prints over nothing, the
        body's socket is a cavity closing in over itself, so it is the upper
        half that overhangs there. It answers to ``edge_break`` the way the
        collar's underside answers to the plate break, and being the steeper of
        the two it is what the printability check has to be run against.
        """
        return math.degrees(
            math.atan(self.interlock / (self.band / 2 - self.edge_break))
        )

    @property
    def bearing_area(self) -> float:
        """Wall the lead's weight lands on inside the collar, in mm^2."""
        return (self.slot_width - 2 * self.slot_corner) * self.band

    # --- the opening ------------------------------------------------------

    @property
    def belly_radius(self) -> float:
        return self.belly / 2

    @property
    def slot_radius(self) -> float:
        return self.slot / 2

    @property
    def ribbon_top(self) -> float:
        return -(self.head_radius - self.joint)

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
    def length(self) -> float:
        return self.head_radius - (self.aperture_bottom - self.wall)

    @property
    def width(self) -> float:
        return max(self.belly + 2 * self.wall, 2 * self.head_radius)

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

    def for_strap(self, name: str) -> Params:
        """The same holder cut for a different lead."""
        return replace(self, strap=webbing.named(name))

    @staticmethod
    def _span(big: float, small: float, angle: float) -> float:
        """Distance between two circles' centres for a tangent line at ``angle``."""
        return (big - small) / math.sin(math.radians(angle))

    def validate(self) -> None:
        if self.plies < 1:
            raise ValueError("a collar that passes no webbing is a hole in a hook")
        if self.wall < MIN_WALL or self.collar_wall < MIN_WALL:
            raise ValueError(
                f"a {min(self.wall, self.collar_wall)}mm ribbon is under "
                f"{MIN_WALL}mm, which is two extrusions on the 0.4mm nozzle. "
                f"Below that it prints as a single bead with no wall either side."
            )
        if self.engagement <= 0:
            raise ValueError(
                f"a {self.interlock}mm swell against a {self.gap}mm gap leaves "
                f"{self.engagement:.2f}mm holding the collar in, which is nothing: "
                f"it lifts straight out of its socket"
            )
        if max(self.lean, self.ceiling) > MAX_LEAN:
            raise ValueError(
                f"swelling {self.interlock}mm leans the joint "
                f"{max(self.lean, self.ceiling):.0f} degrees off vertical -- the "
                f"collar's underside {self.lean:.0f}, the socket's roof "
                f"{self.ceiling:.0f} -- past the {MAX_LEAN:.0f} FDM holds "
                f"unsupported. Raise the band, shrink a break, or swell less and "
                f"take the catch back off the breaks."
            )
        if self.joint >= self.wall:
            raise ValueError(
                f"a {self.joint}mm joint on a {self.wall}mm wall pushes the body "
                f"through the head's ring and into the collar's socket"
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
        if max(self.edge_break, self.base_break) >= self.band / 2:
            raise ValueError(
                f"a {max(self.edge_break, self.base_break)}mm break on a "
                f"{self.band}mm band runs past half its height, so the two ends "
                f"of the swivel's swell meet in the middle and it has no waist "
                f"left to be caught by"
            )
        if max(self.edge_break, self.base_break) >= self.collar_radius:
            raise ValueError(
                f"a {max(self.edge_break, self.base_break)}mm break is deeper "
                f"than the {self.collar_radius:.1f}mm collar is wide"
            )
        if self.slot_ease < 1:
            raise ValueError(
                f"a {self.slot_ease} ease makes the collar's slot smaller than "
                f"the webbing it is cut for"
            )
        if not 0 < self.blend <= self.belly_radius:
            raise ValueError(f"a {self.blend}mm blend is not a bend radius")
        if not 0 < self.neck <= self.head_radius:
            raise ValueError(f"a {self.neck}mm neck fillet is not a bend radius")
        if self.slot_corner > min(self.slot_width, self.slot_height) / 2:
            raise ValueError(
                f"a {self.slot_corner}mm radius in the collar's "
                f"{self.slot_width:.1f} x {self.slot_height:.1f}mm slot leaves no "
                f"flat for the strap to bear on"
            )


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
    """The fixed body's outline: the head's ring, the shoulders, the throat.

    The body is the aperture grown outward by one wall and the aperture taken
    back out of it, which is a constant-width ribbon by construction. The head
    is a disc on top of it -- the socket is cut in three dimensions, because it
    has to swell -- and the two corners where the disc crosses the shoulders
    are filleted into a neck.
    """
    hole = aperture(params)
    body = offset(hole.faces()[0], params.wall, kind=Kind.ARC) - hole
    joined = (body + Circle(params.head_radius)).clean()

    # Where the head's circle crosses the body's outside, solved rather than
    # searched for: two circles of known centres and radii.
    top = params.crown_y + params.crown + params.wall
    dy = -params.crown_y
    reach = (dy**2 + params.head_radius**2 - (params.crown + params.wall) ** 2) / (
        2 * dy
    )
    cross_y = params.crown_y + reach
    cross_x = math.sqrt(max(0.0, params.head_radius**2 - cross_y**2))
    if cross_y >= top:  # the head swallows the crown; nothing to fillet
        return joined
    return fillet(_pick(joined, cross_x, cross_y), radius=params.neck)


def _swell(radius: float, params: Params) -> Part:
    """A disc that is widest at mid height, so it cannot leave its own socket.

    Revolved rather than stacked in slices, because unlike the fidget's rings
    this one is round: a surface of revolution is exact, and it is also the
    only shape a bearing can be if it is going to turn.

    The edge breaks are in the profile rather than cut into it afterwards, and
    that is not tidiness. A rim that is already tapering inward towards the top
    and then has a chamfer taken out of it gets a kink in it where the taper
    suddenly steepens, and a stepped one at that, sitting directly on the
    surface the joint has to turn against. Drawn as one chain of faces it is
    six straight facets from plate to top and the bearing keeps its own shape.

    It also costs nothing in catch. Both breaks fall outside the constriction
    -- the socket's narrowest is still where the swell starts, not at its mouth
    -- so the collar's widest is wider than its socket's tightest by as much as
    it ever was, and the gap between the two is still ``gap`` at every height,
    because both profiles are this same chain shifted radially.
    """
    half = Polygon(
        (0.0, 0.0),
        (radius - params.base_break, 0.0),
        (radius, params.base_break),
        (radius + params.interlock, params.band / 2),
        (radius, params.band - params.edge_break),
        (radius - params.edge_break, params.band),
        (0.0, params.band),
        align=None,
    )
    return revolve(Plane.XZ * half, Axis.Z)


def _break(solid: Part, outline: Sketch, params: Params, grow: bool = False) -> Part:
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
    opinion about how short an edge is allowed to be.

    ``grow`` says which way the boundary runs. An outline is broken by insetting
    it, a hole by letting it out; the collar's slot is the second kind.
    """
    for depth, at_top in ((params.edge_break, True), (params.base_break, False)):
        if depth <= 0:
            continue
        steps = max(1, round(depth / params.slice_height))
        cut = depth / steps
        for k in range(steps):
            back = depth - k * cut if not at_top else (k + 1) * cut
            moved = offset(
                outline, back if grow else -back, kind=Kind.ARC, min_edge_length=0.05
            )
            ring = moved - outline if grow else outline - moved
            z = params.band - depth + k * cut if at_top else k * cut
            solid -= extrude(Plane.XY.offset(z) * ring, amount=cut)
    return solid


def collar(params: Params) -> Part:
    """The turning part: a swollen disc with the lead's slot through it."""
    params.validate()
    slot = RectangleRounded(params.slot_width, params.slot_height, params.slot_corner)
    turning = _swell(params.collar_radius, params) - extrude(
        Plane.XY * slot, amount=params.band
    )
    # Only the slot: the rim's breaks are already in the revolved profile, and
    # cutting them again is exactly what put a kink in the bearing.
    return _break(turning, slot.faces()[0], params, grow=True)


def body(params: Params) -> Part:
    """Everything that does not turn: the head, the shoulders, the throat.

    Chamfered before the socket is cut, deliberately. The socket's narrowest
    place is its two ends, and its ends are what the collar's swell has to be
    wider than; breaking those edges would widen exactly the constriction that
    holds the collar in and give away a third of the catch. They are buried in
    the joint where no thumb reaches anyway.
    """
    face = profile(params).faces()[0]
    outline = _break(extrude(Plane.XY * face, amount=params.band), face, params)
    return outline - _swell(params.socket_radius, params)


def build(params: Params) -> Part:
    """Both bodies, in the places they print in. One plate, one go, no chain."""
    return body(params) + collar(params)


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
    print(f"collar {2 * p.collar_radius:.1f} mm over a {p.slot_width:.1f} x "
          f"{p.slot_height:.1f} mm slot, {p.engagement:.2f} mm engagement, "
          f"{p.lean:.0f} deg lean")
    print(f"wedges bundles from {lo:.0f} to {hi:.0f} mm across")
