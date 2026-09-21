"""Desk switch box: a Gardner Bender GSW-117 toggle under a sit-stand desk.

A box that screws to the underside of the desktop at its front edge, with the
toggle coming out of the front face where a hand reaches under the lip, a
5.5 x 2.1mm DC jack and a strain-relieved lead out of the back, and a lid
that comes off for wiring without unscrewing anything from the desk. Two
parts, both printed flat with nothing overhanging: the shell rim-down so its
mounting flanges are the first layer, the lid face-down so the face you see
is the smooth one.

WHY THE TOGGLE THROWS UP AND DOWN. It is the only mapping that needs no
label: push the bat up and the desk goes up. That puts the switch body in the
front wall with its long axis vertical, and it puts a finger *between the
handle and the desk* every time the desk is sent down -- which is the whole
clearance problem. Everything about the box's height follows from that one
gap: the axis sits ``finger`` below the desk plus however far the bat's tip
rises when it is thrown, and the box is exactly as tall as it has to be to
get the switch body in under that.

WHY IT IS STIFF. The GSW-117 is a 20 A switch with a spring that has to
return a bat handle to centre, and it fights back. The load it puts into the
box is along the throw -- vertical, in the plane of the front wall -- which
is the direction a wall is stiffest in, so the wall does not bend; the only
question is whether the box moves on the desk. It cannot, because it hangs
from three flanges and four wood screws with a 45 degree gusset under every
flange, and the clamping face round the bushing is a 4mm wall with the nut on
it. Print the shell with enough perimeters that the wall round the switch is
solid plastic, not two skins over sparse infill: ``SLICE`` below asks for six.

WHY THE JACK IS ON THE BACK. The supply cable and the motor lead both leave
the box away from the knees, and a plug in the back wall is behind everything
a leg does. The jack and the lead exit are on opposite sides of the back wall
so the plug and the tie-down do not fight for the same air.

WHY IT PRINTS RIM-DOWN. The rim is the face that meets the desk, and the
flanges hang off it, so with the rim on the plate the flanges are the first
layer and nothing anywhere overhangs. Every hole in a wall is a teardrop with
its point away from the plate and a 2mm flat across the tip, which the nozzle
bridges without a thought and the nut and washer cover. The one thing the
orientation costs is a floor: an open rim printed downward means an open end
printed upward, and that end is the lid.

WHAT THE HARDWARE IS. The GSW-117 is Gardner Bender's heavy-duty bat-handle
family: 15/32"-32 bushing in a 1/2" hole, 0.57 x 1.13" body, 0.65" toggle,
screw terminals off the back, and a nut and lock ring in the bag. The DPDT
version, GSW-123, is the same body in the same hole, and see the wiring note
in the tool: a single SPDT cannot reverse a motor on its own. The jack is the
threaded 5.5 x 2.1mm panel type sold in fives with a pigtail; they vary
between 11 and 12.3mm across the thread, so measure yours against
``Jack.hole``. The motor is Greartisan's 37mm gearmotor or worm gearmotor,
which only matters to this box as a pair of leads through a notch.

Print it in ASA. A nut torqued onto PLA loosens over a summer as the plastic
creeps out from under it; ASA holds a preload.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from build123d import (
    Box,
    Circle,
    Cone,
    Cylinder,
    Part,
    Plane,
    Polygon,
    Pos,
    Rectangle,
    RectangleRounded,
    Rot,
    Sketch,
    extrude,
)

MM_PER_INCH = 25.4

MAX_LEAN = 45.0
"""Degrees off vertical a downward face may lean unsupported."""

MIN_WALL = 0.8
"""Two extrusions on the 0.4mm nozzle."""

BRIDGE = 2.0
"""Flat across the tip of every teardrop hole, in mm. A bridge this short is
one line the nozzle draws through air, and it lets the hole stop a little
above the circle it started as instead of a full radius above it, so the nut
still covers what the teardrop added."""


@dataclass(frozen=True)
class Switch:
    """A heavy-duty bat-handle toggle, as its panel and its box see it.

    Defaults are the GSW-117 from Gardner Bender's own figures where they give
    them (body, toggle, bushing thread, hole) and the industry-standard
    15/32"-32 family where they do not (bushing length, throw, lock ring).
    """

    name: str = "GSW-117"
    bushing: float = 15 / 32 * MM_PER_INCH
    """Diameter across the thread, 11.9mm."""
    bushing_length: float = 12.0
    """How far the thread stands off the body. Panel plus washer plus nut have
    to fit in it with thread to spare."""
    hole: float = 13.0
    """Panel hole. The nominal is 1/2" (12.7); a printed hole comes out a
    couple of tenths under, and the bushing is 0.8mm under it anyway."""
    nut_flats: float = 9 / 16 * MM_PER_INCH
    """Across the flats of the 15/32"-32 nut, 14.3mm."""
    nut_thickness: float = 3.2
    washer: float = 1.0
    """Lock ring thickness, between the nut and the panel."""
    body_width: float = 0.57 * MM_PER_INCH
    """Across the body, perpendicular to the throw: 14.5mm."""
    body_length: float = 1.13 * MM_PER_INCH
    """Along the throw: 28.7mm."""
    body_depth: float = 30.0
    """Behind the panel to the tips of the terminal screws. Not published;
    generous for this family, which runs 26 to 30."""
    toggle: float = 0.65 * MM_PER_INCH
    """Bat handle, from the panel face to its tip: 16.5mm."""
    toggle_diameter: float = 4.8
    throw: float = 30.0
    """Degrees each side of centre. Heavy-duty toggles run 25 to 30; the
    bigger number is the safe one, since it only ever adds clearance."""
    tab_offset: float = 0.376 * MM_PER_INCH
    """Lock ring's anti-rotation tab, from the bushing centre: 9.55mm. Zero
    leaves the hole out."""
    tab_hole: float = 2.8

    @property
    def tip_rise(self) -> float:
        """How far above its centreline the bat reaches when thrown."""
        return self.toggle * math.sin(math.radians(self.throw)) + self.toggle_diameter / 2

    @property
    def nut_corners(self) -> float:
        """Across the corners of the nut."""
        return self.nut_flats / math.cos(math.radians(30))


@dataclass(frozen=True)
class Jack:
    """A threaded 5.5 x 2.1mm panel-mount DC jack with a pigtail."""

    hole: float = 12.2
    """Clears the 11.4 and 12mm threads these are sold with. Measure yours."""
    thread: float = 11.4
    nut_flats: float = 14.0
    nut_thickness: float = 2.5
    body: float = 12.5
    """Diameter of what sits behind the panel."""
    body_depth: float = 22.0
    """Behind the panel to the end of the solder cups, before the pigtail."""
    plug: float = 35.0
    """Length of a mated plug and its strain relief outside the wall. Only
    the preview draws it."""

    @property
    def nut_corners(self) -> float:
        return self.nut_flats / math.cos(math.radians(30))


GSW_117 = Switch()
GSW_123 = replace(GSW_117, name="GSW-123")
"""The DPDT one. Same body, same hole, and the one that reverses a motor."""


@dataclass(frozen=True)
class Params:
    """Everything the GUI would expose as a slider.

    The frame is the shell's print orientation: Z = 0 is the desk's underside
    and Z grows downward into the room, so "deeper" means further from the
    desk. X runs across the front of the box and Y from the front face back
    under the desk. The lid is modelled in the same frame, in place, and
    flipped for printing.
    """

    switch: Switch = GSW_117
    jack: Jack = Jack()

    finger: float = 22.0
    """Clear space between the thrown-up bat and the desk, for the finger
    that pushes it down. A bare index finger is 16 to 20mm through."""
    width: float = 66.0
    """Inside, across the front."""
    depth: float = 46.0
    """Inside, front wall to back wall."""
    clearance: float = 3.0
    """Air between the switch body and the lid."""

    wall: float = 3.2
    panel: float = 4.0
    """The front wall, which the switch is clamped through."""
    corner: float = 3.0
    """Radius on the shell's vertical corners and the flange's tips."""

    flange: float = 14.0
    """How far the mounting flange reaches out from the side and back walls."""
    flange_thickness: float = 5.0
    gusset: float = 3.0
    """45 degree fillet under each flange, so it prints and so it does not
    hinge at the wall."""
    screw: float = 4.5
    """Clearance for a #8 wood screw."""
    screw_head: float = 8.5
    """A #8 pan head, which the gusset has to stay clear of."""
    screw_inset: float = 10.0
    """Screw centres this far from the front and back ends of the side flanges."""

    lid: float = 3.0
    lip: float = 1.5
    """How far the lid's locating lip reaches into the shell."""
    lip_width: float = 2.0
    lip_clearance: float = 0.25
    boss: float = 7.0
    """Square corner posts the lid screws into, full height for stiffness."""
    boss_hole: float = 2.5
    """Pilot for an M3 thread-forming screw."""
    boss_depth: float = 10.0
    lid_screw: float = 3.4
    countersink: float = 6.5

    jack_x: float = 16.0
    """Jack centre across the back wall, from the middle. Positive is the
    side a right-handed viewer at the desk calls right."""
    jack_depth: float = 20.0
    """Jack centre below the desk."""
    notch_x: float = -16.0
    """Motor lead exit across the back wall, opposite side from the jack."""
    notch: float = 7.0
    """Width of the lead exit; 6mm of two-core cable and a little."""
    notch_height: float = 6.0
    tie_slot: tuple[float, float] = (1.8, 5.2)
    """A cable tie's band, on edge: thickness by width."""
    tie_spread: float = 11.0
    """Between the two tie slots, either side of the lead."""
    tie_back: float = 10.0
    """Tie slots this far in from the back wall's inner face."""

    # --- what follows from those ------------------------------------------

    @property
    def axis_depth(self) -> float:
        """Toggle centre below the desk: the finger, plus the bat's rise."""
        return self.finger + self.switch.tip_rise

    @property
    def height(self) -> float:
        """Shell, rim to open edge. The switch body has to fit under the axis
        with air under it, and then the lid's lip comes in."""
        return (
            self.axis_depth + self.switch.body_length / 2 + self.clearance + self.lip
        )

    @property
    def overall_height(self) -> float:
        """Desk to the outside of the lid."""
        return self.height + self.lid

    @property
    def outer_width(self) -> float:
        return self.width + 2 * self.wall

    @property
    def outer_depth(self) -> float:
        return self.panel + self.depth + self.wall

    @property
    def footprint(self) -> tuple[float, float]:
        """What the desk sees: shell plus flanges."""
        return self.outer_width + 2 * self.flange, self.outer_depth + self.flange

    @property
    def back_inner(self) -> float:
        """Y of the back wall's inside face."""
        return self.panel + self.depth

    @property
    def screw_positions(self) -> tuple[tuple[float, float], ...]:
        """(x, y) of the four wood screws, in the side flanges."""
        x = self.outer_width / 2 + self.flange / 2 + 1.5
        ys = (self.screw_inset, self.outer_depth - self.screw_inset)
        return tuple((sx, y) for sx in (-x, x) for y in ys)

    @property
    def boss_positions(self) -> tuple[tuple[float, float], ...]:
        """(x, y) of the four corner posts."""
        x = self.width / 2 - self.boss / 2
        ys = (self.panel + self.boss / 2, self.back_inner - self.boss / 2)
        return tuple((sx, y) for sx in (-x, x) for y in ys)

    @property
    def tie_y(self) -> float:
        return self.back_inner - self.tie_back

    @property
    def handle_top(self) -> float:
        """Depth of the highest point the bat reaches, thrown toward the desk."""
        return self.axis_depth - self.switch.tip_rise

    def validate(self) -> None:
        sw, jk = self.switch, self.jack
        if self.finger < 15:
            raise ValueError(
                f"{self.finger}mm is not room for a finger between the bat and "
                f"the desk; nothing under 15 is"
            )
        if min(self.wall, self.lid) < MIN_WALL:
            raise ValueError(
                f"a {min(self.wall, self.lid)}mm wall is under {MIN_WALL}mm, "
                f"which is two extrusions on the 0.4mm nozzle"
            )
        if self.panel < self.wall:
            raise ValueError(
                f"the {self.panel}mm front wall is thinner than the {self.wall}mm "
                f"sides, and it is the wall the switch is clamped through"
            )
        thread_left = sw.bushing_length - (self.panel + sw.washer + sw.nut_thickness)
        if thread_left < 1.0:
            raise ValueError(
                f"a {self.panel}mm panel plus the {sw.name}'s washer and nut is "
                f"{self.panel + sw.washer + sw.nut_thickness:.1f}mm on a "
                f"{sw.bushing_length}mm bushing, which leaves "
                f"{thread_left:.1f}mm of thread to bite; thin the panel"
            )
        if sw.hole <= sw.bushing + 0.3:
            raise ValueError(
                f"a {sw.hole}mm hole will not pass an {sw.bushing:.1f}mm bushing "
                f"once the print shrinks it"
            )
        if sw.hole * math.sqrt(2) - BRIDGE / 2 > sw.nut_corners / 2 + sw.hole / 2 + 3:
            raise ValueError(
                f"a {sw.hole}mm teardrop reaches past what a "
                f"{sw.nut_flats:.1f}mm nut covers"
            )
        # The switch has to sit in the wall with the whole nut on flat plastic:
        # nothing above it but wall, nothing below it but wall, bosses clear.
        top = self.axis_depth - sw.nut_corners / 2
        if sw.tab_offset:
            top = min(top, self.axis_depth - sw.tab_offset - sw.tab_hole / 2)
            # The tab is on the desk side of the bushing and the teardrop's
            # point goes the other way, so it is the round edge it has to clear.
            if sw.tab_offset - sw.tab_hole / 2 - sw.hole / 2 < 1.0:
                raise ValueError(
                    f"the lock ring's tab hole at {sw.tab_offset:.1f}mm leaves "
                    f"under a millimetre of wall to the {sw.hole}mm switch hole"
                )
        if top < self.flange_thickness + self.gusset + 1.0:
            raise ValueError(
                f"the switch hole comes within {top:.1f}mm of the desk, inside the "
                f"flange; a bigger finger gap moves it down"
            )
        if sw.body_width / 2 + 2.0 > self.width / 2 - self.boss:
            raise ValueError(
                f"a {sw.body_width:.1f}mm switch body between {self.boss}mm corner "
                f"posts needs the inside wider than {self.width}mm"
            )
        if sw.body_depth + 10.0 > self.depth:
            raise ValueError(
                f"{sw.body_depth}mm of switch behind the panel leaves no room to "
                f"bend a wire in a {self.depth}mm deep box; make it "
                f"{sw.body_depth + 10:.0f} or more"
            )
        # The jack: in the back wall, out of the flange, out of the corner
        # post, and its body clear of the switch body sideways.
        jr = jk.nut_corners / 2
        if self.jack_depth - jr < self.flange_thickness + self.gusset + 1.0:
            raise ValueError(
                f"the jack at {self.jack_depth}mm below the desk puts its nut in "
                f"the flange gusset; drop it to "
                f"{self.flange_thickness + self.gusset + jr + 1:.0f} or more"
            )
        if self.jack_depth + jr > self.height - self.notch_height - 1.0:
            raise ValueError("the jack is too low: its nut reaches the lid edge")
        if abs(self.jack_x) + jr > self.width / 2 - self.boss - 1.0:
            raise ValueError(
                f"the jack at x={self.jack_x} puts its nut into the corner post; "
                f"bring it in or widen the box"
            )
        if abs(self.jack_x) - jk.body / 2 < sw.body_width / 2 + 2.0:
            raise ValueError(
                f"the jack body at x={self.jack_x} sits over the switch body; "
                f"move it out past {sw.body_width / 2 + jk.body / 2 + 2:.0f}"
            )
        if self.jack_x * self.notch_x >= 0:
            raise ValueError("the jack and the lead exit want opposite sides")
        reach = max(self.notch / 2, self.tie_spread / 2 + self.tie_slot[0] / 2)
        if abs(self.notch_x) + reach > self.width / 2 - self.boss - 1.0:
            raise ValueError("the lead exit and its tie slots run into the corner post")
        # Screws and bosses.
        if self.flange < self.gusset + self.screw_head + 1.0:
            raise ValueError(
                f"a {self.flange}mm flange with a {self.gusset}mm gusset has no "
                f"flat left for a {self.screw_head}mm screw head; widen it to "
                f"{self.gusset + self.screw_head + 1:.0f}"
            )
        if self.screw_inset - self.screw / 2 < 1.5:
            raise ValueError("a screw that close to the end of the flange breaks out")
        if self.boss < self.boss_hole + 3.0:
            raise ValueError(
                f"a {self.boss}mm post round a {self.boss_hole}mm pilot is under "
                f"two extrusions a side"
            )
        if self.countersink > 2 * self.boss:
            raise ValueError("the countersink is wider than the post it sits on")
        if self.lip_width + self.lip_clearance > self.boss:
            raise ValueError("the lid's lip is wider than the posts it dodges")
        if self.lip <= 0 or self.lip > self.height / 4:
            raise ValueError(f"a {self.lip}mm lip is not a lip")


# --- 2D pieces ---------------------------------------------------------------


def teardrop(diameter: float, bridge: float = BRIDGE) -> Sketch:
    """A hole for a vertical wall: a circle with a 45 degree roof.

    The roof meets at a flat ``bridge`` wide instead of a point, so the hole
    ends ``bridge / 2`` short of a full teardrop's apex -- less for the nut to
    have to cover, and a bridge that short is nothing to the printer. A hole
    under twice ``bridge`` across is left round: the roof it would get is
    lower than the circle it replaces, and what its top overhangs is a span
    the printer bridges anyway. Local +y is the direction the printer builds
    in.
    """
    r = diameter / 2
    if diameter <= 2 * bridge:
        return Circle(r)
    c = bridge / 2
    s = r / math.sqrt(2)
    roof = Polygon((s, s), (c, r * math.sqrt(2) - c), (-c, r * math.sqrt(2) - c), (-s, s), align=None)
    return Circle(r) + roof


def _footprint(params: Params, grow: float, radius: float) -> Sketch:
    """The shell's plan outline, grown on the sides and back but not the front.

    The front face is where the switch is, and it is flush with the desk's
    edge; flanges and gussets belong on the other three sides.
    """
    w = params.outer_width + 2 * grow
    d = params.outer_depth + grow
    return Pos(0, d / 2) * RectangleRounded(w, d, radius)


def _in_wall(sketch: Sketch, x: float, z: float, y0: float, y1: float) -> Part:
    """Extrude a hole sketch through a Y-facing wall between y0 and y1.

    The sketch's local y is global Z, so a teardrop drawn point-up comes out
    pointing away from the plate.
    """
    plane = Plane.XZ.offset(-(y1 + 0.5))
    return extrude(plane * (Pos(x, z) * sketch), amount=(y1 - y0) + 1.0)


# --- the parts ----------------------------------------------------------------


def shell(params: Params) -> Part:
    """The box, as printed: rim and flanges on the plate, open end up."""
    params.validate()
    p, sw, jk = params, params.switch, params.jack
    h = p.height

    body = extrude(Plane.XY * _footprint(p, 0.0, p.corner), amount=h)
    flange = extrude(Plane.XY * _footprint(p, p.flange, p.corner + 1), amount=p.flange_thickness)
    # A 45 degree taper is a gusset that prints, and one that cannot hinge:
    # extruded from the flange's top, it shrinks by exactly its own height.
    gusset = extrude(
        Plane.XY.offset(p.flange_thickness) * _footprint(p, p.gusset, p.corner + 0.5),
        amount=p.gusset,
        taper=45,
    )
    part = body + flange + gusset

    cavity = Pos(0, p.panel + p.depth / 2, h / 2) * Box(p.width, p.depth, h + 2)
    part -= cavity
    for x, y in p.boss_positions:
        part += Pos(x, y, h / 2) * Box(p.boss, p.boss, h)
        part -= Pos(x, y, h - p.boss_depth / 2 + 0.5) * Cylinder(p.boss_hole / 2, p.boss_depth + 1)
    for x, y in p.screw_positions:
        part -= Pos(x, y, p.flange_thickness / 2) * Cylinder(
            p.screw / 2, p.flange_thickness + p.gusset + 2
        )

    # The switch, through the front wall, and the lock ring's tab above it.
    part -= _in_wall(teardrop(sw.hole), 0.0, p.axis_depth, 0.0, p.panel)
    if sw.tab_offset:
        part -= _in_wall(
            teardrop(sw.tab_hole), 0.0, p.axis_depth - sw.tab_offset, 0.0, p.panel
        )
    # The jack, through the back wall; the lead, out of its open edge.
    part -= _in_wall(teardrop(jk.hole), p.jack_x, p.jack_depth, p.back_inner, p.outer_depth)
    part -= Pos(p.notch_x, p.back_inner + p.wall / 2, h - p.notch_height / 2 + 0.5) * Box(
        p.notch, p.wall + 1.0, p.notch_height + 1.0
    )
    return part


def _lid_in_place(params: Params) -> Part:
    """The lid where it lives: closing the shell's open end, lip inward."""
    p = params
    h, t = p.height, p.lid
    plate = extrude(Plane.XY.offset(h) * _footprint(p, 0.0, p.corner), amount=t)

    c = p.lip_clearance
    ring = Pos(0, p.panel + p.depth / 2) * (
        Rectangle(p.width - 2 * c, p.depth - 2 * c)
        - Rectangle(p.width - 2 * (c + p.lip_width), p.depth - 2 * (c + p.lip_width))
    )
    for x, y in p.boss_positions:
        ring -= Pos(x, y) * Rectangle(p.boss + 2 * c, p.boss + 2 * c)
    # A gap under the lead exit, so the lead lies flat on the lid through
    # the notch instead of climbing the lip.
    ring -= Pos(p.notch_x, p.back_inner - p.lip_width / 2) * Rectangle(
        p.notch + 2 * c, p.lip_width + 2 * c + 1.0
    )
    lip = extrude(Plane.XY.offset(h - p.lip) * ring, amount=p.lip)
    part = plate + lip

    cs_h = (p.countersink - p.lid_screw) / 2  # a 90 degree countersink
    for x, y in p.boss_positions:
        part -= Pos(x, y, h + t / 2) * Cylinder(p.lid_screw / 2, t + 2 * p.lip + 2)
        # Grown by the same amount in radius and height, so it stays a 45
        # degree cone and only its mouth reaches out past the face.
        part -= Pos(x, y, h + t - cs_h / 2 + 0.5) * Cone(
            p.lid_screw / 2, p.countersink / 2 + 1.0, cs_h + 1.0
        )
    sx, sy = p.tie_slot
    for x in (p.notch_x - p.tie_spread / 2, p.notch_x + p.tie_spread / 2):
        part -= Pos(x, p.tie_y, h + t / 2) * Box(sx, sy, t + 2 * p.lip + 2)
    return part


def lid(params: Params) -> Part:
    """The lid, as printed: outside face on the plate, lip and countersinks up.

    Half a turn about Y, not a mirror, so the tie slots and the countersinks
    land where the shell expects them once it is turned back over.
    """
    params.validate()
    p = params
    return Pos(0, 0, p.height + p.lid) * (Rot(0, 180, 0) * _lid_in_place(p))


def assembly(params: Params) -> Part:
    """Shell and lid together, in the shell's frame: what hangs off the desk."""
    return shell(params) + _lid_in_place(params)


# --- the hardware, as solids, for the tests and the preview ---------------------


def switch_body(params: Params) -> Part:
    """The switch at home: bushing through the wall, body behind it, nut on the front."""
    p, sw = params, params.switch
    z = p.axis_depth
    bushing = Pos(0, p.panel - sw.bushing_length / 2, z) * Rot(90, 0, 0) * Cylinder(
        sw.bushing / 2, sw.bushing_length
    )
    body = Pos(0, p.panel + sw.body_depth / 2, z) * Box(sw.body_width, sw.body_depth, sw.body_length)
    return bushing + body


def switch_nut(params: Params) -> Part:
    """Washer and nut stacked on the front face, drawn round at the corners."""
    p, sw = params, params.switch
    stack = sw.washer + sw.nut_thickness
    return Pos(0, -stack / 2, p.axis_depth) * Rot(90, 0, 0) * Cylinder(sw.nut_corners / 2, stack)


def toggle(params: Params, thrown: float = 0.0) -> Part:
    """The bat, pivoting at the panel face; ``thrown`` in degrees, negative
    toward the desk."""
    p, sw = params, params.switch
    bat = Pos(0, -sw.toggle / 2, 0) * Rot(90, 0, 0) * Cylinder(sw.toggle_diameter / 2, sw.toggle)
    return Pos(0, 0, p.axis_depth) * (Rot(thrown, 0, 0) * bat)


def toggle_sweep(params: Params) -> Part:
    """Everywhere the bat ever is: at rest and at either end of its throw."""
    t = params.switch.throw
    return toggle(params) + toggle(params, -t) + toggle(params, t)


def finger(params: Params) -> Part:
    """A finger laid along the desk's underside where it pushes the bat down.

    A cylinder ``finger`` through, tangent to the desk and to the front face,
    lying across the front of the box: it has to clear the thrown-up bat.
    """
    p = params
    return Pos(0, -p.finger / 2, p.finger / 2) * Rot(0, 90, 0) * Cylinder(p.finger / 2, p.width)


def jack_body(params: Params) -> Part:
    """The jack at home: thread through the back wall, body inside, nut outside."""
    p, jk = params, params.jack
    x, z = p.jack_x, p.jack_depth
    thread = Pos(x, p.back_inner + p.wall / 2, z) * Rot(90, 0, 0) * Cylinder(
        jk.thread / 2, p.wall + 2 * jk.nut_thickness
    )
    body = Pos(x, p.back_inner - jk.body_depth / 2, z) * Rot(90, 0, 0) * Cylinder(
        jk.body / 2, jk.body_depth
    )
    nut = Pos(x, p.outer_depth + jk.nut_thickness / 2, z) * Rot(90, 0, 0) * Cylinder(
        jk.nut_corners / 2, jk.nut_thickness
    )
    return thread + body + nut


def screw_heads(params: Params) -> Part:
    """Pan heads sitting on the flanges, where the driver has to get to them."""
    p = params
    heads = Part()
    for x, y in p.screw_positions:
        heads += Pos(x, y, p.flange_thickness + 2.0) * Cylinder(p.screw_head / 2, 4.0)
    return heads


SLICE = {
    # The wall round the switch bushing is where the whole part is loaded, and
    # a 4mm wall at two loops is two skins over sparse infill: the nut crushes
    # it. Six loops make anything under 5mm solid.
    "wall_loops": "6",
    "sparse_infill_density": "20%",
    "enable_support": "0",
}


if __name__ == "__main__":
    from pathlib import Path

    from build123d import export_stl

    p = Params()
    box, cover = shell(p), lid(p)
    out = Path("out")
    out.mkdir(exist_ok=True)
    export_stl(box, str(out / "desk_switch_shell.stl"))
    export_stl(cover, str(out / "desk_switch_lid.stl"))
    for name, part in (("shell", box), ("lid", cover)):
        bb = part.bounding_box()
        print(
            f"{name}: valid={part.is_valid} solids={len(part.solids())} "
            f"{bb.size.X:.1f} x {bb.size.Y:.1f} x {bb.size.Z:.1f} mm"
        )
    print(f"toggle axis {p.axis_depth:.1f} mm below the desk, {p.finger} mm finger gap")
