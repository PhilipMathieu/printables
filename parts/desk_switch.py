"""Desk switch box: a KCD4 rocker under a sit-stand desk.

A box that screws to the underside of the desktop at its front edge, with a
snap-in KCD4 rocker in the front face where a hand reaches under the lip, a
5.5 x 2.1mm DC jack and a strain-relieved lead out of the back, and a lid
that comes off for wiring without unscrewing anything from the desk. Two
parts, both printed flat with no supports: the shell rim-down so its
mounting flanges are the first layer, the lid face-down so the face you see
is the smooth one.

WHY THE ROCKER ROCKS UP AND DOWN. It is the only mapping that needs no
label: press the top and the desk goes up, press the bottom and it comes
down. That stands the rocker's long axis vertical in the front wall, and it
puts a fingertip on the top half of the rocker just under the desk's edge,
which is what sets the box's height: the bezel sits ``headroom`` below the
desk, and the box is exactly as tall as it takes to get the switch body in
under that.

WHY THE WALL IS THIN ROUND THE HOLE. A KCD4 has no nut. It snaps in, and
its spring clips -- on the short ends of the body, top and bottom here --
only grab a panel 1 to 3mm thick. The front wall is 4mm so the rest of it is
stiff, and a pocket on its inside face takes it down to ``clip_panel`` round
the cutout, with room past the cutout's desk end for the top clip to spring
back out behind the web. The pocket runs on out of the shell's open edge, so
it has no far end to overhang and the bottom clip has the box to spring into.

WHY IT IS STIFF. Pressing a rocker pushes into the front wall, and the bezel
spreads that over the face round the hole; the box resists it as a whole,
hanging from three flanges and four wood screws with a 45 degree gusset
under every flange. ``SLICE`` asks for six wall loops so the web and the
wall round it are solid plastic rather than two skins over infill.

WHY THE JACK IS ON THE BACK. The supply cable and the motor lead both leave
the box away from the knees. The jack and the lead exit are on opposite
sides of the back wall so the plug and the tie-down do not fight for the
same air, and the jack sits off to the side of the switch so its body and
the switch's crimped terminals do not either.

WHY IT PRINTS RIM-DOWN. The rim is the face that meets the desk, and the
flanges hang off it, so with the rim on the plate the flanges are the first
layer. The jack hole is a teardrop with a 2mm flat across its tip, which the
nut covers. The rocker's cutout is a rectangle, because the bezel only
overlaps it by a millimetre and a teardrop would show; its far end is a
straight bridge across the 2mm web, which a P2S draws without sagging.

WHAT THE HARDWARE IS. The switch is the 6-pin KCD4 sold as (on)-off-(on)
momentary DPDT -- VEXUNGA's pack of six among many -- rated 16 A 250 V. KCD4
is a family name, not a drawing: cutouts run 26 x 22 to 30 x 22mm between
makers, so measure the body behind the bezel and set ``Rocker.cutout``, or
print the ``coupon`` first. The jack is the threaded 5.5 x 2.1mm panel type
sold in fives with a pigtail; they vary between 11 and 12.3mm across the
thread, so measure yours against ``Jack.hole``. The motor is Greartisan's
37mm gearmotor or worm gearmotor, which only matters to this box as a pair
of leads through a notch.

Print it in ASA: the clips hold the switch by pressing on the web for as
long as it is in, and ASA keeps that grip through a summer where PLA creeps.
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
class Rocker:
    """A snap-in rocker switch, as its panel and its box see it.

    Defaults are the 30 x 22mm-cutout KCD4 most US sellers list for the 16 A
    6-pin parts. Every "length" runs along the rocking direction, which in
    this box is vertical; every "width" is across it.
    """

    name: str = "KCD4"
    cutout: tuple[float, float] = (22.2, 30.2)
    """Panel hole, width by length. Nominal 22 x 30 plus two tenths, which a
    printed hole gives back; the clips want it snug, not loose."""
    bezel: tuple[float, float] = (25.0, 32.0)
    """The flange on the front of the panel, width by length."""
    bezel_thickness: float = 2.5
    rocker_rise: float = 6.0
    """How far the rocker's pressed-out end stands proud of the bezel."""
    body: tuple[float, float] = (21.8, 29.8)
    """Housing behind the panel, width by length: the cutout, less a bit."""
    body_depth: float = 29.0
    """Behind the panel to the tips of the 6.3mm tabs."""
    clips: float = 2.0
    """How far each clip springs out past the cutout's end, behind the panel."""
    panel_range: tuple[float, float] = (1.0, 3.0)
    """Panel thicknesses the clips grab."""


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


KCD4 = Rocker()
KCD4_26 = replace(KCD4, name="KCD4 26x22", cutout=(22.2, 26.2), bezel=(25.0, 28.0), body=(21.8, 25.8))
"""The other common KCD4, a 26 x 22mm cutout."""


@dataclass(frozen=True)
class Params:
    """Everything the GUI would expose as a slider.

    The frame is the shell's print orientation: Z = 0 is the desk's underside
    and Z grows downward into the room, so "deeper" means further from the
    desk. X runs across the front of the box and Y from the front face back
    under the desk. The lid is modelled in the same frame, in place, and
    flipped for printing.
    """

    switch: Rocker = KCD4
    jack: Jack = Jack()

    headroom: float = 10.0
    """Desk to the top of the bezel: room for the knuckle above a fingertip
    pressing the rocker's top half."""
    finger: float = 18.0
    """A fingertip, through. Only checked: it has to fit between the desk and
    the middle of the rocker's top half."""
    width: float = 74.0
    """Inside, across the front."""
    depth: float = 48.0
    """Inside, front wall to back wall."""
    clearance: float = 3.0
    """Air between the switch body and the lid."""
    wiring: float = 16.0
    """Behind the switch's tabs: a crimped quick-connect and the bend in its
    wire. Solder and heat-shrink take less."""

    wall: float = 3.2
    panel: float = 4.0
    """The front wall, away from the switch."""
    clip_panel: float = 2.0
    """The front wall where the switch's clips grab it."""
    pocket_margin: float = 1.5
    """Pocket past the cutout's sides, beyond what the clips need at its ends."""
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

    jack_x: float = 20.5
    """Jack centre across the back wall, from the middle. Positive is the
    side a right-handed viewer at the desk calls right."""
    jack_depth: float = 20.0
    """Jack centre below the desk."""
    notch_x: float = -20.5
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
        """Rocker centre below the desk: the headroom, plus half the bezel."""
        return self.headroom + self.switch.bezel[1] / 2

    @property
    def press_depth(self) -> float:
        """Middle of the rocker's top half, where a fingertip sends it up."""
        return self.axis_depth - self.switch.cutout[1] / 4

    @property
    def height(self) -> float:
        """Shell, rim to open edge. The switch body has to fit under the axis
        with air under it, and then the lid's lip comes in."""
        return self.axis_depth + self.switch.body[1] / 2 + self.clearance + self.lip

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
    def pocket(self) -> tuple[float, float]:
        """The thinned strip on the front wall's inside: its width, and the
        depth below the desk it starts at, a clip and a millimetre above the
        cutout. It runs from there out of the open edge."""
        w, length = self.switch.cutout
        return w + 2 * self.pocket_margin, self.axis_depth - length / 2 - self.switch.clips - 1.0

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

    def validate(self) -> None:
        sw, jk = self.switch, self.jack
        cw, cl = sw.cutout
        if self.press_depth < self.finger / 2:
            raise ValueError(
                f"a {self.finger}mm fingertip on the rocker's top half at "
                f"{self.press_depth:.1f}mm below the desk hits the desk; more "
                f"headroom moves the switch down"
            )
        if self.headroom < 3.0:
            raise ValueError(
                f"{self.headroom}mm of headroom puts the bezel against the desk"
            )
        if min(self.wall, self.lid, self.clip_panel) < MIN_WALL:
            raise ValueError(
                f"a {min(self.wall, self.lid, self.clip_panel)}mm wall is under "
                f"{MIN_WALL}mm, which is two extrusions on the 0.4mm nozzle"
            )
        if self.panel < self.wall:
            raise ValueError(
                f"the {self.panel}mm front wall is thinner than the {self.wall}mm "
                f"sides, and it is the wall the switch is pressed into"
            )
        lo, hi = sw.panel_range
        if not lo <= self.clip_panel <= hi:
            raise ValueError(
                f"the {sw.name}'s clips grab a {lo}-{hi}mm panel, not "
                f"{self.clip_panel}mm"
            )
        if self.clip_panel > self.panel:
            raise ValueError("the web under the clips is thicker than the wall")
        if sw.bezel[0] < cw + 1.0 or sw.bezel[1] < cl + 1.0:
            raise ValueError(
                f"a {sw.bezel[0]} x {sw.bezel[1]}mm bezel does not cover a "
                f"{cw} x {cl}mm cutout"
            )
        if sw.body[0] > cw or sw.body[1] > cl:
            raise ValueError(
                f"a {sw.body[0]} x {sw.body[1]}mm body will not pass a "
                f"{cw} x {cl}mm cutout"
            )
        # The pocket stays clear of the corner posts and short of the rim.
        pw, top = self.pocket
        if pw / 2 > self.width / 2 - self.boss - 1.0:
            raise ValueError(
                f"a {sw.name} between {self.boss}mm corner posts needs the inside "
                f"wider than {self.width}mm"
            )
        if top < 1.0:
            raise ValueError("the pocket round the switch runs out of the rim")
        if sw.body_depth + self.wiring > self.depth:
            raise ValueError(
                f"{sw.body_depth}mm of switch behind the panel and "
                f"{self.wiring}mm of wiring need a box {sw.body_depth + self.wiring:.0f} "
                f"deep, not {self.depth}"
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
        if abs(self.jack_x) - jk.body / 2 < sw.body[0] / 2 + 2.0:
            raise ValueError(
                f"the jack body at x={self.jack_x} sits over the switch body; "
                f"move it out past {sw.body[0] / 2 + jk.body / 2 + 2:.0f}"
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


def _pocket(params: Params) -> Part:
    """The thinned strip on the front wall's inside, out to the open edge."""
    p = params
    pw, top = p.pocket
    back = p.panel + 0.5
    length = p.height + 1.0 - top
    return Pos(0, (p.clip_panel + back) / 2, top + length / 2) * Box(pw, back - p.clip_panel, length)


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

    # The switch, through the front wall, thinned round it for the clips.
    cw, cl = sw.cutout
    part -= _pocket(p)
    part -= Pos(0, p.panel / 2, p.axis_depth) * Box(cw, p.panel + 1.0, cl)
    # The jack, through the back wall; the lead, out of its open edge.
    part -= _in_wall(teardrop(jk.hole), p.jack_x, p.jack_depth, p.back_inner, p.outer_depth)
    part -= Pos(p.notch_x, p.back_inner + p.wall / 2, h - p.notch_height / 2 + 0.5) * Box(
        p.notch, p.wall + 1.0, p.notch_height + 1.0
    )
    return part


def coupon(params: Params) -> Part:
    """The front of the shell alone, to test-fit the switch before the box.

    Everything from the front face back past the front corner posts, printed
    the same way up as the shell, so the cutout and the web come out exactly
    as they will in the box. Twenty minutes of plastic against two hours.
    """
    p = params
    w = p.footprint[0] + 2
    keep = p.panel + p.boss + 1.0
    return shell(p) & Pos(0, keep / 2 - 0.5, p.height / 2) * Box(w, keep + 1.0, p.height + 2)


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
    """The housing at home: through the cutout from the face, tabs and all."""
    p, sw = params, params.switch
    bw, bl = sw.body
    return Pos(0, sw.body_depth / 2, p.axis_depth) * Box(bw, sw.body_depth, bl)


def switch_front(params: Params) -> Part:
    """Bezel on the front face, and the rocker standing proud of it."""
    p, sw = params, params.switch
    (bw, bl), t = sw.bezel, sw.bezel_thickness
    bezel = Pos(0, -t / 2, p.axis_depth) * Box(bw, t, bl)
    cw, cl = sw.cutout
    rocker = Pos(0, -t - sw.rocker_rise / 2, p.axis_depth) * Box(cw - 3.0, sw.rocker_rise, cl - 3.0)
    return bezel + rocker


def clips(params: Params) -> Part:
    """The two spring clips, sprung out behind the web at the cutout's ends."""
    p, sw = params, params.switch
    cw, cl = sw.cutout
    t = 1.0
    y = p.clip_panel + t / 2 + 0.05
    return sum(
        (
            Pos(0, y, p.axis_depth + s * (cl / 2 + sw.clips / 2)) * Box(cw - 4.0, t, sw.clips)
            for s in (-1, 1)
        ),
        Part(),
    )


def fingertip(params: Params) -> Part:
    """A fingertip pointed at the box, on the middle of the rocker's top half."""
    p = params
    sw = p.switch
    reach = 40.0
    front = sw.bezel_thickness + sw.rocker_rise
    return Pos(0, -front - reach / 2, p.press_depth) * Rot(90, 0, 0) * Cylinder(p.finger / 2, reach)


def desk(params: Params) -> Part:
    """The desktop above the box, flush with its front face."""
    w, d = params.footprint
    return Pos(0, d / 2 + 20, -10) * Box(w + 40, d + 40, 20)


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
    # The web the clips grab is 2mm and the wall round it 4mm; six loops make
    # both solid plastic instead of skins over sparse infill.
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
    print(f"rocker centre {p.axis_depth:.1f} mm below the desk, {p.headroom} mm over the bezel")
