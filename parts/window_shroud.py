"""Window shroud: lets an indoor camera look out through a window at night.

Put a camera behind glass and it sees two things that are not outside. The
room, reflected off the inner face of the pane, and -- once it is dark -- its
own infrared LEDs, which light the glass a few centimetres away far brighter
than anything in the yard, so the picture goes to a white haze. The cure is the
one photographers use against shop windows: put the lens right up to the glass
and black out everything around it.

So this is two tubes, one inside the other, both pressed against the glass.

THE SKIRT is the outer one. It slides over the front of the camera's head, the
head's white rim sits on a lip inside it, and its front edge seals against the
pane. With the skirt in place no room light reaches the patch of glass the
camera looks through.

THE HOOD is the inner one, and it is what fixes night vision. It is a short
black tube that runs from the face panel, around the lens bezel, to the glass.
Infrared from the LEDs leaves the face outside the hood, goes through the pane
and lights the yard; what the inner surface of the glass reflects comes back
outside the hood too, and the hood's wall stands between it and the lens. The
LEDs, the light sensor and the status light all sit in the space between the
two tubes, looking straight out, and three thin ribs -- left, right and up --
hold the hood in place without standing in front of any of them. On the C100
the IR LEDs are at the top corners and the sensor and a port are below, so the
diagonals and straight down are taken.

WHY THE LENS SITS SO CLOSE TO THE GLASS. The C100's lens is close to a
fisheye: 117 degrees across the diagonal. A hood has to be wider than the
picture at its glass end, and the picture spreads at about 60 degrees off-axis
at the corners, so every millimetre of standoff widens the hood's mouth by
three or so. At 5mm the whole picture fits inside a tube that only just clears
the bezel, which leaves the rest of the face free for the LEDs; at 15mm the
hood would have to be nearly as wide as the face and would swallow them.
``standoff`` is the one number to tune, and ``validate`` refuses values where
the hood would cover the face's sensors or lean past what prints.

WHAT IT CANNOT FIX. In a double-glazed window a little of the LED light enters
the inner pane outside the hood, crosses the gap and reflects off the outer
pane's surfaces, and comes back in through the hood's mouth. On low-e glass in
particular that can still show as a faint ghost. No hood stops it, because it
arrives through the glass the camera has to look through. If it is bad, turn
the camera's infrared off and light the yard instead.

HOW IT PRINTS. Glass face down. The front edge of the skirt and of the hood
both come off the plate, which is the flattest, smoothest face the part has and
the one that has to seal. Everything above that is vertical walls or upward-
facing ledges; the only lean is the hood's inner wall when a long standoff
makes it flare, and that is held under 40 degrees. Matte black, PETG or ASA:
PLA softens in a sunny window.

The camera's dimensions come from geom.cameras, which says which are sourced
and which are estimates off a photo. ``coupon`` builds just the sleeve band, a
ten-minute print, to check the fit before committing to the whole thing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from build123d import (
    Box,
    Part,
    Plane,
    Pos,
    RectangleRounded,
    Rot,
    extrude,
    loft,
)

from geom import cameras
from geom.cameras import Camera

MIN_WALL = 0.8
"""Two extrusions on the 0.4mm nozzle. Thinner than this a wall is one bead,
and one bead of black plastic is not light-tight."""

MAX_LEAN = 40.0
"""Degrees off vertical for any wall that overhangs, five inside the 45 FDM
holds, because this is the inside of a tube where a sag is the edge the lens
looks past."""

EPS = 0.01


@dataclass(frozen=True)
class Params:
    """Everything the GUI would expose as a slider."""

    camera: Camera = cameras.TAPO_C100
    """The head this is cut for. Its estimated fields are the ones to measure."""

    standoff: float = 5.0
    """Face panel to glass, in mm. Less is better optically -- see the module
    note -- but the lens bezel stands proud of the face and has to clear the
    pane, so this is the bezel plus a couple of millimetres."""
    margin: float = 3.0
    """Degrees added all round the published field of view when sizing the
    hood's mouth, so a camera a little off-square in the sleeve still sees no
    hood."""

    wall: float = 2.0
    """Skirt and sleeve wall."""
    fit: float = 0.3
    """Gap each side between the head and the sleeve. A slide fit; the rim
    sitting on the lip is what locates the camera, not the sleeve."""
    sleeve: float = 8.0
    """How far back the sleeve reaches over the head. Short enough to stay
    clear of the microSD slot and the stalk."""

    hood_wall: float = 1.2
    hood_clearance: float = 1.0
    """Gap each side between the lens bezel and the hood. Covers the lens
    position being an estimate as much as it covers the print."""

    ribs: tuple[float, ...] = field(default=(0.0, 90.0, 180.0))
    """Angles, from the camera's right going up, of the ribs that hold the
    hood. Left, right and up, because on the C100 the IR LEDs sit on the upper
    diagonals and the light sensor and a port are below the lens."""
    rib_width: float = 1.2
    rib_gap: float = 1.0
    """Ribs stop this far short of the face, so they shade nothing on it and
    the only things touching the face are the hood and the lip."""

    flange: float = 6.0
    """Extra width at the glass end, all round, for adhesive. 0 for none, if the
    camera's own stand holds it to the glass."""
    flange_thickness: float = 2.0

    stalk_notch: float = 14.0
    """Width of a slot in the sleeve's bottom wall for the stand's stalk, open
    at the back. 0 for none."""
    notch_keep: float = 3.0
    """How much of the sleeve's bottom wall the notch leaves behind the lip."""

    corner_gap: float = 1.0
    """Clearance between the hood's mouth and the edge of the face opening."""

    # --- what follows from those ------------------------------------------

    @property
    def lens(self) -> tuple[float, float]:
        """Lens centre in the shroud's x, y: mirrored, because the shroud is
        modelled from behind the camera, looking out through it."""
        return -self.camera.lens_x, self.camera.lens_y

    @property
    def pupil_z(self) -> float:
        """Height of the lens's entrance pupil above the glass."""
        c = self.camera
        return self.standoff - c.bezel_proud + c.pupil_depth

    @property
    def hood_base(self) -> float:
        """Inner radius of the hood where it meets the face."""
        return self.camera.bezel_d / 2 + self.hood_clearance

    @property
    def hood_mouth(self) -> tuple[float, float]:
        """Inner half-width and half-height of the hood on the glass: the
        picture's footprint there, but never narrower than the base."""
        fx, fy = self.camera.footprint(self.pupil_z, self.margin)
        return max(fx, self.hood_base), max(fy, self.hood_base)

    @property
    def hood_lean(self) -> float:
        """Degrees the hood's inner wall leans in going up from the glass."""
        hx, hy = self.hood_mouth
        return math.degrees(math.atan((max(hx, hy) - self.hood_base) / self.standoff))

    @property
    def sleeve_size(self) -> tuple[float, float, float]:
        c = self.camera
        return c.head_w + 2 * self.fit, c.head_h + 2 * self.fit, c.head_corner + self.fit

    @property
    def outer_size(self) -> tuple[float, float, float]:
        w, h, r = self.sleeve_size
        return w + 2 * self.wall, h + 2 * self.wall, r + self.wall

    @property
    def height(self) -> float:
        return self.standoff + self.sleeve

    @property
    def features(self) -> tuple[tuple[str, float, float, float], ...]:
        """Face features in shroud coordinates: name, x, y, diameter."""
        lx, ly = self.lens
        return tuple((f.name, lx - f.x, ly + f.y, f.diameter) for f in self.camera.features)

    def rib_clearance(self, x: float, y: float) -> float:
        """Distance in plan from a point to the nearest rib's centreline,
        counting only the length of rib outside the hood."""
        lx, ly = self.lens
        best = math.inf
        for a in self.ribs:
            ux, uy = math.cos(math.radians(a)), math.sin(math.radians(a))
            along = (x - lx) * ux + (y - ly) * uy
            if along < self.hood_base:
                continue
            best = min(best, abs((x - lx) * uy - (y - ly) * ux))
        return best

    def validate(self) -> None:
        c = self.camera
        for name, value in (("wall", self.wall), ("hood_wall", self.hood_wall),
                            ("rib_width", self.rib_width)):
            if value < MIN_WALL:
                raise ValueError(
                    f"a {value}mm {name} is under {MIN_WALL}mm, two extrusions on "
                    f"the 0.4mm nozzle; one bead of black plastic still lets "
                    f"light through"
                )
        if self.standoff < c.bezel_proud + 0.5:
            raise ValueError(
                f"a {self.standoff}mm standoff puts the {c.name}'s lens bezel, "
                f"which stands {c.bezel_proud}mm proud of the face, against the "
                f"glass. Give it at least {c.bezel_proud + 0.5}mm."
            )
        if self.hood_lean > MAX_LEAN:
            raise ValueError(
                f"at a {self.standoff}mm standoff the picture is "
                f"{2 * max(self.hood_mouth):.0f}mm wide at the glass, so the hood "
                f"would lean {self.hood_lean:.0f} degrees to reach it from the "
                f"bezel, past the {MAX_LEAN:.0f} that prints clean. Bring the "
                f"camera closer to the glass."
            )
        if not (c.panel_w < c.head_w and c.panel_h < c.head_h):
            raise ValueError("the face panel has to be smaller than the head, or "
                             "there is no rim for the lip to bear on")
        lx, ly = self.lens
        hx, hy = self.hood_mouth
        reach_x = abs(lx) + hx + self.hood_wall + self.corner_gap
        reach_y = abs(ly) + hy + self.hood_wall + self.corner_gap
        if reach_x > c.panel_w / 2 or reach_y > c.panel_h / 2:
            raise ValueError(
                f"the hood's mouth reaches {max(reach_x, reach_y):.1f}mm from the "
                f"centre and the face opening only {c.panel_w / 2:.1f}mm. Bring "
                f"the camera closer to the glass."
            )
        for name, x, y, d in self.features:
            if math.hypot(x - lx, y - ly) < self.hood_base + self.hood_wall + d / 2:
                raise ValueError(
                    f"the hood would sit on the {name}. Its outside is "
                    f"{self.hood_base + self.hood_wall:.1f}mm from the lens and "
                    f"the {name} is {math.hypot(x - lx, y - ly):.1f}mm away; "
                    f"tighten hood_clearance or check the measurements."
                )
            if self.rib_clearance(x, y) < self.rib_width / 2 + d / 2:
                raise ValueError(
                    f"a rib runs in front of the {name}; move it off "
                    f"{self.ribs}"
                )
        if not 0 < self.rib_gap < self.standoff - 0.5:
            raise ValueError(
                f"a {self.rib_gap}mm rib gap leaves the ribs "
                f"{self.standoff - self.rib_gap:.1f}mm tall on a "
                f"{self.standoff}mm standoff; keep them at least 0.5mm"
            )
        if self.sleeve > c.head_depth - 2:
            raise ValueError(
                f"a {self.sleeve}mm sleeve reaches past the back of a "
                f"{c.head_depth}mm-deep head"
            )
        if self.stalk_notch and self.notch_keep >= self.sleeve:
            raise ValueError("the stalk notch starts behind the end of the sleeve")


def _rounded(w: float, h: float, r: float):
    return RectangleRounded(w, h, min(r, w / 2 - EPS, h / 2 - EPS))


def _prism(w: float, h: float, r: float, z0: float, z1: float, at=(0.0, 0.0)) -> Part:
    return Pos(at[0], at[1], z0) * extrude(_rounded(w, h, r), amount=z1 - z0)


def _hood_section(params: Params, grow: float, z: float) -> tuple[float, float, float]:
    """Width, height and corner radius of the hood's inner surface (or, with
    ``grow`` = the wall, its outer one) at height ``z``: the mouth on the glass,
    the bezel circle at the face, straight between and straight on past
    either end, so a cutter can overrun the part without leaving a step."""
    hx, hy = params.hood_mouth
    rb = params.hood_base
    t = z / params.standoff
    x = hx + (rb - hx) * t + grow
    y = hy + (rb - hy) * t + grow
    return 2 * x, 2 * y, min(x, y)


def _hood(params: Params, grow: float, z0: float, z1: float) -> Part:
    """The hood between two heights: a loft when the mouth is wider than the
    base, a straight tube when it is not."""
    lo, hi = _hood_section(params, grow, z0), _hood_section(params, grow, z1)
    if abs(lo[0] - hi[0]) < EPS and abs(lo[1] - hi[1]) < EPS:
        return _prism(*lo, z0, z1, params.lens)
    lx, ly = params.lens
    return Pos(lx, ly, 0) * loft([
        Plane.XY.offset(z0) * _rounded(*lo),
        Plane.XY.offset(z1) * _rounded(*hi),
    ])


def skirt(params: Params) -> Part:
    """The outer tube on its own: the lip, the sleeve and the stalk notch."""
    c = params.camera
    top = params.height
    sw, sh, sr = params.sleeve_size
    part = _prism(*params.outer_size, 0, top)
    part -= _prism(c.panel_w, c.panel_h, c.panel_corner, -1, params.standoff + EPS)
    part -= _prism(sw, sh, sr, params.standoff, top + 1)
    if params.stalk_notch:
        _, oh, _ = params.outer_size
        z0 = params.standoff + params.notch_keep
        part -= Pos(0, -(oh + sh) / 4, (z0 + top + 1) / 2) * Box(
            params.stalk_notch, (oh - sh) / 2 + 2, top + 1 - z0)
    return part


def build(params: Params) -> Part:
    """The shroud, glass face on the plate, camera end up."""
    params.validate()
    c = params.camera
    part = skirt(params)

    ow, oh, orr = params.outer_size
    if params.flange:
        f = params.flange
        ring = _prism(ow + 2 * f, oh + 2 * f, orr + f, 0, params.flange_thickness)
        ring -= _prism(c.panel_w, c.panel_h, c.panel_corner, -1, params.flange_thickness + 1)
        part += ring

    part += _hood(params, params.hood_wall, 0, params.standoff)

    rib_top = params.standoff - params.rib_gap
    reach = math.hypot(ow, oh)
    lx, ly = params.lens
    for a in params.ribs:
        rib = Pos(lx, ly, rib_top / 2) * Rot(0, 0, a) * Pos(reach / 2, 0, 0) * Box(
            reach, params.rib_width, rib_top)
        part += rib & _prism(ow - EPS, oh - EPS, orr, 0, rib_top)

    # Cut last, and a millimetre past both ends, so neither a rib nor the
    # flange leaves a skin across the picture.
    return part - _hood(params, 0, -1, params.standoff + 1)


def coupon(params: Params, lip: float = 1.2) -> Part:
    """The sleeve and a sliver of lip, sat on the plate: prints in minutes and
    says whether the head fits before the whole shroud is committed to."""
    params.validate()
    z0 = params.standoff - lip
    ow, oh, _ = params.outer_size
    band = skirt(params) & Pos(0, 0, (z0 + params.height) / 2) * Box(
        ow + 2, oh + 2, params.height - z0)
    return Pos(0, 0, -z0) * band


if __name__ == "__main__":
    from pathlib import Path

    from build123d import export_stl

    p = Params()
    part = build(p)
    out = Path("out")
    out.mkdir(exist_ok=True)
    export_stl(part, str(out / "window_shroud.stl"))
    bb = part.bounding_box()
    hx, hy = p.hood_mouth
    print(f"valid={part.is_valid} solids={len(part.solids())}")
    print(f"bbox={bb.size.X:.1f} x {bb.size.Y:.1f} x {bb.size.Z:.1f} mm")
    print(f"hood mouth {2 * hx:.1f} x {2 * hy:.1f} mm, lean {p.hood_lean:.1f} deg")
