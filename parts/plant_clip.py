"""Parametric plant clip: an adhesive pad with a C that a vine presses into.

Aimed at pothos and hoya, which is to say stems somewhere between 4 and 12mm
through, held against a wall by a Command strip rather than a screw. The whole
part is a pad and a C, and every number below follows from one of three things:
the strip's footprint, the stem's diameter, or the fact that this has to print
on a bed with no support under it.

WHY THE MOUTH FACES AWAY FROM THE WALL. It is the only direction that both
prints and installs. Facing sideways, the C would have to be modelled standing
on edge and the tips would print over air. Facing the wall, it could not be
loaded at all once it is stuck down. Facing out, the arms lean inward as they
rise -- an overhang, bounded by the wrap angle -- and the vine is pressed home
towards the wall, which is the direction a hand pushes anyway.

WHY ``wrap`` IS CAPPED AT 270 DEGREES. On a circular arc the wall's lean from
vertical equals the angle past the equator, so a C that wraps 270 degrees ends
its tips leaning 45 degrees, and 45 degrees is where FDM stops holding on. Wrap
further and the tips curl over and print badly; wrap less and the mouth opens
wider than the stem and holds nothing in. Everything useful is in that band,
which is why the default sits a little inside it rather than at the edge.

WHAT SIZE STEM A CLIP TAKES. Two numbers bracket it. The bore is the largest
stem that fits, and the mouth -- the narrowest part of the opening, at the bore
radius -- is the smallest stem that stays captive; below that it rests in the
cradle but can lift straight out. That is a band a couple of millimetres wide,
not one size fits all, so ``STEMS`` names a few that between them cover the
range. Erring loose is deliberate: a clip that grips a growing stem hard enough
not to move is a clip that scars it.

The pad is ``strip.bond`` long, not ``strip.length`` -- see geom.strips -- so
the pull tab hangs off one end where a thumb can reach it. It is modelled flat
on the bed because the adhesive wants the smooth face, which on a printed part
is the one that was pressed against the plate.

Print it in ASA, not PLA. The load here is small but it never goes away, and
PLA creeps under a sustained load at room temperature: the arms would open over
a season and hand the vine back.
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

from geom import strips
from geom.strips import Strip

MAX_WRAP = 270.0
"""Degrees. Past this the arm tips lean more than 45 degrees. See the module note."""

MIN_WALL = 0.8
"""Two extrusions on the 0.4mm nozzle. Thinner than this the arms are a single
bead with no wall either side, and they snap the first time one is opened."""

STEMS = (5.0, 7.0, 9.0, 11.0)
"""Stem sizes that, at the default wrap, tile the 4-12mm pothos/hoya range with
no meaningful gap between one clip's mouth and the next one's bore."""


@dataclass(frozen=True)
class Params:
    """Everything the GUI would expose as a slider."""

    stem: float = 8.0
    """Diameter of the stem this clip is cut for, in mm. See ``STEMS``."""
    strip: Strip = strips.SMALL
    """Adhesive strip the pad is built around. MEDIUM is the same width and
    twice the length, so a pad sized for it takes two clips and a heavier vine
    without any other number changing."""

    clearance: float = 0.6
    """Added to ``stem`` to get the bore. Slack, not a fit: the stem is alive
    and gets thicker, and the clip should let it."""
    wrap: float = 260.0
    """Degrees of stem the C wraps. Retention, printability and how hard it is
    to clip in all trade off against this one number."""
    wall: float = 1.6
    """Thickness of the C around the bore."""
    clip_width: float = 10.0
    """How much of the stem's length the C holds, along the pad."""
    count: int = 1
    """Clips on one pad, spaced evenly along it. Two on a MEDIUM strip hold a
    run of vine straight instead of letting it pivot about a single point."""

    pad_thickness: float = 1.6
    """Thin enough to stay flat off the plate, thick enough not to peel with
    the strip when the clip is levered."""
    pad_margin: float = 1.2
    """How far the pad oversails the strip on each side, so the adhesive has
    backing right to its edge."""
    flare: float = 1.5
    """How much wider the C's base is than the C itself, each side. A taper
    rather than a fillet: it gussets the joint and still prints as a plain
    inward lean."""
    corner: float = 3.0
    """Plan radius on the pad's corners. A sharp corner on a thin pad is the
    thing that catches a sleeve and starts the peel."""

    # --- what follows from those ------------------------------------------

    @property
    def bore(self) -> float:
        """Largest stem that fits: the biggest end of this clip's range."""
        return self.stem + self.clearance

    @property
    def bore_radius(self) -> float:
        return self.bore / 2

    @property
    def outer_radius(self) -> float:
        return self.bore_radius + self.wall

    @property
    def axis_z(self) -> float:
        """Height of the stem's centreline above the bed.

        Set so the cradle floor is one wall thick, which puts the stem as close
        to the wall as the C's own section allows.
        """
        return self.pad_thickness + self.wall + self.bore_radius

    @property
    def mouth_angle(self) -> float:
        """Half the opening, in degrees off vertical."""
        return (360.0 - self.wrap) / 2

    @property
    def mouth(self) -> float:
        """Narrowest part of the opening, in mm.

        Measured at the bore radius, where the two arms come closest. Beyond
        that the cut faces are radial, so the gap widens with radius and the
        mouth leads the stem in.
        """
        return 2 * self.bore_radius * math.sin(math.radians(self.mouth_angle))

    @property
    def grip(self) -> tuple[float, float]:
        """Stem diameters this clip holds captive, smallest to largest."""
        return self.mouth, self.bore

    @property
    def pad_length(self) -> float:
        return self.strip.bond

    @property
    def pad_width(self) -> float:
        return self.strip.width + 2 * self.pad_margin

    @property
    def positions(self) -> tuple[float, ...]:
        """Where each clip sits along the pad, evenly spaced about the centre."""
        n = self.count
        return tuple(self.pad_length * ((i + 0.5) / n - 0.5) for i in range(n))

    def for_stem(self, diameter: float) -> Params:
        """The same clip cut for a different stem."""
        return replace(self, stem=diameter)

    def validate(self) -> None:
        if self.stem <= 0:
            raise ValueError(f"a {self.stem}mm stem is not a stem")
        if self.wall < MIN_WALL:
            raise ValueError(
                f"a {self.wall}mm wall is under {MIN_WALL}mm, which is two "
                f"extrusions on the 0.4mm nozzle. Below that the arms print as a "
                f"single bead and break the first time the clip is opened."
            )
        if self.wrap <= 180:
            raise ValueError(
                f"wrapping {self.wrap} degrees makes a shelf, not a clip: the "
                f"opening is as wide as the bore and the stem lifts straight out"
            )
        if self.wrap > MAX_WRAP:
            raise ValueError(
                f"wrapping {self.wrap} degrees leans the arm tips "
                f"{self.wrap / 2 - 90:.0f} degrees off vertical, past the "
                f"{MAX_WRAP / 2 - 90:.0f} degrees FDM holds. Cap the wrap at "
                f"{MAX_WRAP} or print it on its side and lose the pad's plate face."
            )
        if self.mouth >= self.stem:
            raise ValueError(
                f"a {self.mouth:.2f}mm mouth is wider than the {self.stem}mm stem "
                f"it is meant to hold, so nothing keeps the stem in. Wrap further "
                f"than {self.wrap} degrees, or cut this clip for a bigger stem."
            )
        if self.count < 1:
            raise ValueError("a clip with no clips on it is a sticker")
        if self.count * self.clip_width > self.pad_length:
            raise ValueError(
                f"{self.count} clips {self.clip_width}mm wide overrun the "
                f"{self.pad_length:.1f}mm the {self.strip.name} strip's bond "
                f"leaves for them. Use fewer, narrow them, or move up a strip."
            )
        if 2 * (self.outer_radius + self.flare) > self.pad_width:
            raise ValueError(
                f"the C's base is {2 * (self.outer_radius + self.flare):.1f}mm "
                f"across and the pad only {self.pad_width:.1f}mm, so it hangs "
                f"over the adhesive. Widen pad_margin, or use a wider strip."
            )
        if self.corner > self.pad_width / 2:
            raise ValueError(
                f"a {self.corner}mm corner radius on a {self.pad_width:.1f}mm pad "
                f"leaves no straight edge between the corners"
            )


def _mouth_cutter(params: Params) -> Sketch:
    """The wedge taken out of the top of the C, in the C's own section plane.

    Radial faces, so the opening is narrowest at the bore and flares outward
    from there. Its apex sits at the stem's centre, inside the bore that has
    already been removed, so the cut leaves no knife edge behind.
    """
    half = math.radians(params.mouth_angle)
    ro, zc = params.outer_radius, params.axis_z
    reach = ro * 1.05  # just past the outer surface; the rest of the cut is air
    y, z = reach * math.sin(half), reach * math.cos(half)
    top = zc + 2 * ro
    return Polygon(
        (0.0, zc), (y, zc + z), (y, top), (-y, top), (-y, zc + z), align=None
    )


def _section(params: Params) -> Sketch:
    """Half-section of one clip in (across the pad, up from the bed).

    A trapezoid up to the stem's centreline, a disc above it, the bore taken
    out of both and the mouth cut out of the top. The trapezoid's sides lean
    inward going up and the disc's outer surface only ever leans further in, so
    the single overhang in the whole part is the bore's own arms.
    """
    ro, zc = params.outer_radius, params.axis_z
    base = Polygon(
        (-(ro + params.flare), 0.0),
        (ro + params.flare, 0.0),
        (ro, zc),
        (-ro, zc),
        align=None,
    )
    body = base + Pos(0, zc) * Circle(ro)
    return body - Pos(0, zc) * Circle(params.bore_radius) - _mouth_cutter(params)


def pad(params: Params) -> Part:
    """The mounting plate on its own: the face the adhesive gets."""
    params.validate()
    plate = RectangleRounded(params.pad_length, params.pad_width, params.corner)
    return extrude(Plane.XY * plate, amount=params.pad_thickness)


def build(params: Params) -> Part:
    """Pad plus its clips, modelled flat on the bed with the stem along X."""
    part = pad(params)
    # Drawn in the YZ plane and extruded along the pad, which is the axis the
    # stem runs down: the section is the design and the third dimension is just
    # how much of the stem it grips.
    profile = Plane.YZ * _section(params)
    for x in params.positions:
        part += Pos(x, 0, 0) * extrude(profile, amount=params.clip_width / 2, both=True)
    return part


if __name__ == "__main__":
    from pathlib import Path

    from build123d import export_stl

    p = Params()
    part = build(p)
    out = Path("out")
    out.mkdir(exist_ok=True)
    export_stl(part, str(out / "plant_clip.stl"))
    bb = part.bounding_box()
    lo, hi = p.grip
    print(f"valid={part.is_valid} solids={len(part.solids())}")
    print(f"bbox={bb.size.X:.1f} x {bb.size.Y:.1f} x {bb.size.Z:.1f} mm")
    print(f"holds {lo:.1f}-{hi:.1f} mm stems, mouth {p.mouth:.2f} mm")
