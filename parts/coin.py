"""Parametric commemorative coin: a disc, a rim, and a mark on each face.

Give it a motif per face and a diameter; everything else has a default that
prints. The body is one revolved profile rather than a stack of booleans, so
the rim, the recessed field and the chamfered edge all come out of a single
sweep and there are no seams between them to go wrong.

WHY THE TWO FACES ARE NOT TREATED ALIKE. A coin wants raised relief on both
sides, and on an FDM printer only the top can have it. Recessing the underside
to sink a field there would leave that field spanning the whole disc with air
beneath it -- a 30mm bridge, printed as the first visible surface of the part.
Raising the reverse motif instead puts the design on the plate and asks the
field above to bridge over it, which is the same problem wearing a hat. So the
obverse is struck in relief inside a recessed field, and the reverse is
engraved into a flat underside. That prints with no support at all, and the
engraved side comes out glossy off a smooth plate, which suits a coin.

The reverse motif is mirrored before it is cut. Sketches are authored looking
down the +Z axis, but the underside is read from below, so a mark cut as drawn
would come out backwards -- which is fine for a squiggle and very much not
fine for a date.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from build123d import (
    Axis,
    Cylinder,
    Part,
    Plane,
    Polyline,
    Pos,
    extrude,
    make_face,
    mirror,
    revolve,
)

from geom.motif import Motif, combine

Faces = Motif | tuple[Motif, ...]


@dataclass(frozen=True)
class Params:
    """Everything the GUI would expose as a slider."""

    obverse: Faces = ()
    """Motif(s) struck in relief on the top face."""
    reverse: Faces = ()
    """Motif(s) engraved into the bottom face."""

    diameter: float = 38.0
    """Overall diameter in mm. 38 is challenge-coin scale -- it fills a palm
    and still gives a legend room to be legible at 3mm cap height."""
    thickness: float = 3.0
    """Overall height in mm, rim to underside."""
    rim: float = 2.2
    """Width of the raised border. 0 leaves the top face flat and full, and the
    relief then stands off it with nothing to protect it, so the coin comes out
    ``thickness + relief`` tall rather than ``thickness``."""
    field_depth: float = 0.7
    """How far the top field is recessed below the rim."""
    relief: float = 0.6
    """Height the obverse motif stands proud of the recessed field."""
    engrave: float = 0.5
    """Depth the reverse motif is cut into the underside."""
    chamfer: float = 0.6
    """45-degree break on both outer edges, so the coin has no sharp arris."""
    reeds: int = 0
    """Milled grooves around the edge, as on a coin's collar. 0 leaves it plain."""
    reed_depth: float = 0.35
    """Radius of each groove; also how deep it cuts."""

    def scaled(self, factor: float) -> Params:
        """The same coin at a different size, for a test print.

        Every millimetre-valued field scales and the counts do not. The motifs
        need no attention at all: a motif is specified in fractions of the
        field and in arbitrary drawing units, and both are scale-free. So this
        lands exactly where the slicer's scale box would, except that the
        parameters still describe the part -- which is what keeps validate()
        and the preview honest about what is going to print.
        """
        mm = ("diameter", "thickness", "rim", "field_depth", "relief",
              "engrave", "chamfer", "reed_depth")
        return replace(self, **{f: getattr(self, f) * factor for f in mm})

    @property
    def radius(self) -> float:
        return self.diameter / 2

    @property
    def field(self) -> float:
        """Diameter of the usable area inside the rim."""
        margin = self.rim if self.rim else self.chamfer
        return self.diameter - 2 * margin

    @property
    def field_z(self) -> float:
        """Height of the recessed top field, where the relief starts."""
        return self.thickness - self.field_depth if self.rim else self.thickness

    @property
    def reed_pitch(self) -> float:
        """Arc between groove centres, in mm."""
        return math.pi * self.diameter / self.reeds if self.reeds else 0.0

    def validate(self) -> None:
        if self.rim and self.rim <= self.field_depth + self.chamfer:
            raise ValueError(
                f"a {self.rim}mm rim is not wide enough to hold a "
                f"{self.field_depth}mm recess behind a {self.chamfer}mm chamfer: "
                f"the 45-degree ramp into the field would start outside the "
                f"chamfer and eat the rim's flat top. Widen the rim past "
                f"{self.field_depth + self.chamfer}mm."
            )
        if self.rim and self.relief > self.field_depth:
            raise ValueError(
                f"{self.relief}mm of relief stands out of a {self.field_depth}mm "
                f"recess, so the motif is proud of the rim and takes the wear "
                f"the rim exists to take. Deepen the field or lower the relief."
            )
        if self.field <= 0:
            raise ValueError(
                f"a {self.rim}mm rim on a {self.diameter}mm coin leaves no field"
            )
        if self.engrave >= self.thickness - self.field_depth:
            raise ValueError(
                f"engraving {self.engrave}mm into a coin only "
                f"{self.thickness - self.field_depth:.2f}mm thick under its "
                f"field would cut through to the other side"
            )
        if 2 * self.chamfer >= self.thickness:
            raise ValueError(
                f"two {self.chamfer}mm chamfers meet inside a "
                f"{self.thickness}mm coin, leaving no edge between them"
            )
        if self.reeds and self.reed_pitch <= 2 * self.reed_depth:
            raise ValueError(
                f"{self.reeds} grooves sit {self.reed_pitch:.2f}mm apart, closer "
                f"than the {2 * self.reed_depth}mm they are wide, so they merge "
                f"and the edge is just a smaller circle. Use at most "
                f"{int(math.pi * self.diameter / (2 * self.reed_depth))} reeds."
            )


def _profile(params: Params) -> list[tuple[float, float]]:
    """Half-section of the blank in (radius, height), revolved to make the body.

    Walked from the centre of the underside outwards, up the edge, and back
    along the top. The ramp out of the recessed field is deliberately 45
    degrees: a vertical wall there would be a lip printing over air.
    """
    r, t, c = params.radius, params.thickness, params.chamfer
    pts = [(0.0, 0.0), (r - c, 0.0), (r, c), (r, t - c), (r - c, t)]
    if params.rim:
        d = params.field_depth
        pts += [(r - params.rim + d, t), (r - params.rim, t - d), (0.0, t - d)]
    else:
        pts += [(0.0, t)]
    return pts


def blank(params: Params) -> Part:
    """The coin with no marks on it: disc, rim, chamfers, reeded edge."""
    params.validate()
    section = make_face(Polyline(*_profile(params), close=True).wire())
    body = revolve(Plane.XZ * section, Axis.Z)
    if params.reeds:
        # Cut in one boolean rather than one per groove: a hundred sequential
        # subtractions off a revolved solid takes minutes and this takes
        # seconds, for the identical result.
        cutters = [
            Pos(
                params.radius * math.cos(2 * math.pi * i / params.reeds),
                params.radius * math.sin(2 * math.pi * i / params.reeds),
                params.thickness / 2,
            )
            * Cylinder(params.reed_depth, params.thickness + 2)
            for i in range(params.reeds)
        ]
        body -= cutters
    return body


def build(params: Params) -> Part:
    """The struck coin: blank, obverse in relief, reverse engraved."""
    coin = blank(params)
    if params.obverse:
        mark = combine(params.obverse, params.field)
        coin += extrude(Plane.XY.offset(params.field_z) * mark, amount=params.relief)
    if params.reverse:
        mark = combine(params.reverse, params.field)
        # Mirrored so it reads correctly from the side it is seen from.
        coin -= extrude(Plane.XY * mirror(mark, about=Plane.YZ), amount=params.engrave)
    return coin


if __name__ == "__main__":
    from pathlib import Path

    from build123d import export_stl

    from geom.motif import Text

    p = Params(obverse=Text(text="26"), reverse=Text(text="A"), reeds=90)
    part = build(p)
    out = Path("out")
    out.mkdir(exist_ok=True)
    export_stl(part, str(out / "coin.stl"))
    bb = part.bounding_box()
    print(f"valid={part.is_valid} solids={len(part.solids())}")
    print(f"bbox={bb.size.X:.1f} x {bb.size.Y:.1f} x {bb.size.Z:.1f} mm")
    print(f"volume={part.volume/1000:.2f} cm^3")
