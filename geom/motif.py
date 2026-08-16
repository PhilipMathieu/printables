"""Marks to press into a coin face: text, stroked paths, filled outlines, SVG.

A motif is written in whatever units are convenient -- a squiggle sketched on a
0..30 grid, a glyph at its natural size, an SVG at whatever the illustrator
saved -- and ``sketch(field)`` scales and centres it to land on a coin face of
the given diameter. Nothing downstream has to know how the mark was authored,
so adding a new kind of mark means adding one ``_raw`` method.

Fitting works off how far the mark actually reaches from its centre, sampled
along its edges, rather than off its bounding box. The field is a circle, so
the box is the wrong shape to measure against: a round logo touches its box at
four points and its corners are empty, and fitting the box into the field
would leave the logo at 71% of the diameter it could have had. Measuring the
reach directly means ``fill`` says what it sounds like -- the fraction of the
field's radius the mark comes out to -- for a disc, a word and a squiggle
alike.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from build123d import (
    Align,
    Edge,
    Face,
    FontStyle,
    Kind,
    Plane,
    Pos,
    Rot,
    Side,
    Sketch,
    Wire,
    import_svg,
    make_face,
    offset,
    scale,
)
from build123d import Polygon as _PolygonObject
from build123d import Polyline as _PolylineObject
from build123d import Spline as _SplineObject
from build123d import Text as _TextObject

Point = tuple[float, float]

DEFAULT_FONT = "Copperplate"
"""The engraver's face, which is the right one for a struck coin twice over.

It was cut for engraved plates, so it looks the part; and its strokes are of
uniform weight, which is what an extruder wants. A high-contrast face like
Didot measures 0.17mm through its thin strokes at legend size -- well under one
0.42mm extrusion, so the slicer simply drops them and the inscription comes out
in pieces. Copperplate holds 0.52mm all the way round. It also sets lowercase
as small caps, which suits an inscription and surprises anyone who types one.
"""

TOP = "top"
BOTTOM = "bottom"


class MotifError(ValueError):
    pass


@dataclass(frozen=True)
class Motif:
    """A 2D mark that knows how to place itself on a coin face of a given size."""

    rotate: float = 0.0
    """Degrees counter-clockwise."""

    def sketch(self, field: float) -> Sketch:
        raise NotImplementedError


@dataclass(frozen=True)
class Fitted(Motif):
    """A mark drawn at any size, then centred and scaled onto the field.

    Subclasses implement ``_raw`` on whatever grid suits them. Centring,
    rotation and scaling are written once, here. A Legend is the one motif that
    is not fitted this way -- it is placed against the rim instead -- which is
    why ``fill`` lives on this class and not on Motif.
    """

    fill: float = 0.9
    """Fraction of the field's radius the mark reaches out to."""
    shift: tuple[float, float] = (0.0, 0.0)
    """Nudge off centre, as a fraction of the field's radius.

    For balancing a face by eye: a design that shares the field with a legend
    at one end rarely looks centred when it is centred. Deliberately not
    allowed to shrink the mark to make room for itself -- a shift that pushes
    the mark past the rim raises instead, so ``fill`` keeps meaning exactly
    what it says and the two knobs stay independent.
    """

    def _raw(self) -> Sketch:
        raise NotImplementedError

    def sketch(self, field: float) -> Sketch:
        """The mark, centred on the origin, fitted to a ``field``mm circle."""
        if not 0 < self.fill <= 1:
            raise MotifError(f"fill {self.fill} is not a fraction of the field")
        raw = self._raw()
        if self.rotate:
            raw = Rot(Z=self.rotate) * raw
        if raw.area <= 0:
            raise MotifError(f"{type(self).__name__} produced an empty mark")
        # Centred on the bounding box rather than on the centre of area: a
        # squiggle with a heavy left half should still sit in the middle of the
        # coin, which is what the eye reads as centred.
        bbox = raw.bounding_box()
        centred = Pos(-bbox.center().X, -bbox.center().Y) * raw
        # ``about`` has to be spelled out: scale() otherwise works about the
        # object's location, which the centring move has just displaced, and
        # the mark slides off the field by however far it was drawn from zero.
        fitted = scale(
            centred, field / 2 * self.fill / reach(centred), about=(0, 0, 0)
        )
        if self.shift == (0.0, 0.0):
            return fitted
        moved = Pos(*(s * field / 2 for s in self.shift)) * fitted
        overrun = reach(moved) - field / 2
        if overrun > 1e-9:
            raise MotifError(
                f"shifted by {self.shift}, the mark overruns the field by "
                f"{overrun:.2f}mm and would climb the rim. Drop fill to "
                f"{self.fill * (1 - overrun / (field / 2)):.2f}, or shift less."
            )
        return moved


@dataclass(frozen=True)
class Text(Fitted):
    """A word or a number, set in a system font."""

    text: str = ""
    font: str = DEFAULT_FONT
    style: FontStyle = FontStyle.BOLD
    fill: float = 0.66
    """Text reads as an inscription rather than a pattern, so it is set well
    inside the field even when nothing else shares the face."""

    def _raw(self) -> Sketch:
        if not self.text.strip():
            raise MotifError("empty text")
        # Set at an arbitrary size; sketch() rescales to the field anyway.
        return _TextObject(
            self.text, font_size=10.0, font=self.font, font_style=self.style,
            align=(Align.CENTER, Align.CENTER),
        )


@dataclass(frozen=True)
class Stroke(Fitted):
    """A mark drawn as a line of constant width -- a signature, a squiggle.

    ``points`` and ``width`` share one arbitrary unit; only their ratio
    survives fitting. ``smooth`` runs a spline through the points, which is
    what makes a squiggle look drawn rather than folded.
    """

    points: tuple[Point, ...] = ()
    width: float = 1.0
    smooth: bool = True
    closed: bool = False

    def _raw(self) -> Sketch:
        if len(self.points) < 2:
            raise MotifError("a stroke needs at least two points")
        pts = list(self.points)
        if self.closed and pts[0] != pts[-1]:
            pts.append(pts[0])
        curve = (_SplineObject if self.smooth else _PolylineObject)(*pts)
        # Kind.ARC rounds the joins and caps, so the stroke reads as one
        # continuous pen line instead of a chain of mitred segments.
        if self.closed:
            # A closed line has an inside, so Side.BOTH on the wire only ever
            # hands back the outer loop. Grow and shrink the enclosed face
            # instead and keep the band between them.
            enclosed = make_face(Wire(curve.edges()))
            return offset(enclosed, self.width / 2, kind=Kind.ARC) - offset(
                enclosed, -self.width / 2, kind=Kind.ARC
            )
        wires = offset(
            Wire(curve.edges()), self.width / 2, side=Side.BOTH, kind=Kind.ARC
        ).wires()
        if len(wires) != 1:
            raise MotifError(
                f"stroke offset to {len(wires)} loops; width {self.width} is "
                f"probably wide enough that the line overlaps itself"
            )
        return make_face(wires[0])


@dataclass(frozen=True)
class Outline(Fitted):
    """A filled polygon -- the hat monotile, a state, a logo traced by hand."""

    points: tuple[Point, ...] = ()

    def _raw(self) -> Sketch:
        if len(self.points) < 3:
            raise MotifError("an outline needs at least three points")
        return make_face(_PolygonObject(*self.points, align=None).wire())


@dataclass(frozen=True)
class Svg(Fitted):
    """Whatever an illustrator saved, as long as its paths are closed.

    Holes are recovered by even-odd nesting: loops are taken largest first, and
    a loop that lands inside what is already filled cuts a hole instead of
    adding to it. That handles a letter 'o' or a ring; a hole inside a hole
    inside a hole is beyond it, so trace those by hand as an Outline.
    """

    path: Path | None = None

    def _raw(self) -> Sketch:
        if self.path is None:
            raise MotifError("no svg path given")
        if not self.path.exists():
            raise MotifError(f"no such svg: {self.path}")
        shapes = import_svg(self.path)
        faces = [s if isinstance(s, Face) else make_face(s) for s in shapes]
        if not faces:
            raise MotifError(f"{self.path} has no closed paths to fill")
        result = None
        for face in sorted(faces, key=lambda f: -f.area):
            if result is None:
                result = face
            elif (result & face).area > face.area / 2:
                result -= face
            else:
                result += face
        return result


@dataclass(frozen=True)
class Legend(Motif):
    """Text curved around the field's edge, the way a coin carries its date.

    Placed against the rim rather than through the shared fitting path: a
    legend's job is to hug the edge at a legible size, so it is sized by
    ``height`` and positioned by ``margin``, and never scaled to fill anything.
    """

    text: str = ""
    font: str = DEFAULT_FONT
    style: FontStyle = FontStyle.BOLD
    """Bold, not regular, and not as a matter of taste. Measured as 2*area over
    perimeter -- exact for a ribbon, and a letter is a ribbon bent about -- a
    legend set regular at this size runs 0.20-0.36mm through the stroke in
    every face on this machine, under the 0.42mm the nozzle lays down. Bold
    plus the size below is what carries it over."""
    height: float = 0.13
    """Type size as a fraction of the field diameter; 0.13 of a 34mm field is
    about 4.4mm, which holds a 0.5mm stroke in a face of uniform weight."""
    at: str = TOP
    """TOP arches the legend over the field; BOTTOM sits it in the exergue."""
    margin: float = 0.06
    """Clear space kept between the legend and the rim, as a fraction of field."""

    def sketch(self, field: float) -> Sketch:
        if not self.text.strip():
            raise MotifError("empty legend")
        if self.at not in (TOP, BOTTOM):
            raise MotifError(f"legend goes {TOP!r} or {BOTTOM!r}, not {self.at!r}")
        size = field * self.height
        limit = field / 2 - field * self.margin
        if size <= 0 or limit <= 0:
            raise MotifError(f"legend height {self.height} leaves no room")
        # Glyphs stand off the baseline circle -- outward along the top, inward
        # along the bottom -- by an amount only the font knows. So set it once
        # at the limit, measure how far it actually reached, and pull the
        # baseline in by the overshoot. One correction is exact to the extent
        # the standoff does not itself depend on the radius, which over a
        # fraction of a millimetre it does not.
        radius = limit - (reach(self._on_circle(limit, size)) - limit)
        if radius <= 0:
            raise MotifError(
                f"a legend {self.height:.2f} of the field tall does not fit "
                f"inside it at all; the glyphs alone overrun the rim"
            )
        # Set flat, the same string measures how much of the circle it will
        # take once it is bent round. Past about seven eighths of the way the
        # end of the legend runs into its own beginning.
        span = _flat_width(self, size) / radius
        if span > 1.75 * math.pi:
            raise MotifError(
                f"{self.text!r} at {self.height:.2f} of the field wraps "
                f"{math.degrees(span):.0f} degrees round it and does not fit: "
                f"the end of the legend meets its own start. Shorten it, or "
                f"set height below {self.height * 1.75 * math.pi / span:.3f}."
            )
        text = self._on_circle(radius, size)
        return Rot(Z=self.rotate) * text if self.rotate else text

    def _on_circle(self, radius: float, size: float) -> Sketch:
        if radius <= 0:
            raise MotifError(f"legend {self.text!r} does not fit the field")
        # A circle drawn on Plane.XY runs counter-clockwise, which stands the
        # glyphs upright along the *bottom* of the coin. Flipping the plane's
        # normal reverses the run and arches them over the top instead. Either
        # way three-quarters of the way round is where the text centres.
        plane = (
            Plane.XY if self.at == BOTTOM
            else Plane((0, 0, 0), (1, 0, 0), (0, 0, -1))
        )
        return _TextObject(
            self.text, font_size=size, font=self.font, font_style=self.style,
            path=Edge.make_circle(radius, plane), position_on_path=0.75,
            align=None,
        )


def reach(sketch: Sketch, samples: int = 16) -> float:
    """How far the mark gets from the origin, in mm.

    Sampled along the edges rather than taken from the vertices: an arc bulges
    between its two ends, and a mark fitted to its vertices would push that
    bulge over the rim.
    """
    return max(
        math.hypot(p.X, p.Y)
        for edge in sketch.edges()
        for p in (edge.position_at(i / samples) for i in range(samples + 1))
    )


def _flat_width(legend: Legend, size: float) -> float:
    """Width of the legend's text set on a straight line, in mm."""
    return _TextObject(
        legend.text, font_size=size, font=legend.font, font_style=legend.style,
        align=(Align.CENTER, Align.CENTER),
    ).bounding_box().size.X


def combine(motifs: Motif | tuple[Motif, ...], field: float) -> Sketch:
    """Union of every motif on one face, fitted to the same field."""
    marks = (motifs,) if isinstance(motifs, Motif) else tuple(motifs)
    if not marks:
        raise MotifError("a face needs at least one motif")
    result = marks[0].sketch(field)
    for mark in marks[1:]:
        result += mark.sketch(field)
    return result
