"""Raised lettering: outlines in, bevelled solid out.

Extracted from the MUSTANG badge, where the bevel was worked out the hard way,
and then rebuilt for the Lamborghini script, where the MUSTANG's method turned
out not to survive traced artwork. Both lessons are here because both are
expensive to rediscover.

EVERY NATIVE WAY OF BEVELLING LETTERFORMS FAILS. ``chamfer()`` on the top edges
gives "try a smaller length"; ``extrude(taper=...)`` gives "incoherent
intersection"; ``loft`` between an outline and a smaller one gives "the number
of holes must be the same" as soon as a counter closes. That leaves offsetting,
and the bevel is built by stacking offsets.

OCCT'S OFFSET IS NOT DEPENDABLE ON REAL OUTLINES, WHICH IS WHY SHAPELY IS HERE.
``offset(..., Kind.ARC)`` manages the MUSTANG's glyphs up to 0.8mm and refuses
past it. On the Lamborghini script it does something worse: it succeeds, hands
back a face that reports ``is_valid`` as true, and that face cannot be meshed
at any tessellation setting -- "3mf mesh is invalid" -- because the offset has
quietly self-intersected somewhere along a 1668-vertex outline. A part that
builds, validates, and then cannot be exported is the worst failure shape
available, so offsetting is done by Shapely's ``buffer`` instead, which
resolves self-intersections properly and produces outlines that mesh at every
amount tried.

CURVES ARE THE PROBLEM, SO CURVES ARE REMOVED. Straight off an SVG the script
is 187 Bezier edges over six wires, and in that form OCCT will not offset it,
will not chamfer it, and fuses an overlapping shape into it by cutting slivers
rather than merging -- three faces in, nine disjoint faces out. Sampled into
dense polylines, the identical region fuses to a single face. The tolerance
that makes this a good trade is a printed 0.4mm bead: at ``POLY_STEP`` the area
error is under 0.01%.

THE BEVEL IS A FLARE DOWNWARD, NOT A TAPER UPWARD. Same solid, described from
the other end, and worth the swap:

  - the top face is the untouched outline, so no thin stroke can be eaten by
    the bevel however large it gets -- which matters most for exactly the
    delicate artwork you would want a bevel on;
  - every layer is smaller than the one below, so nothing overhangs and it
    prints face up with support off.

The flare is a staircase of ``LAYER``-high steps rather than a ramp. That is
not an approximation in the finished part: a 45 degree bevel printed at 0.2mm
layers IS a staircase of 0.2mm steps.
"""

from __future__ import annotations

import numpy as np
from build123d import Face, Part, Polygon, Pos, Sketch, extrude
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

POLY_STEP = 0.4
"""mm. Segment length when replacing curves with polylines. Reproduces the
Lamborghini script's true area to better than 0.01% -- the chord error on a 2mm
radius is about 10 microns -- and halving it buys no measurable accuracy while
adding a thousand edges and most of a minute to the build."""

QUAD_SEGS = 4
"""Segments per quarter circle in a rounded offset corner. Four puts the chord
error on a 0.6mm fillet at about 5 microns."""

LAYER = 0.2
"""mm. Step height of the flare, set to the print's layer height so the steps
modelled here and the steps printed are the same steps."""

MAX_GROW = 0.8
"""mm. Kept as guidance rather than a limit: it is where OCCT's own offset gave
up on the MUSTANG glyphs. Shapely goes further, but a bevel is still bounded by
the GAPS between strokes -- close those and the outline merges into itself,
which is a design failure rather than a numerical one."""


def _rings(face: Face, step: float) -> ShapelyPolygon:
    """A face as a Shapely polygon, curves sampled into polylines.

    The largest ring is the outer boundary and the rest are holes, which is
    always right for letterforms: a counter is enclosed by its glyph.
    """
    rings = []
    for wire in face.wires():
        count = max(16, int(wire.length / step))
        points = [wire @ t for t in np.linspace(0, 1, count, endpoint=False)]
        rings.append([(p.X, p.Y) for p in points])
    if not rings:
        raise ValueError("face has no wires")
    rings.sort(key=len, reverse=True)
    return ShapelyPolygon(rings[0], rings[1:])


def _to_sketch(geom) -> Sketch:
    """Shapely geometry back into a build123d sketch, holes preserved.

    Rings are re-oriented counter-clockwise first. Winding decides the face
    normal, and a clockwise ring gives a face pointing at -Z, which makes
    ``extrude(amount=+h)`` build downwards -- the badge comes out spanning
    -3.4 to 4.0 instead of 0 to 4.0. Shapely does not promise an orientation,
    so it has to be imposed here.
    """
    parts = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    sketch = None
    for poly in parts:
        if poly.is_empty:
            continue
        poly = orient(poly, sign=1.0)
        face = Polygon(*list(poly.exterior.coords)[:-1], align=None)
        for hole in poly.interiors:
            face = face - Polygon(*list(hole.coords)[:-1], align=None)
        sketch = face if sketch is None else sketch + face
    if sketch is None:
        raise ValueError("offset produced nothing")
    return sketch


def polygonise(face: Face, step: float = POLY_STEP) -> Sketch:
    """Rebuild a face with every curve replaced by a dense polyline.

    The enabling trick for traced artwork -- reach for it the moment a boolean
    or an offset misbehaves on an imported outline. See the module docstring
    for what it fixes.
    """
    return _to_sketch(_rings(face, step))


def grow(sketch: Sketch, amount: float, step: float = POLY_STEP) -> Sketch:
    """Outward offset, via Shapely rather than OCCT.

    Rounded joins, matching ``Kind.ARC``: the alternative extends both edges
    until they meet, and that is what goes wrong where outline edges are nearly
    parallel or very short.
    """
    if amount <= 1e-9:
        return sketch
    merged = unary_union([_rings(f, step) for f in sketch.faces()])
    grown = merged.buffer(amount, join_style=1, quad_segs=QUAD_SEGS)
    if grown.is_empty:
        raise ValueError(f"offsetting by {amount}mm produced nothing")
    return _to_sketch(grown)


def flared(sketch: Sketch, top: float, bevel: float, layer: float = LAYER) -> Part:
    """A solid whose top face at ``top`` is exactly ``sketch``, flaring outward
    on the way down to the plate.

    One piece, one height. The MUSTANG badge needed a two-height variant for
    its rail and paid for it: fusing a grown outline to a grown rectangle
    returns a partial result on some inputs, silently, and the extruded slab
    then annihilates its neighbour. Keep to one height where the design allows.
    """
    if bevel <= 0:
        return extrude(sketch, amount=top)

    steps = max(1, round(bevel / layer))
    rise = bevel / steps
    shoulder = top - bevel
    if shoulder <= 0:
        raise ValueError(
            f"a {bevel}mm bevel does not fit under a face {top}mm off the plate"
        )

    solid = extrude(grow(sketch, bevel), amount=shoulder)
    for k in range(steps):
        # Offset that puts each slab's TOP on the ideal bevel, so the topmost
        # slab carries the untouched outline.
        section = grow(sketch, bevel - (k + 1) * rise)
        solid += Pos(0, 0, shoulder + k * rise) * extrude(section, amount=rise)
    return solid
