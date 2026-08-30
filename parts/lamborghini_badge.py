"""Lamborghini script badge: a traced wordmark, bevelled, taped to a panel.

Same job as the MUSTANG badge and the same construction underneath -- outlines
flared downward into a bevelled solid, printed face up, held on with automotive
tape -- but the lettering comes from a vector source rather than a font, and
that changes what is hard about it.

WHERE THE LETTERING COMES FROM. There is no font for this. The flowing
"Lamborghini" script is custom artwork; the nearest published lookalike is
La Macchina, and "nearest lookalike" is exactly the compromise the MUSTANG
badge had to accept because its letterforms only exist as a typeface at one
remove. Here the real outlines are available, so they are used: the wordmark is
vendored as ``assets/lamborghini_wordmark.svg`` and imported directly. Nothing
is traced from a photograph and nothing is approximated by a similar face.

Worth being clear that this is the brand wordmark, which is a trademark. Fine
for a badge on your own car; not something to sell.

WHY THE SCRIPT PRINTS BETTER THAN BLOCK CAPITALS. Set MUSTANG in capitals and
you get seven islands that have to be tied together with a rail, and above that
rail every layer is seven separate small shapes. That cost two failed prints --
not because it printed badly, but because the printer's clumping detector fires
on layer 10 and could not make sense of a part that had just gone from one
island to seven. A script has no such transition. The strokes join, so this
badge is ONE piece from the plate to the top face, with two exceptions.

THE TWO EXCEPTIONS ARE THE DOTS ON THE i's. They float 3.3mm clear of the body,
5mm across, and left alone they would print as two loose discs to be aligned by
eye on the car. They are joined with a short stem each, cut along the line of
closest approach so it follows the stroke rather than crossing it -- the script
is italic, so "straight down" would be visibly wrong; the measured directions
are 95 and 104 degrees. ``dot_neck`` sets the width and zero leaves them
detached, which is a legitimate choice if you would rather place them by hand.

The stem is full height rather than a low-relief web. A web would be less
visible, but it would mean two pieces at two different heights, and the MUSTANG
badge established what that costs: fusing a grown outline to a grown rectangle
returns a partial result on some inputs, silently, and the bad slab annihilates
its neighbour on the next union. One height, one union, no silent failures.

HOW THIN IT GETS. A script has hairlines, so this was measured rather than
assumed -- and it had to be measured on a raster, because OCCT will not erode
this outline at any amount. Filling the shape and taking a distance transform
gives, at 150mm wide: median stroke 3.5mm, tenth percentile 1.8mm, and only the
calligraphic taper TIPS below 1.2mm. Tips taper to a point by definition and
simply round off at the nozzle. Nothing needs thickening at this size; below
about 110mm it would.

Note that the flare helps here rather than hurting: it is applied outward and
downward, so a 1.8mm stroke keeps its 1.8mm top face and gains a wider base.
Tapering inward from the base would have thinned every stroke by twice the
bevel and eaten the hairlines outright.

WHY ASA, AND THE TAPE. Both as the MUSTANG badge: direct sun for years, so ASA
rather than PLA; loaded in shear on a vertical panel, which is the mode a
pressure-sensitive adhesive is strongest in, so the real questions are heat,
airflow and surface prep rather than bond area. Use a proper automotive
attachment tape and clean the paint with isopropyl first.

PRINTED 2026-08-17, first attempt, and it came out almost perfect. Two marks
left, both characteristic of the process rather than of this design:

  - a slight ridge where the ironing pass meets the perimeter. Ironing runs
    0.21mm inside the wall by default, at 10% flow, so it drags melt outward and
    piles it against the boundary. ``ironing_inset`` in the tool's SLICE is
    raised to hold the pass further off.
  - visible layer lines on the flanks. Unavoidable here: the bevel is a
    staircase of 0.2mm steps by construction, and the flanks are that staircase.
    A finer layer height would soften it at the cost of print time; nothing
    about the geometry can.

Worth knowing what the print settles: the badge held to the textured plate with
NO BRIM, which together with the MUSTANG's three brimless prints is enough to
stop asking for one. The tool's slice settings now match what actually printed
rather than the heavier defaults inherited from the MUSTANG badge.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from build123d import (
    Axis,
    Cylinder,
    Part,
    Polygon,
    Pos,
    Rotation,
    Sketch,
    import_svg,
    scale,
)

from geom.relief import MAX_GROW, flared, grow, polygonise

ASA_DENSITY = 1.07
"""g/cm^3 for Bambu ASA."""

MIN_WALL = 1.2
"""mm. Below this a feature is fewer than three extrusions wide on a 0.4mm
nozzle. The script's stroke tips go below it and that is fine -- see the module
docstring -- but a sustained stroke should not."""

WORDMARK = Path(__file__).parent / "assets/lamborghini_wordmark.svg"

SAMPLES = 2000
"""Points along an outline when looking for the closest approach between the
body and a dot. The body's outline is 733mm long at final size, so this is
roughly a sample every 0.4mm -- fine enough that the stem lands on the stroke
rather than near it."""


@dataclass
class Params:
    """Everything the badge is. Defaults are a 150mm script badge."""

    source: Path = WORDMARK
    length: float = 150.0
    """Overall width, bevel included. The dimension you measure off the car.
    The script is 4.26:1, so 150mm comes out 35mm tall -- taller than a capitals
    badge of the same width, because of the L's loop and the descenders."""

    thickness: float = 4.0
    """Letter thickness, bond face to letter face."""

    bevel: float = 0.6
    """Width of the flare at the base of the strokes. Zero gives slab sides.

    Bounded by the gaps between strokes rather than by the strokes themselves:
    the flare closes those gaps, and once two strokes meet the offset fails.
    Measured on this artwork, the ceiling is about ``length / 190`` -- 0.8mm at
    150mm, 0.6mm at 110-130mm, 0.4mm at 90mm."""

    dot_neck: float = 1.6
    """Width of the stem joining each i-dot to the body. Zero leaves the dots
    as separate pieces, to be placed by hand."""

    neck_bite: float = 0.6
    """How far the stem reaches into the body and into the dot, so the union is
    a real overlap rather than two shapes touching along a line."""

    sag: float = 0.0
    """Measured hollow needed in the back for a crowned panel. Zero is flat."""

    tape: float = 1.1
    """Automotive attachment tape thickness, for reporting only."""

    def validate(self) -> None:
        if self.length <= 0:
            raise ValueError("length must be positive")
        if not Path(self.source).is_file():
            raise ValueError(f"no wordmark at {self.source}")
        if self.thickness < MIN_WALL:
            raise ValueError(
                f"thickness is {self.thickness}mm, under the {MIN_WALL}mm that "
                f"prints as a solid on a 0.4mm nozzle"
            )
        if self.bevel < 0:
            raise ValueError("bevel is a width, so it cannot be negative")
        if self.bevel >= self.thickness:
            raise ValueError(
                f"bevel ({self.bevel}mm) is at least the thickness "
                f"({self.thickness}mm), so there would be no flat face left"
            )
        if self.bevel > MAX_GROW:
            raise ValueError(
                f"bevel {self.bevel}mm is past the {MAX_GROW}mm where outward "
                f"offsets on letterforms start failing outright"
            )
        if self.dot_neck < 0:
            raise ValueError("dot_neck is a width")
        if 0 < self.dot_neck < MIN_WALL:
            raise ValueError(
                f"dot_neck is {self.dot_neck}mm, under the {MIN_WALL}mm that "
                f"prints as a solid. Use 0 to leave the dots detached instead"
            )
        if self.sag < 0:
            raise ValueError("sag is a hollow depth, so it cannot be negative")

    @property
    def radius(self) -> float | None:
        """Panel radius implied by the measured sag, across ``length``.

        The sagitta relation R = c^2/(8s) + s/2, because nobody can measure a
        radius on a car and everybody can measure a gap under a straightedge.
        """
        if self.sag <= 0:
            return None
        return self.length**2 / (8 * self.sag) + self.sag / 2


def artwork(params: Params) -> Sketch:
    """The wordmark at final size, sitting on the origin.

    Solved to ``length`` minus the bevel it will grow on each side, since the
    flare adds material outside this outline. Otherwise every badge prints wide.
    """
    raw = Sketch() + list(import_svg(str(params.source)))
    box = raw.bounding_box()
    if box.size.X <= 0:
        raise ValueError(f"{params.source} imported as nothing")
    target = params.length - 2 * params.bevel
    if target <= 0:
        raise ValueError(
            f"a {params.length}mm badge is narrower than the {params.bevel}mm "
            f"bevel on each end leaves room for"
        )
    sized = scale(raw, target / box.size.X)
    box = sized.bounding_box()
    return Pos(-box.min.X, -box.min.Y) * sized


def _outline(face, count: int) -> np.ndarray:
    """Points spaced along a face's outer boundary."""
    wire = face.outer_wire()
    return np.array(
        [(p.X, p.Y) for p in (wire @ t for t in np.linspace(0, 1, count))]
    )


def _stem(body, dot, width: float, bite: float) -> Sketch:
    """A bar joining ``dot`` to ``body`` along their line of closest approach.

    Along that line rather than straight down: the script is italic, so the
    stem has to lean with the stroke or it reads as a mistake.
    """
    bp = _outline(body, SAMPLES)
    dp = _outline(dot, SAMPLES // 8)
    gaps = np.linalg.norm(bp[:, None, :] - dp[None, :, :], axis=2)
    i, j = np.unravel_index(np.argmin(gaps), gaps.shape)
    a, b = bp[i], dp[j]

    span = np.linalg.norm(a - b)
    if span < 1e-9:
        return None  # already touching
    along = (a - b) / span
    across = np.array([-along[1], along[0]]) * (width / 2)
    start, end = b - along * bite, a + along * bite
    return Polygon(
        tuple(start - across), tuple(end - across),
        tuple(end + across), tuple(start + across),
        align=None,
    )


def joined(params: Params) -> Sketch:
    """The wordmark with its dots stemmed to the body: one connected outline.

    Everything is polygonised on the way through, and that is not a detail.
    Straight off the SVG this outline cannot be offset by any amount, cannot be
    chamfered, and fuses a stem into itself by cutting slivers instead of
    merging -- 3 faces in, 9 disjoint faces out. Polygonised, the same union is
    a single face and the bevel works. See ``geom.relief.polygonise``.
    """
    art = artwork(params)
    faces = art.faces()
    if not faces:
        raise ValueError("the wordmark has no faces")

    body = max(faces, key=lambda f: f.area)
    dots = [f for f in faces if f is not body]

    sketch = polygonise(body)
    for dot in dots:
        sketch = sketch + polygonise(dot)
        if params.dot_neck > 0:
            stem = _stem(body, dot, params.dot_neck, params.neck_bite)
            if stem is not None:
                sketch = sketch + stem

    if params.dot_neck > 0 and len(sketch.faces()) != 1:
        raise ValueError(
            f"the badge came out as {len(sketch.faces())} regions instead of "
            f"one. A stem is not reaching -- raise neck_bite, or set dot_neck "
            f"to 0 to accept the dots as separate pieces"
        )
    return sketch


def _counters(sketch: Sketch) -> int:
    """How many enclosed loops the outline has -- the holes in a, o, g, b.

    Counted as inner wires. A face always has exactly one outer wire, so the
    rest are counters.
    """
    return sum(max(0, len(f.wires()) - 1) for f in sketch.faces())


def _hollow(params: Params, part: Part) -> Part:
    """Cut a cylindrical valley into the back for a crowned panel."""
    if params.sag <= 0:
        return part
    radius = params.radius
    assert radius is not None
    span = part.bounding_box().size.Y * 4
    cutter = Rotation(90, 0, 0) * Cylinder(radius=radius, height=span)
    return part - Pos(0, 0, params.sag - radius) * cutter


def build(params: Params | None = None) -> Part:
    """The badge, bond face down on z=0, script growing in +Z."""
    params = params or Params()
    params.validate()

    sketch = joined(params)
    if params.bevel > 0:
        # The flare widens the strokes AND narrows the gaps between them, so on
        # a small enough badge it closes the counters -- the enclosed loops in
        # the a, o, g and so on. The top face keeps them, the base does not, and
        # the result is a letter whose hole does not go through. Shapely
        # resolves that merge without complaining, which makes it a silent
        # quality failure rather than a crash, so it is counted here.
        # Roughly, this bites below length/190: 0.6mm is fine at 150mm and not
        # at 90mm.
        before = _counters(sketch)
        after = _counters(grow(sketch, params.bevel))
        if after != before:
            change = (
                f"closes {before - after}" if after < before
                else f"traps {after - before} new"
            )
            raise ValueError(
                f"a {params.bevel}mm bevel {change} counter(s) on a "
                f"{params.length}mm badge ({before} -> {after}). Either a letter "
                f"would have a hole in its face and solid material behind it, or "
                f"two strokes have merged and trapped a pocket between them. "
                f"Keep the bevel under about length/190 "
                f"({params.length / 190:.2f}mm here), or make the badge bigger"
            )

    badge = flared(sketch, params.thickness, params.bevel)

    box = badge.bounding_box()
    if abs(box.min.Z) > 1e-6 or abs(box.max.Z - params.thickness) > 1e-6:
        raise ValueError(
            f"the badge spans z {box.min.Z:.3f}..{box.max.Z:.3f} instead of "
            f"0..{params.thickness}. A slab was lost assembling the stack"
        )

    badge = _hollow(params, badge)
    expected = 1 if params.dot_neck > 0 else 3
    if len(badge.solids()) != expected:
        raise ValueError(
            f"the badge came out as {len(badge.solids())} pieces, expected "
            f"{expected}"
        )
    if not badge.is_valid:
        raise ValueError("the badge is not a valid solid")
    return badge


def plated(params: Params | None = None, part: Part | None = None) -> Part:
    """Print orientation: bond face on the plate, script up."""
    params = params or Params()
    part = build(params) if part is None else part
    return Pos(0, 0, -part.bounding_box().min.Z) * part


def report(params: Params | None = None, part: Part | None = None) -> dict:
    """Numbers worth knowing before committing filament to it."""
    params = params or Params()
    part = build(params) if part is None else part
    box = part.bounding_box()
    volume = part.volume / 1000.0
    back = box.min.Z
    bond = sum(
        f.area
        for f in part.faces().filter_by_position(Axis.Z, back - 0.01, back + 0.01)
    )
    return {
        "length": box.size.X,
        "height": box.size.Y,
        "thickness": box.size.Z,
        "aspect": box.size.X / box.size.Y if box.size.Y else 0.0,
        "volume_cm3": volume,
        "mass_g": volume * ASA_DENSITY,
        "bond_area_mm2": bond,
        "pieces": len(part.solids()),
        "panel_radius": params.radius,
    }


def variant(params: Params | None = None, **changes) -> Params:
    return replace(params or Params(), **changes)
