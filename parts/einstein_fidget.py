"""Nested print-in-place fidget built from the 'hat' aperiodic monotile.

Same idea as the nested-hexagon fidgets, but the rings follow the hat outline
instead of a hexagon.

Every ring surface bulges outward by ``interlock`` at mid height and returns to
its nominal offset at the top and bottom faces. Because inner and outer
surfaces move together, neighbouring rings keep a constant gap while the
V-profile captures them: a ring can shift about ``gap / slope`` vertically
before it wedges, which is enough travel to feel and not enough to escape.

Getting that profile built is the whole difficulty. A Z-varying cross-section
looks like a job for a loft, but a loft joins sampled points between two
*different* offsets, and those samples do not correspond -- by arc length or
by ray-casting from a fixed centre, measured clearances collapsed to 0.13mm
and 0.005mm against a 0.3mm nominal, tight enough to fuse the rings solid.
Straight prisms fixed the gaps but left nothing holding the rings in, so they
fell out independently and it stopped being a fidget.

So the profile is built as a stack of thin prisms instead. Every slice is a
true offset of the outline, so at any height two neighbouring surfaces are two
true offsets differing by exactly ``gap`` -- exact clearance by construction,
at every height, with no correspondence to get wrong. The staircase is a
non-issue: the slicer discretises to layers regardless, so the steps are set
to one layer.

Corners are rounded (``Kind.ARC``) rather than mitered: rounding survives
about 20% deeper before the offset fails, which is what makes five rings fit
where mitering managed three. The outline stops looking much like a hat
several rings in, which is the accepted trade -- nesting is the point.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from build123d import (
    Kind,
    Part,
    Plane,
    Polygon,
    Sketch,
    extrude,
    make_face,
    offset,
)

from geom.hat import MAX_INSET_UNITS, find_hat


@dataclass(frozen=True)
class Params:
    """Everything the GUI would expose as a slider."""

    unit: float = 12.0
    """mm per hat unit; the tile's short edge is 1 unit."""
    rings: int = 5
    """Movable rings inside the base outline, between it and the core."""
    outer_rings: int = 15
    """Movable rings added outside the base outline, to reach coaster size.

    Growing outward is the cheap direction: an outward offset of a simple
    polygon is always simple, so there is no collapse limit the way there is
    going inward. The outline does round off as it grows -- the hat's notch
    fills in after a few rings -- but the tile stays legible at the centre.
    """
    thickness: float = 5.0
    """Overall height in mm."""
    wall: float = 1.2
    """Radial wall thickness of each ring; keep a multiple of line width."""
    gap: float = 0.3
    """Print-in-place clearance between neighbouring rings."""
    interlock: float = 0.8
    """Mid-height bulge capturing the rings. Must exceed ``gap`` to hold.

    0.6 printed as "possible to pop out, but takes a bit of effort" on a
    5-ring set. Raised for the 20-ring coaster, where each ring only has to
    give a little for the play to add up across the stack. Engagement past the
    clearance goes 0.3 -> 0.5mm; the flare stays 18 degrees off vertical, well
    inside what prints unsupported.
    """
    slice_height: float = 0.2
    """Step of the stacked profile; one layer, so the steps never show."""

    @property
    def pitch(self) -> float:
        return self.wall + self.gap

    @property
    def deepest_inset(self) -> float:
        """Inset of the innermost surface, in mm."""
        return self.rings * self.pitch

    @property
    def total_rings(self) -> int:
        return self.outer_rings + self.rings

    @property
    def size(self) -> tuple[float, float]:
        """Overall footprint in mm, before the mid-height bulge."""
        pts = [(x * self.unit, y * self.unit) for x, y in find_hat()]
        grown = 2 * self.outer_rings * self.pitch
        span = lambda k: max(p[k] for p in pts) - min(p[k] for p in pts)
        return (span(0) + grown, span(1) + grown)

    @property
    def step(self) -> float:
        """Sideways jog of the profile from one slice to the next.

        Must stay well under ``gap``: the stack holds the clearance within a
        slice, but a step that approaches the gap lets one ring's slab hang
        over its neighbour's slab a level down, and they fuse.
        """
        return self.interlock * self.slice_height / (self.thickness / 2)

    @property
    def engagement(self) -> float:
        """How much of the bulge actually captures, past the clearance."""
        return self.interlock - self.gap

    @property
    def travel(self) -> float:
        """Vertical movement a ring gets before the profile wedges it."""
        slope = self.interlock / (self.thickness / 2)
        return self.gap / slope

    def validate(self) -> None:
        ratio = self.deepest_inset / self.unit
        if ratio > MAX_INSET_UNITS:
            need = self.deepest_inset / MAX_INSET_UNITS
            raise ValueError(
                f"{self.rings} rings at {self.pitch}mm pitch inset "
                f"{self.deepest_inset:.2f}mm = {ratio:.2f} hat units, past the "
                f"{MAX_INSET_UNITS} ceiling where there is no interior left to "
                f"offset into. Raise unit to >= {need:.1f}mm, or drop a ring."
            )
        if self.engagement <= 0:
            raise ValueError(
                f"interlock {self.interlock}mm does not exceed the {self.gap}mm "
                f"gap, so nothing captures the rings and they fall out"
            )
        if self.step >= self.gap / 2:
            allowed = (self.gap / 2) * (self.thickness / 2) / self.interlock
            raise ValueError(
                f"slices {self.slice_height}mm tall step the profile sideways by "
                f"{self.step:.3f}mm, too close to the {self.gap}mm gap. The stack "
                f"only holds the gap within a slice; once a step approaches it, a "
                f"ring's slab overhangs its neighbour's slab one level down and "
                f"the two fuse into one solid. Use slice_height <= {allowed:.2f}mm."
            )


@lru_cache(maxsize=None)
def _outline(params: Params, inset: float) -> Sketch:
    """Hat outline scaled to mm and offset by ``inset`` (negative = outward)."""
    pts = [(x * params.unit, y * params.unit) for x, y in find_hat()]
    face = make_face(Polygon(*pts, align=None).wire())
    if inset == 0:
        return face
    face = offset(face, -inset, kind=Kind.ARC)
    if inset > 0:
        # A split or vanished outline cannot be a ring at all.
        if len(face.wires()) != 1:
            raise ValueError(
                f"inset {inset:.2f}mm split the outline into "
                f"{len(face.wires())} loops"
            )
        if face.area < (2 * params.wall) ** 2:
            raise ValueError(f"inset {inset:.2f}mm left too little area to ring")
    return face


def _bulge(params: Params, z: float) -> float:
    """Outward displacement of every surface at height ``z``.

    Zero at both faces, ``interlock`` at mid height. Identical for inner and
    outer surfaces, which is what keeps the gap constant while still capturing.
    """
    half = params.thickness / 2
    return params.interlock * (1.0 - abs(z - half) / half)


def _levels(params: Params) -> list[tuple[float, float]]:
    """(bottom z, height) of each slice in the stack."""
    n = max(1, round(params.thickness / params.slice_height))
    step = params.thickness / n
    return [(i * step, step) for i in range(n)]


def _ring(params: Params, outer_inset: float, inner_inset: float | None) -> Part:
    """One body, stacked slice by slice.

    ``inner_inset`` of None builds the solid core rather than a ring.
    """
    body = None
    for z0, height in _levels(params):
        # Both surfaces of this slice are evaluated at the same height, so they
        # stay exact offsets of one another -- and so do the facing surfaces of
        # the neighbouring bodies.
        shift = _bulge(params, z0 + height / 2)
        face = _outline(params, round(outer_inset - shift, 6))
        if inner_inset is not None:
            face = face - _outline(params, round(inner_inset - shift, 6))
        slab = extrude(Plane.XY.offset(z0) * face, amount=height)
        body = slab if body is None else body + slab
    return body


def build(params: Params = Params()) -> Part:
    """The whole nested assembly, as printed: rings already interleaved."""
    params.validate()
    bodies = [
        _ring(params, i * params.pitch, i * params.pitch + params.wall)
        for i in range(-params.outer_rings, params.rings)
    ]
    bodies.append(_ring(params, params.deepest_inset, None))
    result = bodies[0]
    for body in bodies[1:]:
        result = result + body
    return result


if __name__ == "__main__":
    from pathlib import Path

    from build123d import export_stl

    p = Params()
    part = build(p)
    out = Path("out")
    out.mkdir(exist_ok=True)
    export_stl(part, str(out / "einstein_fidget.stl"))
    bb = part.bounding_box()
    print(f"valid={part.is_valid} solids={len(part.solids())}")
    print(f"bbox={bb.size.X:.1f} x {bb.size.Y:.1f} x {bb.size.Z:.1f} mm")
    print(f"volume={part.volume/1000:.2f} cm^3")
    print(f"engagement={p.engagement:.2f} mm, travel=+/-{p.travel:.2f} mm")
