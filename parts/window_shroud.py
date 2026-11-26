"""Window shroud: lets an indoor camera look out through a window at night.

Put a camera behind glass and it sees two things that are not outside. The
room, reflected off the inner face of the pane, and -- once it is dark -- its
own infrared LEDs, which light the glass a few centimetres away far brighter
than anything in the yard, so the picture goes to a white haze. The cure is the
one photographers use against shop windows: put the lens right up to the glass
and black out everything around it.

THE TILT. The camera is held looking ``tilt`` degrees down through the pane,
25 by default, which is what a camera on an upstairs sill wants. That decides
most of the shape. The head's top edge tips towards the glass and its bottom
edge away, so the shroud is a wedge: thin at the top, deep at the bottom, with
the camera's face as its sloping back wall. And the lens can come no closer to
the glass than the top of the head allows, about 14mm on a C100 at 25 degrees,
where untilted it could sit 4mm off. The C100's lens is close to a fisheye --
117 degrees across the diagonal -- so the picture's lower corners now meet the
pane at 75 degrees and the patch of glass the camera looks through is about
86mm wide, half again as wide as the camera.

So it is two chambers in one convex body, both open on the glass.

THE LENS CHAMBER is exactly the space the camera looks through: the convex
hull of the picture's footprint on the glass and the lower part of the face,
around the bezel. Nothing of the shroud can be in the picture, because the
picture is inside a convex hole and the shroud is outside it. The light sensor,
the port and the status light are on the face below the lens, so they look out
through this chamber too.

THE LED CHAMBER is the space in front of the face above the lens, where the
two IR LEDs are. A partition between the chambers runs from the face, just
above the bezel and below the LEDs, down to the glass. Infrared that the inner
surface of the pane reflects comes back into the LED chamber and the partition
stands between it and the lens. On the C100 the LEDs' own axes reach the glass
right at the partition, since they are only 10mm above the lens on the face and
the top of the picture rises off the lens axis at 30 degrees, so the lower part
of each LED's beam lands on the partition rather than the glass. That is the
light that would otherwise land on the glass inside the picture, where it
hazes worst; the yard is lit by the upper part.

THE SKIRT is everything outside the chambers: a sleeve that slides over the
front of the camera's head, a lip inside it that the head's white rim sits on,
and the wedge's walls down to the glass, which seal the pane against room
light.

WHAT IT CANNOT FIX. In a double-glazed window a little of the LED light enters
the inner pane outside the lens chamber, crosses the gap and reflects off the
outer pane, and comes back in through the chamber's mouth. On low-e glass in
particular that can still show as a faint ghost. No shroud stops it, because it
arrives through the glass the camera has to look through. If it is bad, turn
the camera's infrared off and light the yard instead. The status light, inside
the lens chamber, is best switched off in the app.

HOW IT PRINTS. Glass face down. Both chambers and the skirt come off the
plate, the flattest face the part has and the one that has to seal. The
wedge's outer walls lean outwards going down, so they are no overhang. What
leans the other way is the sleeve, at the tilt, and the inside of the lens
chamber and the partition, at whatever the footprint asks; ``validate``
refuses tilts past 40 degrees and the tests hold every face under 45. Matte
black, PETG or ASA: PLA softens in a sunny window.

The camera's dimensions come from geom.cameras, which says which are sourced,
which measured and which estimated. ``coupon`` builds just the sleeve band, a
ten-minute print, to check the fit before committing to the whole thing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from build123d import (
    Box,
    Face,
    Part,
    Plane,
    Pos,
    Shell,
    Solid,
    Vector,
    Wire,
    extrude,
)
from scipy.spatial import ConvexHull
from shapely.geometry import MultiPoint, Point, Polygon, box
from shapely.geometry.polygon import orient

from geom import cameras
from geom.cameras import Camera

MIN_WALL = 0.8
"""Two extrusions on the 0.4mm nozzle. Thinner than this a wall is one bead,
and one bead of black plastic is not light-tight."""

MAX_GRAZE = 80.0
"""Degrees off square to the glass for the steepest ray of the picture. Past
this the picture's lower corners run along the pane and the footprint, and the
shroud with it, grows without bound."""

MAX_LEAN = 44.0
"""Degrees off vertical for the lens chamber's inside walls, a hair inside the
45 FDM holds."""

MAX_TILT = 40.0
"""Degrees. The sleeve leans by the tilt, and past 40 its underside is closer
to the 45 FDM holds than is comfortable for a face the camera has to slide
into."""

EPS = 0.01
CUT = 0.2
"""How far the chambers overrun the glass and the face, so neither leaves a
skin."""


@dataclass(frozen=True)
class Params:
    """Everything the GUI would expose as a slider."""

    camera: Camera = cameras.TAPO_C100
    """The head this is cut for. Its estimated fields are the ones to measure."""

    tilt: float = 25.0
    """Degrees the camera looks down through the glass. 0 looks straight out."""
    standoff: float | None = None
    """Height of the face's centre above the glass, mm. None puts it as close
    as ``edge_gap`` and ``glass_gap`` allow, which is best optically: every
    millimetre further back widens the patch of glass the picture spans."""
    edge_gap: float = 1.2
    """Plastic between the glass and the sleeve's top edge, which is the part
    of the head the tilt brings nearest the pane."""
    glass_gap: float = 2.0
    """Clearance between the lens bezel's top edge and the glass."""
    margin: float = 2.5
    """Degrees added all round the published field of view when sizing the
    lens chamber, so a camera a little off-square in the sleeve still sees no
    shroud. A 0.3mm fit lets the head rock about half a degree, so 2.5 is
    plenty; any more and, at 25 degrees, the chamber's lower corners lean past
    what prints, because the squircle face pulls its corners in."""

    wall: float = 2.0
    """Skirt and sleeve wall, and the skirt's width round the chambers on the
    glass."""
    fit: float = 0.3
    """Gap each side between the head and the sleeve. A slide fit; the rim
    sitting on the lip is what locates the camera, not the sleeve."""
    sleeve: float = 8.0
    """How far back the sleeve reaches over the head. Short enough to stay
    clear of the microSD slot and the stalk."""

    partition: float = 1.2
    """Thickness of the wall between the lens chamber and the LED chamber."""
    hood_clearance: float = 1.0
    """Gap between the lens bezel and the partition where it meets the face.
    Covers the lens position being an estimate as much as it covers the
    print."""
    partition_drop: float = 3.0
    """How far below the lens centre the partition meets the face away from
    the bezel. Lower leaves the IR LEDs more room above it; higher keeps
    more of the face in the lens chamber."""

    flange: float = 6.0
    """Extra width at the glass end, all round, for adhesive. 0 for none, if the
    camera's own stand holds it to the glass."""
    flange_thickness: float = 2.0

    stalk_notch: float = 14.0
    """Width of a slot in the sleeve's bottom wall for the stand's stalk, open
    at the back. 0 for none."""
    notch_keep: float = 3.0
    """How much of the sleeve's bottom wall the notch leaves behind the lip."""

    pad: float = 0.3
    """Added round the picture's footprint on the glass, for the print."""

    # --- the frame the camera's face sits in -------------------------------
    #
    # The shroud's own coordinates: the glass is z = 0 and the part prints on
    # it; z goes back into the room, y up the window, x along it. The face is a
    # plane through ``centre`` with the camera's ``up`` and the shroud's x in
    # it, and the camera looks along ``forward``.

    @property
    def up(self) -> np.ndarray:
        t = math.radians(self.tilt)
        return np.array([0.0, math.cos(t), -math.sin(t)])

    @property
    def forward(self) -> np.ndarray:
        t = math.radians(self.tilt)
        return np.array([0.0, -math.sin(t), -math.cos(t)])

    @property
    def normal(self) -> np.ndarray:
        """Out of the face, back towards the camera: the sleeve's axis."""
        return -self.forward

    @property
    def lens2d(self) -> tuple[float, float]:
        """Lens centre on the face, in the face's own x and up: x mirrored,
        because the shroud is modelled from behind the camera."""
        return -self.camera.lens_x, self.camera.lens_y

    @property
    def head(self) -> Polygon:
        """The head's outline, as you face the camera (it is symmetric, so
        mirroring it changes nothing)."""
        return Polygon(self.camera.outline("head"))

    @property
    def sleeve_inner(self) -> Polygon:
        return self.head.buffer(self.fit, quad_segs=16)

    @property
    def sleeve_outer(self) -> Polygon:
        return self.head.buffer(self.fit + self.wall, quad_segs=16)

    @property
    def sleeve_size(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.sleeve_inner.bounds
        return x1 - x0, y1 - y0

    @property
    def outer_size(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.sleeve_outer.bounds
        return x1 - x0, y1 - y0

    @property
    def closest(self) -> float:
        """The least face-centre height that keeps the sleeve's top edge and
        the bezel's top edge off the glass."""
        t = math.radians(self.tilt)
        c = self.camera
        _, ly = self.lens2d
        edge = self.edge_gap + self.outer_size[1] / 2 * math.sin(t)
        bezel = (self.glass_gap + (ly + c.bezel_d / 2) * math.sin(t)
                 + c.bezel_proud * math.cos(t))
        return max(edge, bezel)

    @property
    def centre_z(self) -> float:
        return self.closest if self.standoff is None else self.standoff

    @property
    def centre(self) -> np.ndarray:
        return np.array([0.0, 0.0, self.centre_z])

    def on_face(self, xy, offset: float = 0.0) -> np.ndarray:
        """Face coordinates (x, up) to the shroud's, ``offset`` back along the
        normal."""
        xy = np.atleast_2d(np.asarray(xy, float))
        return (self.centre + xy[:, :1] * np.array([1.0, 0, 0])
                + xy[:, 1:2] * self.up + offset * self.normal)

    def to_world(self, d: np.ndarray) -> np.ndarray:
        """Camera-frame directions (x right, y up, -z forward) to the
        shroud's."""
        d = np.atleast_2d(d)
        return (d[:, :1] * np.array([1.0, 0, 0]) + d[:, 1:2] * self.up
                + d[:, 2:3] * self.normal)

    @property
    def lens(self) -> np.ndarray:
        """The lens centre, on the face."""
        return self.on_face(self.lens2d)[0]

    @property
    def pupil(self) -> np.ndarray:
        """The entrance pupil: where the picture fans out from."""
        c = self.camera
        return self.lens + (c.bezel_proud - c.pupil_depth) * self.forward

    @property
    def pupil_height(self) -> float:
        """The pupil's height above the glass."""
        return float(self.pupil[2])

    def rays(self, n: int = 24, margin: float | None = None):
        """Origins round the pupil's rim and directions round the picture's
        edge, every pairing, in the shroud's coordinates."""
        c = self.camera
        margin = self.margin if margin is None else margin
        u, v = c.border(n)
        dirs = self.to_world(c.direction(u, v, margin))
        r = c.pupil_diameter / 2
        a = np.linspace(0, 2 * math.pi, 8, endpoint=False)
        rim = (self.pupil + r * np.outer(np.cos(a), [1.0, 0, 0])
               + r * np.outer(np.sin(a), self.up))
        origins = np.repeat(rim, len(dirs), axis=0)
        return origins, np.tile(dirs, (len(rim), 1))

    @property
    def chamber_lean(self) -> float:
        """Degrees off vertical of the lens chamber's steepest ceiling: the
        inside walls where the picture, spreading towards the glass, is wider
        below than the face is above."""
        pts = _lens_chamber_points(self)
        h = ConvexHull(pts)
        worst = 0.0
        for simplex, eq in zip(h.simplices, h.equations):
            z = pts[simplex, 2]
            if np.all(z < 0) or abs(eq[:3] @ self.normal - 1) < 1e-6:
                continue  # the glass, or the face: both open
            worst = max(worst, math.degrees(math.asin(np.clip(eq[2], -1, 1))))
        return worst

    @property
    def graze(self) -> float:
        """Degrees off square to the glass of the picture's steepest ray."""
        _, d = self.rays(48)
        return float(np.degrees(np.arccos(np.clip(-d[:, 2].max(), -1, 1))))

    # --- outlines ----------------------------------------------------------

    @property
    def footprint(self) -> Polygon:
        """Where the picture lands on the glass, with the margin and the pad."""
        o, d = self.rays(48)
        s = -o[:, 2] / d[:, 2]
        hits = o[:, :2] + s[:, None] * d[:, :2]
        return MultiPoint([tuple(p) for p in hits]).convex_hull.buffer(self.pad)

    @property
    def panel(self) -> Polygon:
        return Polygon(self.camera.outline("panel"))

    @property
    def lens_face(self) -> Polygon:
        """The part of the face inside the lens chamber: the panel below the
        partition, swept up round the bezel."""
        c = self.camera
        lx, ly = self.lens2d
        cut = ly - self.partition_drop
        low = self.panel.intersection(box(-c.panel_w, -c.panel_h, c.panel_w, cut))
        bezel = Point(lx, ly).buffer(c.bezel_d / 2 + self.hood_clearance, quad_segs=32)
        return low.union(bezel).convex_hull.intersection(self.panel)

    @property
    def lens_face_walled(self) -> Polygon:
        return self.lens_face.buffer(self.partition, quad_segs=16)

    @property
    def led_face(self) -> Polygon:
        """The part of the face inside the LED chamber."""
        return self.panel.difference(self.lens_face_walled)

    @property
    def features(self) -> tuple[tuple[str, float, float, float], ...]:
        """Face features in face coordinates: name, x, up, diameter."""
        lx, ly = self.lens2d
        return tuple((f.name, lx - f.x, ly + f.y, f.diameter) for f in self.camera.features)

    def validate(self) -> None:
        c = self.camera
        for name, value in (("wall", self.wall), ("partition", self.partition)):
            if value < MIN_WALL:
                raise ValueError(
                    f"a {value}mm {name} is under {MIN_WALL}mm, two extrusions on "
                    f"the 0.4mm nozzle; one bead of black plastic still lets "
                    f"light through"
                )
        if not 0 <= self.tilt <= MAX_TILT:
            raise ValueError(
                f"a {self.tilt} degree tilt is outside 0 to {MAX_TILT:.0f}: the "
                f"sleeve leans by the tilt, and past {MAX_TILT:.0f} its "
                f"underside is too close to what prints without support"
            )
        graze = self.graze
        if graze > MAX_GRAZE:
            raise ValueError(
                f"at a {self.tilt:.0f} degree tilt the picture's lower corners "
                f"meet the glass {graze:.0f} degrees off square, past "
                f"{MAX_GRAZE:.0f}: they all but run along the pane, so the "
                f"patch of glass the camera sees, and the shroud round it, "
                f"would be enormous. Tilt less or lower the margin."
            )
        lean = self.chamber_lean
        if lean > MAX_LEAN:
            raise ValueError(
                f"at a {self.tilt:.0f} degree tilt the picture is so much wider "
                f"on the glass than the face that the inside of the lens "
                f"chamber would overhang {lean:.0f} degrees, past the "
                f"{MAX_LEAN:.0f} that prints without support. Tilt less, or "
                f"lower the margin."
            )
        if self.edge_gap < MIN_WALL:
            raise ValueError(
                f"a {self.edge_gap}mm edge gap leaves less than a wall's worth "
                f"of plastic between the sleeve's top edge and the glass"
            )
        if self.centre_z < self.closest - 1e-9:
            raise ValueError(
                f"a {self.centre_z:.1f}mm standoff puts the head or the "
                f"{c.name}'s lens bezel against the glass at a {self.tilt:.0f} "
                f"degree tilt. Give it at least {self.closest:.1f}mm, or leave "
                f"it as None."
            )
        if not self.head.buffer(-0.5).contains(self.panel):
            raise ValueError("the face panel has to be smaller than the head, or "
                             "there is no rim for the lip to bear on")
        if self.sleeve > c.head_depth - 2:
            raise ValueError(
                f"a {self.sleeve}mm sleeve reaches past the back of a "
                f"{c.head_depth}mm-deep head"
            )
        if self.stalk_notch and self.notch_keep >= self.sleeve:
            raise ValueError("the stalk notch starts behind the end of the sleeve")
        if self.led_face.is_empty or self.led_face.area < 1:
            raise ValueError(
                "the lens chamber takes the whole face, leaving no room above "
                "the bezel for the LED chamber; lower partition_drop or "
                "hood_clearance"
            )
        lens, leds = self.lens_face, self.led_face
        for name, x, y, d in self.features:
            spot = Point(x, y).buffer(d / 2, quad_segs=16)
            if "IR LED" in name and not leds.contains(spot):
                raise ValueError(
                    f"the partition would sit on the {name}: it meets the face "
                    f"{self.lens_face_walled.exterior.distance(Point(x, y)):.1f}mm "
                    f"from its centre, inside its {d / 2:.1f}mm radius. Lower "
                    f"partition_drop or tighten hood_clearance, or check the "
                    f"measurements."
                )
            if not (lens.contains(spot) or leds.contains(spot)):
                raise ValueError(
                    f"the {name} is under the partition or the lip, so it would "
                    f"see plastic instead of the glass; move partition_drop or "
                    f"check the measurements"
                )


def _head_outline(params: Params) -> Polygon:
    """The sleeve's outside, a hair proud of it, so the body under the sleeve
    never leaves the sleeve's edge hanging over it."""
    return params.sleeve_outer.buffer(0.05, quad_segs=16)


def _ring(poly: Polygon) -> np.ndarray:
    return np.asarray(poly.exterior.coords)[:-1]


def _flat(poly: Polygon, z: float) -> np.ndarray:
    xy = _ring(poly)
    return np.column_stack([xy, np.full(len(xy), z)])


def hull(points) -> Solid:
    """The convex hull of a cloud of points, as a solid. Facets that share a
    plane are merged, so the flat glass face comes out as one face."""
    pts = np.asarray(points, float)
    h = ConvexHull(pts)
    groups: list[tuple[np.ndarray, set[int]]] = []
    for simplex, eq in zip(h.simplices, h.equations):
        for g_eq, members in groups:
            if np.allclose(g_eq, eq, atol=1e-7):
                members.update(int(i) for i in simplex)
                break
        else:
            groups.append((eq, {int(i) for i in simplex}))
    faces = []
    for eq, members in groups:
        n = eq[:3]
        idx = np.array(sorted(members))
        p = pts[idx]
        mid = p.mean(axis=0)
        e1 = p[0] - mid
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        ang = np.arctan2((p - mid) @ e2, (p - mid) @ e1)
        loop = p[np.argsort(ang)]
        faces.append(Face(Wire.make_polygon([Vector(*q) for q in loop], close=True)))
    return Solid(Shell(faces))


def _face_plane(params: Params, offset: float = 0.0) -> Plane:
    """A plane on the face, its y the camera's up and its z back along the
    sleeve."""
    origin = params.centre + offset * params.normal
    return Plane(origin=Vector(*origin), x_dir=(1, 0, 0), z_dir=Vector(*params.normal))


def glass_outline(params: Params) -> Polygon:
    """The skirt's outside on the glass: the footprint and the LED chamber
    with a wall round them, and the head's own outline dropped straight down,
    so no outer wall leans out over the plate."""
    head = _ring(_head_outline(params))
    dropped = params.on_face(head)[:, :2]
    led = params.on_face(_ring(params.led_face))[:, :2]
    inner = MultiPoint([tuple(q) for q in led]).union(params.footprint).convex_hull
    return inner.buffer(params.wall, quad_segs=12).union(
        MultiPoint([tuple(q) for q in dropped]).convex_hull).convex_hull


def _lens_chamber_points(params: Params, grow: float = 0.0) -> np.ndarray:
    glass = params.footprint.buffer(grow, quad_segs=12)
    face = params.lens_face.buffer(grow, quad_segs=12)
    return np.vstack([_flat(glass, -CUT), params.on_face(_ring(face), CUT)])


def lens_chamber(params: Params, grow: float = 0.0) -> Solid:
    """What the lens looks through, overrun past the glass and the face. With
    ``grow``, the same a wall's thickness bigger all round, to cut the LED
    chamber away from."""
    return hull(_lens_chamber_points(params, grow))


def led_chamber(params: Params) -> Part:
    """Straight down from the face above the partition to the glass, with
    the lens chamber and its partition taken out. Straight down, and not out
    along the LEDs' axes, so its walls are vertical and nothing of it runs
    in under the lens chamber, which would leave the partition hanging."""
    below = params.on_face(_ring(params.led_face))[:, :2]
    outline = Wire.make_polygon([Vector(x, y, -CUT) for x, y in below], close=True)
    prism = extrude(Face(outline), amount=params.centre_z + params.outer_size[1] + 10,
                    dir=(0, 0, 1))
    behind = _face_plane(params, CUT) * Pos(0, 0, 500) * Box(1000, 1000, 1000)
    return prism - behind - lens_chamber(params, params.partition)


def _sleeve(params: Params, plane: Plane, lip: float) -> Part:
    """The sleeve band on a face plane: ``lip`` of solid ring in front of the
    face with the panel's opening through it, then the sleeve behind."""
    _, oh = params.outer_size
    _, sh = params.sleeve_size
    top = params.sleeve
    band = plane * Pos(0, 0, -lip) * extrude(_sketch(params.sleeve_outer), amount=lip + top)
    band -= plane * Pos(0, 0, -lip - 1) * extrude(_sketch(params.panel),
                                                  amount=lip + 1 + EPS)
    band -= plane * extrude(_sketch(params.sleeve_inner), amount=top + 1)
    if params.stalk_notch:
        z0 = params.notch_keep
        band -= plane * Pos(0, -(oh + sh) / 4, (z0 + top + 1) / 2) * Box(
            params.stalk_notch, (oh - sh) / 2 + 2, top + 1 - z0)
    return band


def _sketch(poly: Polygon) -> Face:
    """A shapely outline as a flat face on XY, wound so it extrudes up."""
    ring = _ring(orient(poly, 1.0))
    return Face(Wire.make_polygon([Vector(x, y, 0) for x, y in ring], close=True))


def skirt(params: Params) -> Part:
    """The body and sleeve with no chambers cut: the convex wedge from the
    glass outline to the head's outline on the face, and the sleeve on it."""
    body = hull(np.vstack([
        _flat(glass_outline(params), 0.0),
        params.on_face(_ring(_head_outline(params))),
    ]))
    return body + _sleeve(params, _face_plane(params), 0.0)


def build(params: Params) -> Part:
    """The shroud, glass face on the plate, camera end up."""
    params.validate()
    part = skirt(params)
    if params.flange:
        f = params.flange
        ring = glass_outline(params).buffer(f, quad_segs=12)
        part += hull(np.vstack([_flat(ring, 0.0), _flat(ring, params.flange_thickness)]))
    part -= lens_chamber(params)
    part -= led_chamber(params)
    return part


def coupon(params: Params, lip: float = 1.2) -> Part:
    """The sleeve and a sliver of lip, square to the plate: prints in minutes
    and says whether the head fits before the whole shroud is committed to."""
    params.validate()
    return _sleeve(params, Plane.XY.offset(lip), lip)


if __name__ == "__main__":
    from pathlib import Path

    from build123d import export_stl

    p = Params()
    part = build(p)
    out = Path("out")
    out.mkdir(exist_ok=True)
    export_stl(part, str(out / "window_shroud.stl"))
    bb = part.bounding_box()
    fp = p.footprint.bounds
    print(f"valid={part.is_valid} solids={len(part.solids())}")
    print(f"bbox={bb.size.X:.1f} x {bb.size.Y:.1f} x {bb.size.Z:.1f} mm")
    print(f"lens {p.pupil_height:.1f} mm off the glass, footprint "
          f"{fp[2] - fp[0]:.1f} x {fp[3] - fp[1]:.1f} mm")
