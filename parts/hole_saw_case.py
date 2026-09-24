"""Parametric case for a hole saw set: a tray of pockets and a hinged lid.

Built around Harbor Freight's Warrior 57523 -- eight carbon steel saws from 1
to 2-1/2 inches, the large mandrel with its pilot bit, and the hex key -- but
nothing in it knows that except the catalogue entry in ``geom.hole_saws``. Every
dimension below follows from the pieces it holds, the clearances around them,
and the fact that both halves print without support.

WHY IT IS ONE STACK. The saws nest: each drops into the cup of the next size
up and stands on its back plate, a couple of millimetres prouder than the one
round it. All eight together stand 0.37 in taller than one saw on its own,
in the floor space of the biggest. So the set is one pocket, not eight, and the
case is barely taller than a case of flat saws while its footprint is set by
the mandrel -- which is the longest thing in it and does not nest in anything.
Stood teeth-up, because the teeth are the one part of a saw that should touch
nothing. ``nest`` splits the set into shorter stacks, down to 1 for every saw
in its own pocket, which trades floor for height.

HOW THE POCKETS ARE LAID OUT. The mandrel lies along the front wall, half
buried in a cradle turned to its own profile. Then the stacks are dropped in,
largest first, each to the lowest place it fits and then the leftmost -- the
classic bottom-left rule -- and the key goes wherever of its eight ways of
lying lets it sit lowest, all inside a tray of some width. Which width is the
whole question, so ``layout`` tries every one in whole millimetres and keeps
the smallest case. That is a greedy packing, not an optimal one; with eight
flat saws a constrained optimiser found about five percent more, which is not
worth a solver as a dependency or a layout that moves when a clearance does.

HOW IT STAYS SHUT. A hinge along the back on a length of 1.75mm filament, and a
bead along the lid's front lip that clicks into a groove in the front wall. The
pin is a press fit in the tray's knuckles and runs free in the lid's, so it
cannot walk out and the lid still swings. ``STEEL_PIN`` swaps the filament for
2mm steel rod, which does not creep, wear or bend, for a case that gets thrown
in a truck; nothing changes but the bores. The bead is small because it is the
tray's wall that gives, not the lip: the wall stands clear of the tray for the
top half of its height and has a long span either side of the snap to bow
over.

WHY THE LID HAS NO LIP ALONG THE BACK. The hinge locates that edge already,
and a lip there would swing down into the tray as the lid opens -- into
whatever is standing along the back wall.

The lid is modelled closed, in place over the tray, because that is where the
tests want it; ``lid_for_print`` turns it over onto its face for the plate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

from build123d import (
    Align,
    Box,
    Circle,
    Cylinder,
    FontStyle,
    Part,
    Plane,
    Polygon,
    Pos,
    RectangleRounded,
    Rot,
    Text,
    extrude,
)
from shapely import affinity
from shapely.geometry import Point as _Point
from shapely.geometry import box as _box
from shapely.ops import unary_union

from geom.hole_saws import WARRIOR_57523, HoleSaw, HoleSawSet

MIN_WEB = 0.8
"""Two extrusion widths at 0.4mm. The narrowest wall that is still a wall
rather than a single line the slicer may or may not keep."""

MIN_WALL = 1.2

MAX_SIDE = 246.0
"""The P2S plate is 256mm square; a 5mm margin each side. The two halves are
printed one at a time, so each has the whole plate to itself."""


@dataclass(frozen=True)
class Pin:
    """What the hinge turns on, and what is added to it for each half's bores.

    Printed holes come out a tenth or two undersize, so a bore a little over
    the pin grips it and one well over lets it turn.
    """

    name: str
    diameter: float
    press: float
    """Added to the pin for the tray's knuckles, which hold it."""
    play: float
    """Added to the pin for the lid's, which turn on it."""


FILAMENT_PIN = Pin("1.75mm filament", 1.75, 0.1, 0.4)
STEEL_PIN = Pin("2mm steel rod", 2.0, 0.15, 0.4)
"""5/64in music wire or a 2mm drill blank, cut to length. Filament gives a
little as it is pushed in and steel gives nothing, so the tray's knuckles get
an extra twentieth to keep the press from splitting them."""
PINS = {"filament": FILAMENT_PIN, "steel": STEEL_PIN}


@dataclass(frozen=True)
class Params:
    kit: HoleSawSet = WARRIOR_57523

    clearance: float = 0.8
    """Added to every saw's diameter for its pocket. A saw dropped into a
    pocket should fall to the bottom, not have to be pushed."""
    mandrel_clearance: float = 0.6
    """Radial, all along the mandrel's cradle, and at each end."""
    key_clearance: float = 0.5
    web: float = 1.2
    """Least plastic between any two pockets, or a pocket and the margin."""

    wall: float = 2.0
    floor: float = 1.6
    headroom: float = 1.0
    """Between the tips of the tallest saw and the underside of the lid."""
    corner: float = 6.0
    """Outer corner radius of the case, in plan."""

    lid: float = 2.4
    lip: float = 5.0
    """How far the lid's lip drops inside the walls."""
    lip_wall: float = 1.2
    lip_gap: float = 0.2
    snap: float = 0.4
    """How far the bead stands proud of the lip -- and so how far the front
    wall bows to let it past."""
    snap_play: float = 0.15

    notch: float = 8.0
    """Radius of the thumb notch in the front wall's top edge."""
    notch_depth: float = 4.0

    pin: Pin = FILAMENT_PIN
    """The hinge pin: a length of filament, or ``STEEL_PIN``."""
    knuckle: float = 4.0
    """Radius of the hinge knuckles."""
    knuckles: int = 5
    knuckle_length: float = 11.0
    knuckle_gap: float = 0.4
    hinge_gap: float = 0.6
    """Clear air between the back wall and the knuckles' round."""

    push: float = 16.0
    """Diameter of the hole under each saw, for pushing it up out of its pocket
    from below. Capped at a fraction of the pocket, so the floor keeps a ring."""
    grip: float = 8.0
    """How far the mandrel's cradle is dug out at flange depth either side of the
    flange, so a finger and thumb can get round it."""
    key_scoop: float = 11.0
    """Diameter of the finger well at the end of the key's slot."""
    label: float = 0.4
    """Depth each saw's size is engraved into its pocket floor. 0 for none."""
    font: str = "DejaVu Sans"

    nest: int = 8
    """Most saws to a stack. The whole set nests, so the default is one
    stack; 1 lays every saw out flat in its own pocket."""

    max_side: float = MAX_SIDE

    # ---- heights ----------------------------------------------------------

    @property
    def cradle_radius(self) -> float:
        return self.kit.mandrel_diameter / 2 + self.mandrel_clearance

    @property
    def seat(self) -> float:
        """Top of the tray: the mandrel's axis, so it lies half buried.

        Deep for a saw pocket, which is fine -- a saw is lifted by a finger
        inside its cup or pushed from below, never pinched from the side.
        """
        return self.floor + self.cradle_radius

    @property
    def rim(self) -> float:
        """Top of the walls, where the lid sits."""
        tallest = max(max(s.height for s in self.stacks), 2 * self.cradle_radius)
        return self.floor + tallest + self.headroom

    @property
    def height(self) -> float:
        """The closed case, less the knuckles, which stay inside it."""
        return self.rim + self.lid

    @property
    def margin(self) -> float:
        """Kept clear inside the walls, all round, for the lid's lip to drop
        into past everything standing taller than the lip's bottom edge."""
        return self.lip_gap + self.lip_wall + 0.6

    # ---- hinge ------------------------------------------------------------

    @property
    def axis_z(self) -> float:
        """The hinge line sits one knuckle radius below the top of the lid,
        less a flat, so the lid's knuckles land on the plate face down."""
        return self.height - self.knuckle + 0.4

    @property
    def axis_offset(self) -> float:
        """Behind the back wall's outside face."""
        return self.knuckle + self.hinge_gap

    @property
    def stacks(self) -> tuple[Stack, ...]:
        """The saws, largest first, nested ``nest`` to a stack."""
        saws = sorted(self.kit.saws, key=lambda s: -s.inches)
        return tuple(Stack(tuple(saws[i:i + self.nest]), self.kit.rise)
                     for i in range(0, len(saws), self.nest))

    def pocket(self, stack: Stack) -> float:
        """Only the outermost saw sits in the pocket; it holds the rest."""
        return stack.outer.diameter + self.clearance

    def push_hole(self, stack: Stack) -> float:
        return min(self.push, 0.45 * self.pocket(stack))

    def validate(self) -> None:
        if self.web < MIN_WEB:
            raise ValueError(f"a {self.web}mm web is thinner than two extrusions")
        if self.wall < MIN_WALL or self.lip_wall < MIN_WEB:
            raise ValueError("walls under three extrusions do not hold a snap")
        if min(self.clearance, self.mandrel_clearance, self.key_clearance) < 0:
            raise ValueError("a negative clearance is an interference fit")
        if self.lip >= self.rim - self.seat:
            raise ValueError(
                f"a {self.lip}mm lip reaches down past the top of the tray"
            )
        if self.snap + self.snap_play >= self.wall - MIN_WEB:
            raise ValueError("the snap groove would cut through the front wall")
        if self.snap_play < 0 or self.snap <= 0:
            raise ValueError("a bead that stands in nothing is not a snap")
        if self.pin.diameter <= 0 or self.pin.press < 0:
            raise ValueError("a pin needs a diameter and a bore at least its size")
        if self.pin.play <= self.pin.press:
            raise ValueError("the lid's knuckles would grip the pin as hard as the tray's")
        hole = self.pin.diameter + self.pin.play
        if self.knuckle - hole / 2 < 2 * MIN_WEB:
            raise ValueError(
                f"a {self.knuckle}mm knuckle leaves too little round a "
                f"{hole:.2f}mm pin hole"
            )
        if self.knuckle > self.height - self.floor:
            raise ValueError("knuckles taller than the case")
        if self.knuckles < 3 or self.knuckles % 2 == 0:
            raise ValueError(
                "the hinge wants an odd number of knuckles, three or more, so "
                "the tray holds the pin at both ends"
            )
        if self.notch_depth >= self.notch or self.notch_depth >= self.lip:
            raise ValueError("the thumb notch is deeper than it is round")
        if self.label < 0 or self.label >= self.floor - 2 * 0.2:
            raise ValueError("labels engraved through the floor")
        if self.nest < 1:
            raise ValueError("a stack holds at least one saw")
        for stack in self.stacks:
            for big, small in zip(stack.saws, stack.saws[1:]):
                if big.inches - small.inches < self.kit.nest_step - 1e-9:
                    raise ValueError(
                        f"a {small.nominal} saw does not nest in a {big.nominal}"
                    )


# ---- layout -----------------------------------------------------------------


@dataclass(frozen=True)
class Stack:
    """Saws nested one inside the next, the largest outermost."""

    saws: tuple[HoleSaw, ...]
    rise: float

    @property
    def outer(self) -> HoleSaw:
        return self.saws[0]

    @property
    def height(self) -> float:
        """Floor to the highest tooth: each saw in stands ``rise`` prouder."""
        return max(s.height + i * self.rise for i, s in enumerate(self.saws))

    @property
    def label(self) -> str:
        small, big = (s.nominal.removesuffix(" in") for s in (self.saws[-1], self.saws[0]))
        return f'{big}"' if len(self.saws) == 1 else f'{small}–{big}"'


@dataclass(frozen=True)
class Placed:
    stack: Stack
    x: float
    y: float


@dataclass(frozen=True)
class Layout:
    """Where everything sits, in the tray's interior: (0, 0) is the inside of
    the front-left corner, y runs back towards the hinge."""

    width: float
    depth: float
    stacks: tuple[Placed, ...]
    mandrel: tuple[float, float]
    """The mandrel's chuck end, on its axis."""
    key_bars: tuple[tuple[float, float, float, float], ...]
    """The key's two legs as (x0, y0, x1, y1), clearance in."""
    key_scoop: tuple[float, float]
    """Centre of the finger well at the end of the key's long leg."""
    reserved: object = field(repr=False, compare=False)
    """The mandrel's and key's footprints, as shapely geometry, clearances in."""


def _cradle_runs(params: Params) -> list[tuple[float, float, float]]:
    """(start, end, radius) along the mandrel's axis, clearance and grip in."""
    c = params.mandrel_clearance
    runs, x = [], c
    flange = max(params.kit.mandrel, key=lambda s: s.diameter)
    for seg in params.kit.mandrel:
        runs.append((x, x + seg.length, seg.diameter / 2 + c))
        if seg is flange:
            runs.append((x - params.grip, x + seg.length + params.grip,
                         seg.diameter / 2 + c))
        x += seg.length
    runs[0] = (0.0, runs[0][1], runs[0][2])
    runs[-1] = (runs[-1][0], x + c, runs[-1][2])
    return [(max(0.0, a), b, r) for a, b, r in runs]


def _cradle_plan(params: Params) -> list:
    """The cradle in plan, as the convex rectangles it is made of."""
    y = params.cradle_radius
    return [_box(a, y - r, b, y + r) for a, b, r in _cradle_runs(params)]


Rounded = tuple[float, float, float, float, float]
"""(x0, y0, x1, y1, r): a box grown by r. A point is a box of no size, so a
circle is one too, and every piece of the key and every obstacle it has to miss
is one of these."""


def _shape(g: Rounded):
    x0, y0, x1, y1, r = g
    core = (_Point(x0, y0) if x0 == x1 and y0 == y1
            else _box(x0, y0, x1, y1))
    return core.buffer(r, 16) if r > 0 else core


def _key_parts(params: Params, turn: int, flip: bool) -> list[Rounded]:
    """The key's slot and finger well about the outside corner of its bend.
    Unturned, the long leg runs along +x and the short one along +y; ``turn``
    quarter turns and a ``flip`` -- the key laid on its other face -- give the
    eight ways it can lie."""
    k, c = params.kit.key, params.key_clearance
    bar, long = k.across + 2 * c, k.long + 2 * c
    parts = [(0, 0, long, bar, 0.0), (0, 0, bar, k.short + 2 * c, 0.0),
             (long, bar / 2, long, bar / 2, params.key_scoop / 2)]
    out = []
    for x0, y0, x1, y1, r in parts:
        corners = [(x0, y0), (x1, y1)]
        if flip:
            corners = [(x, -y) for x, y in corners]
        for _ in range(turn):
            corners = [(-y, x) for x, y in corners]
        (ax, ay), (bx, by) = corners
        out.append((min(ax, bx), min(ay, by), max(ax, bx), max(ay, by), r))
    return out


def _lowest(region, key=lambda p: (round(p[1], 3), p[0])):
    """Bottom-most, then left-most, vertex of a shapely region."""
    pts = []
    for poly in getattr(region, "geoms", [region]):
        if poly.is_empty:
            continue
        pts += list(poly.exterior.coords)
        for ring in poly.interiors:
            pts += list(ring.coords)
    return min(pts, key=key)


def _place_key(params: Params, width: float, obstacles: list[Rounded]):
    """The key, in whichever of its eight lies lets its top sit lowest.

    Not a bottom-left drop of one point, because the key is not a circle: a
    point p is free for it exactly when p is outside every obstacle grown by
    the key's own shape turned backwards -- a Minkowski sum. Between two grown
    boxes that sum is itself a grown box, the boxes added and the radii added,
    so the free region is exact and cheap.
    """
    big = 10 * params.max_side
    best = None
    for turn in range(4):
        for flip in (False, True):
            parts = _key_parts(params, turn, flip)
            room = _box(-big, -big, big, big)
            for x0, y0, x1, y1, r in parts:
                room = room.intersection(
                    _box(r - x0, r - y0, width - x1 - r, big - y1))
            if room.is_empty:
                continue
            blocked = unary_union([
                _shape((ox0 - kx1, oy0 - ky1, ox1 - kx0, oy1 - ky0,
                        orad + krad + params.web + 0.01))
                for ox0, oy0, ox1, oy1, orad in obstacles
                for kx0, ky0, kx1, ky1, krad in parts
            ])
            free = room.difference(blocked)
            if free.is_empty:
                continue
            top = max(y1 + r for _, _, _, y1, r in parts)
            x, y = _lowest(free, key=lambda p: (round(p[1] + top, 3), p[0]))
            if best is None or (y + top, x) < best[0]:
                best = ((y + top, x), [(x0 + x, y0 + y, x1 + x, y1 + y, r)
                                       for x0, y0, x1, y1, r in parts])
    return None if best is None else best[1]


def _drop(params: Params, width: float):
    """Bottom-left fill of a tray ``width`` wide: the stacks largest first,
    each to the lowest place it fits and then the leftmost, then the key."""
    cradle = _cradle_plan(params)
    taken, placed = list(cradle), []
    solid: list[Rounded] = [(*g.bounds, 0.0) for g in cradle]
    for stack in sorted(params.stacks, key=lambda s: -s.outer.diameter):
        r = params.pocket(stack) / 2
        room = _box(r, r, width - r, 10 * params.max_side)
        if room.is_empty:
            return None
        # A hair over the web: the buffers are polygons, and they cut inside
        # the circles they stand for.
        blocked = unary_union([g.buffer(r + params.web + 0.01, 48) for g in taken])
        free = room.difference(blocked)
        if free.is_empty:
            return None
        x, y = _lowest(free)
        placed.append(Placed(stack, x, y))
        taken.append(_Point(x, y).buffer(r, 48))
        solid.append((x, y, x, y, r))
    key = _place_key(params, width, solid)
    if key is None:
        return None
    return tuple(placed), cradle, [_shape(g) for g in key]


@lru_cache(maxsize=None)
def layout(params: Params) -> Layout:
    """The smallest case the bottom-left rule finds, over every tray width.

    A width is skipped without packing it when even the shallowest tray it
    could be is bigger than the best found: no shallower than the biggest
    piece, and no smaller in area than the pieces themselves.
    """
    params.validate()
    border = 2 * (params.margin + params.wall)
    cradle = unary_union(_cradle_plan(params))
    pockets = [params.pocket(s) for s in params.stacks]
    shallowest = max([2 * params.cradle_radius] + pockets)
    pieces = cradle.area + sum(math.pi * d * d / 4 for d in pockets)
    narrowest = math.ceil(max(cradle.bounds[2], max(pockets)))
    best = None
    for width in range(narrowest, int(params.max_side - border) + 1):
        depth = max(shallowest, pieces / width)
        if best is not None and (width + border) * (depth + border) >= best[0]:
            continue
        dropped = _drop(params, float(width))
        if dropped is None:
            continue
        placed, cradle, key = dropped
        used = unary_union([*cradle, *key,
                            *(_Point(p.x, p.y).buffer(params.pocket(p.stack) / 2, 48)
                              for p in placed)]).bounds
        area = (used[2] + border) * (used[3] + border)
        if used[3] + border <= params.max_side and (best is None or area < best[0]):
            best = (area, used, placed, cradle, key)
    if best is None:
        raise ValueError(f"{params.kit.name} does not fit a {params.max_side:.0f}mm plate")
    _, used, placed, cradle, key = best
    m = params.margin
    moved = [affinity.translate(g, m, m) for g in key]
    return Layout(
        width=used[2] + 2 * m,
        depth=used[3] + 2 * m,
        stacks=tuple(Placed(p.stack, p.x + m, p.y + m) for p in placed),
        mandrel=(m, m + params.cradle_radius),
        key_bars=tuple(g.bounds for g in moved[:2]),
        key_scoop=(moved[2].centroid.x, moved[2].centroid.y),
        reserved=affinity.translate(unary_union([*cradle, *key]), m, m),
    )


# ---- solids -----------------------------------------------------------------


def _plan(params: Params, grow: float = 0.0):
    """The case's interior outline, let out by ``grow``."""
    lay = layout(params)
    radius = max(0.5, params.corner - params.wall + grow)
    return Pos(lay.width / 2, lay.depth / 2) * RectangleRounded(
        lay.width + 2 * grow, lay.depth + 2 * grow, radius
    )


def _pocket_cuts(params: Params) -> Part:
    cuts = Part()
    for p in layout(params).stacks:
        d, hole = params.pocket(p.stack), params.push_hole(p.stack)
        cuts += Pos(p.x, p.y, params.floor) * Cylinder(
            d / 2, params.rim, align=(Align.CENTER, Align.CENTER, Align.MIN)
        )
        cuts += Pos(p.x, p.y, -1) * Cylinder(
            hole / 2, params.floor + 2, align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
        if params.label > 0:
            ring = (d - hole) / 2
            size = min(6.0, 0.5 * ring)
            y = p.y + (hole / 2 + d / 2) / 2
            text = Text(p.stack.label, font_size=size, font=params.font,
                        font_style=FontStyle.BOLD)
            # Shrunk to the chord it sits on, if a long range would not fit.
            chord = 2 * math.sqrt(max(0.0, (d / 2) ** 2 - (y - p.y + size / 2) ** 2))
            wide = text.bounding_box().size.X
            if wide > 0.85 * chord:
                text = Text(p.stack.label, font_size=size * 0.85 * chord / wide,
                            font=params.font, font_style=FontStyle.BOLD)
            cuts += extrude(
                Plane.XY.offset(params.floor - params.label) * Pos(p.x, y) * text,
                amount=params.label + 0.5,
            )
    return cuts


def _cradle(params: Params) -> Part:
    x0, y = layout(params).mandrel
    cut = Part()
    for a, b, r in _cradle_runs(params):
        cut += Pos(x0 + a + (b - a) / 2, y, params.seat) * Rot(0, 90, 0) \
            * Cylinder(r, b - a)
    return cut


def key_depth(params: Params) -> float:
    """How far the key's slot goes down from the top of the tray."""
    return params.kit.key.across + params.key_clearance + 0.4


def _key_slot(params: Params) -> Part:
    lay = layout(params)
    depth = key_depth(params)
    cut = Part()
    for a, b, c, d in lay.key_bars:
        cut += Pos((a + c) / 2, (b + d) / 2, params.seat - depth) * Box(
            c - a, d - b, depth + 1, align=(Align.CENTER, Align.CENTER, Align.MIN)
        )
    cx, cy = lay.key_scoop
    cut += Pos(cx, cy, params.seat - depth - 1) * Cylinder(
        params.key_scoop / 2, depth + 2, align=(Align.CENTER, Align.CENTER, Align.MIN)
    )
    return cut


def _snap_runs(params: Params) -> list[tuple[float, float]]:
    """Two runs of bead along the front, either side of the thumb notch."""
    w = layout(params).width
    inner = params.notch + 3.0
    outer = 0.36 * w
    return [(w / 2 - outer, w / 2 - inner), (w / 2 + inner, w / 2 + outer)]


def _snap_z(params: Params) -> float:
    return params.rim - params.lip / 2


def _triangle(params: Params, y0: float, reach: float, run: tuple[float, float]) -> Part:
    """A 45-degree ridge or groove along x, on the plane y = y0, pointing at -y."""
    z = _snap_z(params)
    tri = Polygon((y0, z - reach), (y0 - reach, z), (y0, z + reach), align=None)
    return _along_x(tri, *run)


def _along_x(sketch, start: float, end: float) -> Part:
    """A (y, z) profile run along x from ``start`` to ``end``.

    With the direction given: a polygon wound clockwise faces -x, and extruding
    along its own normal runs it backwards from ``start``.
    """
    return extrude(Plane.YZ.offset(start) * sketch, amount=end - start, dir=(1, 0, 0))


def _hinge_xs(params: Params) -> list[tuple[float, float, bool]]:
    """(start, end, belongs to the tray) for each knuckle, centred on the back."""
    w = layout(params).width
    n, k, g = params.knuckles, params.knuckle_length, params.knuckle_gap
    span = n * k + (n - 1) * g
    x = (w - span) / 2
    out = []
    for i in range(n):
        out.append((x, x + k, i % 2 == 0))
        x += k + g
    return out


def hinge_axis(params: Params) -> tuple[float, float]:
    """(y, z) of the hinge line."""
    return layout(params).depth + params.wall + params.axis_offset, params.axis_z


def _knuckle(params: Params, start: float, end: float, tray: bool) -> Part:
    ya, za = hinge_axis(params)
    r, back = params.knuckle, layout(params).depth + params.wall
    s = r / math.sqrt(2)
    ring = _along_x(Pos(ya, za) * Circle(r), start, end)
    if tray:
        # A 45-degree teardrop under the round, and a 45-degree gusset from it
        # back to the wall: nothing here overhangs, printed standing up.
        drop = Polygon((ya - s, za - s), (ya, za - r * math.sqrt(2)), (ya + s, za - s),
                       (ya, za), align=None)
        run = ya - back + 0.5
        gusset = Polygon((back - 0.5, za), (ya, za), (ya, za - r * math.sqrt(2)),
                         (back - 0.5, za - r * math.sqrt(2) - run), align=None)
        body = ring
        for sk in (drop, gusset):
            body += _along_x(sk, start, end)
        hole = params.pin.diameter + params.pin.press
    else:
        # Printed face down, so its teardrop points at the lid's top face, and
        # a bridge ties it back to the lid over the tray's wall.
        peak = Polygon((ya - s, za + s), (ya, za + r * math.sqrt(2)), (ya + s, za + s),
                       (ya, za), align=None)
        bridge = Polygon((back - 0.5, params.rim), (ya, params.rim),
                         (ya, params.height), (back - 0.5, params.height), align=None)
        body = ring
        for sk in (peak, bridge):
            body += _along_x(sk, start, end)
        hole = params.pin.diameter + params.pin.play
    body = body & Pos(0, 0, params.height / 2) * Box(
        10 * params.max_side, 10 * params.max_side, params.height
    )
    bore = Pos((start + end) / 2, ya, za) * Rot(0, 90, 0) * Cylinder(
        hole / 2, end - start + 2
    )
    return body - bore


def tray(params: Params) -> Part:
    """The bottom half, as printed: floor on the plate, knuckles at the back."""
    params.validate()
    lay = layout(params)
    outer = extrude(_plan(params, params.wall), amount=params.rim)
    shell = outer - extrude(Plane.XY.offset(params.floor) * _plan(params), amount=params.rim)
    shell += extrude(Plane.XY.offset(params.floor) * _plan(params),
                     amount=params.seat - params.floor)
    shell -= _pocket_cuts(params)
    shell -= _cradle(params)
    shell -= _key_slot(params)
    for run in _snap_runs(params):
        shell -= _triangle(params, 0.0, params.snap + params.snap_play, run)
    shell -= Pos(lay.width / 2, 0, params.rim + params.notch - params.notch_depth) \
        * Rot(90, 0, 0) * Cylinder(params.notch, 4 * params.wall)
    for start, end, own in _hinge_xs(params):
        if own:
            shell += _knuckle(params, start, end, tray=True)
    return shell


def lid(params: Params) -> Part:
    """The top half, closed over the tray -- see ``lid_for_print``."""
    params.validate()
    lay = layout(params)
    g, t = params.lip_gap, params.lip_wall
    top = extrude(Plane.XY.offset(params.rim) * _plan(params, params.wall), amount=params.lid)
    ring = _plan(params, -g) - _plan(params, -g - t)
    lip = extrude(Plane.XY.offset(params.rim - params.lip) * ring, amount=params.lip)
    # No lip along the back: see the module note.
    lip -= Pos(lay.width / 2, lay.depth, params.rim - params.lip) * Box(
        lay.width + 2, 2 * (params.corner + 1), params.lip,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )
    body = top + lip
    for run in _snap_runs(params):
        body += _triangle(params, g, params.snap + g, run)
    for start, end, own in _hinge_xs(params):
        if not own:
            body += _knuckle(params, start, end, tray=False)
    return body


def opened(params: Params, degrees: float, part: Part | None = None) -> Part:
    """The lid swung open about the hinge by ``degrees``."""
    ya, za = hinge_axis(params)
    part = lid(params) if part is None else part
    return Pos(0, ya, za) * Rot(-degrees, 0, 0) * Pos(0, -ya, -za) * part


def lid_for_print(params: Params, part: Part | None = None) -> Part:
    """Turned over onto its top face, on the plate at z = 0."""
    part = lid(params) if part is None else part
    return Pos(0, 0, params.height) * Rot(180, 0, 0) * part


def pin_length(params: Params) -> float:
    xs = _hinge_xs(params)
    return xs[-1][1] - xs[0][0]


def build(params: Params) -> Part:
    """Both halves, closed: for looking at, not for printing."""
    return tray(params) + lid(params)


if __name__ == "__main__":
    from pathlib import Path

    from build123d import export_stl

    p = Params()
    out = Path("out")
    out.mkdir(exist_ok=True)
    t, lid_part = tray(p), lid_for_print(p)
    export_stl(t, str(out / "hole_saw_case_tray.stl"))
    export_stl(lid_part, str(out / "hole_saw_case_lid.stl"))
    bb = t.bounding_box()
    print(f"tray {bb.size.X:.1f} x {bb.size.Y:.1f} x {bb.size.Z:.1f} mm, "
          f"valid={t.is_valid}, lid valid={lid_part.is_valid}")
