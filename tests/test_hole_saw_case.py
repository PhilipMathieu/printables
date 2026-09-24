"""Geometric assertions for the hole saw case.

What decides whether this works is not visible in a render: that every piece
of the set goes into its place and nothing of the case is in the way of it,
that the lid closes over them and swings open again without touching either
the tray or a saw, that the snap actually catches, and that both halves print
without support. Each is measured off the built solids, with the set stood in
them as plain shapes the size the catalogue says they are -- the nested stack
as one cylinder, as wide as its outer saw and as tall as its innermost stands.
"""

from __future__ import annotations

import dataclasses
import itertools
import math

import numpy as np
import pytest
from build123d import Align, Box, Cylinder, Part, Pos, Rot
from shapely import affinity
from shapely.geometry import Point, box
from shapely.ops import unary_union

from geom.hole_saws import MM_PER_INCH, WARRIOR_57523
from parts.hole_saw_case import (
    MAX_SIDE,
    STEEL_PIN,
    Params,
    _cradle_plan,
    _cradle_runs,
    key_depth,
    _snap_runs,
    _snap_z,
    hinge_axis,
    layout,
    lid,
    lid_for_print,
    opened,
    pin_length,
    tray,
)
from tools.render import mesh

TOUCH = 1e-3
"""Cubic millimetres of overlap that count as none: boolean noise."""


@pytest.fixture(scope="module")
def params():
    return Params()


@pytest.fixture(scope="module")
def bottom(params):
    return tray(params)


@pytest.fixture(scope="module")
def top(params):
    return lid(params)


@pytest.fixture(scope="module")
def contents(params):
    """The set, in its places: the stack on its pocket floor, the mandrel
    lying in its cradle, the key in its slot."""
    lay = layout(params)
    pieces = Part()
    for p in lay.stacks:
        pieces += Pos(p.x, p.y, params.floor) * Cylinder(
            p.stack.outer.diameter / 2, p.stack.height,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
    x, y = lay.mandrel
    z = params.floor + params.kit.mandrel_diameter / 2
    along = x + params.mandrel_clearance
    for seg in params.kit.mandrel:
        pieces += Pos(along + seg.length / 2, y, z) * Rot(0, 90, 0) * Cylinder(
            seg.diameter / 2, seg.length
        )
        along += seg.length
    c, across = params.key_clearance, params.kit.key.across
    bottom = params.seat - key_depth(params)
    for x0, y0, x1, y1 in lay.key_bars:
        pieces += Pos(x0 + c, y0 + c, bottom) * Box(
            x1 - x0 - 2 * c, y1 - y0 - 2 * c, across,
            align=(Align.MIN, Align.MIN, Align.MIN),
        )
    return pieces


# --- the set ---------------------------------------------------------------


def test_the_set_is_the_one_on_the_shelf():
    """Ten pieces: eight saws, 1 to 2-1/2 inches, a mandrel and a key."""
    kit = WARRIOR_57523
    assert kit.pieces == 10
    assert [s.inches for s in kit.saws] == [1, 1.25, 1.5, 1.75, 2, 2.125, 2.25, 2.5]
    for saw in kit.saws:
        assert saw.diameter >= saw.inches * MM_PER_INCH


def test_every_saw_has_a_place(params):
    placed = [saw.nominal for p in layout(params).stacks for saw in p.stack.saws]
    assert sorted(placed) == sorted(s.nominal for s in params.kit.saws)


# --- they nest ---------------------------------------------------------------


def test_the_whole_set_is_one_stack_half_an_inch_proud(params):
    """Measured on the set: all eight nest, and the 1 inch in the middle
    stands about half an inch proud of the 2-1/2 round the outside."""
    (only,) = layout(params).stacks
    assert [s.inches for s in only.stack.saws] == [
        2.5, 2.25, 2.125, 2, 1.75, 1.5, 1.25, 1
    ]
    assert only.stack.height == pytest.approx(
        params.kit.tallest + 0.5 * MM_PER_INCH
    )


@pytest.mark.parametrize("nest", [1, 2, 3, 4, 8])
def test_every_saw_nests_in_the_one_round_it(nest):
    params = Params(nest=nest)
    for stack in params.stacks:
        assert len(stack.saws) <= nest
        for big, small in zip(stack.saws, stack.saws[1:]):
            assert big.inches - small.inches >= params.kit.nest_step - 1e-9


def test_validate_refuses_a_nest_the_set_does_not_make():
    kit = dataclasses.replace(WARRIOR_57523, nest_step=0.25)
    with pytest.raises(ValueError, match="does not nest"):
        Params(kit=kit).validate()


def test_nesting_is_what_makes_it_compact(params):
    """One stack against every saw in its own pocket: under two thirds the
    volume, for at most the half inch the stack stands proud -- less, since
    the mandrel already stands taller than a single saw."""
    flat = Params(nest=1)

    def volume(p):
        lay = layout(p)
        return (lay.width + 2 * p.wall) * (lay.depth + 2 * p.wall) * p.height

    assert volume(params) < 0.65 * volume(flat)
    assert params.height - flat.height <= 0.5 * MM_PER_INCH + 1e-6


# --- everything goes in ----------------------------------------------------


@pytest.mark.parametrize("nest", [8, 2, 1])
def test_the_pockets_never_run_into_one_another(nest):
    """At least one web of plastic between any two pockets, and between any
    pocket and the mandrel's cradle or the key's slot."""
    params = Params(nest=nest)
    lay = layout(params)
    for a, b in itertools.combinations(lay.stacks, 2):
        gap = math.dist((a.x, a.y), (b.x, b.y)) - (
            params.pocket(a.stack) + params.pocket(b.stack)
        ) / 2
        assert gap >= params.web - 1e-6, (a.stack.label, b.stack.label)
    for p in lay.stacks:
        gap = Point(p.x, p.y).distance(lay.reserved) - params.pocket(p.stack) / 2
        assert gap >= params.web - 0.05, p.stack.label


@pytest.mark.parametrize("nest", [8, 2, 1])
def test_the_key_keeps_clear_of_the_cradle_and_the_walls(nest):
    params = Params(nest=nest)
    lay = layout(params)
    cradle = affinity.translate(unary_union(_cradle_plan(params)),
                                params.margin, params.margin)
    inside = box(params.margin, params.margin, lay.width - params.margin,
                 lay.depth - params.margin)
    scoop = Point(lay.key_scoop).buffer(params.key_scoop / 2)
    for g in (*(box(*bar) for bar in lay.key_bars), scoop):
        assert g.distance(cradle) >= params.web - 0.05
        assert inside.buffer(0.05).contains(g)


def test_the_set_goes_in_without_touching_the_case(bottom, contents):
    assert (bottom & contents).volume < TOUCH


def test_the_stack_stands_on_its_pocket_floor(params, bottom):
    """The pocket's own floor, not the tray's top: the outer saw sinks in to
    its full depth, and so is held by the whole of it."""
    for p in layout(params).stacks:
        edge = p.x + params.pocket(p.stack) / 2 - 1.5
        probe = Pos(edge, p.y, params.floor - 0.1) * Box(0.4, 0.4, 0.2)
        assert (bottom & probe).volume > 0, p.stack.label
        above = Pos(edge, p.y, params.floor + 0.2) * Box(0.4, 0.4, 0.2)
        assert (bottom & above).volume < TOUCH, p.stack.label


def test_the_mandrel_lies_half_buried(params, bottom):
    """The cradle's rims are at the mandrel's axis: deep enough that it cannot
    roll out, shallow enough that it is lifted rather than prised."""
    x, y = layout(params).mandrel
    runs = _cradle_runs(params)
    a, b, r = max(runs, key=lambda run: run[2])
    probe = Pos(x + (a + b) / 2, y + r + 0.5, params.seat - 0.2) * Box(0.4, 0.4, 0.2)
    assert (bottom & probe).volume > 0
    assert params.seat == pytest.approx(params.floor + r)


# --- it closes, and opens --------------------------------------------------


def test_the_lid_closes_over_everything(bottom, top, contents):
    assert (bottom & top).volume < TOUCH
    assert (top & contents).volume < TOUCH


def test_the_lid_sits_on_the_rim(params, bottom, top):
    """Not on a saw, and not on the snap: a sliver below the rim is the tray,
    a sliver above it is the lid, all the way round."""
    lay = layout(params)
    edge = Pos(lay.width / 2, -params.wall / 2, params.rim) * Box(20, 0.6, 0.2)
    assert (bottom & (Pos(0, 0, -0.15) * edge)).volume > 0
    assert (top & (Pos(0, 0, 0.15) * edge)).volume > 0


@pytest.mark.parametrize("degrees", [15, 45, 75, 105, 135, 165, 180])
def test_the_lid_swings_clear(params, bottom, top, contents, degrees):
    """All the way over and flat behind, touching neither the tray nor the
    stack on the way."""
    swung = opened(params, degrees, top)
    assert (swung & bottom).volume < TOUCH
    assert (swung & contents).volume < TOUCH


def test_the_snap_catches(params, top):
    """The bead reaches into the wall by ``snap``: past the inside face of the
    wall, which is what it has to click back over to open."""
    for a, b in _snap_runs(params):
        slab = Pos((a + b) / 2, 0, _snap_z(params)) * Box(b - a - 2, 4, 0.05)
        reach = (top & slab).bounding_box().min.Y
        assert reach == pytest.approx(-params.snap, abs=0.02)


def _check_pin_bores(params, bottom, top):
    ya, za = hinge_axis(params)
    span = pin_length(params)
    x0 = layout(params).width / 2

    def rod(diameter):
        return Pos(x0, ya, za) * Rot(0, 90, 0) * Cylinder(diameter / 2, span)

    held = params.pin.diameter + params.pin.press
    free = params.pin.diameter + params.pin.play
    assert (bottom & rod(held - 0.04)).volume < TOUCH
    assert (bottom & rod(held + 0.04)).volume > 0
    assert (top & rod(held + 0.04)).volume < TOUCH
    assert (top & rod(free - 0.04)).volume < TOUCH
    assert (top & rod(free + 0.04)).volume > 0


def test_the_hinge_pin_is_held_by_the_tray_and_turns_in_the_lid(params, bottom, top):
    """The tray's bores are a tenth over the filament and the lid's are four:
    printed holes usually come out a tenth or two undersize, which makes the
    first a press fit and leaves the second running free. Probed with a
    cylinder a hair over each bore -- it touches that half, and a cylinder of
    the tray's bore does not touch the lid at all."""
    _check_pin_bores(params, bottom, top)


def test_a_steel_pin_is_bored_for_and_moves_nothing_else(params):
    """The steel upgrade changes the bores and nothing else: same hinge line,
    same case, and the bores fit the rod the same way they fit filament."""
    steel = dataclasses.replace(params, pin=STEEL_PIN)
    assert hinge_axis(steel) == hinge_axis(params)
    assert steel.height == params.height
    assert layout(steel) == layout(params)
    _check_pin_bores(steel, tray(steel), lid(steel))


def test_validate_refuses_a_pin_the_lid_would_grip(params):
    tight = dataclasses.replace(STEEL_PIN, play=STEEL_PIN.press)
    with pytest.raises(ValueError, match="grip"):
        dataclasses.replace(params, pin=tight).validate()
    fat = dataclasses.replace(STEEL_PIN, diameter=5.0)
    with pytest.raises(ValueError, match="knuckle"):
        dataclasses.replace(params, pin=fat).validate()


# --- it prints ---------------------------------------------------------------


def _overhangs(solid, bores: list[tuple[float, float, float]] = ()) -> np.ndarray:
    """Centroids of downward facets steeper than 45 degrees off vertical,
    leaving out the plate and the pin bores, which are small enough to bridge."""
    points, tris = mesh(solid, tol=0.05)
    tri = points[tris]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(n, axis=1)
    keep = length > 1e-9
    n = n[keep] / length[keep, None]
    centre = tri[keep].mean(axis=1)
    bad = (n[:, 2] < -math.sin(math.radians(45)) - 0.02) & (centre[:, 2] > 0.05)
    for y, z, r in bores:
        bad &= ~(np.hypot(centre[:, 1] - y, centre[:, 2] - z) < r + 0.05)
    return centre[bad]


def test_the_tray_prints_without_support(params, bottom):
    ya, za = hinge_axis(params)
    bore = (ya, za, (params.pin.diameter + params.pin.press) / 2)
    assert len(_overhangs(bottom, [bore])) == 0


def test_the_lid_prints_face_down_without_support(params, top):
    ya, za = hinge_axis(params)
    # Turned over about the x axis and dropped onto the plate, which moves the
    # bore to (-ya, height - za).
    bore = (-ya, params.height - za, (params.pin.diameter + params.pin.play) / 2)
    flipped = lid_for_print(params, top)
    assert flipped.bounding_box().min.Z == pytest.approx(0, abs=1e-6)
    assert len(_overhangs(flipped, [bore])) == 0


def test_each_half_fits_the_plate(params, bottom, top):
    for half in (bottom, lid_for_print(params, top)):
        size = half.bounding_box().size
        assert max(size.X, size.Y) <= MAX_SIDE


def test_the_case_is_as_short_as_the_set_allows(params, bottom, top):
    """The tallest piece -- the nested stack -- with a floor under it,
    headroom over it and a lid: the closed case is that and not a millimetre
    more, knuckles included."""
    closed = (bottom + top).bounding_box()
    tallest = max([s.height for s in params.stacks]
                  + [params.kit.mandrel_diameter + 2 * params.mandrel_clearance])
    assert closed.size.Z == pytest.approx(
        params.floor + tallest + params.headroom + params.lid, abs=1e-3
    )


def test_validate_refuses_an_even_hinge():
    with pytest.raises(ValueError):
        Params(knuckles=4).validate()


def test_validate_refuses_a_lip_that_reaches_the_tray():
    with pytest.raises(ValueError):
        Params(lip=30.0).validate()
