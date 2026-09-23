"""Geometric assertions for the hole saw case.

What decides whether this works is not visible in a render: that every piece
of the set goes into its place and nothing of the case is in the way of it,
that the lid closes over them and swings open again without touching either
the tray or a saw, that the snap actually catches, and that both halves print
without support. Each is measured off the built solids, with the set stood in
them as plain cylinders the size the catalogue says they are.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from build123d import Align, Box, Cylinder, Part, Pos, Rot

from geom.hole_saws import MM_PER_INCH, WARRIOR_57523
from parts.hole_saw_case import (
    MAX_SIDE,
    Params,
    _cradle_runs,
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
    """The set, in its places: saws standing on the pocket floors, the mandrel
    lying in its cradle, the key in its slot."""
    lay = layout(params)
    pieces = Part()
    for p in lay.saws:
        pieces += Pos(p.x, p.y, params.floor) * Cylinder(
            p.saw.diameter / 2, p.saw.height,
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
    kx, ky = lay.key
    k, c = params.kit.key, params.key_clearance
    depth = params.seat - k.across - c - 0.4
    pieces += Pos(kx + c, ky + c, depth) * Box(
        k.long, k.across, k.across, align=(Align.MIN, Align.MIN, Align.MIN)
    )
    pieces += Pos(kx + c, ky + c, depth) * Box(
        k.across, k.short, k.across, align=(Align.MIN, Align.MIN, Align.MIN)
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


def test_every_piece_has_a_place(params):
    lay = layout(params)
    assert sorted(p.saw.nominal for p in lay.saws) == sorted(
        s.nominal for s in params.kit.saws
    )


# --- everything goes in ----------------------------------------------------


def test_the_pockets_never_run_into_one_another(params):
    """At least one web of plastic between any two pockets, and between any
    pocket and the mandrel's cradle or the key's slot."""
    lay = layout(params)
    for a, b in itertools.combinations(lay.saws, 2):
        gap = math.dist((a.x, a.y), (b.x, b.y)) - (
            params.pocket(a.saw) + params.pocket(b.saw)
        ) / 2
        assert gap >= params.web - 1e-6, (a.saw.nominal, b.saw.nominal)
    from shapely.geometry import Point

    for p in lay.saws:
        gap = Point(p.x, p.y).distance(lay.reserved) - params.pocket(p.saw) / 2
        assert gap >= params.web - 0.05, p.saw.nominal


def test_the_set_goes_in_without_touching_the_case(bottom, contents):
    assert (bottom & contents).volume < TOUCH


def test_every_saw_stands_on_its_pocket_floor(params, bottom):
    """The pocket's own floor, not the tray's top: each saw sinks in to its
    full depth, and so is held by the whole of it."""
    for p in layout(params).saws:
        probe = Pos(p.x + params.pocket(p.saw) / 2 - 1.5, p.y, params.floor - 0.1) \
            * Box(0.4, 0.4, 0.2)
        assert (bottom & probe).volume > 0, p.saw.nominal
        above = Pos(p.x + params.pocket(p.saw) / 2 - 1.5, p.y, params.floor + 0.2) \
            * Box(0.4, 0.4, 0.2)
        assert (bottom & above).volume < TOUCH, p.saw.nominal


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
    saws on the way."""
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


def test_the_hinge_pin_is_held_by_the_tray_and_turns_in_the_lid(params, bottom, top):
    """The tray's bores are a tenth over the filament and the lid's are four:
    printed holes usually come out a tenth or two undersize, which makes the
    first a press fit and leaves the second running free. Probed with a
    cylinder a hair over each bore -- it touches that half, and a cylinder of
    the tray's bore does not touch the lid at all."""
    ya, za = hinge_axis(params)
    span = pin_length(params)
    x0 = layout(params).width / 2

    def rod(diameter):
        return Pos(x0, ya, za) * Rot(0, 90, 0) * Cylinder(diameter / 2, span)

    held = params.pin + params.pin_press
    free = params.pin + params.pin_play
    assert (bottom & rod(held - 0.04)).volume < TOUCH
    assert (bottom & rod(held + 0.04)).volume > 0
    assert (top & rod(held + 0.04)).volume < TOUCH
    assert (top & rod(free - 0.04)).volume < TOUCH
    assert (top & rod(free + 0.04)).volume > 0


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
    bore = (ya, za, (params.pin + params.pin_press) / 2)
    assert len(_overhangs(bottom, [bore])) == 0


def test_the_lid_prints_face_down_without_support(params, top):
    ya, za = hinge_axis(params)
    # Turned over about the x axis and dropped onto the plate, which moves the
    # bore to (-ya, height - za).
    bore = (-ya, params.height - za, (params.pin + params.pin_play) / 2)
    flipped = lid_for_print(params, top)
    assert flipped.bounding_box().min.Z == pytest.approx(0, abs=1e-6)
    assert len(_overhangs(flipped, [bore])) == 0


def test_each_half_fits_the_plate(params, bottom, top):
    for half in (bottom, lid_for_print(params, top)):
        size = half.bounding_box().size
        assert max(size.X, size.Y) <= MAX_SIDE


def test_the_case_is_as_short_as_the_set_allows(params, bottom, top):
    """The tallest piece, a floor under it, headroom over it and a lid: the
    closed case is that and not a millimetre more, knuckles included."""
    closed = (bottom + top).bounding_box()
    tallest = max(params.kit.tallest, params.kit.mandrel_diameter
                  + 2 * params.mandrel_clearance)
    assert closed.size.Z == pytest.approx(
        params.floor + tallest + params.headroom + params.lid, abs=1e-3
    )


def test_the_packing_leaves_no_row_of_air(params):
    """The pieces, in plan, cover well over half the tray: about 64 percent.
    Four by two equal cells sized to the biggest saw, with the mandrel along
    the front, covers 39 -- and at 264mm wide does not fit the plate at all."""
    lay = layout(params)
    pieces = sum(math.pi * params.pocket(p.saw) ** 2 / 4 for p in lay.saws)
    pieces += lay.reserved.area
    assert pieces / (lay.width * lay.depth) > 0.6


def test_validate_refuses_an_even_hinge():
    with pytest.raises(ValueError):
        Params(knuckles=4).validate()


def test_validate_refuses_a_lip_that_reaches_the_tray():
    with pytest.raises(ValueError):
        Params(lip=30.0).validate()
