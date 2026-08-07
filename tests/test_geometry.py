"""Geometric assertions -- the substitute for being able to look at the print."""

from __future__ import annotations

import pytest
from OCP.BRepExtrema import BRepExtrema_DistShapeShape

from geom.hat import HAT_AREA, HAT_EDGE_LENGTHS, area, edge_lengths, find_hat
from p2s import profiles
from parts.einstein_fidget import Params, _bulge, _levels, _outline, build


@pytest.fixture(scope="module")
def hat():
    return find_hat()


def test_hat_is_a_13_gon(hat):
    assert len(hat) == 13


def test_hat_area_is_eight_kites(hat):
    assert area(hat) == pytest.approx(HAT_AREA, abs=1e-6)


def test_hat_edge_lengths_match_the_published_tile(hat):
    tally: dict[float, int] = {}
    for length in edge_lengths(hat):
        match = min(HAT_EDGE_LENGTHS, key=lambda w: abs(w - length))
        assert length == pytest.approx(match, abs=1e-6)
        tally[match] = tally.get(match, 0) + 1
    assert tally == HAT_EDGE_LENGTHS


def _wire_gap(params: Params, inset_a: float, inset_b: float) -> float:
    a = _outline(params, round(inset_a, 6)).wires()[0]
    b = _outline(params, round(inset_b, 6)).wires()[0]
    dist = BRepExtrema_DistShapeShape(a.wrapped, b.wrapped)
    dist.Perform()
    return dist.Value()


def test_clearance_holds_at_every_height():
    """The one property the whole construction exists to guarantee.

    Checked per slice rather than between finished solids: it is the same
    invariant, and a distance query against 25-slice stacks takes minutes.
    """
    params = Params()
    for z0, height in _levels(params):
        shift = _bulge(params, z0 + height / 2)
        for i in range(params.rings):
            gap = _wire_gap(
                params,
                i * params.pitch + params.wall - shift,
                (i + 1) * params.pitch - shift,
            )
            assert gap == pytest.approx(params.gap, abs=1e-6), f"at z={z0}"


def test_walls_hold_at_every_height():
    params = Params()
    for z0, height in _levels(params):
        shift = _bulge(params, z0 + height / 2)
        for i in range(params.rings):
            wall = _wire_gap(
                params, i * params.pitch - shift, i * params.pitch + params.wall - shift
            )
            assert wall == pytest.approx(params.wall, abs=1e-6), f"at z={z0}"


def test_rings_are_captured():
    """A printed set with no engagement fell out ring by ring and stopped
    being a fidget, so the bulge has to beat the clearance."""
    params = Params()
    assert params.engagement > 0
    assert params.interlock > params.gap
    assert 0 < params.travel < params.thickness / 2


def test_coarse_slices_are_rejected():
    """A step near the gap fuses the rings; caught by construction, not luck."""
    with pytest.raises(ValueError, match="fuse"):
        build(Params(slice_height=1.0))


def test_step_is_well_under_the_gap():
    assert Params().step < Params().gap / 2


def test_no_capture_is_rejected():
    with pytest.raises(ValueError, match="fall out"):
        build(Params(interlock=0.2, gap=0.3))


def test_too_many_rings_is_rejected():
    """The defaults that shipped the star-shaped bug must not build."""
    with pytest.raises(ValueError, match="hat units"):
        build(Params(unit=9.0, rings=4, wall=2.4, gap=0.35))


def test_walls_are_at_least_two_extrusions():
    assert Params().wall >= 2 * profiles.machine(0.4).line_width


@pytest.fixture(scope="module")
def coarse():
    """A small build: same construction, quick enough to assert on.

    Fewer rings and taller slices, but slice_height still has to satisfy the
    step rule -- at 1.0mm the profile jogs 0.32mm against a 0.3mm gap and every
    body fuses into one solid, which is what this fixture used to do.
    """
    params = Params(outer_rings=2, rings=2, slice_height=0.4)
    return params, build(params)


def test_rings_stay_separate_bodies(coarse):
    """A print-in-place fidget is only free if the rings never fuse."""
    params, part = coarse
    assert part.is_valid
    assert len(part.solids()) == params.total_rings + 1


def test_fits_the_p2s_bed(coarse):
    _, part = coarse
    size = part.bounding_box().size
    assert profiles.machine(0.4).fits((size.X, size.Y, size.Z))
