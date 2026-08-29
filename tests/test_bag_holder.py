"""Geometric assertions for the bag holder.

Four things decide whether this works, and none of them is visible in a render:
that the hole the bag goes into is a hole and not a hook, that the funnel stops
each bundle of handles at its own size and none of them at the bottom, that
there is no corner anywhere inside the aperture, and that the thing hangs plumb
off its eye. Each is measured off the built solid rather than trusted to the
arithmetic that made it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from build123d import Box, CenterOf, Cylinder, Pos

from p2s import profiles
from parts.bag_holder import (
    MAX_FUNNEL,
    MIN_EYE,
    MIN_SLOT,
    MIN_WALL,
    Params,
    aperture,
    build,
    profile,
)

NOZZLE = 0.4

PROBE = 0.05
"""Step the bundle sweeps down in, in mm. Everything measured that way is
quoted to it."""


@pytest.fixture(scope="module")
def holder():
    return build(Params())


@pytest.fixture(scope="module")
def hole():
    """The aperture as a solid, for measuring what fits down it."""
    from build123d import Plane, extrude

    params = Params()
    return extrude(Plane.XY * aperture(params), amount=params.band)


def _bundle(params: Params, y: float, across: float):
    """A tied bundle of handles, on the axis at some height."""
    return Pos(0, y, params.band / 2) * Cylinder(across / 2, params.band * 3)


def _rests_at(part, params: Params, across: float) -> float:
    """Lowest height a bundle that wide reaches before the ribbon stops it."""
    y = params.belly_y
    while y > params.aperture_bottom:
        if (part & _bundle(params, y, across)).volume > 1e-6:
            return y + PROBE
        y -= PROBE
    return y


def _wire_points(wire, samples: int) -> np.ndarray:
    return np.array(
        [(p.X, p.Y) for p in (wire.position_at(i / samples) for i in range(samples))]
    )


# --- the hole is a hole ----------------------------------------------------


def test_the_profile_is_one_face_with_two_holes_in_it():
    """The whole retention argument, and the check that the eye, the body and
    the tail landed on one another rather than beside one another. Two closed
    inner wires: the eye's bore and the aperture. A hook would be neither."""
    face = profile(Params()).faces()[0]
    assert len(profile(Params()).faces()) == 1
    assert len(face.inner_wires()) == 2


def test_the_ribbon_is_never_thinner_than_one_wall(holder):
    """Which is what says the aperture is walled the whole way round, and that
    the ribbon really is a constant-width offset and not a shape with a hole
    roughly in the middle of it."""
    params = Params()
    face = profile(params).faces()[0]
    inner = sorted(face.inner_wires(), key=lambda w: w.length)[1]
    across = _wire_points(inner, 900)[:, None, :] - _wire_points(
        face.outer_wire(), 2400
    )[None, :, :]
    assert np.linalg.norm(across, axis=2).min() == pytest.approx(params.wall, abs=0.02)


def test_the_eye_keeps_its_whole_bore(holder):
    """The body reaches into the eye's ring far enough to fuse and no further;
    a bore with the body poking into it takes no chain."""
    params = Params()
    bore = sorted(profile(params).faces()[0].inner_wires(), key=lambda w: w.length)[0]
    assert bore.length == pytest.approx(2 * math.pi * params.eye_radius)


# --- the funnel ------------------------------------------------------------


@pytest.mark.parametrize("across", [34.0, 26.0, 18.0, 12.0])
def test_a_bundle_wedges_where_the_arithmetic_says(holder, across):
    """``seats`` is the design; this is where the solid actually stops it."""
    params = Params()
    assert params.touches(across) >= params.blend_start, "not on the straight taper"
    assert params.seats(across) == pytest.approx(
        _rests_at(holder, params, across), abs=2 * PROBE
    )


def test_the_funnel_sorts_by_size(holder):
    """The mechanism, in one assertion: a bigger bundle stops higher up."""
    params = Params()
    sizes = [34.0, 26.0, 18.0, 12.0, 8.0]
    depths = [_rests_at(holder, params, across) for across in sizes]
    assert depths == sorted(depths, reverse=True), "a smaller bundle stopped higher"
    assert params.seats(sizes[0]) > params.seats(sizes[-1])


def test_the_funnel_only_ever_narrows_on_the_way_down(hole):
    """What makes it sort rather than rattle. Measured across the hole itself,
    from the belly to the bottom of the throat: never once wider than it was a
    step above."""
    params = Params()
    widths = []
    for y in np.linspace(params.belly_y, params.throat_bottom_y, 160):
        slab = Pos(0, y, params.band / 2) * Box(4 * params.width, 0.2, params.band)
        cut = hole & slab
        widths.append(cut.bounding_box().size.X if cut.volume > 1e-9 else 0.0)
    assert widths[0] == pytest.approx(params.belly, abs=0.1)
    assert widths[-1] == pytest.approx(params.slot, abs=0.1)
    assert np.all(np.diff(widths) <= 1e-6), "the hole widens somewhere on the way down"


def test_everything_it_grips_stops_inside_the_funnel():
    """Nothing in the range reaches the throat's floor, so nothing in the range
    is hanging off the bottom of the part where it can be knocked about."""
    params = Params()
    lo, hi = params.grip
    assert (lo, hi) == (params.slot, params.belly)
    # Open at the bottom: a bundle exactly the throat's width fits the throat
    # and travels down it, which is the case the next test is about.
    for across in np.linspace(lo, hi, 25)[1:]:
        assert params.throat_top_y <= params.seats(across) <= params.belly_y


def test_a_bundle_under_the_throat_is_still_held(holder):
    """It stops being wedged and starts simply lying in the bottom, which is
    closed. Losing the grip is not the same as losing the bag."""
    params = Params()
    small = params.slot / 2
    assert params.seats(small) == pytest.approx(params.aperture_bottom + small / 2)
    assert params.seats(small) < params.throat_bottom_y
    assert (holder & _bundle(params, params.seats(small), small)).volume == pytest.approx(
        0, abs=1e-6
    )


def test_a_bundle_too_big_for_the_belly_is_refused():
    with pytest.raises(ValueError, match="does not go through"):
        Params().seats(Params().belly + 1)


# --- no corners ------------------------------------------------------------


def test_the_aperture_has_no_corner_to_snag_or_crack_at():
    """Every join in the hole is tangent, including the one where the funnel
    runs into the throat, which only is because it is filleted. Walk the wire
    and the tangent has to turn no faster than the tightest arc in it does --
    an unblended corner there would turn a whole ``funnel`` at one step."""
    params = Params()
    inner = sorted(
        profile(params).faces()[0].inner_wires(), key=lambda w: w.length
    )[1]
    samples = 3000
    tangents = np.array(
        [(v.X, v.Y) for v in (inner.tangent_at(i / samples) for i in range(samples))]
    )
    turn = np.degrees(
        np.arccos(np.clip((tangents[:-1] * tangents[1:]).sum(axis=1), -1, 1))
    )
    # The tightest radius in the aperture is the throat's own, so that arc sets
    # how fast the tangent can legitimately turn at this sampling.
    curvature = math.degrees(inner.length / samples / params.slot_radius)
    assert turn.max() < 3 * curvature
    assert turn.max() < params.funnel / 4


# --- printability ----------------------------------------------------------


def test_nothing_in_the_part_overhangs_at_all(holder):
    """Not 'inside 45 degrees' -- none. The part is one profile extruded, so
    every face is the plate, the top, or a wall square to both, and there is
    nothing anywhere for a support to be under."""
    from tools.render import mesh

    points, tris = mesh(holder, tol=0.02)
    tri = points[tris]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    flat = np.isclose(np.abs(n[:, 2]), 1.0, atol=1e-9)
    assert np.abs(n[~flat, 2]).max() < 1e-9
    assert flat.sum() > 0 and (~flat).sum() > 0


def test_the_part_is_exactly_its_profile_extruded(holder):
    """Which is the claim the overhang test rests on, so it is worth making
    separately: no draft, no fillet in Z, no feature that is not in the
    drawing."""
    params = Params()
    face = profile(params).faces()[0]
    assert holder.volume == pytest.approx(face.area * params.band, rel=1e-6)
    assert holder.bounding_box().size.Z == pytest.approx(params.band)


def test_the_ribbon_is_thicker_than_two_extrusions():
    assert Params().wall >= 2 * profiles.machine(NOZZLE).line_width


def test_the_throat_is_deeper_than_it_is_wide():
    """Or the bundle rolls out of the side of the pinch instead of being held
    by it."""
    assert Params().band >= Params().slot


def test_a_plate_of_them_fits_the_bed():
    from tools.render import plate

    laid = plate([build(Params()) for _ in range(4)])
    size = laid.bounding_box().size
    assert profiles.machine(NOZZLE).fits((size.X, size.Y, size.Z))


# --- how it hangs ----------------------------------------------------------


def test_it_hangs_plumb_from_the_eye(holder):
    """Which is the whole reason it goes on a chain rather than clamping to the
    lead: hung from a point, it puts the funnel upright by itself."""
    params = Params()
    centre = holder.center(CenterOf.MASS)
    # 1e-4 mm is the mass-property solver's own tolerance on a part this size,
    # not an offset: the profile is mirror-symmetric by construction.
    assert centre.X == pytest.approx(0, abs=1e-4)
    assert centre.Y < -params.eye_outer


def test_the_tail_is_what_puts_the_mass_low():
    """It is not styling. A holder with a longer tail hangs from the same eye
    with its centre further below it, which is what keeps the funnel up when
    the lead is swinging."""
    stub = build(Params(tail=8.0)).center(CenterOf.MASS).Y
    long = build(Params(tail=24.0)).center(CenterOf.MASS).Y
    assert long < stub


def test_the_holder_is_one_sound_solid(holder):
    assert holder.is_valid
    assert len(holder.solids()) == 1


# --- the parameter guards --------------------------------------------------


def test_a_ribbon_of_one_extrusion_is_rejected():
    with pytest.raises(ValueError, match="extrusions"):
        build(Params(wall=MIN_WALL / 2))


def test_an_eye_nothing_clips_to_is_rejected():
    with pytest.raises(ValueError, match="carabiner gate"):
        build(Params(eye=MIN_EYE - 1))


def test_a_joint_that_blocks_the_bore_is_rejected():
    with pytest.raises(ValueError, match="into its bore"):
        build(Params(joint=4.0))


def test_a_throat_that_has_to_be_threaded_is_rejected():
    with pytest.raises(ValueError, match="two-handed"):
        build(Params(slot=MIN_SLOT - 1))


def test_a_throat_as_wide_as_the_belly_is_rejected():
    with pytest.raises(ValueError, match="slot with a bulge"):
        build(Params(slot=40.0))


def test_a_funnel_past_the_cap_is_rejected():
    with pytest.raises(ValueError, match="guides a bundle down"):
        build(Params(funnel=MAX_FUNNEL + 5))


def test_a_crown_wider_than_the_belly_is_rejected():
    with pytest.raises(ValueError, match="flare inward"):
        build(Params(crown=20.0))


def test_a_pinch_wider_than_it_is_deep_is_rejected():
    with pytest.raises(ValueError, match="rolls out"):
        build(Params(band=4.0))


def test_a_tail_shorter_than_it_is_wide_is_rejected():
    with pytest.raises(ValueError, match="bump on the bottom"):
        build(Params(tail=2.0))


def test_a_tail_wider_than_the_throat_is_rejected():
    with pytest.raises(ValueError, match="stands proud"):
        build(Params(tail_width=14.0))
