"""Geometric assertions for the bag holder.

Four things decide whether this works, and none of them is visible in a render:
that the hole the bag goes into is a hole and not a hook, that the funnel stops
each bundle of handles at its own size and none of them at the bottom, that
there is no corner anywhere inside the aperture, and that the collar comes off
the plate turning and cannot be lifted out. Each is measured off the built
solid rather than trusted to the arithmetic that made it.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from build123d import Box, CenterOf, Cylinder, Pos, Rot

from geom import webbing
from p2s import profiles
from parts.bag_holder import (
    MAX_FUNNEL,
    MAX_LEAN,
    MIN_SLOT,
    MIN_WALL,
    Params,
    _swell,
    aperture,
    body,
    build,
    collar,
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


def _radius_at(solid, z: float) -> float:
    """Outermost radius of a body of revolution at one height."""
    cut = solid & (Pos(0, 0, z) * Box(400, 400, 0.02))
    return cut.bounding_box().size.X / 2 if cut.volume > 1e-12 else 0.0


def _wire_points(wire, samples: int) -> np.ndarray:
    return np.array(
        [(p.X, p.Y) for p in (wire.position_at(i / samples) for i in range(samples))]
    )


# --- the hole is a hole ----------------------------------------------------


def test_the_profile_is_one_face_with_one_hole_in_it():
    """The whole retention argument, and the check that the head landed on the
    shoulders rather than beside them. One closed inner wire: the aperture. A
    hook would not be closed, and two faces would mean the neck missed."""
    face = profile(Params()).faces()[0]
    assert len(profile(Params()).faces()) == 1
    assert len(face.inner_wires()) == 1


def test_the_ribbon_is_never_thinner_than_one_wall(holder):
    """Which is what says the aperture is walled the whole way round, and that
    the ribbon really is a constant-width offset and not a shape with a hole
    roughly in the middle of it."""
    params = Params()
    face = profile(params).faces()[0]
    inner = face.inner_wires()[0]
    across = _wire_points(inner, 900)[:, None, :] - _wire_points(
        face.outer_wire(), 2400
    )[None, :, :]
    assert np.linalg.norm(across, axis=2).min() == pytest.approx(params.wall, abs=0.02)


# --- the webbing -----------------------------------------------------------


@pytest.mark.parametrize("strap", webbing.CATALOGUE.values(), ids=webbing.CATALOGUE)
def test_a_width_is_the_inches_it_is_sold_as(strap):
    """The width is the one number a lead is actually held to -- it is woven on
    a loom -- so it has to be the nominal size converted, not a caliper reading.
    Thickness is the caliper reading, and runs thick on purpose."""
    inches = float(Fraction(strap.nominal.split()[0]))
    assert strap.width == pytest.approx(inches * webbing.MM_PER_INCH)
    assert strap.handle == pytest.approx(2 * strap.thickness)


def test_an_unknown_width_says_what_there_is():
    with pytest.raises(KeyError, match="standard"):
        webbing.named("paracord")


# --- the swivel ------------------------------------------------------------


def test_the_collar_and_the_body_are_two_free_bodies(holder):
    """Printed in one go on one plate, touching nowhere. If they touched they
    would fuse, and a fused swivel is a lump."""
    params = Params()
    assert len(holder.solids()) == 2
    assert (body(params) & collar(params)).volume == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize("turn", [7.0, 45.0, 90.0, 180.0])
def test_the_collar_turns(turn):
    """Which is the whole point of it. Spun to any angle it still touches
    nothing, because both it and its socket are surfaces of revolution."""
    params = Params()
    assert (body(params) & (Rot(0, 0, turn) * collar(params))).volume == pytest.approx(
        0, abs=1e-9
    )


def test_the_collar_cannot_be_lifted_out_of_its_socket(holder):
    """It is fatter in the middle than either end of the hole it sits in, which
    is the only thing holding it in and the reason the swell has to lean."""
    params = Params()
    fat = _radius_at(collar(params), params.band / 2)
    mouth = _radius_at(_swell(params.socket_radius, params), 0.05)
    assert fat > mouth
    assert fat - mouth == pytest.approx(params.engagement, abs=0.05)


def test_the_gap_is_the_same_at_every_height():
    """The one number that decides whether it comes off the plate turning. Both
    surfaces are the same profile shifted radially, so it has to be -- measured
    here rather than assumed, because a loft would not have been."""
    params = Params()
    turning, socket = collar(params), _swell(params.socket_radius, params)
    for z in (0.05, 1.0, 2.5, 4.0, 5.5, 7.0, params.band - 0.05):
        gap = _radius_at(socket, z) - _radius_at(turning, z)
        assert gap == pytest.approx(params.gap, abs=0.02), f"at z={z}"


def test_the_collar_passes_the_lead_s_folded_handle():
    """The only way onto a lead that does not go past the snap hook."""
    params = Params()
    handle = Box(
        params.strap.width, params.strap.stack(params.plies), params.band * 3
    )
    assert (collar(params) & handle).volume == pytest.approx(0, abs=1e-6)
    thicker = Box(
        params.strap.width, params.strap.stack(params.plies + 1), params.band * 3
    )
    assert (collar(params) & thicker).volume > 1.0


def test_a_wider_lead_changes_the_collar_and_the_head_and_nothing_else():
    """The opening holds a bag, and a bag is the same size whatever the dog is
    on. So only the top of the part knows what the lead is."""
    sized = [Params().for_strap(name) for name in webbing.CATALOGUE]
    assert len({(p.belly, p.slot, p.funnel, round(p.seats(20.0) - p.belly_y, 9))
                for p in sized}) == 1
    radii = [p.collar_radius for p in sized]
    assert radii == sorted(radii)


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
    inner = profile(params).faces()[0].inner_wires()[0]
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


def test_the_only_thing_that_leans_anywhere_is_the_swivel(holder):
    """Everything else is a profile extruded straight up, so the steepest thing
    in the part is the swell that makes the collar captive -- and that has to
    stay inside what FDM bridges, because there is no support under a swivel."""
    from tools.render import mesh

    params = Params()
    points, tris = mesh(holder, tol=0.02)
    tri = points[tris]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    upright = np.isclose(np.abs(n[:, 2]), 1.0, atol=1e-9)
    lean = np.degrees(np.arcsin(np.clip(np.abs(n[~upright, 2]), -1, 1)))
    assert lean.max() < MAX_LEAN
    # A degree of slack: these are mesh facets, and a chord cuts inside the arc
    # it stands for, so a faceted cone reads a fraction steeper than it is.
    assert lean.max() == pytest.approx(params.lean, abs=1.0)

    # And it is only the swivel that leans: everything sloped is within a
    # millimetre of the collar's own radius, out at the joint.
    centre = tri[~upright].mean(axis=1)
    sloped = np.linalg.norm(centre[lean > 1.0][:, :2], axis=1)
    assert sloped.min() > params.collar_radius - 1
    assert sloped.max() < params.socket_radius + params.interlock + 1


def test_the_body_is_its_profile_extruded_less_the_socket(holder):
    """Which is the claim the lean test rests on, so it is worth making
    separately: no draft, no fillet in Z, nothing in the fixed body that is not
    either in the drawing or the hole the collar turns in."""
    params = Params()
    face = profile(params).faces()[0]
    socket = _swell(params.socket_radius, params)
    assert body(params).volume + socket.volume == pytest.approx(
        face.area * params.band, rel=1e-6
    )
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


def test_the_body_hangs_plumb_under_the_collar():
    """Which is what the swivel buys. The collar goes where the lead puts it;
    the body turns under it until its own weight is below the joint, and its
    weight is below the joint from every angle because it is all below it."""
    params = Params()
    centre = body(params).center(CenterOf.MASS)
    # 1e-4 mm is the mass-property solver's own tolerance on a part this size,
    # not an offset: the profile is mirror-symmetric by construction.
    assert centre.X == pytest.approx(0, abs=1e-4)
    assert centre.Y < -params.head_radius


def test_both_bodies_are_sound(holder):
    assert holder.is_valid
    assert all(s.is_valid for s in holder.solids())
    assert len(holder.solids()) == 2


# --- the parameter guards --------------------------------------------------


def test_a_ribbon_of_one_extrusion_is_rejected():
    with pytest.raises(ValueError, match="extrusions"):
        build(Params(wall=MIN_WALL / 2))


def test_a_swell_the_gap_swallows_is_rejected():
    with pytest.raises(ValueError, match="lifts straight out"):
        build(Params(interlock=0.3, gap=0.35))


def test_a_swell_too_steep_to_print_is_rejected():
    with pytest.raises(ValueError, match="off vertical"):
        build(Params(interlock=6.0))


def test_a_joint_that_reaches_the_socket_is_rejected():
    with pytest.raises(ValueError, match="into the collar's socket"):
        build(Params(joint=4.0))


def test_a_collar_that_passes_no_webbing_is_rejected():
    with pytest.raises(ValueError, match="hole in a hook"):
        build(Params(plies=0))


def test_a_throat_that_has_to_be_threaded_is_rejected():
    with pytest.raises(ValueError, match="two-handed"):
        build(Params(slot=MIN_SLOT - 1))


def test_a_throat_as_wide_as_the_belly_is_rejected():
    with pytest.raises(ValueError, match="slot with a bulge"):
        build(Params(slot=40.0))


def test_a_funnel_past_the_cap_is_rejected():
    with pytest.raises(ValueError, match="guides a bundle down"):
        build(Params(funnel=MAX_FUNNEL + 5))


def test_an_opening_that_does_not_widen_in_order_is_rejected():
    with pytest.raises(ValueError, match="flares inward"):
        build(Params(cheek=9.0))


def test_a_pinch_wider_than_it_is_deep_is_rejected():
    with pytest.raises(ValueError, match="rolls out"):
        build(Params(band=4.0))


def test_a_slot_radiused_away_to_nothing_is_rejected():
    with pytest.raises(ValueError, match="no flat"):
        build(Params(slot_corner=4.0))
