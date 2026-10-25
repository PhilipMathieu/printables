"""Geometric assertions for the bag holder.

Four things decide whether this works, and none of them is visible in a render:
that the hole the bag goes into is a hole and not a hook, that the funnel stops
each bundle of handles at its own size and none of them at the bottom, that
there is no corner anywhere inside the aperture, and that a mini carabiner goes
through the loop and hangs it plumb. Each is measured off the built solid
rather than trusted to the arithmetic that made it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from build123d import Box, CenterOf, Cylinder, Pos

from p2s import profiles
from parts.bag_holder import (
    MAX_FUNNEL,
    MINI_GATE,
    MINI_ROD,
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
    """Built once: the breaks are a stack of boolean insets and not cheap."""
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
    """Half the width of a solid at one height."""
    cut = solid & (Pos(0, 0, z) * Box(400, 400, 0.02))
    return cut.bounding_box().size.X / 2 if cut.volume > 1e-12 else 0.0


def _area_at(solid, z: float, thick: float = 0.02) -> float:
    """Cross-sectional area of a solid at one height."""
    slab = Pos(0, 0, z) * Box(400, 400, thick)
    return (solid & slab).volume / thick


def _wire_points(wire, samples: int) -> np.ndarray:
    return np.array(
        [(p.X, p.Y) for p in (wire.position_at(i / samples) for i in range(samples))]
    )


# --- the hole is a hole ----------------------------------------------------


def _holes(face):
    """The aperture and the loop's bore, biggest first."""
    return sorted(face.inner_wires(), key=lambda w: w.length, reverse=True)


def test_the_profile_is_one_face_with_two_holes_in_it():
    """The whole retention argument, and the check that the loop landed on the
    crown rather than beside it. Two closed inner wires: the aperture and the
    loop's bore. A hook would not be closed, and two faces would mean the neck
    missed."""
    face = profile(Params()).faces()[0]
    assert len(profile(Params()).faces()) == 1
    assert len(face.inner_wires()) == 2


def test_the_ribbon_is_never_thinner_than_one_wall():
    """Which is what says the aperture and the loop are walled the whole way
    round, that the ribbon really is a constant-width offset and not a shape
    with a hole roughly in the middle of it, and that the loop's bore and the
    aperture never come closer to each other than to the outside."""
    params = Params()
    face = profile(params).faces()[0]
    outer = _wire_points(face.outer_wire(), 2400)
    aperture_, bore = (_wire_points(w, 900) for w in _holes(face))

    def closest(a, b):
        return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2).min()

    assert closest(aperture_, outer) == pytest.approx(params.wall, abs=0.02)
    assert closest(bore, outer) == pytest.approx(params.wall, abs=0.02)
    assert closest(bore, aperture_) >= params.wall


# --- the loop ------------------------------------------------------------


def test_the_loop_takes_a_mini_carabiner_s_rod_with_room_to_swing(holder):
    """A rod in the bore touches nothing even pushed against the bottom of it,
    with a millimetre to spare -- the slack that lets the holder pivot on it
    rather than hang however it was last pushed."""
    params = Params()
    rod = MINI_ROD / 2
    for y in (params.eye_radius - rod - 0.01, -(params.eye_radius - rod - 0.01)):
        pin = Pos(0, y, params.band / 2) * Cylinder(rod, params.band * 3)
        assert (holder & pin).volume == pytest.approx(0, abs=1e-6)
    assert params.eye - MINI_ROD >= 1


def test_the_loop_s_ribbon_goes_through_a_mini_carabiner_s_gate():
    """It has to go in sideways to be clipped on at all, and the section the
    gate has to pass is the ribbon's own, because the loop is the same ribbon
    as the rest."""
    params = Params()
    assert min(params.wall, params.band) < MINI_GATE


def test_the_loop_is_on_the_axis_above_everything_else(holder):
    """The loop is the one place it hangs from, so it has to be the top of the
    part and centred over the opening."""
    params = Params()
    bb = holder.bounding_box()
    assert bb.max.Y == pytest.approx(params.eye_outer, abs=1e-3)
    assert bb.min.X == pytest.approx(-bb.max.X, abs=1e-3)


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
    inner = _holes(profile(params).faces()[0])[0]
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


def test_nothing_overhangs_past_what_fdm_bridges(holder):
    """Nothing in the part leans at all. It is a profile extruded straight up,
    and both breaks are staircases a layer to a step rather than chamfers, so
    every facet is either vertical or flat. The swivel's swell was the only
    sloped thing it ever had."""
    from tools.render import mesh

    params = Params()
    points, tris = mesh(holder, tol=0.02)
    tri = points[tris]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    # Overhang is a downward-facing surface, so the sign matters: the top break
    # leans as far as the plate break does and faces the other way, which is free.
    on_plate = tri[:, :, 2].max(axis=1) < 1e-6
    lean = np.degrees(np.arcsin(np.clip(-n[:, 2], -1, 1)))
    centre = tri.mean(axis=1)

    # The plate break is a staircase one layer to a step, so its risers read as
    # flat downward ledges. Those are not overhangs in any sense that matters --
    # every sloped surface an FDM printer makes is exactly this -- but they are
    # 90 degrees to a mesh, so the slope test is about the sloped facets.
    ledge = (~on_plate) & (lean > 89)
    assert centre[ledge][:, 2].max() <= params.base_break + 1e-6, (
        "an unsupported flat has appeared somewhere other than the plate break"
    )

    # Five degrees rather than nought for the same reason the ledges are
    # excluded above: a flat ring around a curve triangulates a degree or two
    # off level, and those facets are the ledges seen edge on.
    sloped = (~on_plate) & (lean > 5.0) & (lean <= 89)
    assert not sloped.any(), (
        f"a facet leans {lean[sloped].max():.1f} degrees in a part with no slopes"
    )


def test_the_breaks_step_in_one_layer_at_a_time():
    """Which is what makes the plate break's ledges harmless: each is a single
    layer's inset, the same step the slicer would have made of a chamfer."""
    params = Params()
    for depth in (params.edge_break, params.base_break):
        steps = max(1, round(depth / params.slice_height))
        assert depth / steps == pytest.approx(params.slice_height, abs=0.05)


def test_the_part_is_its_profile_extruded(holder):
    """Which is the claim the lean test rests on, so it is worth making
    separately: no draft, no feature that is not in the drawing.

    Measured on a slice between the breaks rather than on the volume, since
    the breaks take a chamfer's worth off both ends.
    """
    params = Params()
    face = profile(params).faces()[0]
    for z in np.linspace(params.base_break + 0.05, params.band - params.edge_break - 0.05, 4):
        assert _area_at(holder, z) == pytest.approx(face.area, rel=1e-3)
    assert holder.bounding_box().size.Z == pytest.approx(params.band)


def test_the_ribbon_s_section_is_an_octagon(holder):
    """Broken at both ends rather than one, which is what a print asked for: a
    single chamfer on the top does not read as rounded in the hand.

    The widest thing in the part is the belly, so measuring its half width at
    three heights measures the section: drawn in one chamfer at the plate, full
    width through the middle, drawn in the other at the top.
    """
    params = Params()
    middle = (params.base_break + params.band / 2) / 2
    full = _radius_at(holder, middle)
    assert _radius_at(holder, 0.02) == pytest.approx(
        full - params.base_break, abs=0.05
    )
    assert _radius_at(holder, params.band - 0.02) == pytest.approx(
        full - params.edge_break, abs=0.05
    )
    assert _area_at(holder, 0.02) < _area_at(holder, middle)
    assert _area_at(holder, params.band - 0.02) < _area_at(holder, middle)


def test_the_ribbon_is_thicker_than_two_extrusions():
    assert Params().wall >= 2 * profiles.machine(NOZZLE).line_width


def test_a_plate_of_them_fits_the_bed():
    from tools.render import plate

    laid = plate([build(Params()) for _ in range(2)])
    size = laid.bounding_box().size
    assert profiles.machine(NOZZLE).fits((size.X, size.Y, size.Z))


# --- how it hangs ----------------------------------------------------------


def test_it_hangs_plumb_under_the_loop(holder):
    """Which is what the carabiner buys. Hung from the top of the loop's bore
    it pivots until its weight is under the rod, and its weight is on the axis
    and well below the rod, so plumb is with the funnel pointing up."""
    params = Params()
    centre = holder.center(CenterOf.MASS)
    # A micron of slack for the solver: the profile is mirror-symmetric by
    # construction, but the breaks are cut by a stack of booleans and mass
    # properties come back a few microns off centre.
    assert centre.X == pytest.approx(0, abs=5e-3)
    assert centre.Y < params.crown_y


def test_it_is_one_sound_body(holder):
    assert holder.is_valid
    assert len(holder.solids()) == 1


# --- the parameter guards --------------------------------------------------


def test_a_ribbon_of_one_extrusion_is_rejected():
    with pytest.raises(ValueError, match="extrusions"):
        build(Params(wall=MIN_WALL / 2))


def test_a_joint_that_reaches_the_bore_is_rejected():
    with pytest.raises(ValueError, match="into its bore"):
        build(Params(joint=4.0))


def test_a_loop_too_tight_for_the_rod_is_rejected():
    with pytest.raises(ValueError, match="no room to swing"):
        build(Params(eye=MINI_ROD + 0.5))


def test_a_ribbon_too_fat_for_the_gate_is_rejected():
    with pytest.raises(ValueError, match="carabiner gate"):
        build(Params(wall=MINI_GATE, band=MINI_GATE))


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


def test_breaks_that_eat_the_whole_ribbon_are_rejected():
    with pytest.raises(ValueError, match="no flat on top"):
        build(Params(edge_break=3.0))


def test_breaks_that_meet_across_the_band_are_rejected():
    with pytest.raises(ValueError, match="no side wall"):
        build(Params(edge_break=1.6, base_break=1.6, band=3.0))


def test_a_negative_break_is_rejected():
    with pytest.raises(ValueError, match="burr"):
        build(Params(base_break=-0.2))

