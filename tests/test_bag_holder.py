"""Geometric assertions for the bag holder.

Four things decide whether this works, and none of them is visible in a render:
that the slot passes a folded handle and no more, that the part really is one
profile with nothing overhanging, that the only way out of the horn is above
the horn's centre, and that the whole thing hangs the right way up on its own.
Each is measured off the built solid rather than trusted to the arithmetic that
made it.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from build123d import Box, CenterOf, Cylinder, Plane, Pos

from geom import webbing
from p2s import profiles
from parts.bag_holder import (
    MAX_ESCAPE_BEARING,
    MIN_GATE,
    MIN_SEAT,
    MIN_WALL,
    Params,
    build,
    profile,
)

NOZZLE = 0.4

KNOT = 6.0
"""Diameter of a tied bag's handles, squeezed. Deliberately well under the
gate: the question these tests ask is where an opening is, not how big."""


@pytest.fixture(scope="module")
def holder():
    return build(Params())


def _handle(params: Params, plies: int | None = None):
    """The lead's folded handle as a solid, standing in the slot.

    Runs along Z because that is the direction the lead runs: the part is the
    profile, and the strap goes through it.
    """
    plies = params.plies if plies is None else plies
    return Box(params.strap.width, params.strap.stack(plies), params.band * 3)


def _knot(params: Params, bearing: float, radius: float | None = None):
    """A tied bag's handles, on the seat circle at some bearing off straight up."""
    radius = params.seat_radius if radius is None else radius
    a = math.radians(bearing)
    return Pos(
        radius * math.sin(a), params.horn_y + radius * math.cos(a), params.band / 2
    ) * Cylinder(KNOT / 2, params.band * 3)


# --- the webbing -----------------------------------------------------------


@pytest.mark.parametrize("strap", webbing.CATALOGUE.values(), ids=webbing.CATALOGUE)
def test_a_width_is_the_inches_it_is_sold_as(strap):
    """The width is the one number a lead is actually held to, so it has to be
    the nominal size converted and not a figure off a caliper."""
    inches = float(Fraction(strap.nominal.split()[0]))
    assert strap.width == pytest.approx(inches * webbing.MM_PER_INCH)
    assert strap.handle == pytest.approx(2 * strap.thickness)


def test_the_catalogue_runs_thickest_at_its_widest():
    """Wider leads are heavier leads, and heavier webbing is thicker webbing.
    A catalogue that broke that would size a wide lead's slot off a thin one."""
    order = [webbing.TOY, webbing.SMALL, webbing.STANDARD, webbing.WIDE]
    assert [w.width for w in order] == sorted(w.width for w in order)
    assert [w.thickness for w in order] == sorted(w.thickness for w in order)


def test_an_unknown_width_says_what_there_is():
    with pytest.raises(KeyError, match="standard"):
        webbing.named("paracord")


# --- getting it onto the lead ----------------------------------------------


def test_the_slot_passes_the_folded_handle(holder):
    """The only way onto a lead that does not go past the snap hook."""
    assert (holder & _handle(Params())).volume == pytest.approx(0, abs=1e-6)


def test_the_slot_is_cut_for_the_fold_and_not_a_millimetre_more():
    """The clearance is for the stitching down a handle's fold, not for a
    third ply -- a slot loose enough for one is a holder that rattles."""
    params = Params()
    assert (build(params) & _handle(params, plies=params.plies + 1)).volume > 1.0
    assert params.slot_height == pytest.approx(
        params.strap.handle + params.slot_clearance
    )


def test_the_strap_bears_on_a_wall_and_not_an_edge(holder):
    """Webbing wears on edges. The top of the slot is the face that carries the
    holder's whole weight, so it has to be a face: as wide as the slot's flat
    and as long as the part is tall."""
    params = Params()
    tops = [
        f
        for f in holder.faces().filter_by(Plane.XZ)
        if f.center().Y == pytest.approx(params.slot_height / 2)
    ]
    assert len(tops) == 1
    assert tops[0].area == pytest.approx(params.bearing_area)
    assert params.bearing_area > 100


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
    separately: no draft, no fillet, no feature that is not in the drawing."""
    params = Params()
    face = profile(params).faces()[0]
    assert holder.volume == pytest.approx(face.area * params.band, rel=1e-6)
    assert holder.bounding_box().size.Z == pytest.approx(params.band)


def test_the_head_the_neck_and_the_horn_land_on_one_another():
    """One face with one hole in it. Three faces would mean the neck missed."""
    face = profile(Params()).faces()[0]
    assert len(profile(Params()).faces()) == 1
    assert len(face.inner_wires()) == 1


def test_the_ribbon_is_thicker_than_two_extrusions():
    assert Params().wall >= 2 * profiles.machine(NOZZLE).line_width


def test_a_plate_of_the_whole_catalogue_fits_the_bed():
    from tools.render import plate

    laid = plate(
        [build(Params().for_strap(name)) for name in webbing.CATALOGUE for _ in range(2)]
    )
    size = laid.bounding_box().size
    assert profiles.machine(NOZZLE).fits((size.X, size.Y, size.Z))


# --- holding a bag ---------------------------------------------------------


def test_the_bag_sits_in_the_seat_without_touching(holder):
    """At rest the knot lies in the bottom of the seat, clear of the ribbon."""
    params = Params()
    resting = _knot(params, 180.0, radius=params.seat_radius - KNOT / 2)
    assert (holder & resting).volume == pytest.approx(0, abs=1e-6)


def test_the_only_way_out_of_the_horn_is_the_gate_and_it_is_above_the_centre(holder):
    """The design, measured: walk a knot right round the seat and see where the
    horn lets it through. Everything that opens has to be inside the gate, and
    everything inside the gate has to be above the horn's centreline -- which
    is what makes leaving a climb rather than a swing."""
    params = Params()
    ways_out = [
        b for b in range(0, 360, 5)
        if (holder & _knot(params, b)).volume == pytest.approx(0, abs=1e-6)
    ]
    assert ways_out, "the horn is closed; nothing can be hooked into it"
    for bearing in ways_out:
        assert (
            params.gate_bearing - params.gate_half
            <= bearing
            <= params.escape_bearing
        ), f"the horn opens at {bearing} degrees, outside its own gate"
        assert math.cos(math.radians(bearing)) > 0, (
            f"the horn opens at {bearing} degrees, at or below its centreline, "
            f"so a bag could swing out without rising"
        )


def test_leaving_the_seat_is_a_climb_and_the_climb_is_most_of_the_seat():
    params = Params()
    assert params.lift == pytest.approx(
        params.seat_radius * (1 + math.cos(math.radians(params.escape_bearing)))
    )
    assert params.lift > params.seat_radius
    assert params.escape_bearing < MAX_ESCAPE_BEARING


def test_the_gate_clears_the_neck():
    """Otherwise the way in is blocked by the thing holding the horn on."""
    params = Params()
    assert params.gate_bearing - params.gate_half > params.neck_bearing


# --- how it hangs ----------------------------------------------------------


def test_the_holder_hangs_the_right_way_up_on_its_own(holder):
    """Nothing holds the holder square to the lead -- it pivots on the strap,
    and a flat strap twists. So the horn has to be what rights it: the whole
    mass below the slot makes it a pendulum, and a pendulum hangs with its
    gate up."""
    params = Params()
    assert holder.center(CenterOf.MASS).Y < -params.slot_height / 2
    assert holder.bounding_box().min.Y < params.horn_y


def test_the_holder_is_one_sound_solid(holder):
    assert holder.is_valid
    assert len(holder.solids()) == 1


# --- one horn, four leads --------------------------------------------------


def test_a_wider_lead_changes_the_head_and_nothing_else():
    """Which is why a set is worth printing: whatever is by the door, the part
    that holds the bag is the same part, and only the slot moves."""
    sized = [Params().for_strap(name) for name in webbing.CATALOGUE]
    assert len({(p.seat, p.gate, p.gate_bearing, round(p.lift, 9)) for p in sized}) == 1
    widths = [p.head_width for p in sized]
    assert widths == sorted(widths)
    volumes = [build(p).volume for p in sized]
    assert volumes == sorted(volumes)


# --- the parameter guards --------------------------------------------------


def test_a_gate_that_reaches_below_the_centreline_is_rejected():
    with pytest.raises(ValueError, match="without ever rising"):
        build(Params(gate_bearing=MAX_ESCAPE_BEARING - 5))


def test_a_gate_that_opens_into_the_neck_is_rejected():
    with pytest.raises(ValueError, match="opens into the neck"):
        build(Params(gate_bearing=26.0))


def test_a_gate_too_narrow_to_push_a_knot_through_is_rejected():
    with pytest.raises(ValueError, match="two-handed"):
        build(Params(gate=MIN_GATE - 1))


def test_a_seat_with_no_room_for_a_finger_is_rejected():
    with pytest.raises(ValueError, match="one-handed"):
        build(Params(seat=MIN_SEAT - 1))


def test_a_ribbon_of_one_extrusion_is_rejected():
    with pytest.raises(ValueError, match="extrusions"):
        build(Params(wall=MIN_WALL / 2))


def test_a_band_shorter_than_the_slot_is_deep_is_rejected():
    with pytest.raises(ValueError, match="cocks"):
        build(Params(band=4.0))


def test_a_slot_that_passes_no_webbing_is_rejected():
    with pytest.raises(ValueError, match="hole in a hook"):
        build(Params(plies=0))


def test_a_slot_radiused_away_to_nothing_is_rejected():
    with pytest.raises(ValueError, match="no flat"):
        build(Params(slot_corner=4.0))


def test_a_neck_that_lands_on_the_head_s_corners_is_rejected():
    with pytest.raises(ValueError, match="rounded corners"):
        build(Params(flare=9.0))
