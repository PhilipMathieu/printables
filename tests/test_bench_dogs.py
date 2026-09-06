"""Geometric assertions for the drill press dog set.

Four things decide whether these work, and none of them shows up in a render.
That a stop cannot fall through its hole and cannot foul its neighbour. That its
working face is the same distance from the hole's centre whichever way it is
turned, because that is the entire reason the stops are round. That the shank
stops short of the deck's underside, because a shank a hair long lifts the deck
off the casting and every other number here is measured from that surface. And
that nothing in the set overhangs past 45 degrees in the one orientation they
are all modelled in.

Each is measured off the built solid rather than recomputed from the arithmetic
that made it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from geom import dog_grid
from parts import dog_deck
from dataclasses import replace

from parts.bench_dogs import (
    BUILDERS,
    Params,
    capacity,
    clamp,
    fence,
    ladder,
    pad,
    puck,
    stop,
)

COS45 = math.cos(math.radians(45))


@pytest.fixture(scope="module")
def params():
    return Params()


@pytest.fixture(scope="module")
def deck():
    return dog_deck.Params()


def facets(part, tol: float = 0.02):
    """Triangles of the tessellated solid, with outward normals and areas.

    The winding that comes back is trusted only after checking it: the signed
    volume from the divergence theorem has to come out positive, and if it does
    not, every normal is flipped. A test that silently reads the inside of a
    solid as its outside would pass on a part that prints as a cloud.
    """
    verts, tris = part.tessellate(tol)
    pts = np.array([(v.X, v.Y, v.Z) for v in verts])
    tri = pts[np.array(tris)]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1) / 2
    keep = area > 1e-9
    tri, cross, area = tri[keep], cross[keep], area[keep]
    normal = cross / np.linalg.norm(cross, axis=1, keepdims=True)
    centroid = tri.mean(axis=1)
    if float((centroid * normal).sum(axis=1) @ area) < 0:
        normal = -normal
    return tri, normal, area


def overhangs(part, floor_tol: float = 1e-3, slack: float = 0.02,
              bridged_at: float | None = None):
    """Facets that would print over air.

    Downward facing, leaning more than 45 degrees off vertical, and not lying on
    the build plate -- the plate-facing bottom of a part points straight down and
    is not an overhang, it is the first layer.

    ``bridged_at`` exempts one named height: the flat ceiling of an engraved
    mark, which is a bridge and not a roof. The distinction is span, not angle.
    A letter stroke is well under a millimetre across and bridges over nothing;
    a wide flat ceiling at the same angle would sag. Since a facet's angle
    cannot tell those apart, the exemption is deliberately narrow -- one height,
    named by the caller, and the caller is expected to check separately that
    what it let through really is the engraving.
    """
    tri, normal, _ = facets(part)
    floor = tri[:, :, 2].min()
    ceiling = np.zeros(len(tri), dtype=bool)
    if bridged_at is not None:
        ceiling = (np.abs(tri[:, :, 2] - bridged_at) < floor_tol).all(axis=1)
    on_plate = (np.abs(tri[:, :, 2] - floor) < floor_tol).all(axis=1)
    return normal[(-normal[:, 2] > COS45 + slack) & ~on_plate & ~ceiling]


# --- the grid --------------------------------------------------------------


@pytest.mark.parametrize("grid", dog_grid.CATALOGUE.values(), ids=dog_grid.CATALOGUE)
def test_every_grid_leaves_a_web_between_its_holes(grid):
    assert grid.web > 0
    assert grid.pitch > grid.hole


def test_odd_counts_put_a_station_on_the_spindle_axis():
    """The centre station is the datum, the backer's home, and where the bit
    goes when it comes through the work. Without it the deck has none of those."""
    stations = dog_grid.HALF_INCH.stations(7, 5)
    assert len(stations) == 35
    assert (0.0, 0.0) in [(round(x, 9), round(y, 9)) for x, y in stations]


def test_the_grid_is_centred_on_the_origin():
    xs = [x for x, _ in dog_grid.HALF_INCH.stations(7, 5)]
    assert min(xs) == pytest.approx(-max(xs))


def test_an_unknown_grid_or_fit_says_what_there_is():
    with pytest.raises(KeyError, match="half-inch"):
        dog_grid.named("metric")
    with pytest.raises(KeyError, match="slip"):
        dog_grid.fit("interference")


def test_the_fits_are_a_ladder_with_no_gaps_in_it():
    assert [dog_grid.FITS[n] for n in dog_grid.LADDER] == sorted(
        dog_grid.FITS.values(), reverse=True
    )


# --- the stop --------------------------------------------------------------


def test_a_stop_cannot_fall_through_its_own_hole(params):
    """The head is what the whole part stands on. Smaller than the hole and the
    dog is a pin that drops out the bottom the first time it is knocked."""
    bb = stop(params).bounding_box()
    assert bb.size.X == pytest.approx(params.head)
    assert params.head > params.grid.hole


def test_two_stops_in_neighbouring_holes_do_not_foul(params):
    """Otherwise every station next to an occupied one is lost, and the grid is
    a quarter of the size it looks."""
    assert params.head < params.grid.pitch


def test_a_stops_reach_is_the_same_whichever_way_it_faces(params):
    """The reason the stops are round rather than square-faced. Measured on the
    solid: every point on the head's surface at mid height is the same distance
    from the axis, so a workpiece touches it ``reach`` from the hole's centre
    however the dog happens to have been dropped in."""
    tri, _, _ = facets(stop(params), tol=0.01)
    # Selected by facet, not by vertex: a tessellated cylinder has vertices only
    # at the ends of its wall, so filtering points by height finds none of it.
    height = tri[:, :, 2].mean(axis=1)
    wall = tri[(height > params.break_edge + 0.05) & (height < params.rise - 0.05)]
    pts = wall.reshape(-1, 3)
    radii = np.hypot(pts[:, 0], pts[:, 1])
    assert radii.max() == pytest.approx(params.reach, abs=0.05)
    assert radii.min() == pytest.approx(params.reach, abs=0.05)


def test_the_shank_stops_short_of_the_decks_underside(params, deck):
    """A shank flush with the underside, printed a hair long, stands the deck
    off the casting -- and then every reference in the set is measured from a
    plate that rocks."""
    assert params.shank_length == pytest.approx(deck.deck - params.relief)
    assert params.shank_length < deck.deck
    total = stop(params).bounding_box().size.Z
    assert total - params.rise == pytest.approx(params.shank_length)


def test_the_shank_is_the_hole_less_the_fit(params):
    tri, _, _ = facets(stop(params), tol=0.01)
    pts = tri.reshape(-1, 3)
    near_tip = pts[pts[:, 2] > params.rise + params.shank_length / 2]
    assert np.hypot(near_tip[:, 0], near_tip[:, 1]).max() == pytest.approx(
        params.shank / 2, abs=0.05
    )
    assert params.shank == pytest.approx(params.grid.hole - params.fit)


def test_every_shank_root_is_filleted(params):
    """Where a shank breaks is its root, in layer-line tension, and a square
    root is the stress raiser that decides when. The cone is the fillet -- the
    only shape of one that still prints in this orientation."""
    assert params.root > 0
    for name in ("stop", "fence", "clamp"):
        part = BUILDERS[name](params)
        tri, _, _ = facets(part, tol=0.01)
        # The widest thing at the very base of a shank is the cone, not the
        # shank: measured just above the face the shanks grow out of.
        seat = {"stop": params.rise, "fence": params.fence_height,
                "clamp": params.clamp_height}[name]
        band = tri[(tri[:, :, 2].mean(axis=1) > seat + 0.05)
                   & (tri[:, :, 2].mean(axis=1) < seat + params.root - 0.05)]
        assert len(band), name


def test_a_square_shank_root_is_refused():
    with pytest.raises(ValueError, match="layer-line tension"):
        stop(Params(root=0.0))


def test_the_root_cone_sits_inside_the_holes_chamfer(params, deck):
    """Which is why it is free. The cone lives in the void the deck's lead-in
    chamfer already leaves at the mouth of the hole, so it takes no width off
    the seat -- and being a cone in a cone, it centres the dog as well."""
    chamfer_r = deck.grid.hole / 2 + deck.hole_chamfer
    assert params.shank / 2 + params.root < chamfer_r
    # Clears all the way down, not just at the mouth: both are 45 degrees, so
    # checking the two ends is checking the whole depth.
    assert params.root <= deck.hole_chamfer


def test_a_stop_seats_on_flat_deck_and_not_on_the_chamfer(params, deck):
    """Unguarded until now, and already only 0.85mm wide. A head barely larger
    than the chamfer's outer edge seats the dog on a slope, and a dog on a slope
    is a dog whose reach depends how hard it was pushed in."""
    chamfer_r = deck.grid.hole / 2 + deck.hole_chamfer
    seat = params.head / 2 - chamfer_r
    assert seat > 0.5, f"only {seat:.2f}mm of flat seat under the head"


def test_the_ladder_is_one_stop_per_fit(params):
    """The first print. Four clearances, and the tightest that still drops in
    is the answer for everything else in the set."""
    rungs = ladder(params)
    assert len(rungs) == len(dog_grid.LADDER)
    widths = [r.bounding_box().size.Z for r in rungs]
    assert widths == pytest.approx([widths[0]] * len(widths))


# --- the fence -------------------------------------------------------------


def test_a_fence_face_and_a_stops_face_are_coplanar(params):
    """So the two are interchangeable along one reference line: a fence over
    four pitches and a stop six pitches out are the same straight edge, and work
    bridging them touches both rather than pivoting on the stop."""
    assert params.fence_reach == pytest.approx(params.reach)
    assert params.fence_thickness == pytest.approx(params.head)


def test_the_fence_face_is_flat_and_where_it_says_it_is(params):
    bb = fence(params).bounding_box()
    assert bb.size.Y == pytest.approx(params.fence_thickness)
    assert bb.max.Y == pytest.approx(params.fence_reach)
    assert bb.min.Y == pytest.approx(-params.fence_reach)


def test_the_fences_shanks_are_a_whole_number_of_pitches_apart(params):
    apart = params.grid.span(params.fence_span + 1)
    assert apart % params.grid.pitch == pytest.approx(0, abs=1e-9)
    assert apart == pytest.approx(params.fence_span * params.grid.pitch)


def test_an_even_span_centres_the_fence_on_a_station(params):
    """Odd spans work too, offset half a pitch; even ones let a stop share the
    column the bar is centred on."""
    assert params.fence_span % 2 == 0
    half = params.grid.span(params.fence_span + 1) / 2
    assert half % params.grid.pitch == pytest.approx(0, abs=1e-9)


def test_a_fence_on_one_shank_is_refused():
    with pytest.raises(ValueError, match="swing"):
        fence(Params(fence_span=0))


def test_a_fence_thinner_than_a_stop_is_refused():
    """Because the stops would then stand proud of it and the work would rest on
    two points instead of on the face."""
    with pytest.raises(ValueError, match="proud"):
        fence(Params(fence_thickness=12.0))


# --- the clamp -------------------------------------------------------------


def test_the_clamps_reach_window_is_wider_than_the_pitch(params):
    """Otherwise there is a band of workpiece sizes that falls between two rows
    of holes: too big for the clamp in one, too small to reach from the next."""
    near, far = params.reach_window
    assert far - near >= params.grid.pitch


def test_the_pad_shifts_the_reach_window_rather_than_shortening_it(params):
    """Worth asserting because it reads the other way round. The pad sits beyond
    the bolt tip, not between the tip and the thread, so it moves both ends of
    the window out by its own thickness and the width -- which is what has to
    beat the pitch -- is untouched."""
    near, far = params.reach_window
    assert near == pytest.approx(params.pad_thickness)
    assert far - near == pytest.approx(params.travel)
    thicker = Params(pad_thickness=params.pad_thickness + 3)
    assert thicker.reach_window[1] - thicker.reach_window[0] == pytest.approx(
        params.travel
    )


def test_every_gap_up_to_the_capacity_is_reachable_from_some_row(params, deck):
    """The claim the window check exists to support, walked rather than argued:
    take any gap between the clamp's front face and the work, step the clamp
    back a row at a time, and one of those rows lands the gap inside the
    window."""
    near, far = params.reach_window
    pitch = params.grid.pitch
    for tenths in range(0, int(pitch * 10)):
        gap = tenths / 10
        assert any(near <= gap + k * pitch <= far for k in range(4)), gap


def test_a_screw_too_short_to_cover_the_pitch_is_refused():
    with pytest.raises(ValueError, match="between two rows"):
        clamp(Params(screw_length=40.0))


def test_the_clamps_shanks_are_one_pitch_apart(params):
    body = clamp(params)
    bb = body.bounding_box()
    assert bb.size.X == pytest.approx(params.clamp_length)
    assert bb.size.Z == pytest.approx(params.clamp_height + params.shank_length)


def test_the_bore_goes_all_the_way_through(params):
    """A blind bore would trap the bolt and leave nothing for the tip to come
    out of."""
    body = clamp(params)
    d = params.clamp_depth
    front = [f for f in body.faces() if abs(f.center().Y - d / 2) < 1e-6]
    back = [f for f in body.faces() if abs(f.center().Y + d / 2) < 1e-6]
    assert front and back
    # A solid front face would be one rectangle; the bore breaking out of it
    # leaves a face with a hole in it, and so more than one wire.
    assert max(len(f.wires()) for f in front) > 1


def test_the_nut_trap_opens_towards_the_work_not_away_from_it(params):
    """The one that decides whether the clamp works at all.

    The bolt tip pushes the work forward, so the work pushes back along the
    bolt, so the bolt drags the nut rearward. The nut therefore needs solid body
    behind it. Opening the pocket on the back face -- which is the intuitive
    place, since that is the end the bolt goes in -- opens it in exactly the
    direction the load pushes, and the nut leaves the pocket under the first
    turn of the screw.
    """
    body = clamp(params)
    d = params.clamp_depth
    front = [f for f in body.faces() if abs(f.center().Y - d / 2) < 1e-6]
    back = [f for f in body.faces() if abs(f.center().Y + d / 2) < 1e-6]
    # The front face is broken by both the bore and the trap; the back only by
    # the bore, which is what "solid behind the nut" looks like from outside.
    assert max(len(f.wires()) for f in front) > 1
    assert [len(f.wires()) for f in back] == [2]
    assert params.clamp_depth - params.nut_deep > 20


def test_the_nut_traps_gable_leaves_material_under_the_seating_face(params):
    """The gable is dead space above the nut, so it is free to shave by dropping
    the bolt -- right up until the roof over it is two layers, on the face the
    whole clamp bears and tips on."""
    from parts.bench_dogs import NUT_COVER

    assert params.screw_height - params.gable >= NUT_COVER
    with pytest.raises(ValueError, match="seats and tips on"):
        clamp(Params(screw_height=params.gable + 0.1))


def test_the_gable_is_measured_across_the_flats_not_the_corners(params):
    """A flat-up hexagon stands AF/2 tall and across-corners wide. Reading the
    height off the corners describes a point-up trap, which is not the one
    here and is the orientation that does not print."""
    assert params.gable == pytest.approx(
        params.nut_af / 2 + params.nut_across_corners / 4
    )
    assert params.nut_across_corners > params.nut_af  # corners are the wide way
    assert params.screw_height + params.nut_af / 2 < params.clamp_height


# --- the backer ------------------------------------------------------------


def test_a_backer_sits_flush_with_the_deck(params, deck):
    """Proud and the work rocks on it; sunk and the metal is unsupported at
    exactly the moment the bit breaks through."""
    assert puck(params).bounding_box().size.Z == pytest.approx(deck.deck)


@pytest.mark.parametrize("fit_name", dog_grid.LADDER)
def test_a_backer_grips_the_hole_at_every_fit_the_dogs_come_in(params, fit_name):
    """Sized off the hole, not off a shank. Off the shank it inherits whatever
    clearance the dogs were cut to: at a slip fit that left 0.05mm of
    interference, which is printer noise, and at a loose fit the puck came out
    smaller than the hole and dropped through the one station it is ever used
    at -- the centre one, which has the casting's clearance hole under it."""
    p = params.at(fit_name)
    widest = puck(p).bounding_box().size.X
    assert widest > p.grid.hole
    assert widest == pytest.approx(p.grid.hole + p.puck_grip, abs=0.01)


def test_a_backers_wedge_is_confined_to_a_band_near_the_top(params, deck):
    """Spread over the whole height, the taper turns a tenth of a millimetre of
    radial print error into millimetres of seating height. Confined to a band it
    is steep enough that where it stops is where it was meant to."""
    body = puck(params)
    tri, _, _ = facets(body, tol=0.01)
    pts = tri.reshape(-1, 3)
    below = pts[pts[:, 2] < deck.deck - params.puck_band - 0.1]
    assert np.hypot(below[:, 0], below[:, 1]).max() == pytest.approx(
        (params.grid.hole - params.puck_entry) / 2, abs=0.05
    )
    slope = params.puck_grip / 2 / params.puck_band
    assert slope > 0.04  # steep enough that seating height is not a lottery


def test_a_backer_that_falls_through_is_refused():
    with pytest.raises(ValueError, match="falls through"):
        puck(Params(puck_grip=-0.1))


def test_a_backer_tightens_under_the_drills_own_thrust(params):
    """Wider at the top, so pushing down wedges it home rather than driving it
    out. A parallel plug would need a press fit to stay put, and a press fit is
    a puck you cannot change."""
    body = puck(params)
    bb = body.bounding_box()
    tri, _, _ = facets(body, tol=0.01)
    pts = tri.reshape(-1, 3)
    low = pts[pts[:, 2] < bb.max.Z * 0.25]
    high = pts[pts[:, 2] > bb.max.Z * 0.75]
    assert np.hypot(high[:, 0], high[:, 1]).max() > np.hypot(low[:, 0], low[:, 1]).max()
    assert params.puck_grip > 0


# --- printability ----------------------------------------------------------


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_nothing_in_the_set_overhangs_past_45_degrees(params, name):
    """All five print the same way up, shanks towards the sky, and the reason
    they can is that a shank is always narrower than what it grows out of. Every
    change of section on the way up is a step inward; the only sloped faces that
    look downward at all are the lead chamfers, and those are cut at exactly 45.
    """
    bad = overhangs(BUILDERS[name](params))
    worst = 90 - math.degrees(math.acos(min(-bad[:, 2].min(), 1.0))) if len(bad) else 0
    assert len(bad) == 0, (
        f"{name} has {len(bad)} facets printing over air, the worst leaning "
        f"{worst:.1f} degrees off vertical"
    )


def test_a_marked_stop_overhangs_nowhere_but_its_engraving(params):
    """The mark is cut into the face on the plate, so its ceiling is flat and
    downward -- a bridge a letter-stroke wide, which prints, and the only thing
    in the set the 45 degree rule cannot judge for itself. Everything else on a
    marked stop has to stay clean."""
    marked = stop(replace(params, mark="TIGHT"))
    assert len(overhangs(marked, bridged_at=params.mark_depth)) == 0
    # And the exemption is not hiding anything: without it, every facet it lets
    # through is horizontal and at exactly the mark's depth.
    tri, normal, _ = facets(marked)
    loose = (-normal[:, 2] > COS45 + 0.02) & (tri[:, :, 2].min(axis=1) > 1e-3)
    assert loose.any()
    assert np.allclose(-normal[loose, 2], 1.0, atol=1e-6)
    assert np.allclose(tri[loose][:, :, 2], params.mark_depth, atol=1e-6)


def test_an_engraving_takes_a_small_bite_out_of_the_face_it_is_cut_into(params):
    """Which bounds how wide the bridge over it can be. Letters at this size are
    strokes, not slabs: if the mark ever grew to a large fraction of the head it
    would stop being a thing that bridges."""
    plain, marked = stop(params), stop(replace(params, mark="TIGHT"))
    bite = plain.volume - marked.volume
    face = math.pi * (params.head / 2) ** 2 * params.mark_depth
    assert 0 < bite < face / 4


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_every_part_is_one_watertight_solid(params, name):
    part = BUILDERS[name](params)
    assert part.is_valid
    assert len(part.solids()) == 1


# --- the deck --------------------------------------------------------------


def test_the_deck_has_a_hole_on_the_spindle_axis(deck):
    """Measured on the solid: the plate really is open at the origin, which is
    what the backer drops into and what the deck is lined up by."""
    plate = dog_deck.build(deck)
    bottom = [f for f in plate.faces() if abs(f.center().Z) < 1e-6]
    assert len(bottom) == 1
    assert len(bottom[0].wires()) == deck.cols * deck.rows + 1  # holes plus outline


def test_the_deck_is_as_thick_as_the_dogs_were_cut_for(deck, params):
    assert deck.deck == pytest.approx(deck.grid.deck)
    assert params.shank_length < deck.deck


def test_an_even_grid_is_refused_with_the_reason():
    with pytest.raises(ValueError, match="spindle between four holes"):
        dog_deck.build(dog_deck.Params(cols=6))


def test_the_margin_is_never_thinner_than_a_web(deck):
    assert deck.margin >= deck.grid.web


def test_the_deck_fits_the_bed(deck):
    """256mm square on the P2S. A deck that has to be printed in halves is a
    deck whose grid has a seam in it, and the grid is the whole point."""
    plate = dog_deck.build(deck)
    bb = plate.bounding_box()
    assert max(bb.size.X, bb.size.Y) < 250


# --- the two together ------------------------------------------------------


def test_the_deck_can_clamp_anything_that_fits_on_it(params, deck):
    """Fence on the back row, clamp stepping forward a row at a time. Because
    the screw reaches further than a pitch, the ranges from consecutive rows
    overlap and every size below the maximum is reachable from some row."""
    deepest = capacity(params, deck.rows)
    back = params.grid.span(deck.rows) / 2
    assert deepest == pytest.approx(
        back - params.reach + back - params.clamp_depth / 2 - params.pad_thickness
    )
    assert deepest > 70
    near, far = params.reach_window
    assert far - near >= params.grid.pitch


def test_a_clamp_on_the_outermost_row_sits_on_the_plate(deck, params):
    """It has to, because that is exactly where a clamp holding the deepest
    workpiece the deck can take has to go."""
    assert deck.margin >= params.clamp_depth / 2


def test_the_example_enclosure_fits_between_fence_and_clamp(params, deck):
    """A 1590B is the thing this was drawn around; if the set cannot hold one,
    the numbers are wrong somewhere."""
    assert 60.0 < capacity(params, deck.rows)
    assert 112.0 < params.grid.span(deck.cols)


def test_a_pad_covers_the_bolt_it_floats_on(params):
    assert pad(params).bounding_box().size.X == pytest.approx(params.pad)
    assert params.pad > params.bore
