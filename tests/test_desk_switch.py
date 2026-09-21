"""Geometric assertions for the desk switch box.

What decides whether this works is not visible in a render: that a finger fits
between the thrown bat and the desk, that the switch and the jack sit in their
walls with the nuts on flat plastic and nothing touching, that a driver can
reach every screw, and that both parts print flat with nothing overhanging.
Each is measured against the solids and the hardware modelled beside them,
not against the arithmetic that placed them.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from build123d import Box, Cylinder, Pos, Rot

from p2s import profiles
from parts.desk_switch import (
    BRIDGE,
    GSW_117,
    GSW_123,
    MAX_LEAN,
    Params,
    _lid_in_place,
    assembly,
    finger,
    jack_body,
    lid,
    screw_heads,
    shell,
    switch_body,
    switch_nut,
    teardrop,
    toggle,
    toggle_sweep,
)

NOZZLE = 0.4


@pytest.fixture(scope="module")
def params():
    return Params()


@pytest.fixture(scope="module")
def box(params):
    return shell(params)


@pytest.fixture(scope="module")
def cover(params):
    return lid(params)


@pytest.fixture(scope="module")
def cover_in_place(params):
    return _lid_in_place(params)


def _touch(a, b) -> float:
    return (a & b).volume


def _lean(part):
    """Per-facet lean off vertical of every downward face, off the mesh."""
    from tools.render import mesh

    points, tris = mesh(part, tol=0.02)
    tri = points[tris]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    on_plate = tri[:, :, 2].max(axis=1) < 1e-6
    lean = np.degrees(np.arcsin(np.clip(-n[:, 2], -1, 1)))
    return tri, on_plate, lean


# --- the finger ------------------------------------------------------------


def test_a_finger_fits_between_the_thrown_bat_and_the_desk(params, box):
    """The one number the whole height follows from. A finger-thick cylinder
    lying along the desk in front of the box clears the bat at the top of its
    throw, the nut, and the box itself."""
    probe = finger(params)
    assert _touch(probe, toggle_sweep(params)) == pytest.approx(0, abs=1e-6)
    assert _touch(probe, switch_nut(params)) == pytest.approx(0, abs=1e-6)
    assert _touch(probe, box) == pytest.approx(0, abs=1e-6)
    # And it is exactly that: the bat's highest point is ``finger`` down.
    assert params.handle_top == pytest.approx(params.finger)


def test_the_bat_thrown_down_stays_clear_of_the_lid(params, cover_in_place):
    down = toggle(params, params.switch.throw)
    assert _touch(down, cover_in_place) == pytest.approx(0, abs=1e-6)
    assert down.bounding_box().max.Z < params.height


# --- the switch ------------------------------------------------------------


def test_the_switch_sits_in_the_wall_touching_nothing(params, box, cover_in_place):
    body = switch_body(params)
    assert _touch(body, box) == pytest.approx(0, abs=1e-6)
    assert _touch(body, cover_in_place) == pytest.approx(0, abs=1e-6)
    assert _touch(body, jack_body(params)) == pytest.approx(0, abs=1e-6)
    assert _touch(switch_nut(params), box) == pytest.approx(0, abs=1e-6)
    assert _touch(toggle_sweep(params), box) == pytest.approx(0, abs=1e-6)


def test_the_nut_bears_on_flat_wall_and_covers_the_teardrop(params, box):
    """The bushing's clamp is what makes the switch feel solid. Inside the
    nut's circle the wall is whole except for the hole itself, and outside the
    face nothing stands proud for the nut to rock on."""
    sw = params.switch
    r = sw.nut_corners / 2
    inside = Pos(0, 0.25, params.axis_depth) * Rot(90, 0, 0) * Cylinder(r, 0.5)
    hole = teardrop(sw.hole).area
    assert _touch(box, inside) / inside.volume == pytest.approx(
        1 - hole / (math.pi * r * r), rel=0.02
    )
    outside = Pos(0, -0.5, params.axis_depth) * Rot(90, 0, 0) * Cylinder(r * 1.5, 1.0)
    assert _touch(box, outside) == pytest.approx(0, abs=1e-6)
    # The teardrop's flat is within the nut's reach, not out past it.
    assert sw.hole / 2 * math.sqrt(2) - BRIDGE / 2 < r + 1.0


def test_the_bushing_has_thread_to_spare(params):
    sw = params.switch
    stack = params.panel + sw.washer + sw.nut_thickness
    assert sw.bushing_length - stack >= 1.0
    with pytest.raises(ValueError, match="thread"):
        shell(Params(panel=8.0))


def test_the_tab_hole_is_on_the_desk_side_and_clear_of_the_bushing(params, box):
    sw = params.switch
    z = params.axis_depth - sw.tab_offset
    assert z < params.axis_depth
    pin = Pos(0, params.panel / 2, z) * Rot(90, 0, 0) * Cylinder(sw.tab_hole / 2 - 0.1, params.panel + 2)
    assert _touch(box, pin) == pytest.approx(0, abs=1e-6)
    # And the wall between the two holes is still wall.
    between = Pos(0, params.panel / 2, (z + sw.tab_hole / 2 + params.axis_depth - sw.hole / 2) / 2) * Box(1.0, params.panel, 0.4)
    assert _touch(box, between) == pytest.approx(between.volume, rel=1e-3)


def test_the_dpdt_switch_drops_into_the_same_box(params, box):
    """GSW-123 is the same body in the same hole, and the one that actually
    reverses a motor."""
    assert GSW_123.hole == GSW_117.hole and GSW_123.body_length == GSW_117.body_length
    assert shell(Params(switch=GSW_123)).volume == pytest.approx(box.volume)


# --- the jack and the lead -------------------------------------------------


def test_the_jack_sits_in_the_back_wall_touching_nothing(params, box, cover_in_place):
    jack = jack_body(params)
    assert _touch(jack, box) == pytest.approx(0, abs=1e-6)
    assert _touch(jack, cover_in_place) == pytest.approx(0, abs=1e-6)
    r = params.jack.nut_corners / 2
    inside = Pos(params.jack_x, params.back_inner + 0.25, params.jack_depth) * Rot(90, 0, 0) * Cylinder(r, 0.5)
    hole = teardrop(params.jack.hole).area
    assert _touch(box, inside) / inside.volume == pytest.approx(
        1 - hole / (math.pi * r * r), rel=0.02
    )
    outside = Pos(params.jack_x, params.outer_depth + 0.5, params.jack_depth) * Rot(90, 0, 0) * Cylinder(r + 1.0, 1.0)
    assert _touch(box, outside) == pytest.approx(0, abs=1e-6)


def test_the_lead_gets_out_and_ties_down(params, box, cover_in_place):
    """A 6mm two-core lead lies on the lid, through the notch in the back
    wall, and a cable tie goes down through the lid either side of it."""
    lead = Pos(params.notch_x, params.back_inner - 5.0, params.height - 3.0) * Rot(90, 0, 0) * Cylinder(2.75, 30.0)
    assert _touch(lead, box) == pytest.approx(0, abs=1e-6)
    assert _touch(lead, cover_in_place) == pytest.approx(0, abs=1e-6)
    sx, sy = params.tie_slot
    for x in (params.notch_x - params.tie_spread / 2, params.notch_x + params.tie_spread / 2):
        band = Pos(x, params.tie_y, params.height + params.lid / 2) * Box(sx - 0.2, sy - 0.2, params.lid + 6)
        assert _touch(band, cover_in_place) == pytest.approx(0, abs=1e-6)
        assert _touch(band, box) == pytest.approx(0, abs=1e-6)


# --- the screws ------------------------------------------------------------


def test_a_driver_reaches_every_wood_screw(params, box):
    """Pan heads sit on the flange beyond the gusset, and the holes go through."""
    assert _touch(screw_heads(params), box) == pytest.approx(0, abs=1e-6)
    for x, y in params.screw_positions:
        shank = Pos(x, y, params.flange_thickness / 2) * Cylinder(params.screw / 2 - 0.1, params.flange_thickness + 4)
        assert _touch(shank, box) == pytest.approx(0, abs=1e-6)
        assert abs(x) - params.screw_head / 2 > params.outer_width / 2 + params.gusset


def test_the_lid_screws_line_up_with_the_posts(params, box, cover_in_place):
    whole = box + cover_in_place
    for x, y in params.boss_positions:
        pin = Pos(x, y, params.height + params.lid - params.boss_depth / 2 + 0.5) * Cylinder(
            params.boss_hole / 2 - 0.1, params.boss_depth + params.lid + 1
        )
        assert _touch(pin, whole) == pytest.approx(0, abs=1e-6)
        # Under the pilot the post is solid down to the desk.
        post = Pos(x, y, (params.height - params.boss_depth) / 2) * Box(1.0, 1.0, params.height - params.boss_depth - 0.5)
        assert _touch(post, box) == pytest.approx(post.volume, rel=1e-3)


# --- the lid ---------------------------------------------------------------


def test_the_lid_closes_the_shell_and_locates_in_it(params, box, cover_in_place):
    assert _touch(cover_in_place, box) == pytest.approx(0, abs=1e-6)
    bb = cover_in_place.bounding_box()
    assert bb.size.X == pytest.approx(params.outer_width)
    assert bb.size.Y == pytest.approx(params.outer_depth)
    assert bb.min.Z == pytest.approx(params.height - params.lip)
    assert bb.max.Z == pytest.approx(params.overall_height)
    # The lip is inside the walls by its clearance: a ring of air the height
    # of the lip fits between them, away from the posts.
    at = Pos(0, params.panel + params.depth / 2, params.height - params.lip / 2)
    gap = at * Box(params.width - 0.1, params.depth - 0.1, params.lip - 0.1) - at * Box(
        params.width - 2 * params.lip_clearance + 0.1, params.depth - 2 * params.lip_clearance + 0.1, params.lip
    )
    for x, y in params.boss_positions:
        gap -= Pos(x, y, params.height - params.lip / 2) * Box(params.boss + 1, params.boss + 1, params.lip)
    assert gap.volume > 0
    assert _touch(gap, cover_in_place) == pytest.approx(0, abs=1e-6)
    assert _touch(gap, box) == pytest.approx(0, abs=1e-6)


def test_the_printed_lid_is_the_same_part_turned_over(params, cover, cover_in_place):
    """Turned, not mirrored, so the tie slots come out where the notch is."""
    assert cover.volume == pytest.approx(cover_in_place.volume)
    bb = cover.bounding_box()
    assert bb.min.Z == pytest.approx(0)
    assert bb.max.Z == pytest.approx(params.lid + params.lip)
    sx, sy = params.tie_slot
    for x in (params.notch_x - params.tie_spread / 2, params.notch_x + params.tie_spread / 2):
        band = Pos(-x, params.tie_y, params.lid / 2) * Box(sx - 0.2, sy - 0.2, params.lid + 6)
        assert _touch(band, cover) == pytest.approx(0, abs=1e-6)


# --- printability ----------------------------------------------------------


def test_nothing_in_the_shell_overhangs_past_what_fdm_bridges(params, box):
    """Rim-down, every wall hole is a teardrop and every flange is on the
    plate. What is left leaning past 45 degrees is the top of the 2.8mm tab
    hole, which is round because its own teardrop would be smaller than it,
    and the flats are the 2mm bridges across the teardrops' tips."""
    tri, on_plate, lean = _lean(box)
    centre = tri.mean(axis=1)
    sw = params.switch

    flat = (~on_plate) & (lean > 89)
    spans = tri[flat][:, :, 0].max(axis=1) - tri[flat][:, :, 0].min(axis=1)
    assert spans.max() <= BRIDGE + 1e-6

    sloped = (~on_plate) & (lean > 5.0) & (lean <= 89)
    steep = sloped & (lean > MAX_LEAN + 0.5)
    tab = np.array([0.0, params.axis_depth - sw.tab_offset])
    off = np.linalg.norm(centre[steep][:, [0, 2]] - tab, axis=1)
    assert off.max() <= sw.tab_hole / 2 + 0.1, "something other than the tab hole overhangs"
    assert (centre[steep][:, 1] < params.panel + 1e-6).all()


def test_nothing_in_the_lid_overhangs_past_what_fdm_bridges(cover):
    """Face-down the countersinks are 45 degree cones and everything else is
    vertical."""
    tri, on_plate, lean = _lean(cover)
    sloped = (~on_plate) & (lean > 5.0)
    assert lean[sloped].max() <= MAX_LEAN + 0.5


def test_the_walls_are_thicker_than_two_extrusions(params):
    lw = profiles.machine(NOZZLE).line_width
    assert min(params.wall, params.lid, params.panel) >= 2 * lw


def test_a_plate_of_both_parts_fits_the_bed(box, cover):
    from tools.render import plate

    size = plate([box, cover]).bounding_box().size
    assert profiles.machine(NOZZLE).fits((size.X, size.Y, size.Z))


def test_both_parts_are_sound(box, cover):
    for part in (box, cover):
        assert part.is_valid
        assert len(part.solids()) == 1


def test_the_box_is_the_size_the_arithmetic_says(params, box):
    bb = box.bounding_box()
    w, d = params.footprint
    assert bb.size.X == pytest.approx(w)
    assert bb.size.Y == pytest.approx(d)
    assert bb.size.Z == pytest.approx(params.height)
    assert assembly(params).bounding_box().max.Z == pytest.approx(params.overall_height)


# --- the parameter guards --------------------------------------------------


def test_a_gap_no_finger_fits_is_rejected():
    with pytest.raises(ValueError, match="finger"):
        shell(Params(finger=10))


def test_a_hole_the_bushing_will_not_pass_is_rejected():
    from dataclasses import replace

    with pytest.raises(ValueError, match="will not pass"):
        shell(Params(switch=replace(GSW_117, hole=12.0)))


def test_a_box_too_narrow_for_the_switch_is_rejected():
    with pytest.raises(ValueError, match="wider"):
        shell(Params(width=30))


def test_a_jack_over_the_switch_body_is_rejected():
    with pytest.raises(ValueError, match="over the switch body"):
        shell(Params(jack_x=5.0, notch_x=-16.0))


def test_a_jack_in_the_corner_post_is_rejected():
    with pytest.raises(ValueError, match="corner post"):
        shell(Params(jack_x=30.0))


def test_a_jack_in_the_flange_is_rejected():
    with pytest.raises(ValueError, match="gusset"):
        shell(Params(jack_depth=8.0))


def test_a_jack_and_lead_on_the_same_side_are_rejected():
    with pytest.raises(ValueError, match="opposite"):
        shell(Params(notch_x=16.0))


def test_a_flange_with_no_room_for_a_screw_head_is_rejected():
    with pytest.raises(ValueError, match="flat left"):
        shell(Params(flange=8.0))


def test_a_lid_validates_the_same_way():
    with pytest.raises(ValueError, match="finger"):
        lid(Params(finger=10))
