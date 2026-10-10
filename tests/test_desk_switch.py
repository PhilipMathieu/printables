"""Geometric assertions for the desk switch box.

What decides whether this works is not visible in a render: that a fingertip
reaches the rocker's top half without meeting the desk, that the switch's
clips have a web they can grab and room to spring into, that the jack sits in
its wall with the nut on flat plastic and nothing touching, that a driver can
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
    KCD4,
    KCD4_26,
    MAX_LEAN,
    Params,
    _lid_in_place,
    assembly,
    clips,
    coupon,
    desk,
    fingertip,
    jack_body,
    lid,
    screw_heads,
    shell,
    switch_body,
    switch_front,
    teardrop,
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


def test_a_fingertip_reaches_the_top_half_without_meeting_the_desk(params, box):
    """Up is the top half, right under the desk's edge. A fingertip pointed at
    the middle of it clears the desk and the box, and lands on the rocker."""
    probe = fingertip(params)
    assert _touch(probe, desk(params)) == pytest.approx(0, abs=1e-6)
    assert _touch(probe, box) == pytest.approx(0, abs=1e-6)
    cl = params.switch.cutout[1]
    assert params.axis_depth - cl / 2 < params.press_depth < params.axis_depth
    # Nudged forward, it is on the rocker: nothing else is in the way.
    pushed = Pos(0, 1.0, 0) * probe
    assert _touch(pushed, switch_front(params)) > 0


def test_the_bezel_is_below_the_desk_by_the_headroom(params):
    bb = switch_front(params).bounding_box()
    assert bb.min.Z == pytest.approx(params.headroom)
    assert _touch(switch_front(params), desk(params)) == pytest.approx(0, abs=1e-6)


# --- the switch ------------------------------------------------------------


def test_the_switch_sits_in_the_wall_touching_nothing(params, box, cover_in_place):
    body = switch_body(params)
    assert _touch(body, box) == pytest.approx(0, abs=1e-6)
    assert _touch(body, cover_in_place) == pytest.approx(0, abs=1e-6)
    assert _touch(body, jack_body(params)) == pytest.approx(0, abs=1e-6)
    assert _touch(switch_front(params), box) == pytest.approx(0, abs=1e-6)
    # Air under the body before the lid's lip.
    assert body.bounding_box().max.Z + params.clearance <= params.height - params.lip + 1e-6


def test_the_bezel_bears_on_flat_wall_all_round_the_cutout(params, box):
    """The bezel is what a press pushes on. Behind its footprint the face is
    whole plastic except for the cutout, and nothing stands proud of it."""
    (bw, bl), (cw, cl) = params.switch.bezel, params.switch.cutout
    skin = Pos(0, 0.25, params.axis_depth) * Box(bw, 0.5, bl)
    assert _touch(box, skin) == pytest.approx(skin.volume * (1 - cw * cl / (bw * bl)), rel=1e-3)
    proud = Pos(0, -0.5, params.axis_depth) * Box(bw + 4, 1.0, bl + 4)
    assert _touch(box, proud) == pytest.approx(0, abs=1e-6)


def test_the_clips_grab_a_web_they_were_made_for(params, box):
    """Through the cutout's ends the wall is ``clip_panel`` thick and no more,
    and that is inside the range the clips grab."""
    sw = params.switch
    cw, cl = sw.cutout
    lo, hi = sw.panel_range
    assert lo <= params.clip_panel <= hi
    for s in (-1, 1):
        z = params.axis_depth + s * (cl / 2 + sw.clips / 2)
        web = Pos(0, params.clip_panel / 2, z) * Box(cw - 4.0, params.clip_panel - 0.02, 0.5)
        assert _touch(box, web) == pytest.approx(web.volume, rel=1e-3)
        behind = Pos(0, params.clip_panel + 0.5, z) * Box(cw - 4.0, 0.9, 0.5)
        assert _touch(box, behind) == pytest.approx(0, abs=1e-6)


def test_the_clips_spring_out_into_air(params, box, cover_in_place):
    assert _touch(clips(params), box) == pytest.approx(0, abs=1e-6)
    assert _touch(clips(params), cover_in_place) == pytest.approx(0, abs=1e-6)


def test_the_wall_away_from_the_switch_is_full_thickness(params, box):
    """The pocket is a strip round the switch, not the whole front."""
    pw, _ = params.pocket
    x = pw / 2 + 2.0
    full = Pos(x, params.panel / 2, params.height / 2) * Box(1.0, params.panel - 0.02, params.height - 2)
    assert _touch(box, full) == pytest.approx(full.volume, rel=1e-3)


def test_the_smaller_kcd4_drops_into_the_same_box_design(params):
    """The 26 x 22 cutout is the other common one; it only moves the hole."""
    other = Params(switch=KCD4_26)
    assert shell(other).is_valid
    assert other.height < params.height


def test_the_coupon_is_the_front_of_the_shell(params, box):
    """Same cutout, same web, same way up: what fits it fits the box."""
    c = coupon(params)
    assert c.is_valid and len(c.solids()) == 1
    assert c.bounding_box().size.Z == pytest.approx(params.height)
    assert c.bounding_box().size.Y < params.panel + params.boss + 1.5
    assert _touch(c, box) == pytest.approx(c.volume, rel=1e-4)


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
    """Rim-down, the jack hole is a teardrop and every flange is on the
    plate. The flats are the 2mm bridges across the teardrop's tip and the
    one across the far end of the rocker's cutout, which spans only the web."""
    tri, on_plate, lean = _lean(box)
    centre = tri.mean(axis=1)
    cw, cl = params.switch.cutout

    flat = (~on_plate) & (lean > 89)
    spans = tri[flat][:, :, 0].max(axis=1) - tri[flat][:, :, 0].min(axis=1)
    cutout_end = (
        (np.abs(centre[flat][:, 2] - (params.axis_depth + cl / 2)) < 1e-3)
        & (centre[flat][:, 1] < params.clip_panel + 1e-6)
    )
    assert spans[~cutout_end].max() <= BRIDGE + 1e-6
    assert cutout_end.any() and spans[cutout_end].max() <= cw + 1e-6

    sloped = (~on_plate) & (lean > 5.0) & (lean <= 89)
    assert lean[sloped].max() <= MAX_LEAN + 0.5


def test_nothing_in_the_lid_overhangs_past_what_fdm_bridges(cover):
    """Face-down the countersinks are 45 degree cones and everything else is
    vertical."""
    tri, on_plate, lean = _lean(cover)
    sloped = (~on_plate) & (lean > 5.0)
    assert lean[sloped].max() <= MAX_LEAN + 0.5


def test_the_walls_are_thicker_than_two_extrusions(params):
    lw = profiles.machine(NOZZLE).line_width
    assert min(params.wall, params.lid, params.clip_panel) >= 2 * lw


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


def test_a_switch_so_close_to_the_desk_no_finger_fits_is_rejected():
    with pytest.raises(ValueError, match="fingertip"):
        shell(Params(headroom=3.0, finger=40.0))


def test_a_web_the_clips_cannot_grab_is_rejected():
    with pytest.raises(ValueError, match="clips grab"):
        shell(Params(clip_panel=3.5))


def test_a_cutout_the_body_will_not_pass_is_rejected():
    from dataclasses import replace

    with pytest.raises(ValueError, match="will not pass"):
        shell(Params(switch=replace(KCD4, cutout=(21.0, 30.2))))


def test_a_box_too_shallow_to_wire_is_rejected():
    with pytest.raises(ValueError, match="wiring"):
        shell(Params(depth=38.0))


def test_a_box_too_narrow_for_the_switch_is_rejected():
    with pytest.raises(ValueError, match="wider"):
        shell(Params(width=34))


def test_a_jack_over_the_switch_body_is_rejected():
    with pytest.raises(ValueError, match="over the switch body"):
        shell(Params(jack_x=12.0))


def test_a_jack_in_the_corner_post_is_rejected():
    with pytest.raises(ValueError, match="corner post"):
        shell(Params(jack_x=28.0))


def test_a_jack_in_the_flange_is_rejected():
    with pytest.raises(ValueError, match="gusset"):
        shell(Params(jack_depth=8.0))


def test_a_jack_and_lead_on_the_same_side_are_rejected():
    with pytest.raises(ValueError, match="opposite"):
        shell(Params(notch_x=20.5))


def test_a_flange_with_no_room_for_a_screw_head_is_rejected():
    with pytest.raises(ValueError, match="flat left"):
        shell(Params(flange=8.0))


def test_a_lid_validates_the_same_way():
    with pytest.raises(ValueError, match="clips grab"):
        lid(Params(clip_panel=0.9))


# --- the wiring --------------------------------------------------------------


def test_the_wiring_reverses_the_motor_and_never_shorts():
    """The netlist the diagram is drawn from: one press runs the motor one
    way, the other the other way, and the centre leaves it alone."""
    from tools.desk_switch_wiring import WIRING, check, motor

    check()
    up, down = motor("top"), motor("bottom")
    assert up == ("+", "-") and down == ("-", "+")
    assert motor("off") is None
    # Read from the wrong side, the picture is mirrored left to right. The
    # mirror image of the cross is the same circuit, so it wires the same.
    mirrored = {(row, {"left": "right", "right": "left"}[col]): net for (row, col), net in WIRING.items()}
    check(mirrored)
    assert motor("top", mirrored) == up


def test_a_jumper_on_a_common_is_a_short_and_is_caught():
    from tools.desk_switch_wiring import WIRING, check

    wrong = dict(WIRING)
    wrong[("middle", "left")] = "+"
    with pytest.raises(ValueError, match="short"):
        check(wrong)


def test_supply_on_one_row_drives_nothing_and_is_caught():
    """+ on both top tabs and - on both bottom: each press puts one side of
    the supply on both motor leads."""
    from tools.desk_switch_wiring import WIRING, check

    wrong = dict(WIRING)
    wrong[("top", "right")], wrong[("bottom", "right")] = "+", "-"
    with pytest.raises(ValueError):
        check(wrong)
