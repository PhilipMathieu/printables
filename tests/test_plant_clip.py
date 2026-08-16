"""Geometric assertions for the plant clip.

Three things decide whether these work, and none of them is visible in a render:
that the pad leaves the strip's pull tab clear, that the only overhang in the
part is inside 45 degrees, and that the C actually holds the stem it is cut for
without gripping it. Each has a test below that measures the solid rather than
trusting the arithmetic that built it.
"""

from __future__ import annotations

import math

import pytest
from build123d import Axis, Cylinder, Pos, Rot

from geom import strips
from p2s import profiles
from parts.plant_clip import MAX_WRAP, MIN_WALL, STEMS, Params, build, pad

NOZZLE = 0.4


@pytest.fixture(scope="module")
def clip():
    return build(Params())


def _stem(params: Params, rise: float = 0.0):
    """The stem itself, as a solid, lying in the cradle or lifted out of it."""
    return Pos(0, 0, params.axis_z + rise) * Rot(0, 90, 0) * Cylinder(
        params.stem / 2, params.clip_width * 2
    )


# --- the strips ------------------------------------------------------------


@pytest.mark.parametrize("strip", strips.CATALOGUE.values(), ids=strips.CATALOGUE)
def test_a_strips_bond_and_tab_account_for_all_of_it(strip):
    assert strip.bond + strip.tab == pytest.approx(strip.length)
    assert 0 < strip.bond < strip.length


def test_small_and_medium_share_a_width():
    """Which is why moving up a strip for a heavier vine changes the pad's
    length and nothing else about the design."""
    assert strips.SMALL.width == pytest.approx(strips.MEDIUM.width)


def test_an_unknown_strip_says_what_there_is():
    with pytest.raises(KeyError, match="small"):
        strips.named("enormous")


# --- the pad ---------------------------------------------------------------


def test_the_pad_stops_where_the_pull_tab_starts():
    """A pad the full length of the strip buries the tab, and a strip whose tab
    cannot be pulled has to be cut off the wall instead of stretched off."""
    params = Params()
    assert params.pad_length == pytest.approx(params.strip.bond)
    assert params.pad_length + params.strip.tab == pytest.approx(params.strip.length)


def test_the_pad_backs_the_adhesive_to_its_edges():
    params = Params()
    assert params.pad_width > params.strip.width
    assert build(params).bounding_box().size.Y == pytest.approx(params.pad_width)


def test_the_face_the_adhesive_gets_is_one_flat_face(clip):
    """3M want a smooth, rigid, non-porous backing, and on a printed part that
    is the face that was pressed against the plate -- so it has to be the whole
    underside, unbroken by anything the clip does above it."""
    params = Params()
    bottom = clip.faces().sort_by(Axis.Z)[0]
    corners = (4 - math.pi) * params.corner**2
    assert bottom.center().Z == pytest.approx(0)
    assert bottom.area == pytest.approx(
        params.pad_length * params.pad_width - corners, rel=1e-3
    )


# --- printability ----------------------------------------------------------


def test_nothing_in_the_part_overhangs_more_than_45_degrees(clip):
    """The whole reason the mouth faces away from the wall and the wrap is
    capped: printed pad-down there is no support anywhere, so every downward
    face has to be inside what FDM bridges on its own.

    Measured off the mesh, ignoring the underside itself -- that one is not an
    overhang, it is the build plate.
    """
    from tools.render import mesh

    import numpy as np

    points, tris = mesh(clip, tol=0.02)
    tri = points[tris]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    on_plate = tri[:, :, 2].max(axis=1) < 1e-6
    lean = np.degrees(np.arcsin(np.clip(-n[~on_plate, 2], -1, 1)))
    assert lean.max() < 45.0
    # And it is the arm tips that set it, which is what ``wrap`` controls.
    assert lean.max() == pytest.approx(Params().wrap / 2 - 90, abs=1.5)


def test_the_arms_are_thicker_than_two_extrusions():
    """One bead with no wall either side has no spring in it and snaps."""
    assert Params().wall >= 2 * profiles.machine(NOZZLE).line_width


def test_a_plate_of_the_whole_size_set_fits_the_bed():
    from tools.plant_clip import plate

    laid = plate([build(Params().for_stem(s)) for s in STEMS for _ in range(3)])
    size = laid.bounding_box().size
    assert profiles.machine(NOZZLE).fits((size.X, size.Y, size.Z))


# --- holding a stem --------------------------------------------------------


def test_the_stem_it_is_cut_for_sits_in_the_bore_without_touching(clip):
    """Clearance, not a fit: the stem is alive and gets thicker."""
    assert (clip & _stem(Params())).volume == pytest.approx(0, abs=1e-6)


def test_the_stem_cannot_leave_without_opening_the_arms(clip):
    """What makes it a clip. Lifted towards the mouth the stem runs into the
    tips, so it comes out only if they are sprung apart."""
    params = Params()
    escaping = _stem(params, rise=params.bore_radius * math.cos(
        math.radians(params.mouth_angle)))
    assert (clip & escaping).volume > 1.0


def test_the_mouth_is_narrower_than_the_stem():
    params = Params()
    assert params.mouth < params.stem
    assert params.grip == (params.mouth, params.bore)


def test_the_named_sizes_cover_pothos_and_hoya():
    """Roughly 4 to 12mm between them, with no stem falling between two clips
    by more than the clearance one of them would have swallowed anyway."""
    clips = [Params().for_stem(s) for s in STEMS]
    assert clips[0].grip[0] <= 4.5
    assert clips[-1].grip[1] >= 11.5
    for small, large in zip(clips, clips[1:]):
        assert large.grip[0] - small.grip[1] < small.clearance


# --- more than one clip on a pad -------------------------------------------


def test_two_clips_sit_evenly_along_the_pad_and_stay_on_it():
    params = Params(strip=strips.MEDIUM, count=2)
    left, right = params.positions
    assert left == pytest.approx(-right)
    assert left - params.clip_width / 2 > -params.pad_length / 2
    part = build(params)
    assert part.is_valid and len(part.solids()) == 1
    assert part.volume > build(Params(strip=strips.MEDIUM)).volume


def test_the_clip_is_one_sound_solid(clip):
    assert clip.is_valid
    assert len(clip.solids()) == 1


# --- the parameter guards --------------------------------------------------


def test_a_wrap_that_curls_the_tips_over_is_rejected():
    with pytest.raises(ValueError, match="off vertical"):
        build(Params(wrap=MAX_WRAP + 10))


def test_a_wrap_that_holds_nothing_in_is_rejected():
    with pytest.raises(ValueError, match="shelf"):
        build(Params(wrap=170))


def test_a_mouth_wider_than_its_stem_is_rejected():
    """Reachable inside the printable band: at a legal wrap a big enough bore
    still opens wider than the stem it was cut for."""
    with pytest.raises(ValueError, match="wider than"):
        build(Params(stem=6.0, clearance=3.0, wrap=200))


def test_a_wall_of_one_extrusion_is_rejected():
    with pytest.raises(ValueError, match="extrusions"):
        build(Params(wall=MIN_WALL / 2))


def test_clips_that_overrun_the_pad_are_rejected():
    with pytest.raises(ValueError, match="overrun"):
        build(Params(count=4))


def test_a_clip_wider_than_the_pad_is_rejected():
    with pytest.raises(ValueError, match="hangs"):
        build(Params(stem=16.0))


def test_the_pad_alone_validates_too():
    with pytest.raises(ValueError, match="sticker"):
        pad(Params(count=0))
