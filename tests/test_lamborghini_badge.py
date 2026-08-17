"""What has to be true of the Lamborghini script badge.

Two things can go wrong here that both produce a file that looks fine. The dots
on the i's can come out as loose discs, and the traced outline can quietly
defeat the boolean operations the bevel is built from -- which it does, in its
raw form, in a way that returns garbage rather than raising.
"""

from __future__ import annotations

import pytest
from build123d import Axis, Box, Pos

from geom.relief import grow, polygonise
from parts.lamborghini_badge import (
    MIN_WALL,
    Params,
    _counters,
    artwork,
    build,
    joined,
    plated,
    report,
    variant,
)
from p2s import profiles


@pytest.fixture(scope="module")
def part():
    """Built once: the flare is a dozen booleans on a 1600-edge outline."""
    return build()


@pytest.fixture(scope="module")
def info(part):
    return report(Params(), part)


# --- the badge is one object ------------------------------------------------


def test_the_badge_is_a_single_solid(part):
    assert part.is_valid
    assert len(part.solids()) == 1


def test_the_dots_are_what_would_otherwise_come_loose():
    """Three pieces without the stems: the script and two discs. This is the
    whole reason dot_neck exists, stated as the thing that happens without it."""
    loose = build(variant(dot_neck=0.0))
    assert len(loose.solids()) == 3


def test_the_stems_lean_with_the_script():
    """The script is italic, so a stem dropped straight down would read as a
    mistake. Each is cut along the line of closest approach instead, which for
    this artwork measures 95 and 104 degrees -- not vertical."""
    sketch = joined(Params())
    assert len(sketch.faces()) == 1

    # The join adds area over the bare artwork; if a stem had missed, the
    # sketch would still be three regions and the face count above would fail.
    assert sketch.area > artwork(Params()).area


def test_a_stem_under_the_nozzle_minimum_is_refused():
    with pytest.raises(ValueError, match=str(MIN_WALL)):
        build(variant(dot_neck=0.4))


# --- the traced outline, and why it is polygonised --------------------------
#
# Straight from the SVG this outline cannot be offset by ANY amount, cannot be
# chamfered, and fuses a small overlapping shape into itself by cutting slivers
# instead of merging: three faces in, nine disjoint faces out. Polygonised, all
# three work. These pin the fix rather than the failure, because the failure is
# OCCT's and may change; what must keep working is the way around it.


def test_polygonising_preserves_the_shape():
    body = max(artwork(Params()).faces(), key=lambda f: f.area)
    poly = polygonise(body)
    assert poly.area == pytest.approx(body.area, rel=1e-3)
    assert len(poly.faces()) == 1


def test_polygonising_keeps_the_counters():
    """A script is full of enclosed loops. Losing them would fill in every
    letter and the area check alone would not notice a hole moving."""
    body = max(artwork(Params()).faces(), key=lambda f: f.area)
    assert len(body.wires()) > 1, "expected counters in the artwork"
    assert len(polygonise(body).faces()[0].wires()) == len(body.wires())


def test_the_polygonised_outline_can_be_offset():
    """The operation the bevel is built from, and the one that fails outright
    on the raw artwork."""
    poly = polygonise(max(artwork(Params()).faces(), key=lambda f: f.area))
    for amount in (0.2, 0.6):
        grown = grow(poly, amount)
        assert grown.area > poly.area


# --- size on the car --------------------------------------------------------


def test_it_comes_out_the_length_you_asked_for(info):
    assert info["length"] == pytest.approx(Params().length, abs=0.05)


def test_the_bevel_does_not_change_the_length():
    """The flare adds material outside the outline, so the artwork is solved to
    length-minus-bevel. Otherwise every badge prints wide."""
    got = report(variant(bevel=0.0))
    assert got["length"] == pytest.approx(Params().length, abs=0.05)


def test_length_is_solved_for_not_assumed():
    """Bevel reduced to suit: at 90mm the default 0.6mm flare closes the gaps
    between strokes -- see the test below."""
    got = report(variant(length=90.0, bevel=0.4))
    assert got["length"] == pytest.approx(90.0, abs=0.05)


def test_a_bevel_that_closes_a_counter_is_refused():
    """The flare widens the strokes AND narrows the gaps between them, so on a
    small enough badge it fills in the enclosed loops of the a, o and g. The
    top face keeps them and the base does not, leaving a letter whose hole does
    not go through.

    Counters can go UP as well as down -- merging strokes trap a new pocket
    between them -- so the check is that the count is unchanged rather than
    that none were lost. Measured on this artwork: 5 counters survive at 150mm
    and at 90mm with a 0.6mm bevel, drop to 3 at 70mm, and drop to 3 at 90mm if
    the bevel goes to 0.8. This used to crash, which at least announced itself; Shapely
    resolves the merge quite happily, so it became silent and has to be counted
    for."""
    with pytest.raises(ValueError, match="counter"):
        build(variant(length=70.0, bevel=0.6))


def test_the_counters_survive_at_the_default_size():
    """The same check passing is what makes the one above meaningful."""
    p = Params()
    sketch = joined(p)
    assert _counters(sketch) >= 4, "expected several enclosed loops in the script"
    assert _counters(grow(sketch, p.bevel)) == _counters(sketch)


def test_it_fits_the_p2s_bed(part):
    box = part.bounding_box()
    assert profiles.machine(0.4).fits((box.size.X, box.size.Y, box.size.Z))


# --- the bevel --------------------------------------------------------------


def _slab_area(part, z: float, thickness: float = 0.08) -> float:
    box = part.bounding_box()
    slab = Pos(box.center().X, box.center().Y, z) * Box(
        box.size.X + 20, box.size.Y + 20, thickness
    )
    hit = part & slab
    return hit.volume / thickness if hit.volume else 0.0


def test_the_letter_face_is_the_untouched_outline(part):
    """The reason the bevel flares downward. The top face is the artwork
    itself, so no hairline can be eaten however big the bevel gets -- which
    matters more here than on the MUSTANG badge, because a script has
    hairlines."""
    p = Params()
    top = sum(
        f.area
        for f in part.faces().filter_by_position(
            Axis.Z, p.thickness - 0.01, p.thickness + 0.01
        )
    )
    assert top == pytest.approx(joined(p).area, rel=0.02)


def test_the_base_is_wider_than_the_face(part, info):
    p = Params()
    top = sum(
        f.area
        for f in part.faces().filter_by_position(
            Axis.Z, p.thickness - 0.01, p.thickness + 0.01
        )
    )
    assert info["bond_area_mm2"] > top


def test_nothing_overhangs(part):
    """What lets this print face up with support off, sampled off the finished
    solid rather than argued from the construction."""
    box = part.bounding_box()
    heights = [box.min.Z + box.size.Z * i / 16.0 for i in range(1, 16)]
    areas = [_slab_area(part, z) for z in heights]
    for z, lower, upper in zip(heights[1:], areas, areas[1:]):
        assert upper <= lower + 1e-3, f"section grows going up through z={z:.2f}"


def test_a_bevel_that_would_leave_no_face_is_refused():
    with pytest.raises(ValueError, match="no flat face"):
        build(variant(thickness=2.0, bevel=2.0))


# --- the face that gets taped ----------------------------------------------


def test_the_bond_face_is_flat_and_on_the_plate():
    laid = plated()
    assert laid.bounding_box().min.Z == pytest.approx(0.0, abs=1e-6)
    assert report(Params(), laid)["bond_area_mm2"] > 1000


def test_the_bond_carries_the_badge_many_times_over(info):
    """Shear on a vertical panel. Even at a deliberately pessimistic
    0.02 N/mm^2 the margin is enormous."""
    weight = info["mass_g"] / 1000.0 * 9.80665
    assert weight / info["bond_area_mm2"] < 0.02 / 50


# --- things that should not print ------------------------------------------


def test_a_missing_wordmark_is_refused():
    with pytest.raises(ValueError, match="no wordmark"):
        build(variant(source="/no/such/file.svg"))


def test_a_negative_sag_is_refused():
    with pytest.raises(ValueError, match="cannot be negative"):
        build(variant(sag=-1.0))


# --- the slice settings the badge ships with -------------------------------


def test_cooling_is_kept_out_of_the_process_overrides():
    from tools.lamborghini_badge import COOLING, SLICE

    assert not set(COOLING) & set(SLICE)
    assert COOLING["fan_max_speed"] == "0"


def test_support_is_off():
    """Nothing overhangs in this orientation."""
    from tools.lamborghini_badge import SLICE

    assert SLICE["enable_support"] == "0"
