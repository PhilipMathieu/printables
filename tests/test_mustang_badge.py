"""What has to be true of the badge.

Two things can go wrong here that both produce a file that looks fine. The word
can come out as seven loose letters, and the font can come out as Arial because
the renderer substituted it without saying so. Most of these tests are about
those two.
"""

from __future__ import annotations

import pytest
from build123d import Axis, Pos

from p2s import profiles
from parts.mustang_badge import (
    CLASSIC_ASPECT,
    MAX_BEVEL,
    MIN_WALL,
    REFERENCE_SIZE,
    Params,
    _rail,
    _word,
    build,
    check_font,
    plated,
    proportion_note,
    report,
    serif_top,
)

FONT = "DIN Condensed"  # installed here; only a stand-in for the real wordmark


def params(**changes) -> Params:
    return Params(font=FONT, **changes)


@pytest.fixture(scope="module")
def part():
    return build(params())


@pytest.fixture(scope="module")
def info(part):
    return report(params(), part)


@pytest.fixture(scope="module")
def classic():
    """The real defaults, Superclarendon and all. Built once: the flare stack is
    a dozen booleans on glyph outlines and takes long enough to notice."""
    return build(Params())


@pytest.fixture(scope="module")
def classic_info(classic):
    return report(Params(), classic)


# --- the word is one object -------------------------------------------------


def test_the_badge_is_a_single_solid(part):
    """Seven letters you have to align on the car by eye is not a badge."""
    assert part.is_valid
    assert len(part.solids()) == 1


def test_without_the_rail_the_word_is_seven_loose_letters():
    """What the rail is for, stated as the thing that happens without it."""
    from build123d import extrude

    p = params()
    letters = extrude(_word(p), amount=p.thickness)
    assert len(letters.solids()) == len(p.text)


def test_a_rail_that_misses_the_letters_is_refused():
    """Connectivity is checked, not assumed: lift the rail clear of the caps
    and the build must complain rather than emit eight loose pieces."""
    with pytest.raises(ValueError, match="regions instead of one"):
        build(params(rail_lift=400.0))


def test_letters_stand_proud_of_the_rail(part):
    p = params()
    assert part.bounding_box().size.Z == pytest.approx(p.thickness, abs=1e-3)
    assert p.rail < p.thickness


# --- where the rail sits ----------------------------------------------------
#
# The photographed emblem does not run its bar under the letters; the bar
# crosses them and the serif feet hang below it. That is the detail these cover.


def test_the_rail_crosses_the_letters_rather_than_underlining_them(classic):
    """Serifs below the bar, bar inside the cap height."""
    p = Params()
    word = _word(p)
    bar = _rail(p, word).bounding_box()
    cap = word.bounding_box().size.Y

    assert bar.min.Y > 0, "the rail is sitting on the baseline, not above it"
    assert bar.max.Y < cap, "the rail is taller than the letters it joins"
    # ...and the badge really does extend below the bar: those are the serifs.
    assert classic.bounding_box().min.Y < bar.min.Y


def test_the_rail_clears_the_serifs():
    """It is placed off a measurement of the glyphs, not off a fraction that
    happens to suit one font."""
    p = Params()
    word = _word(p)
    top = serif_top(word)
    assert _rail(p, word).bounding_box().min.Y == pytest.approx(top, abs=1e-6)


def test_serif_top_is_not_fooled_by_the_baseline_slice():
    """Regression. The undersides of the serifs are slightly rounded, so the
    slice sitting exactly on the baseline is the NARROWEST on the whole letter.
    Reading upward from there stops immediately and puts the rail back on the
    baseline -- which is the bug this whole change exists to fix."""
    word = _word(Params())
    cap = word.bounding_box().size.Y
    assert serif_top(word) > cap * 0.05


def test_the_rail_scales_with_the_badge():
    """Rail height is a fraction of cap height, so a small badge does not get a
    bar sized for a big one."""
    big, small = Params(), Params(length=90.0)
    ratio = (
        _rail(small, _word(small)).bounding_box().size.Y
        / _rail(big, _word(big)).bounding_box().size.Y
    )
    assert ratio == pytest.approx(90.0 / 150.0, rel=0.02)


# --- the bevel --------------------------------------------------------------


def test_the_letter_face_is_the_untouched_typeface_outline(classic):
    """The reason the bevel flares downward instead of tapering upward. The top
    face is the glyph outline itself, so no stroke is thinned and nothing thin
    can be consumed however big the bevel gets."""
    p = Params()
    top = classic.faces().filter_by_position(Axis.Z, p.thickness - 0.01,
                                             p.thickness + 0.01)
    assert sum(f.area for f in top) == pytest.approx(_word(p).area, rel=0.02)


def test_the_base_is_wider_than_the_face(classic, classic_info):
    p = Params()
    top = sum(
        f.area
        for f in classic.faces().filter_by_position(
            Axis.Z, p.thickness - 0.01, p.thickness + 0.01
        )
    )
    assert classic_info["bond_area_mm2"] > top


def _slab_area(part, z: float, thickness: float = 0.08) -> float:
    """Cross-section area at height ``z``, measured as the volume of a thin
    slice divided by its thickness. Cheaper and far less fragile than asking
    for a planar section of a shape with this many faces."""
    from build123d import Box

    box = part.bounding_box()
    slab = Pos(box.center().X, box.center().Y, z) * Box(
        box.size.X + 10, box.size.Y + 10, thickness
    )
    hit = part & slab
    return hit.volume / thickness if hit.volume else 0.0


def test_nothing_overhangs(classic):
    """Every layer sits on one at least as big, which is what lets this print
    face up with support switched off. Sampled off the finished solid rather
    than argued from the construction, because the construction is exactly what
    might be wrong."""
    box = classic.bounding_box()
    heights = [box.min.Z + box.size.Z * i / 24.0 for i in range(1, 24)]
    areas = [_slab_area(classic, z) for z in heights]
    for z, lower, upper in zip(heights[1:], areas, areas[1:]):
        assert upper <= lower + 1e-3, (
            f"the section grows from {lower:.1f} to {upper:.1f} mm^2 going up "
            f"through z={z:.2f} -- that is an overhang"
        )


def test_a_bevel_over_the_measured_ceiling_is_refused():
    """The limit is where outward offsets on glyphs start failing, so raising
    the number just moves the crash from validate() into build()."""
    with pytest.raises(ValueError, match="outward offsets"):
        build(params(bevel=MAX_BEVEL + 0.2))


def test_no_bevel_gives_slab_sides():
    """The bevel is optional. Compared above the bar rather than at the bond
    face, because the bond face is letters *and* bar however the sides are cut."""
    p = params(bevel=0.0)
    plain = build(p)
    assert len(plain.solids()) == 1
    assert _slab_area(plain, p.rail + 0.2) == pytest.approx(
        _slab_area(plain, p.thickness - 0.2), rel=1e-3
    )


def test_the_bevel_is_what_makes_the_sides_slope(classic):
    """The same comparison on the real part must fail, or the test above is
    passing for the wrong reason."""
    p = Params()
    assert _slab_area(classic, p.rail + 0.2) > _slab_area(classic, p.thickness - 0.1)


def test_the_bar_is_not_bevelled_on_purpose(classic):
    """A bevelled bar means its outline moves with height, and every height
    then needs its own fuse of grown glyphs to a grown rectangle -- the fuse
    that silently returns a partial result. Pinned as a decision rather than
    left to look like an oversight: the section is constant through the bar."""
    p = Params()
    assert _slab_area(classic, 0.2) == pytest.approx(
        _slab_area(classic, p.rail - 0.2), rel=1e-3
    )


def test_a_dropped_slab_would_be_caught(classic):
    """The failure that started all this was a badge missing its bottom
    millimetre, which every other number in report() described as healthy. The
    span is checked directly."""
    box = classic.bounding_box()
    assert box.min.Z == pytest.approx(0.0, abs=1e-6)
    assert box.max.Z == pytest.approx(Params().thickness, abs=1e-6)


def test_the_bevel_does_not_change_the_length(classic_info):
    """The flare adds material outside the outline, so the word is solved to
    length-minus-bevel. Otherwise every badge prints wide."""
    assert classic_info["length"] == pytest.approx(Params().length, abs=0.05)


# --- the font is the font you asked for -------------------------------------


def test_a_missing_font_is_refused_not_substituted():
    """OCCT prints a warning and silently uses Arial. A badge in the wrong
    letterforms is the one defect that survives every other check here."""
    with pytest.raises(ValueError, match="not installed"):
        check_font(Params(font="No Such Font At All"))


def test_an_installed_font_passes():
    check_font(Params(font=FONT))


def test_an_explicit_font_path_is_trusted():
    """A file either opens or it does not, so there is nothing to second-guess."""
    check_font(Params(font="No Such Font At All", font_path="/some/where.ttf"))


# --- size on the car --------------------------------------------------------


def test_it_comes_out_the_length_you_asked_for(info):
    assert info["length"] == pytest.approx(params().length, abs=0.05)


def test_length_is_solved_for_not_assumed():
    """Font size is derived by measuring, so a different target still lands."""
    got = report(params(length=90.0))
    assert got["length"] == pytest.approx(90.0, abs=0.05)


def test_stretch_widens_without_changing_cap_height():
    plain, wide = report(params()), report(params(stretch=1.4))
    assert wide["length"] == pytest.approx(plain["length"], abs=0.05)
    assert wide["cap_height"] < plain["cap_height"]


def test_it_fits_the_p2s_bed():
    box = plated(params()).bounding_box()
    assert profiles.machine(0.4).fits((box.size.X, box.size.Y, box.size.Z))


def test_a_condensed_sans_is_flagged_as_the_wrong_shape(info):
    """Why the default is not a sans. Set a condensed text face to badge width
    and it comes out far too tall -- the single most obvious way a printed
    badge reads as homemade."""
    assert info["aspect"] < CLASSIC_ASPECT[0]
    assert "too tall" in proportion_note(info)


def test_the_default_font_has_badge_proportions(classic_info):
    """Superclarendon at 150mm lands near 20mm caps, which is what the classic
    emblem measures. Corroborates the identification rather than assuming it."""
    low, high = CLASSIC_ASPECT
    assert low <= classic_info["aspect"] <= high
    assert proportion_note(classic_info) is None
    assert 18.0 <= classic_info["cap_height"] <= 22.0


def test_the_default_font_prints_as_one_badge(classic):
    """Serifs widen the letters, so the gaps shrink and the rail has less to
    do -- but it still has to end up a single object."""
    assert len(classic.solids()) == 1


def test_the_serifs_are_thick_enough_to_print(classic_info):
    """Serif faces put thin features in a part for the first time here. Erode
    the word by 0.6mm a side: every glyph must survive, which means nothing is
    under 1.2mm, or three extrusions on a 0.4mm nozzle."""
    from build123d import Text, offset

    p = Params()
    word = Text(p.text, REFERENCE_SIZE * p.length /
                Text(p.text, REFERENCE_SIZE, font=p.font).bounding_box().size.X,
                font=p.font)
    assert len(offset(word, amount=-0.6).faces()) == len(word.faces())
    assert classic_info["cap_height"] > 0


def test_badge_proportions_pass_the_note_silently():
    assert proportion_note({"aspect": sum(CLASSIC_ASPECT) / 2}) is None


# --- the face that gets taped ----------------------------------------------


def test_the_bond_face_is_flat_and_on_the_plate():
    """Printed back-down, so the tape face is a first layer: the flattest,
    most continuous surface the machine can make."""
    laid = plated(params())
    assert laid.bounding_box().min.Z == pytest.approx(0.0, abs=1e-6)
    assert report(params(), laid)["bond_area_mm2"] > 1000


def test_the_bond_carries_the_badge_many_times_over(info):
    """Shear on a tailgate, not tension under a desk. Even at a deliberately
    pessimistic 0.02 N/mm^2 the margin is enormous, which is why this part is
    about tape selection and surface prep rather than bond area."""
    weight = info["mass_g"] / 1000.0 * 9.80665
    assert weight / info["bond_area_mm2"] < 0.02 / 50


# --- the curved-back escape hatch ------------------------------------------


def test_a_flat_back_is_the_default():
    assert params().sag == 0.0
    assert params().radius is None


def test_sag_hollows_the_back_and_implies_a_radius():
    p = params(sag=1.0)
    assert p.radius == pytest.approx(150.0**2 / 8 + 0.5, rel=1e-6)
    flat, curved = build(params()).volume, build(p).volume
    assert curved < flat  # material removed, not added


def test_a_hollow_that_would_sever_the_rail_is_refused():
    with pytest.raises(ValueError, match="rail thickness"):
        build(params(sag=2.0))


# --- things that should not print ------------------------------------------


def test_a_rail_thicker_than_the_letters_is_refused():
    with pytest.raises(ValueError, match="thinner than the letters"):
        build(params(rail=5.0))


def test_a_wall_under_the_nozzle_minimum_is_refused():
    with pytest.raises(ValueError, match=str(MIN_WALL)):
        build(params(rail=0.5))


def test_an_empty_word_is_refused():
    with pytest.raises(ValueError, match="no badge without a word"):
        build(params(text="   "))


# --- the slice settings the badge ships with -------------------------------


def test_slice_overrides_are_strings():
    from tools.mustang_badge import SLICE

    assert all(isinstance(v, str) for v in SLICE.values())


def test_support_is_off_and_ironing_is_on():
    """Nothing overhangs in this orientation, and the letter faces are top
    surfaces, which is the one case ironing genuinely earns its time."""
    from tools.mustang_badge import SLICE

    assert SLICE["enable_support"] == "0"
    assert SLICE["ironing_type"] == "top"


def test_a_brim_is_specified():
    """ASA, and the first layer is letter-shaped rather than a solid block."""
    from tools.mustang_badge import SLICE

    assert float(SLICE["brim_width"]) >= 5
