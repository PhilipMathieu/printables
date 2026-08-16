"""What has to be true of the badge.

Two things can go wrong here that both produce a file that looks fine. The word
can come out as seven loose letters, and the font can come out as Arial because
the renderer substituted it without saying so. Most of these tests are about
those two.
"""

from __future__ import annotations

import pytest
from build123d import Axis

from p2s import profiles
from parts.mustang_badge import (
    CLASSIC_ASPECT,
    MIN_WALL,
    REFERENCE_SIZE,
    Params,
    build,
    check_font,
    plated,
    proportion_note,
    report,
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


# --- the word is one object -------------------------------------------------


def test_the_badge_is_a_single_solid(part):
    """Seven letters you have to align on the car by eye is not a badge."""
    assert part.is_valid
    assert len(part.solids()) == 1


def test_the_rail_is_what_joins_them():
    """Take the rail away and it should fall apart -- proof the rail is load
    bearing for connectivity and not decoration."""
    with pytest.raises(ValueError, match="separate pieces|rail_bite"):
        build(params(rail_drop=0.0, rail_bite=0.0))


def test_letters_stand_proud_of_the_rail(part):
    p = params()
    assert part.bounding_box().size.Z == pytest.approx(p.thickness, abs=1e-3)
    assert p.rail < p.thickness


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


def test_the_default_font_has_badge_proportions():
    """Superclarendon at 150mm lands near 20mm caps, which is what the classic
    emblem measures. Corroborates the identification rather than assuming it."""
    got = report(Params())
    low, high = CLASSIC_ASPECT
    assert low <= got["aspect"] <= high
    assert proportion_note(got) is None
    assert 18.0 <= got["cap_height"] <= 22.0


def test_the_default_font_prints_as_one_badge():
    """Serifs widen the letters, so the gaps shrink and the rail has less to
    do -- but it still has to end up a single object."""
    assert len(build(Params()).solids()) == 1


def test_the_serifs_are_thick_enough_to_print():
    """Serif faces put thin features in a part for the first time here. Erode
    the word by 0.6mm a side: every glyph must survive, which means nothing is
    under 1.2mm, or three extrusions on a 0.4mm nozzle."""
    from build123d import Text, offset

    p = Params()
    caps = report(p)
    word = Text(p.text, REFERENCE_SIZE * p.length /
                Text(p.text, REFERENCE_SIZE, font=p.font).bounding_box().size.X,
                font=p.font)
    assert len(offset(word, amount=-0.6).faces()) == len(word.faces())
    assert caps["cap_height"] > 0


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
