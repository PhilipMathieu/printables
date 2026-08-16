"""Geometric assertions for the coin utility.

The two things worth asserting are the two that are invisible until the print
is in hand: that nothing a motif draws can overhang the rim, and that the
reverse comes out the right way round.
"""

from __future__ import annotations

import dataclasses
from itertools import combinations

import pytest
from OCP.BRepExtrema import BRepExtrema_DistShapeShape

from geom.motif import (
    BOTTOM,
    Fitted,
    Svg,
    TOP,
    Legend,
    MotifError,
    Outline,
    Stroke,
    Text,
    combine,
    reach,
)
from p2s import profiles
from parts.coin import Params, blank, build
from parts.tremblant_coin import PARAMS as TREMBLANT
from parts.tremblant_coin import SQUIGGLE

FIELD = 30.0

MARKS = {
    "text": Text(text="26"),
    "small text": Text(text="MT", fill=0.4),
    "stroke": Stroke(points=((0, 0), (5, 8), (10, 0), (15, 8), (20, 0)), width=1.5),
    "closed stroke": Stroke(points=((0, 0), (10, 0), (10, 10), (0, 10)), width=1.2,
                            closed=True, smooth=False),
    "outline": Outline(points=((0, 0), (10, 0), (5, 9))),
    "squiggle": SQUIGGLE,
    "legend": Legend(text="MONT-TREMBLANT"),
    "exergue": Legend(text="2026", at=BOTTOM),
}
FITTED = {k: v for k, v in MARKS.items() if isinstance(v, Fitted)}


@pytest.mark.parametrize("mark", MARKS.values(), ids=MARKS.keys())
def test_every_mark_stays_inside_its_field(mark):
    """A mark that reaches past the field lands on the rim, or off the coin."""
    assert reach(mark.sketch(FIELD)) <= FIELD / 2 + 1e-6


@pytest.mark.parametrize("mark", FITTED.values(), ids=FITTED.keys())
def test_fitted_marks_are_centred_and_scaled_to_fill(mark):
    """Fitting is what lets a motif be drawn on any convenient grid."""
    fitted = dataclasses.replace(mark, shift=(0.0, 0.0)).sketch(FIELD)
    bbox = fitted.bounding_box()
    assert bbox.center().X == pytest.approx(0, abs=1e-6)
    assert bbox.center().Y == pytest.approx(0, abs=1e-6)
    assert reach(fitted) == pytest.approx(FIELD / 2 * mark.fill)


@pytest.mark.parametrize("mark", FITTED.values(), ids=FITTED.keys())
def test_a_shift_moves_a_mark_without_resizing_it(mark):
    """The two knobs stay independent: a nudge for balance must not quietly
    cost the design the size that ``fill`` was asked for."""
    still = dataclasses.replace(mark, fill=0.5, shift=(0.0, 0.0)).sketch(FIELD)
    moved = dataclasses.replace(mark, fill=0.5, shift=(0.0, 0.2)).sketch(FIELD)
    assert moved.area == pytest.approx(still.area, rel=1e-9)
    assert moved.bounding_box().center().Y == pytest.approx(
        still.bounding_box().center().Y + 0.2 * FIELD / 2, abs=1e-6
    )


def test_a_shift_that_climbs_the_rim_is_rejected():
    """Silently shrinking the mark to make room would make ``fill`` a lie."""
    with pytest.raises(MotifError, match="overruns the field"):
        Text(text="26", fill=0.9, shift=(0.0, 0.5)).sketch(FIELD)


def test_a_closed_stroke_keeps_its_hole():
    """It is a band, not a filled shape -- the difference is a whole design."""
    band = Stroke(points=((0, 0), (10, 0), (10, 10), (0, 10)), width=1.2,
                  closed=True, smooth=False).sketch(FIELD)
    assert len(band.faces()) == 1
    assert len(band.faces()[0].inner_wires()) == 1


def test_legends_sit_at_opposite_ends_of_the_field():
    """Which is the whole point of arching one over and one under."""
    assert Legend(text="ABC", at=TOP).sketch(FIELD).center().Y > 0
    assert Legend(text="ABC", at=BOTTOM).sketch(FIELD).center().Y < 0


def test_a_legend_that_would_lap_the_coin_is_rejected():
    """Bent round far enough, a legend runs into its own first letter."""
    with pytest.raises(MotifError, match="meets its own start"):
        Legend(text="A VERY LONG INSCRIPTION INDEED", height=0.4).sketch(FIELD)
    with pytest.raises(MotifError, match="meets its own start"):
        Legend(text="MONT-TREMBLANT QUEBEC CANADA 2026 SEPTEMBRE").sketch(FIELD)


DISC_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <path d="M 50 5 A 45 45 0 1 0 50 95 A 45 45 0 1 0 50 5 Z" fill="black"/>
  <path d="M 50 25 A 25 25 0 1 1 50 75 A 25 25 0 1 1 50 25 Z" fill="white"/>
</svg>"""


def test_a_round_mark_fills_the_field_instead_of_its_bounding_box(tmp_path):
    """The reason fitting measures reach: a disc fitted by the diagonal of its
    box comes out at 71% of the diameter the field would have allowed."""
    svg = tmp_path / "ring.svg"
    svg.write_text(DISC_SVG)
    ring = Svg(path=svg, fill=1.0).sketch(FIELD)
    assert ring.bounding_box().size.X == pytest.approx(FIELD, rel=1e-3)
    assert len(ring.faces()[0].inner_wires()) == 1, "the hole survived the import"


def test_an_empty_mark_is_rejected():
    with pytest.raises(MotifError):
        Text(text="  ").sketch(FIELD)


def test_a_face_can_carry_several_marks():
    face = combine((Text(text="26", fill=0.5), Legend(text="2026", at=BOTTOM)), FIELD)
    assert face.area > Text(text="26", fill=0.5).sketch(FIELD).area


# --- the blank -------------------------------------------------------------


@pytest.fixture(scope="module")
def plain():
    return Params(diameter=30.0, thickness=3.0)


def test_the_blank_is_one_sound_solid(plain):
    part = blank(plain)
    assert part.is_valid
    assert len(part.solids()) == 1


def test_the_blank_is_the_size_it_says_it_is(plain):
    size = blank(plain).bounding_box().size
    assert (size.X, size.Y, size.Z) == pytest.approx(
        (plain.diameter, plain.diameter, plain.thickness), abs=1e-6
    )


def test_the_rim_is_the_highest_thing_on_the_coin():
    """What a rim is for: it takes the wear so the relief does not."""
    params = Params(diameter=30.0, obverse=Text(text="26"))
    assert build(params).bounding_box().max.Z == pytest.approx(params.thickness)


def test_reeds_cut_the_edge_without_breaking_it():
    params = Params(diameter=30.0, reeds=60, reed_depth=0.4)
    reeded = blank(params)
    assert reeded.is_valid
    assert len(reeded.solids()) == 1
    assert reeded.volume < blank(Params(diameter=30.0)).volume


# --- striking --------------------------------------------------------------


def test_relief_adds_material_and_engraving_removes_it():
    params = Params(diameter=30.0, obverse=Text(text="8"), reverse=Text(text="8"))
    struck, plain_blank = build(params), blank(params)
    added = params.relief * combine(params.obverse, params.field).area
    cut = params.engrave * combine(params.reverse, params.field).area
    assert struck.volume == pytest.approx(plain_blank.volume + added - cut, rel=1e-3)


def test_the_reverse_is_mirrored_so_it_reads_from_below():
    """Cut as drawn, a date on the underside comes out backwards.

    A right triangle carries its area to one side of its bounding box, so the
    centre of the pocket it cuts has to land on the opposite side of the coin
    from the centre of the mark that cut it.
    """
    lopsided = Outline(points=((0, 0), (10, 0), (10, 10)))
    params = Params(diameter=30.0, reverse=lopsided)
    assert lopsided.sketch(params.field).center().X > 0.5
    pocket = blank(params) - build(params)
    assert pocket.center().X < -0.5


def test_the_engraved_pocket_stops_short_of_the_field():
    params = Params(diameter=30.0, reverse=Text(text="8"))
    pocket = blank(params) - build(params)
    assert pocket.bounding_box().max.Z == pytest.approx(params.engrave)
    assert params.engrave < params.field_z


# --- the parameter guards --------------------------------------------------


def test_a_rim_too_narrow_for_its_recess_is_rejected():
    with pytest.raises(ValueError, match="not wide enough"):
        blank(Params(rim=1.0, field_depth=0.7, chamfer=0.6))


def test_relief_standing_proud_of_the_rim_is_rejected():
    with pytest.raises(ValueError, match="proud of the rim"):
        blank(Params(relief=1.0, field_depth=0.7))


def test_engraving_through_the_coin_is_rejected():
    with pytest.raises(ValueError, match="cut through"):
        blank(Params(thickness=1.0, field_depth=0.7, engrave=0.5))


def test_reeds_that_would_merge_are_rejected():
    with pytest.raises(ValueError, match="merge"):
        blank(Params(diameter=30.0, reeds=400, reed_depth=0.4))


# --- the shipped coin ------------------------------------------------------


@pytest.fixture(scope="module")
def tremblant():
    return build(TREMBLANT)


def test_the_tremblant_coin_is_one_sound_solid(tremblant):
    assert tremblant.is_valid
    assert len(tremblant.solids()) == 1


def test_the_tremblant_coin_fits_the_p2s_bed(tremblant):
    size = tremblant.bounding_box().size
    assert profiles.machine(0.4).fits((size.X, size.Y, size.Z))


def test_scaling_a_coin_scales_all_of_it():
    """A test print at 80% has to be the same coin, not a differently
    proportioned one -- which it is only because motifs are specified in
    fractions of the field and never in millimetres."""
    small = build(TREMBLANT.scaled(0.8))
    big = build(TREMBLANT)
    assert small.bounding_box().size.X == pytest.approx(
        big.bounding_box().size.X * 0.8, rel=1e-6
    )
    assert small.volume == pytest.approx(big.volume * 0.8**3, rel=1e-3)


def test_a_finer_nozzle_is_what_makes_the_lettering_work():
    """The 38mm coin's legends are one-and-a-bit extrusions wide on a 0.4mm
    nozzle, which prints as a single bead with no wall either side. The same
    lettering at 80% on a 0.2mm nozzle is nearly two, which is the whole reason
    a smaller coin can carry finer text than a larger one.
    """
    legend = TREMBLANT.obverse[0]
    coarse = _stroke_width(legend.sketch(TREMBLANT.field)) / profiles.machine(0.4).line_width
    fine = _stroke_width(legend.sketch(TREMBLANT.scaled(0.8).field)) / (
        profiles.machine(0.2).line_width
    )
    assert coarse < 1.5, "the 0.4mm nozzle print was always going to be marginal"
    assert fine > 2.0, "two extrusions is where a stroke gets a wall either side"


def test_the_marks_on_each_face_clear_one_another():
    """Each face now carries a legend at one or both ends as well as a design,
    and a legend that grazes the design is the kind of thing that only shows up
    once the coin is off the plate.
    """
    land = 2 * profiles.machine(0.4).line_width
    for face in (TREMBLANT.obverse, TREMBLANT.reverse):
        marks = [mark.sketch(TREMBLANT.field) for mark in face]
        for a, b in combinations(marks, 2):
            assert (a & b).area == 0, "two marks on one face overlap"
            gap = BRepExtrema_DistShapeShape(a.wrapped, b.wrapped)
            gap.Perform()
            assert gap.Value() > land, "no land left between two marks"


def _stroke_width(sketch) -> float:
    """Average stroke width, as 2 * area / perimeter.

    Exact for a ribbon of constant width, and a letter is a ribbon bent about;
    a hole's edge counts because it is the far side of the stroke.
    """
    return 2 * sketch.area / sum(edge.length for edge in sketch.edges())


def test_every_inscription_is_thicker_than_one_extrusion():
    """Under one extrusion the slicer has nothing to draw the stroke with and
    drops it, so the inscription comes out in pieces.

    This is why the legends are set bold and large in a face of uniform weight.
    Set regular at the size that looks right on the page, every font on this
    machine measured 0.20-0.36mm through the stroke -- all of them too thin.
    """
    line = profiles.machine(0.4).line_width
    for face in (TREMBLANT.obverse, TREMBLANT.reverse):
        for mark in face:
            if isinstance(mark, (Legend, Text)):
                width = _stroke_width(mark.sketch(TREMBLANT.field))
                assert width > line, f"{mark.text!r} strokes {width:.2f}mm"


def test_the_squiggle_engraves_a_channel_wider_than_one_extrusion():
    """Two walls that meet leave no floor, and the engraving reads as a scratch.

    Measured as area over centreline length rather than by trusting the
    authored width, because the width is in the squiggle's own units and only
    fitting to the field decides what it becomes in mm.
    """
    assert _stroke_width(SQUIGGLE.sketch(TREMBLANT.field)) > (
        4 * profiles.machine(0.4).line_width
    )
