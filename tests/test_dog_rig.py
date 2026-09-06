"""Assertions for the test rig and the wooden-deck template.

These parts exist to answer questions the arithmetic cannot, so most of what
matters about them is not geometric. Three things are.

That the template's pilots land on exactly the grid the printed deck has --
a template off by a millimetre produces a wooden deck that every dog in the set
is wrong for, and it would not be obvious until the fence would not sit down.
That the break arm fails at the shank and not at the lever, or it measures the
lever. And that the warp strip is honestly the same print as the deck in the
dimension being tested, or it is not a proxy for anything.
"""

from __future__ import annotations

import math

import pytest
from build123d import Cylinder, Pos

from parts import bench_dogs, dog_deck, dog_rig
from tests.test_bench_dogs import overhangs


@pytest.fixture(scope="module")
def params():
    return bench_dogs.Params()


@pytest.fixture(scope="module")
def deck():
    return dog_deck.Params()


@pytest.fixture(scope="module")
def template(deck):
    return dog_deck.template(deck)


# --- the template ----------------------------------------------------------


def test_the_template_pilots_the_decks_own_grid(deck, template):
    """The whole reason it exists. A wooden deck bored through this has to come
    out with the grid the dogs were cut for, not a grid of its own."""
    t = 2.5
    for x, y in deck.stations:
        at_station = template & (Pos(x, y, t / 2) * Cylinder(1.0, t * 3))
        assert at_station.volume == pytest.approx(0, abs=1e-6), f"solid at {x},{y}"


def test_the_template_is_solid_between_its_pilots(deck, template):
    """Ribs, not a sheet -- but the ribs have to be under every station or the
    pilot has nothing to be drilled through."""
    t = 2.5
    for x, y in deck.stations:
        beside = template & (Pos(x + 4.0, y, t / 2) * Cylinder(1.0, t * 3))
        assert beside.volume > 0, f"no material beside {x},{y}"


def test_the_template_has_one_pilot_per_station_and_is_a_lattice(deck, template):
    """Counted off the solid: its underside is one face whose inner boundaries
    are a pilot at every station plus a window in every square of the lattice.
    The windows are what makes it a fifth of the deck to print, so counting them
    is also how we know it did not quietly come out as a sheet."""
    bottom = [f for f in template.faces() if abs(f.center().Z) < 1e-6]
    assert len(bottom) == 1
    pilots = deck.cols * deck.rows
    windows = (deck.cols - 1) * (deck.rows - 1)
    assert len(bottom[0].wires()) == 1 + pilots + windows


def test_the_template_is_worth_printing_instead_of_the_deck(deck, template):
    """It is only an answer to "why not wood" if it is much cheaper than the
    plate it saves. A solid template of this footprint would be half the deck."""
    assert template.volume < dog_deck.build(deck).volume / 4


def test_the_template_aligns_off_the_decks_own_edges(deck, template):
    """Its outer ribs run through the outermost stations, so setting it back
    ``margin`` from each edge of a board cut to the deck's size squares it up."""
    bb = template.bounding_box()
    assert bb.size.X == pytest.approx(deck.width - 2 * deck.margin + 14.0)
    assert bb.size.Y == pytest.approx(deck.depth - 2 * deck.margin + 14.0)


# --- the break arm ---------------------------------------------------------


def test_the_arm_puts_the_eye_a_known_distance_from_the_shank(params):
    """The moment is the hung mass times this, so if it is not what it says the
    number that comes out means nothing."""
    body = dog_rig.arm(params)
    t = 2.5
    at_eye = body & (Pos(dog_rig.ARM, 0, params.rise / 2) * Cylinder(1.0, params.rise))
    assert at_eye.volume == pytest.approx(0, abs=1e-6)
    assert body.bounding_box().max.X == pytest.approx(dog_rig.ARM + 14.0)


def test_the_arm_is_stronger_than_the_shank_it_is_there_to_break(params):
    """Otherwise the test measures the lever. Twice the section modulus, and
    loaded along its layers where the shank is loaded across them, so the margin
    is larger than the ratio makes it look."""
    width, height = 14.0, params.rise
    arm_z = width * height**2 / 6
    shank_z = math.pi * params.shank**3 / 32
    assert arm_z > 1.5 * shank_z


def _kg_at_the_eye(moment):
    """Mass hung on the arm's eye that puts a given moment into the root."""
    return moment / (9.81 * dog_rig.ARM / 1000)


def test_the_arm_reads_out_in_newton_metres_without_arithmetic(params):
    """The reason ARM is 100 and not 120. At the bench the answer wants to be
    readable off the scale, and at a tenth of a metre a kilogram is 0.981 N.m --
    so the number you hang is the number at the root, to within two percent, and
    nobody has to convert anything with a broken part in their hand."""
    assert _kg_at_the_eye(1.0) == pytest.approx(1.0, abs=0.02)


def test_the_arm_has_a_number_to_beat_and_a_number_to_clear(params):
    """What the test is actually for, which is not "how strong is it" but "can
    the clamp break its own shank". Two thresholds fall out of ``KNOB``: a light
    hand needs 2.7kg or the clamp is unusable at any setting, and a firm hand
    needs 5.4kg or the knob has to be small enough that a firm hand is not
    available. Both sit inside the 3.2 to 5.5kg the arithmetic predicts, which
    is the whole reason this gets printed rather than calculated."""
    light, firm = bench_dogs.KNOB
    unusable = _kg_at_the_eye(light * params.screw_height / 1000)
    comfortable = _kg_at_the_eye(firm * params.screw_height / 1000)
    assert unusable == pytest.approx(2.7, abs=0.1)
    assert comfortable == pytest.approx(5.4, abs=0.1)

    shank_z = math.pi * params.shank**3 / 32
    predicted = [_kg_at_the_eye(shank_z * mpa / 1000) for mpa in (20, 35)]
    assert predicted[0] < comfortable < predicted[1], (
        "the firm-hand threshold has left the predicted range, so the arm no "
        "longer decides anything -- re-read what changed"
    )
    assert unusable < predicted[0], "a light hand now breaks even the best case"


def test_the_expected_break_is_a_weight_a_person_can_hang(params):
    """The point of a 100mm arm: the whole plausible range of answers lands
    between a bottle of water and a bucket of it."""
    shank_z = math.pi * params.shank**3 / 32
    for mpa, expect in ((20, 3.2), (35, 5.5)):
        kg = shank_z * mpa / 1000 / (dog_rig.ARM / 1000) / 9.81
        assert kg == pytest.approx(expect, abs=0.1)


def _moment(params, force):
    """N.m at a shank root from a screw force, the bolt being that far up."""
    return force * params.screw_height / 1000


def _root_holds(params, mpa):
    """N.m a shank root holds at a given layer adhesion."""
    return math.pi * params.shank**3 / 32 * mpa / 1000


def test_a_light_hand_on_the_knob_is_inside_the_weakest_plausible_root(params):
    """The half of the range that is fine, and it is the half the clamp is
    actually used in: 250N is a knob turned to snug, and the root holds it even
    if the layers came out as badly as they plausibly can."""
    light, _ = bench_dogs.KNOB
    assert _moment(params, light) < _root_holds(params, 20)


def test_a_firm_hand_on_the_knob_is_not(params):
    """The correction that made the arm the first thing to print. This was
    written as "200N, a factor of one and a half clear", from a guessed force.
    The preload an M6 actually delivers under a firm hand is 500N, and that
    lands between the two ends of the root's plausible strength -- so whether
    the clamp can break its own shank is not something the arithmetic answers,
    which is the whole argument for breaking one."""
    _, firm = bench_dogs.KNOB
    assert _root_holds(params, 20) < _moment(params, firm) < _root_holds(params, 35)


def test_the_knob_forces_are_the_preload_formula_and_not_a_guess(params):
    """F = T / (K d), nut factor 0.2, on an M6 -- a knob turned lightly at
    0.3 N.m and firmly at 0.6. Asserted because the number they replaced was a
    guess, and a guess is what this is here to stop happening twice."""
    light, firm = bench_dogs.KNOB
    for torque, expect in ((0.3, light), (0.6, firm)):
        assert torque / (0.2 * 0.006) == pytest.approx(expect, rel=0.01)


WALL = 0.42
"""Wall extrusion width for a 0.4mm nozzle, in mm. Every conclusion below is a
ratio, so being a few hundredths out changes none of them."""


def _shelled_z(diameter, walls, width=WALL):
    """Section modulus of a shank whose core is sparse infill, in mm^3.

    Counting the shell only, which is the conservative reading and close to the
    true one: sparse infill in a small core is neither continuous nor well bonded
    across layers, which is the axis this root fails on.
    """
    core = max(diameter - 2 * walls * width, 0.0)
    return math.pi * (diameter**4 - core**4) / (32 * diameter)


def test_four_walls_take_a_quarter_off_the_root(params):
    """The figure quoted everywhere -- 3.1 to 5.4 N.m -- is for a solid root, and
    a shank is only 11.65mm across, so the shell is most of it. At four walls the
    range is really 2.3 to 4.1, which a firm hand on the knob clears at both
    ends. This is the setting the break number is most sensitive to."""
    solid = math.pi * params.shank**3 / 32
    assert _shelled_z(params.shank, 4) / solid == pytest.approx(0.74, abs=0.02)


def test_eight_walls_make_the_shank_effectively_solid(params):
    """And it costs nothing worth counting on parts this size, which is why the
    set is printed at eight rather than at a number chosen for print time."""
    solid = math.pi * params.shank**3 / 32
    assert _shelled_z(params.shank, 8) / solid > 0.95


def test_the_infill_pattern_is_not_what_decides_the_root(params):
    """Asked directly, and worth an assertion because the intuition is wrong. At
    eight walls the core is inside a third of the diameter, which is close to the
    neutral axis, so the whole of it -- pattern, density, all of it -- is worth a
    few percent of the root. Wall count is worth twenty-three points over the
    same span. Choose the pattern for the deck's warp instead; it is the part
    that has a stake in it."""
    solid = math.pi * params.shank**3 / 32
    core_share = (solid - _shelled_z(params.shank, 8)) / solid
    wall_share = (_shelled_z(params.shank, 8) - _shelled_z(params.shank, 4)) / solid
    assert core_share < 0.05
    assert wall_share > 4 * core_share


def test_a_smaller_screw_would_make_the_load_worse_not_better(params):
    """The instinct that a smaller fastener bounds what a hand can do, which is
    backwards: preload goes as 1/d at a given torque, so the same knob on an M5
    delivers more force than on an M6. What bounds it is the knob."""
    _, firm = bench_dogs.KNOB
    on_m5 = 0.6 / (0.2 * 0.005)
    assert on_m5 > firm


# --- the warp strip --------------------------------------------------------


def test_the_strip_is_the_decks_own_length_and_thickness(deck):
    """The two dimensions warp actually depends on. Anything else about it can
    differ; those cannot, or it is not a proxy."""
    body = dog_rig.strip(deck)
    bb = body.bounding_box()
    assert bb.size.X == pytest.approx(deck.width)
    assert bb.size.Z == pytest.approx(deck.deck)


def test_the_strip_is_a_fraction_of_the_deck_to_print(deck):
    body = dog_rig.strip(deck)
    assert body.volume < dog_deck.build(deck).volume / 3


def test_the_strip_carries_a_real_row_of_holes(deck):
    """Same holes, because a plate's holes are where it is thinnest and warp
    finds thin."""
    body = dog_rig.strip(deck)
    bottom = [f for f in body.faces() if abs(f.center().Z) < 1e-6]
    assert sum(len(f.wires()) for f in bottom) == deck.cols + len(bottom)


# --- the puck ladder -------------------------------------------------------


def test_the_puck_ladder_brackets_the_grip_that_was_guessed(params):
    """The default has to be inside the range being tried, or the ladder cannot
    tell you it was wrong in the direction it was wrong in."""
    assert min(dog_rig.PUCK_GRIPS) < params.puck_grip < max(dog_rig.PUCK_GRIPS)
    assert list(dog_rig.PUCK_GRIPS) == sorted(dog_rig.PUCK_GRIPS)


def test_every_rung_of_the_puck_ladder_grips_the_hole(params):
    rungs = dog_rig.puck_ladder(params)
    assert len(rungs) == len(dog_rig.PUCK_GRIPS)
    for rung, grip in zip(rungs, dog_rig.PUCK_GRIPS):
        assert rung.bounding_box().size.X == pytest.approx(
            params.grid.hole + grip, abs=0.01
        )


# --- all of it has to print ------------------------------------------------


def test_the_rig_prints_the_same_way_up_as_the_set(params, deck, template):
    """A rig part that needs support is a rig part whose answer is about the
    support."""
    for name, part in (
        ("arm", dog_rig.arm(params)),
        ("strip", dog_rig.strip(deck)),
        ("template", template),
        ("puck", dog_rig.puck_ladder(params)[-1]),
    ):
        # The backer is the one marked part in the rig, so its engraving's
        # ceiling is exempted by height -- see ``overhangs``.
        bad = overhangs(part, bridged_at=params.mark_depth if name == "puck" else None)
        assert len(bad) == 0, f"{name} has {len(bad)} facets over air"
        assert part.is_valid, name
        assert len(part.solids()) == 1, name
