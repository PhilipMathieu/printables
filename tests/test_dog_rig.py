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


def test_the_expected_break_is_a_weight_a_person_can_hang(params):
    """The point of a 100mm arm: the whole plausible range of answers lands
    between a bottle of water and a bucket of it."""
    shank_z = math.pi * params.shank**3 / 32
    for mpa, expect in ((20, 3.2), (35, 5.5)):
        kg = shank_z * mpa / 1000 / (dog_rig.ARM / 1000) / 9.81
        assert kg == pytest.approx(expect, abs=0.1)


def test_the_clamp_can_load_a_shank_to_within_a_factor_of_a_few(params):
    """Which is why this is worth breaking one to find out, rather than
    asserting it is fine. At 200N -- a mild hand on a knob -- the clamp is
    already inside a factor of three of the weakest plausible root."""
    shank_z = math.pi * params.shank**3 / 32
    weakest = shank_z * 20 / 1000  # N.m at 20 MPa of layer adhesion
    applied = 200 * params.screw_height / 1000
    assert 1 < weakest / applied < 3


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
