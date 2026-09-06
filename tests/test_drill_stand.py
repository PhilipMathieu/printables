"""The measured machine, and the deck checked against it.

Until these numbers existed the deck was checked against arithmetic alone, and
the arithmetic was fed by a photograph that turned out to be a fifth out. What
these assert is the join between the two: that the hardware the slots take is
the hardware the deck is drilled for, that the deck reaches back less far than
the column stands, and that the stations a workpiece is held at are over casting
rather than over air.
"""

from __future__ import annotations

import pytest

from geom import drill_stand
from geom.drill_stand import CRAFTSMAN_33525921 as STAND
from parts import bench_dogs, dog_deck, dog_rig

ACROSS_FLATS = {"1/4-20": 11.11, "5/16-18": 12.70, "M8": 13.00, "3/8-16": 14.29,
                "M5": 8.00}
"""Hex heads, across the flats, in mm."""


@pytest.fixture(scope="module")
def deck():
    return dog_deck.Params()


@pytest.fixture(scope="module")
def params():
    return bench_dogs.Params()


# --- the hardware the slots will take --------------------------------------


def test_the_slot_takes_a_five_sixteenths_head_and_will_not_give_it_back():
    """Both halves of a T-slot's job. A 1/2in head goes through the 13.4mm
    pocket with 0.7mm to spare and is then far too wide to pull back out
    through the 8.6mm slot."""
    assert STAND.head_fits(ACROSS_FLATS["5/16-18"])
    assert ACROSS_FLATS["5/16-18"] < STAND.pocket
    assert ACROSS_FLATS["5/16-18"] > STAND.slot


def test_a_head_too_big_for_the_pocket_is_refused():
    assert not STAND.head_fits(ACROSS_FLATS["3/8-16"])


def test_a_head_that_would_pull_through_the_slot_is_refused():
    """The failure that is not obvious: a head small enough to fit the pocket
    easily is a head that comes straight back out under load."""
    assert ACROSS_FLATS["M5"] < STAND.slot
    assert not STAND.head_fits(ACROSS_FLATS["M5"])


def test_metric_fits_too_but_with_less_room():
    assert STAND.head_fits(ACROSS_FLATS["M8"])
    assert STAND.pocket - ACROSS_FLATS["M8"] < STAND.pocket - ACROSS_FLATS["5/16-18"]


def test_there_is_room_under_the_casting_for_a_head(deck):
    """It has to go in from underneath and then be reachable. 22mm is a lot more
    than the 5mm a 5/16 head is tall."""
    assert STAND.under > 20


def test_an_inch_long_bolt_reaches(deck):
    """Through the casting, through the deck, into a counterbored jam nut."""
    needed = STAND.bolt(deck.deck, sunk=6.0, nut=4.4)
    assert needed < 25.4
    # And a full nut in the same counterbore would leave too little deck under
    # it, which is why the jam nut is specified rather than suggested.
    assert deck.deck - 6.0 == pytest.approx(4.0)


# --- the deck against the machine ------------------------------------------


def test_the_deck_reaches_back_less_far_than_the_column_stands(deck):
    """The measurement that was feared and turned out not to bind at all: 114mm
    of throat against a deck that reaches 67 back."""
    assert STAND.fits(deck.depth)
    assert STAND.throat - deck.depth / 2 > 40


def test_even_a_square_deck_would_clear_the_column(deck):
    """Which is why seven rows is on the table rather than ruled out."""
    square = dog_deck.Params(cols=deck.cols, rows=deck.cols)
    assert STAND.fits(square.depth)


def test_every_station_is_over_casting_and_not_over_air(deck):
    """A stop bears on the deck and the deck bears on the casting, so a station
    hanging past the flat field is a station cantilevered on ten millimetres of
    plastic. None of them is, but the outermost column is close, and that is
    what the 25mm pitch cost.

    Three pitches have been through here. At 25.4 the outer column landed on the
    field's edge to within two tenths of a millimetre. At 24 it had four
    millimetres. At 25 -- the optical-breadboard pitch, taken so that a future
    M6 deck and this set's fixtures are the same grid -- it has one. Inside, but
    the assertion below is the whole of the margin, so any change that moves a
    station outwards has to be a deliberate one."""
    reach = STAND.field / 2
    for x, y in deck.stations:
        assert abs(x) < reach, x
        assert abs(y) < reach, y


def test_the_outermost_holes_still_bear_mostly_on_casting(deck):
    """The centre being inside the field is not enough on its own: a hole has a
    width, and at this pitch the outer column's holes straddle the casting's
    edge. What has to hold is that most of the bearing footprint is still over
    iron and the rest is a stub of overhang no longer than the plate is thick --
    5mm out on 10mm of deck, an aspect ratio at which nothing a hand-fed drill
    stand applies will measurably deflect it."""
    reach = STAND.field / 2
    outer = max(abs(x) for x, _ in deck.stations)
    supported = reach - (outer - deck.grid.hole / 2)
    assert supported > deck.grid.hole / 2, "outer holes are more air than casting"
    overhang = (outer + deck.grid.hole / 2) - reach
    assert overhang < deck.deck


def test_the_deck_covers_the_field_it_bolts_to(deck):
    """Overhanging it in one direction is fine -- nothing bears out there -- but
    falling short of the slots would leave nothing to bolt through."""
    assert deck.width > STAND.field
    assert deck.depth > STAND.field / 2


def test_the_backer_can_be_pushed_out_from_underneath(deck):
    """Which needs the casting's own bore to be wider than a dog hole, under the
    one station a backer is ever used at."""
    assert STAND.bore > deck.grid.hole


def test_the_deck_costs_no_stroke(deck):
    """The carriage clamps at whatever height on the column you set it to, so
    the deck moves the whole stroke up rather than eating into it. What it
    bounds is the work, and 2.5 inches is deeper than anything this is for."""
    assert STAND.stroke > 60
    assert STAND.stroke > 39.5  # a 1590BB stood on end, the tallest likely case


# --- the catalogue ---------------------------------------------------------


def test_an_unknown_stand_says_what_there_is():
    with pytest.raises(KeyError, match="craftsman"):
        drill_stand.named("delta")


def test_the_rig_is_worth_its_time_against_the_deck(deck):
    """Cheap enough to be worth running first, which is the only argument for
    it: together the strip and the template are a third of the plate."""
    plate = dog_deck.build(deck).volume
    assert dog_rig.strip(deck).volume + dog_deck.template(deck).volume < plate / 2
