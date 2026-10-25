"""The lead webbing catalogue."""

from __future__ import annotations

from fractions import Fraction

import pytest

from geom import webbing


@pytest.mark.parametrize("strap", webbing.CATALOGUE.values(), ids=webbing.CATALOGUE)
def test_a_width_is_the_inches_it_is_sold_as(strap):
    """The width is the one number a lead is actually held to -- it is woven on
    a loom -- so it has to be the nominal size converted, not a caliper reading.
    Thickness is the caliper reading, and runs thick on purpose."""
    inches = float(Fraction(strap.nominal.split()[0]))
    assert strap.width == pytest.approx(inches * webbing.MM_PER_INCH)
    assert strap.handle == pytest.approx(2 * strap.thickness)


def test_an_unknown_width_says_what_there_is():
    with pytest.raises(KeyError, match="standard"):
        webbing.named("paracord")
