"""Dog-hole grids: the one thing every fixture in the set has to agree on.

A bench dog system is a hole pattern and nothing else. Every part that plugs
into it -- stops, fences, wedges, sacrificial backers -- is derived from three
numbers: how big the holes are, how far apart they are, and how thick the deck
is. Put them in one place and a fence can never be cut for a grid the dogs are
not, which is the only way this kind of set goes wrong.

WHY TWELVE ON TWENTY-FOUR. Woodworking settled on 3/4 inch dogs at two inch
centres because it is sized for timber. Electronics work is not: a pedal
enclosure is 112mm long and a Pi hat is 65, so a two inch grid gives you two
usable stations under a workpiece and no way to put a stop where you want one.
12mm holes on a 24mm pitch put four times as many stations under the same part,
and a 12mm dog is still far stronger than anything a drill stand this size can
do to it. The 3/4 inch grid is in the catalogue anyway, because the arithmetic
is the same and someone will want to bolt this to a real bench.

The pitch is exactly twice the hole, which is not decoration. It makes the web
between two stations the same width as a station, so the plate is no weaker
between its holes than at them.

WHY METRIC, HAVING FIRST BEEN IMPERIAL. This began as half inch holes on an
inch, reasoning that the machine is imperial and a 1/2 inch Forstner is in every
hardware shop in the country. Both halves were weaker than they sounded. The
machine's coupling to imperial is one bolt size -- its slots measure 8.6mm,
which is a round number in neither system -- and the grid does not touch that.
And no bit is needed to make the printed deck at all: a wooden one is piloted at
3mm through a printed template and opened out, so bit size appears in exactly
one place, where 12mm is as ordinary as half an inch. Everything else in this
repository is millimetres.

What actually decided it was neither of those. At 24mm the seven columns span
144, which puts every station inside the 152mm square of flat casting they bolt
to; at 25.4 the outermost column landed on its edge to within two tenths of a
millimetre.

THE COST, WHICH IS REAL. A 12mm hole takes an 11.65mm shank where a half inch
one took 12.35, and that is a sixth off the section modulus at the root -- the
one place in this design where the margin was already thin. It is 3.2 to 5.5kg
on the break arm rather than 3.8 to 6.6, against 2.1 N.m the clamp can apply.
Which makes the test in ``parts.dog_rig`` matter more than it did, not less.

THE CENTRE STATION IS NOT OPTIONAL. One hole has to sit exactly on the spindle
axis. It is where the sacrificial backer goes, it is what you line the deck up
by, and it is the hole the bit drops into when it comes through the work. A
grid with the spindle landing between four holes has nowhere to put any of
that, which is why the deck insists on odd counts.
"""

from __future__ import annotations

from dataclasses import dataclass

MM_PER_INCH = 25.4


@dataclass(frozen=True)
class Grid:
    """A dog hole pattern, in mm."""

    name: str
    hole: float
    """Nominal diameter of a dog hole."""
    pitch: float
    """Centre to centre, the same in both directions."""
    deck: float
    """Thickness of the deck the holes are bored through. Dogs are cut to this:
    it is how deep a shank can go, and so how much of a side load the hole can
    take before the dog starts to lever rather than bear."""

    @property
    def web(self) -> float:
        """Material left between two neighbouring holes.

        The thing that fails first if the pitch is ever brought down towards the
        hole size, and the reason a half-pitch interleaved grid is not on offer:
        at 12.7mm holes it would leave a 5mm web, which is two walls and some
        infill.
        """
        return self.pitch - self.hole

    def shank(self, fit: float) -> float:
        """Dog shank diameter for a given diametral clearance."""
        return self.hole - fit

    def span(self, holes: int) -> float:
        """Centre-to-centre distance across a run of holes."""
        return (holes - 1) * self.pitch

    def stations(self, cols: int, rows: int) -> tuple[tuple[float, float], ...]:
        """Hole centres for a cols x rows grid, centred on the origin.

        Odd counts put a station on the origin, which the deck requires and the
        module note explains.
        """
        x0, y0 = self.span(cols) / 2, self.span(rows) / 2
        return tuple(
            (c * self.pitch - x0, r * self.pitch - y0)
            for r in range(rows)
            for c in range(cols)
        )


FITS = {"loose": 0.50, "slip": 0.35, "snug": 0.25, "tight": 0.15}
"""Diametral clearance between a dog shank and its hole, in mm.

These are guesses until a plate has been printed, which is what ``--ladder`` is
for: print one stop at each fit, keep the tightest that still drops in under its
own weight, and set that as the default. A dog wants to be a slip fit and no
better -- a tight one has to be tapped out with something, and the something is
usually the workpiece.
"""

LADDER = ("loose", "slip", "snug", "tight")
"""The fits worth printing side by side, loosest first."""


def fit(name: str) -> float:
    try:
        return FITS[name.lower()]
    except KeyError:
        raise KeyError(f"no {name!r} fit; have {', '.join(FITS)}") from None


METRIC = Grid("metric", hole=12.0, pitch=24.0, deck=10.0)
HALF_INCH = Grid("half-inch", hole=MM_PER_INCH / 2, pitch=MM_PER_INCH, deck=10.0)
THREE_QUARTER = Grid(
    "three-quarter", hole=MM_PER_INCH * 0.75, pitch=MM_PER_INCH * 2, deck=18.0
)

DEFAULT = METRIC

CATALOGUE = {g.name: g for g in (METRIC, HALF_INCH, THREE_QUARTER)}


def named(name: str) -> Grid:
    try:
        return CATALOGUE[name.lower()]
    except KeyError:
        raise KeyError(f"no {name!r} grid; have {', '.join(CATALOGUE)}") from None
