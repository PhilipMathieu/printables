"""Dog-hole grids: the one thing every fixture in the set has to agree on.

A bench dog system is a hole pattern and nothing else. Every part that plugs
into it -- stops, fences, wedges, sacrificial backers -- is derived from three
numbers: how big the holes are, how far apart they are, and how thick the deck
is. Put them in one place and a fence can never be cut for a grid the dogs are
not, which is the only way this kind of set goes wrong.

WHY TWELVE ON TWENTY-FIVE. Woodworking settled on 3/4 inch dogs at two inch
centres because it is sized for timber. Electronics work is not: a pedal
enclosure is 112mm long and a Pi hat is 65, so a two inch grid gives you two
usable stations under a workpiece and no way to put a stop where you want one.
12mm holes on a 25mm pitch put four times as many stations under the same part,
and a 12mm dog is still far stronger than anything a drill stand this size can
do to it. The 3/4 inch grid is in the catalogue anyway, because the arithmetic
is the same and someone will want to bolt this to a real bench.

WHY NOT SMALLER HOLES, WHICH IS THE OBVIOUS ECONOMY. Ten on twenty-five is the
tempting version and it is the one thing here that is not a preference. A dog
is a cantilever and its root is a circle, so what it survives goes as the cube
of the diameter: a 9.85mm shank has a section modulus of 94mm3 against 163 for
an 11.85mm one. Across printed layers, at 20 to 35 MPa, that is a root that lets
go somewhere between 176 and 309 newtons applied 10mm up. An M6 in the clamp
turned by hand on a knob delivers 250 to 500. A 10mm dog therefore breaks under
a firm hand, not under abuse, and no amount of pitch cleverness recovers that.
12mm is the smallest hole with the load on the right side of the line -- and
even there the margin is thinner than it looks, which is what the break arm in
``parts.dog_rig`` is for.

WHY TWENTY-FIVE, AND WHAT STANDARD IT IS. There are two metric fixturing
standards and neither covers this size. Festool's MFT is the de facto metric
dog standard -- 20mm holes on a 96mm grid -- and it is timber scale: a 96mm
pitch puts a single station under a pedal enclosure, and 20mm dogs would need a
40mm pitch, which is three columns on this deck. The other is the optical
breadboard: M6 threaded, 25mm grid, and that one *is* this scale. It is the
pitch every small metric fixture plate, camera cheese plate and lab jig already
uses, which is why the deck uses it too.

The article to hold in mind is a Thorlabs MB1515/M or its equivalent: 150 x
150mm of aluminium, 12.7mm thick, M6 on 25mm centres. That is within two
millimetres of this casting's 152mm flat field, and it is flat and stiff in a
way no printed plate will ever be. Which makes it the honest endpoint of this
design rather than a rival to it -- the printed deck is what you use while the
numbers are still moving, and a bought or tapped plate is what it becomes.

The pitch is what carries over. A fence spans a whole number of pitches and a
clamp straddles exactly one, so both are already cut for a breadboard; what
would change is the shank, from an 11.85mm dog to an M6 stud, which is one
feature on each part rather than a redesign. Two details worth knowing before
counting on it. A 150mm board at this pitch holds six holes with 12.5mm edge
margin, an *even* count -- so its centre falls between four holes and it has no
station on the spindle axis, which the printed deck insists on for the datum and
the backer. And its thickness is not this deck's, so dogs cut for one are the
wrong length for the other. Compatibility lives in the pitch, not the bore.

``Grid.takes_insert`` covers the other direction -- M6 in a dog hole, by way of
a bushing with a heat-set insert -- and for a 10mm deck the answer is: only the
short inserts, and only just.

The pitch is no longer exactly twice the hole, which the first metric draft made
a virtue of. It bought a 13mm web instead of 12 -- material, not a loss -- and
the symmetry it cost was decoration.

WHY METRIC, HAVING FIRST BEEN IMPERIAL. This began as half inch holes on an
inch, reasoning that the machine is imperial and a 1/2 inch Forstner is in every
hardware shop in the country. Both halves were weaker than they sounded. The
machine's coupling to imperial is one bolt size -- its slots measure 8.6mm,
which is a round number in neither system -- and the grid does not touch that.
And no bit is needed to make the printed deck at all: a wooden one is piloted at
3mm through a printed template and opened out, so bit size appears in exactly
one place, where 12mm is as ordinary as half an inch. Everything else in this
repository is millimetres.

THE COST, WHICH IS REAL, AND IT IS AT THE EDGE. Seven columns on a 25mm pitch
span 150, against a casting whose flat field is 152 square. The outermost column
sits one millimetre inside that edge, where at 24mm it had four. Its holes still
have more than half their footprint over casting and the rest is a 5mm overhang
of a 10mm plate, which is stiff enough not to matter for anything this machine
can push; but it is the column to leave empty when a setup lets you, and it is a
locating column rather than one to hang a clamp's whole preload off.

The other cost is unchanged and older: a 12mm hole takes an 11.85mm shank where
a half inch one took 12.35, and that is a ninth off the section modulus at the
root. It is 3.3 to 5.8kg on the break arm rather than 3.8 to 6.6 -- and it was a
sixth until the ladder came back tight rather than slip.

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
        at 12mm holes on a 25mm pitch it would leave a 0.5mm web, which is not
        a wall.
        """
        return self.pitch - self.hole

    def shank(self, fit: float) -> float:
        """Dog shank diameter for a given diametral clearance."""
        return self.hole - fit

    def takes_insert(self, od: float, length: float, wall: float = 1.5) -> bool:
        """Whether a station could carry a threaded insert of this size.

        The question the 25mm pitch raises: a breadboard station is an M6
        thread, so could a dog hole simply become one, by way of a printed
        bushing with an insert melted into it? Two things have to hold. The
        insert plus a wall each side has to fit the bore, or there is nothing
        holding it. And it has to be shorter than the deck, or it stands proud
        and the work rocks on it -- which for M6 in a 10mm deck rules out the
        common 12.7mm inserts and leaves only the short ones.

        ``wall`` is the printed material around the insert after melting, not
        before; a heat-set insert displaces its own knurl outwards.
        """
        return od + 2 * wall <= self.hole and length <= self.deck

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


M6_INSERT = (8.0, 9.5)
"""Outside diameter and length of a short M6 heat-set insert, in mm.

The size that decides whether the M6 compatibility the 25mm pitch buys can be
had in a dog hole rather than only in a future deck. Confirm against whatever
inserts you actually own before printing a bushing for them: the same thread is
sold from about 8.0 to 8.7mm across and from 9.5 to 12.7mm long, and the long
ones do not fit a 10mm deck.
"""

FITS = {"loose": 0.50, "slip": 0.35, "snug": 0.25, "tight": 0.15}
"""Diametral clearance between a dog shank and its hole, in mm.

These were guesses until a plate was printed, which is what ``--ladder`` is for:
print one stop at each fit, keep the tightest that still drops in under its own
weight, and set that as the default. Which has now been done, and the answer was
``tight``.

That is two rungs tighter than this first assumed, and the assumption is worth
recording because it was reasoned rather than measured: a dog wants to be a slip
fit and no better, went the argument, because a tight one has to be tapped out
with something and the something is usually the workpiece. On a printed 12mm
hole at 0.4mm nozzle that turned out to be over-cautious -- ``tight`` drops in
under its own weight and comes out by hand. Any new grid or nozzle gets the
ladder printed again rather than inheriting this.
"""

LADDER = ("loose", "slip", "snug", "tight")
"""The fits worth printing side by side, loosest first."""


def fit(name: str) -> float:
    try:
        return FITS[name.lower()]
    except KeyError:
        raise KeyError(f"no {name!r} fit; have {', '.join(FITS)}") from None


METRIC = Grid("metric", hole=12.0, pitch=25.0, deck=10.0)
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
