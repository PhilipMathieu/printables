"""The machine, as measured: a catalogue of drill stands to build fixtures onto.

One entry so far, and every number in it came off the casting with calipers to a
tenth. That matters because the first version of this design was scaled off a
photograph, and the photograph was wrong by a fifth: it put the slot at 7.1mm
when it is 8.6, which is the difference between a 1/4-20 machine and a 5/16-18
one. Nothing here is inferred any more.

WHAT A STAND HAS TO TELL A FIXTURE. Three things, and they are the three that
were guessed at:

How wide its slots are, and how wide the pockets at the ends of them, because
together those decide the only piece of hardware in the whole system -- the
bolt that has to pass head-first through a pocket, slide along, and then not be
able to pull back through the slot.

How far it is from the spindle to the column, because that is the one hard
bound on how deep a deck can be. It turned out to be twice what was feared,
which is why ``fits`` mostly says yes.

How thick the casting is at a slot and how much air there is beneath it,
because that is the arithmetic for how long the bolt has to be and whether its
head has anywhere to sit.

WHAT IT DOES NOT TELL ANYTHING. Where the mounting holes go. They are marked
through the casting's own slots with the deck sitting on it, which needs no
measurement and cannot be off; see ``parts.dog_deck``.
"""

from __future__ import annotations

from dataclasses import dataclass

MM_PER_INCH = 25.4


@dataclass(frozen=True)
class Stand:
    """A drill press stand's base, in mm. Measured, not inferred."""

    name: str
    field: float
    """Side of the flat, slotted square the fixture sits on."""
    slot: float
    """Width of the T-bolt slots. What a bolt's shank has to pass."""
    pocket: float
    """Width of the square opening at each end of a slot. What a bolt's head has
    to pass, going in from above."""
    plate: float
    """Casting thickness at a slot, which is what a bolt head bears against."""
    rib: float
    """Casting thickness including the support alongside the slot -- the lowest
    thing near a slot, and so what really sets how much air is underneath."""
    height: float
    """Top of the casting to the bench it stands on."""
    bore: float
    """The bit clearance hole through the middle."""
    column: float
    """Column diameter."""
    throat: float
    """Nearest face of the column to the centre of the bore. The bound on how
    far back a deck can reach."""
    stroke: float
    """Quill travel. Does not bound the deck's thickness -- the carriage clamps
    at whatever height on the column you set -- but it does bound the work."""

    @property
    def under(self) -> float:
        """Air beneath the casting at a slot, for a bolt head to live in."""
        return self.height - self.rib

    def head_fits(self, across_flats: float) -> bool:
        """Whether a hex head passes a pocket and is then held by the slot.

        Both halves matter. Too big and it will not go in; too small and it
        pulls straight back out through the slot under load, which is the whole
        job of a T-slot.
        """
        return self.slot < across_flats < self.pocket

    def bolt(self, deck: float, sunk: float, nut: float) -> float:
        """Shortest bolt, under the head, that reaches through and takes a nut.

        Head under the casting, up through it and the deck, emerging into a
        counterbore ``sunk`` deep with a ``nut`` thick nut in it. Two threads of
        margin on top, which is what "plus 2" is.
        """
        return self.plate + (deck - sunk) + nut + 2.0

    def fits(self, depth: float) -> bool:
        """Whether a deck reaching ``depth``/2 back from the spindle clears the
        column."""
        return depth / 2 < self.throat


CRAFTSMAN_33525921 = Stand(
    name="craftsman-33525921",
    field=152.0,
    slot=8.6,
    pocket=13.4,
    plate=4.1,
    rib=6.6,
    height=28.9,
    bore=22.4,
    column=25.5,
    throat=114.0,
    stroke=2.5 * MM_PER_INCH,
)
"""Sears/Craftsman 335.25921 -- a stand a 1/4in or 3/8in portable drill clamps
into, not a drill press. Measured to a tenth of a millimetre.

The bore reads 22.4, which is 7/8in to within four hundredths, and the pocket
0.528in, which is a hair over 1/2in -- so a 5/16-18 hex head, half an inch
across the flats, drops through it with 0.7mm to spare and is then far too big
to come back through an 8.6mm slot. M8 works too and leaves only 0.4mm at the
pocket. The slot itself is 0.339in, which is nothing in particular: a nominal
5/16 slot cast generously.
"""

CATALOGUE = {s.name: s for s in (CRAFTSMAN_33525921,)}


def named(name: str) -> Stand:
    try:
        return CATALOGUE[name.lower()]
    except KeyError:
        raise KeyError(f"no {name!r} stand; have {', '.join(CATALOGUE)}") from None
