"""Adhesive mounting strips, as footprints a part can be built around.

The sizes are 3M's published figures for the Command range, converted from the
inches they are specified in. A part that mounts on one of these needs three
numbers from it -- how long, how wide, and how much of the length is pull tab
rather than adhesive -- and the tab is the one 3M does not publish. See ``tab``.

WHAT THE HOLD RATINGS MEAN. They are the pack figures, and they are for a strip
on a smooth painted wall loaded in shear: pulling straight down the strip's long
axis. Hung on a ceiling the same load becomes a peel, which is the direction a
pressure-sensitive adhesive is weakest by an order of magnitude, and 3M's own
guidance is walls only. So ``hold`` is an upper bound for the wall case and no
guide at all overhead. Overhead, spread the load over more strips than the
arithmetic says and accept that it is off-label.

They also want a smooth, rigid, non-porous backing on both sides. A printed
part gives that only on the face that was printed against the build plate, so
anything mounted this way should be modelled with its pad flat on the bed.
"""

from __future__ import annotations

from dataclasses import dataclass

MM_PER_INCH = 25.4


@dataclass(frozen=True)
class Strip:
    """One adhesive strip, in mm."""

    name: str
    length: float
    width: float
    tab: float
    hold: float
    """Kilograms per strip, in shear, on a smooth wall. See the module note."""

    @property
    def bond(self) -> float:
        """Length actually stuck down: the strip less its pull tab.

        This, not ``length``, is how long a mounting pad should be. The tab has
        to hang off the end of the part where a thumb can reach it -- a tab
        trapped between the pad and the wall cannot be pulled, and the strip
        then has to be cut off rather than stretched off, which is the whole
        thing they exist to avoid.
        """
        return self.length - self.tab


def _inches(name: str, length: float, width: float, tab: float, hold: float) -> Strip:
    return Strip(name, length * MM_PER_INCH, width * MM_PER_INCH, tab, hold)


# Lengths and widths from 3M's product information sheet; hold from the pack.
#
# ``tab`` is measured off a strip in hand and is the soft number here. Erring
# long is the safe direction: a pad shorter than the adhesive leaves a little
# glue stuck to the wall side only, which costs some holding and nothing else,
# whereas a pad too long buries the tab and the strip stops being removable.
MINI = _inches("mini", 1.177, 0.500, tab=8.0, hold=0.23)
SMALL = _inches("small", 1.813, 0.625, tab=12.0, hold=0.23)
MEDIUM = _inches("medium", 2.750, 0.625, tab=14.0, hold=1.36)
LARGE = _inches("large", 3.650, 0.750, tab=16.0, hold=2.27)
NARROW = _inches("narrow", 3.650, 0.250, tab=16.0, hold=0.45)

CATALOGUE = {s.name: s for s in (MINI, SMALL, MEDIUM, LARGE, NARROW)}


def named(name: str) -> Strip:
    try:
        return CATALOGUE[name.lower()]
    except KeyError:
        raise KeyError(
            f"no {name!r} strip; have {', '.join(CATALOGUE)}"
        ) from None
