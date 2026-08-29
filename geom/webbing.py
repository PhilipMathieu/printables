"""Leash webbing, as sections a part can be built around.

Flat webbing only. Dog leads are sold by nominal width in inches and the
widths below are those inches converted, which is the number that is actually
held to -- webbing is woven to width on a loom, so a 3/4 inch lead is 19mm and
change whoever made it.

THICKNESS IS THE SOFT NUMBER. It is not published, it varies with the weave
and the coating, and it is measured here off leads in hand. So these run a
little thick on purpose. A part built around webbing gets a slot, and a slot
that came out generous only slides more freely; a slot that came out tight
does not go on at all, and there is no adjusting it after the print.

WHY ``handle`` EXISTS. A lead's handle is the same webbing folded back and
stitched, so it is two plies thick plus the stitching, and it is the fold that
has to go through a slot to get anything onto a lead without unclipping it.
Sizing a slot for one thickness gives a part that can only be threaded on past
the snap hook, which is bigger than any slot worth printing.
"""

from __future__ import annotations

from dataclasses import dataclass

MM_PER_INCH = 25.4


@dataclass(frozen=True)
class Webbing:
    """One width of flat lead webbing, in mm."""

    name: str
    nominal: str
    """The size it is sold as, e.g. '3/4 in'."""
    width: float
    thickness: float
    """Single ply, measured not published. See the module note."""

    @property
    def handle(self) -> float:
        """Thickness of the strap where it folds back into the handle.

        Two plies. The stitching adds a little more and the clearance a slot is
        given should swallow it -- see ``Params.slot_clearance``.
        """
        return 2 * self.thickness

    def stack(self, plies: int) -> float:
        """Thickness of ``plies`` layers of this webbing."""
        return plies * self.thickness


def _inches(name: str, nominal: str, width: float, thickness: float) -> Webbing:
    return Webbing(name, nominal, width * MM_PER_INCH, thickness)


# The four widths that cover essentially every flat lead on a shelf. Anything
# narrower is a cat lead and anything wider is a tug line.
TOY = _inches("toy", "3/8 in", 0.375, thickness=1.6)
SMALL = _inches("small", "5/8 in", 0.625, thickness=2.0)
STANDARD = _inches("standard", "3/4 in", 0.750, thickness=2.4)
WIDE = _inches("wide", "1 in", 1.000, thickness=2.8)

CATALOGUE = {w.name: w for w in (TOY, SMALL, STANDARD, WIDE)}


def named(name: str) -> Webbing:
    try:
        return CATALOGUE[name.lower()]
    except KeyError:
        raise KeyError(
            f"no {name!r} webbing; have {', '.join(CATALOGUE)}"
        ) from None
