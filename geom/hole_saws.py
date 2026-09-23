"""Hole saw sets, as the pieces a case has to hold.

One set so far: Harbor Freight's Warrior 57523, ten pieces -- eight carbon
steel saws from 1 to 2-1/2 inches, one large mandrel with its pilot bit in it,
and the hex key that locks the pilot bit's set screw. The sizes and the 1 inch
cut depth are published. Nothing else is.

WHAT IS PUBLISHED AND WHAT IS NOT. A hole saw's nominal size is the hole it
cuts, which is the diameter across its teeth, so that one is held to: the teeth
are set outward past the cup, and the cup is the smaller of the two. The
allowance on top of it here is for the set being a little proud of nominal on
cheap saws, and it errs large. Height, the mandrel and the key are neither
published nor measured here: they are estimates from the standard parts this
kind of set is made from, each rounded in the direction that fails soft, and
they are the numbers to check with calipers against the set in hand before
printing. A pocket that came out loose rattles a little; one that
came out tight does not take the saw at all.

WHY THE MANDREL IS A STACK OF CYLINDERS. It is a thing of revolution lying on
its side in a case, and what decides the cradle it lies in is its diameter at
each point along it, in order: the hex shank the chuck grips, the flange whose
two pins drive the bigger saws, the threaded nose the saws screw onto, and the
pilot bit standing out of the end. A hex is taken across its corners, because
that is the widest way it can lie.
"""

from __future__ import annotations

from dataclasses import dataclass

MM_PER_INCH = 25.4


@dataclass(frozen=True)
class Segment:
    """One stretch of the mandrel, lying along its axis, in mm."""

    name: str
    length: float
    diameter: float
    """Across the corners, if it is a hex."""


@dataclass(frozen=True)
class HoleSaw:
    """One saw, standing on its back plate with the teeth up, in mm."""

    nominal: str
    """The size it is sold as, e.g. '2-1/8 in'."""
    inches: float
    diameter: float
    """Across the teeth, including ``SET_ALLOWANCE``."""
    height: float
    """Back of the hub to the tips of the teeth."""


@dataclass(frozen=True)
class HexKey:
    """An L-shaped Allen key, in mm."""

    across: float
    """Across the corners."""
    long: float
    short: float
    """Both legs measured outside the bend, so to the far edge of the other leg."""


@dataclass(frozen=True)
class HoleSawSet:
    name: str
    saws: tuple[HoleSaw, ...]
    mandrel: tuple[Segment, ...]
    key: HexKey

    @property
    def mandrel_length(self) -> float:
        return sum(s.length for s in self.mandrel)

    @property
    def mandrel_diameter(self) -> float:
        """Widest point of the mandrel, which is what it rests on."""
        return max(s.diameter for s in self.mandrel)

    @property
    def tallest(self) -> float:
        return max(s.height for s in self.saws)

    @property
    def pieces(self) -> int:
        return len(self.saws) + 2


SET_ALLOWANCE = 0.5
"""Added to every nominal diameter. See the module note."""


def _saw(nominal: str, inches: float, height: float) -> HoleSaw:
    return HoleSaw(f"{nominal} in", inches, inches * MM_PER_INCH + SET_ALLOWANCE,
                   height)


# Carbon steel saws of this kind cut 1 inch deep, which puts the tips of the
# teeth about 30mm above the back plate once the plate and the hub under it are
# counted. 32 is that with the hub's boss erring tall.
WARRIOR_HEIGHT = 32.0

# The large mandrel, chuck end first. A 3/8 inch hex shank; a flange about an
# inch and a quarter across carrying the two drive pins; a 5/8-18 threaded nose;
# a 1/4 inch pilot bit standing out of it. Estimated from the standard large
# mandrel, not measured, and rounded up. Measure yours.
WARRIOR_MANDREL = (
    Segment("shank", 36.0, 11.5),
    Segment("flange", 16.0, 32.0),
    Segment("nose", 14.0, 16.5),
    Segment("pilot", 40.0, 7.0),
)

WARRIOR_57523 = HoleSawSet(
    name="warrior-57523",
    saws=tuple(
        _saw(n, i, WARRIOR_HEIGHT)
        for n, i in (("1", 1.0), ("1-1/4", 1.25), ("1-1/2", 1.5), ("1-3/4", 1.75),
                     ("2", 2.0), ("2-1/8", 2.125), ("2-1/4", 2.25), ("2-1/2", 2.5))
    ),
    mandrel=WARRIOR_MANDREL,
    # A small L-key for the pilot bit's set screw; size estimated, legs rounded up.
    key=HexKey(across=3.7, long=56.0, short=17.0),
)

CATALOGUE = {s.name: s for s in (WARRIOR_57523,)}


def named(name: str) -> HoleSawSet:
    try:
        return CATALOGUE[name.lower()]
    except KeyError:
        raise KeyError(f"no {name!r} set; have {', '.join(CATALOGUE)}") from None
