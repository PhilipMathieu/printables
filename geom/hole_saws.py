"""Hole saw sets, as the pieces a case has to hold.

One set so far: Harbor Freight's Warrior 57523, ten pieces -- eight carbon
steel saws from 1 to 2-1/2 inches, one large mandrel with its pilot bit in it,
and the hex key that locks the pilot bit's set screw. The sizes and the 1 inch
cut depth are published. Nothing else is.

WHAT IS PUBLISHED AND WHAT IS MEASURED. A hole saw's nominal size is the hole
it cuts, which is the diameter across its teeth, so that one is held to: the
teeth are set outward past the cup, and the cup is the smaller of the two. The
allowance on top of it here is for the set being a little proud of nominal on
cheap saws, and it errs large -- the 2-1/2 in measured 2.51 in across its
teeth, a quarter of a millimetre over, against half a millimetre allowed.
Everything else was measured with calipers on the set in hand: the saws'
height, the nested stack, the mandrel and the key. Where a reading could have
been taken across a hex's flats, it is taken up to the corners, because a
pocket that came out loose rattles a little and one that came out tight does
not take the part at all.

HOW THEY NEST. A smaller saw drops inside a bigger one's cup and stands on
its back plate, so a nested saw adds only a back plate and a hub to the height
of the stack, not a whole saw. Two numbers describe it: the smallest step in
size that still drops in (``nest_step``), and how much prouder each saw in
stands than the one it sits in (``rise``). Both are measured, on the set in
hand: all eight nest, down to the 1/8 inch steps between 2, 2-1/8 and 2-1/4,
a single saw stands 1.15 in tall and the whole stack 1.52 in, so the 1 inch at
the middle stands 0.37 in proud of the 2-1/2 at the outside -- seven steps up.

WHY THE MANDREL IS A STACK OF CYLINDERS. It is a thing of revolution lying on
its side in a case, and what decides the cradle it lies in is its diameter at
each point along it, in order: the hex shank the chuck grips, the flange and
threaded nose the saws screw onto -- one run at the flange's diameter, which
leaves room for its drive pins -- and the pilot bit standing out of the end. A
hex is taken across its corners, because that is the widest way it can lie.
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
    nest_step: float
    """Smallest difference in nominal size, in inches, that nests."""
    rise: float
    """How much prouder a nested saw's teeth stand than its host's, in mm."""

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


# Measured: back of the hub to the tips of the teeth, 1.15 in.
WARRIOR_HEIGHT = 1.15 * MM_PER_INCH

# The mandrel, chuck end first, measured. The shank read 10mm, which is across
# the flats of a 3/8 inch hex give or take; its corners are 11mm, so it is cut
# for 11.5. That costs nothing: the mandrel rests on the flange, and the case's
# height is the stack's.
WARRIOR_MANDREL = (
    Segment("shank", 22.0, 11.5),
    Segment("flange and nose", 22.0, 25.0),
    Segment("pilot", 45.0, 6.0),
)

WARRIOR_57523 = HoleSawSet(
    name="warrior-57523",
    saws=tuple(
        _saw(n, i, WARRIOR_HEIGHT)
        for n, i in (("1", 1.0), ("1-1/4", 1.25), ("1-1/2", 1.5), ("1-3/4", 1.75),
                     ("2", 2.0), ("2-1/8", 2.125), ("2-1/4", 2.25), ("2-1/2", 2.5))
    ),
    mandrel=WARRIOR_MANDREL,
    # Measured: 2.46mm across the flats -- a 2.5mm key -- which is 2.84 across
    # the corners, and legs of 56.7 and 20.4.
    key=HexKey(across=2.9, long=56.7, short=20.4),
    # Measured: the whole set nests, eight deep, the stack 1.52 in against one
    # saw's 1.15 -- 0.37 in proud in seven steps.
    nest_step=1 / 8,
    rise=(1.52 - 1.15) * MM_PER_INCH / 7,
)

CATALOGUE = {s.name: s for s in (WARRIOR_57523,)}


def named(name: str) -> HoleSawSet:
    try:
        return CATALOGUE[name.lower()]
    except KeyError:
        raise KeyError(f"no {name!r} set; have {', '.join(CATALOGUE)}") from None
