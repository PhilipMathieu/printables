"""The rig: things printed to find a number out, rather than to use.

Everything in ``bench_dogs`` that could be settled by arithmetic has been, and
the tests check it on every build. What is left is the handful of things
arithmetic cannot reach, and each of them is here as the cheapest object that
answers it. None of these is part of the set; two of them are meant to be
destroyed.

WHAT IS ACTUALLY UNKNOWN, in the order it would hurt:

1. How much a shank root holds. The fillet is a guess, and the numbers say the
   margin is thinner than it feels: a 12.35mm shank has a section modulus of
   185mm^3, so at somewhere between 20 and 35 MPa of layer adhesion it lets go
   between 3.7 and 6.5 N.m -- against 2.1 N.m from the clamp at a modest 200N
   of screw force. Two-ish times, and an M6 wound up with a tool reaches ten
   times that. ``arm`` turns the question into a weight you can hang.

2. Whether a printed plate this long comes off the bed flat. ``strip`` is the
   cheap proxy, with the caveat in its own docstring.

3. Whether a backer actually grips. ``puck_ladder``, four grips, ten minutes.

The two that need no special part: the shank clearance, which
``bench_dogs.ladder`` already covers, and whether the clamp tips its front
shanks out of their holes under load -- print the real clamp for that, because a
truncated one saves a quarter of an hour and changes the thing being measured.
"""

from __future__ import annotations

from dataclasses import replace

from build123d import Align, Box, Cylinder, Part, Plane, Pos, RectangleRounded, extrude

from parts import bench_dogs, dog_deck
from parts.bench_dogs import Params

ARM = 100.0
"""Distance from the shank's axis to the eye, in mm. Round, so the moment is
the hung mass in kilograms times a hundred -- and long enough that the whole
useful range of answers lands between two and seven kilograms, which is a range
you can hit with a bottle and a jug of water rather than a load cell."""

PUCK_GRIPS = (0.15, 0.25, 0.35, 0.45)
"""Diametral interferences worth trying on a backer, loosest first."""


def arm(params: Params, length: float = ARM, width: float = 14.0,
        eye: float = 8.0) -> Part:
    """A stop with a lever on it, for breaking one root on purpose.

    Drop it in a hole, clamp the deck down, hang a bag off the eye and fill the
    bag by weight until something goes. The moment at the root is the mass times
    ``length``, so the number that comes out is directly comparable with what
    the clamp can apply -- which is the only reason to know it.

    The arm is deliberately the stronger end. Its root section modulus is nearly
    twice the shank's, and it is loaded along its layers where the shank is
    loaded across them, so the failure lands at the root of the shank where it
    is wanted. Watch where it actually breaks: at the root means the fillet is
    the limit, up the shank means the fillet did its job and moved the weak
    point somewhere the design does not care about.

    For the comparison, print a second one with ``--root 0.05``. That is a
    square root in all but name, and the difference between the two is what the
    fillet bought.
    """
    params.validate()
    body = bench_dogs.stop(params)
    body += Box(
        length + width, width, params.rise,
        align=(Align.MIN, Align.CENTER, Align.MIN),
    )
    return body - Pos(length, 0, params.rise / 2) * Cylinder(eye / 2, params.rise * 3)


def puck_ladder(params: Params) -> list[Part]:
    """One backer at each grip in ``PUCK_GRIPS``, loosest first.

    Press each into a spare station and push on it. The one to keep is the
    loosest that still seats flush by thumb and does not move when a bit's
    thrust is imitated with a thumb, which is all the load a backer ever sees.
    """
    return [bench_dogs.puck(replace(params, puck_grip=g)) for g in PUCK_GRIPS]


def strip(deck: dog_deck.Params, width: float = 30.0) -> Part:
    """A long thin plate of the deck's own length, to see whether it lifts.

    The deck is the only part here big enough for warp to be the thing that
    ruins it, and it is also the longest print in the set, so finding out the
    expensive way is the wrong order. This is the same length, the same
    thickness, the same holes and the same material, at a fifth of the time.

    Read it honestly, because the proxy is not symmetric. A strip that curls off
    the plate is proof the deck would have: warp scales with length and this has
    all of it. A strip that stays flat is only evidence, not proof -- a wide
    plate carries more shrinkage force than a narrow one, even though it also
    has more stiffness to resist it. Good news here means print the deck and
    watch it; bad news here means do not bother.
    """
    deck.validate()
    plate = extrude(
        Plane.XY * RectangleRounded(deck.width, width, deck.corner),
        amount=deck.deck,
    )
    cutter = dog_deck.hole(deck)
    holes = Part()
    for x in sorted({x for x, _ in deck.stations}):
        holes += Pos(x, 0, 0) * cutter
    return plate - holes


if __name__ == "__main__":
    from pathlib import Path

    from build123d import export_stl

    p, d = Params(), dog_deck.Params()
    out = Path("out")
    out.mkdir(exist_ok=True)
    for name, part in (
        ("arm", arm(p)),
        ("strip", strip(d)),
        ("template", dog_deck.template(d)),
        ("puck_ladder", puck_ladder(p)[0]),
    ):
        export_stl(part, str(out / f"rig_{name}.stl"))
        bb = part.bounding_box()
        print(
            f"{name:12s} valid={part.is_valid} "
            f"{bb.size.X:6.1f} x {bb.size.Y:6.1f} x {bb.size.Z:6.1f} mm "
            f"{part.volume / 1000:6.1f} cm^3"
        )
