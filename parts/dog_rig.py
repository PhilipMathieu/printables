"""The rig: things printed to find a number out, rather than to use.

Everything in ``bench_dogs`` that could be settled by arithmetic has been, and
the tests check it on every build. What is left is the handful of things
arithmetic cannot reach, and each of them is here as the cheapest object that
answers it. None of these is part of the set; two of them are meant to be
destroyed.

WHAT IS ACTUALLY UNKNOWN, in the order it would hurt:

1. How much a shank root holds. The fillet is a guess, and the numbers no longer
   say the margin is thin -- they say there may not be one. An 11.85mm shank has
   a section modulus of 163mm^3, so at somewhere between 20 and 35 MPa of layer
   adhesion it lets go between 3.3 and 5.7 N.m. Against that, the clamp at a
   firm hand on its knob (``bench_dogs.KNOB``, 500N, corrected upwards from a
   guessed 200) puts 5.2 N.m into the root. That is inside the range the root
   fails in, not below it. A light hand at 250N gives 2.6 and is comfortable, so
   the honest statement is that the clamp is usable and its upper half is
   unknown. ``arm`` turns the question into a weight you can hang, and it is now
   the first thing to print rather than the interesting one.

2. ANSWERED. Whether a printed plate this long comes off the bed flat. ``strip``
   was printed in PLA at 182mm and came off flat, which clears the printed deck
   -- and is the one result here that argues *against* the part that produced
   it, since the reason to prefer a wooden deck was never accuracy but stiffness
   and creep. Both routes are open; ``dog_deck.template`` is still the way to
   the wooden one.

3. ANSWERED. Whether a backer actually grips. ``puck_ladder`` came back at the
   default 0.25mm of interference, so nothing moved.

The two that needed no special part: the shank clearance, which came back two
rungs tighter than assumed -- see ``bench_dogs.Params.fit`` -- and which
``bench_dogs.ladder`` already covers, and whether the clamp tips its front
shanks out of their holes under load -- print the real clamp for that, because a
truncated one saves a quarter of an hour and changes the thing being measured.
"""

from __future__ import annotations

import math
from dataclasses import replace

from build123d import Align, Box, Cylinder, Part, Plane, Pos, RectangleRounded, extrude

from parts import bench_dogs, dog_deck
from parts.bench_dogs import Params

LEVER = 100.0
"""Effective moment arm at the shank root, in mm.

Round, so the moment is the hung mass in kilograms times a hundred -- 0.981 N.m
per kilogram, near enough that the scale reads out in the units the design is
argued in. And long enough that the whole useful range of answers lands between
two and seven kilograms, which is a bottle and a jug of water rather than a
load cell.

IT IS NOT THE DISTANCE TO THE EYE, and the difference is the whole reason this
constant is not just called ARM any more. The arm bears on the deck through the
head, and under load it tips onto the head's *outboard* edge -- half a head out
from the axis. Vertical equilibrium puts the deck's whole reaction there, so the
root sees the load times (eye distance minus half a head), not times the eye
distance. The eye therefore sits half a head further out than the number that
comes off the scale. See ``arm``.
"""

STAND = 3.0
"""How far the break arm's lever stands clear of the deck, in mm.

The fix for the flaw in ``arm``, and a limit on the block in the same number --
see ``block_reach``.
"""


def block_reach(params: Params, stand: float = STAND) -> float:
    """How far the block may reach past the hole in the break test, in mm.

    The arm has to be free to rotate through everything between snug and broken
    or the lever touches down and quietly takes the load back. Two things add
    up: the shank cocks in its clearance before the hole resists at all, and
    then the root bends before it lets go. The stand-off has to outlast both at
    the block's outer edge, which is what this solves for.

    Measuring the fit paid for itself here. At the guessed slip clearance the
    arm cocked 2.2 degrees before the hole did anything and the block could only
    reach 53mm; at the measured tight fit it is 1.0 and the block may reach 90,
    which is the difference between specifying an offcut and specifying a
    particular offcut.
    """
    cocking = params.fit / params.shank_length
    stiffness = math.pi * params.shank**4 / 64
    bending = 5.4e3 * params.shank_length / (3000 * stiffness)  # PLA, at the break
    return stand / (cocking + bending)


PUCK_GRIPS = (0.15, 0.25, 0.35, 0.45)
"""Diametral interferences worth trying on a backer, loosest first."""


def arm(params: Params, lever: float = LEVER, width: float = 20.0,
        eye: float = 8.0, stand: float = STAND) -> Part:
    """A stop with a lever on it, for breaking one root on purpose.

    Drop it in a hole, clamp the block down, hang a bag off the eye and fill the
    bag by weight until something goes. The moment at the root is the mass times
    ``lever``, so the number that comes out is directly comparable with what the
    clamp can apply -- which is the only reason to know it.

    THE LEVER STANDS OFF THE DECK, AND THE FIRST DRAWING OF THIS DID NOT. It was
    a flat paddle coplanar with the head's seat, which meant it lay flat on the
    deck -- and a lever lying on the surface its own fulcrum is in hands the
    moment straight to the surface. Taking moments about the shank axis with the
    deck reacting at distance ``a``, the root sees ``W (lever - a)``: on a 50mm
    offcut that is two thirds of the intended load, and on the real 182mm deck,
    with the plate bearing all the way out to its edge, it is *nine percent* of
    it. The arm would have come back reading four times too strong and there
    would have been nothing in the result to say so.

    So the lever is ``stand`` clear of the seat plane and only the head touches.
    That fixes ``a`` at half a head whatever the block is cut to, and the eye
    goes half a head further out so the effective arm is exactly ``lever``.
    ``stand`` is 3mm because the arm has to be free to rotate: shank clearance
    alone lets it cock about 2 degrees before the hole even starts to resist,
    and the root bends through another degree before it breaks, so anything less
    would have the lever touch down mid-test and quietly take the load back.

    The arm is deliberately the stronger end. Its root section modulus is nearly
    twice the shank's, and it is loaded along its layers where the shank is
    loaded across them, so the failure lands at the root of the shank where it
    is wanted. Watch where it actually breaks: at the root means the fillet is
    the limit, up the shank means the fillet did its job and moved the weak
    point somewhere the design does not care about.

    For the comparison, print a second one with ``--root 0.01``. That is a
    square root in all but name, and the difference between the two is what the
    fillet bought.
    """
    params.validate()
    if stand >= params.rise:
        raise ValueError(
            f"a {stand}mm stand-off leaves no lever under a {params.rise}mm head"
        )
    body = bench_dogs.stop(params)
    deep = params.rise - stand
    at_eye = lever + params.head / 2
    body += Box(
        at_eye + width, width, deep, align=(Align.MIN, Align.CENTER, Align.MIN),
    )
    return body - Pos(at_eye, 0, deep / 2) * Cylinder(eye / 2, params.rise * 3)


def eye_at(params: Params, lever: float = LEVER) -> float:
    """Where the eye sits, measured from the shank axis.

    Half a head further out than the effective arm, because that is where the
    deck's reaction lands. Anything that wants to draw or check the arm should
    ask for this rather than assume the two are the same number.
    """
    return lever + params.head / 2


def puck_ladder(params: Params) -> list[Part]:
    """One backer at each grip in ``PUCK_GRIPS``, loosest first.

    Press each into a spare station and push on it. The one to keep is the
    loosest that still seats flush by thumb and does not move when a bit's
    thrust is imitated with a thumb, which is all the load a backer ever sees.

    Each is marked with its grip in hundredths on the end that goes in first,
    which is the end you can still read while deciding.
    """
    return [
        bench_dogs.puck(replace(params, puck_grip=g, mark=f"{g * 100:.0f}"))
        for g in PUCK_GRIPS
    ]


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
