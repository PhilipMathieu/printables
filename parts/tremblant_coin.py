"""Mont-Tremblant coin: 26 struck on the obverse, the squiggle on the reverse.

    python -m tools.coin --part parts.tremblant_coin -o out/tremblant_coin

The squiggle is drawn here rather than imported because there is no artwork to
import -- it is a cursive 'm' of three humps, tallest in the middle, traced
from the mark as described and not from the tourism board's file. If the real
SVG turns up, the whole of this module's drawing collapses to

    Svg(path=Path("tremblant.svg"))

and nothing else changes.

Three humps at 11, 13 and 10 units, rather than three the same, is what stops
it reading as a sine wave: the eye takes an uneven run of peaks as mountains
and an even one as a waveform. The stroke runs at 2.6 units against a 29-unit
span, which lands around 2.2mm once it is fitted to the field -- five extrusion
widths, so the engraved channel has a floor rather than being two walls that
meet.
"""

from __future__ import annotations

from geom.motif import BOTTOM, Legend, Stroke, Text
from parts.coin import Params

SQUIGGLE = Stroke(
    points=((0, 0), (4, 11), (8, 1), (13, 13), (18, 1), (23, 10), (29, 3)),
    width=2.6,
    fill=0.80,
    # Lifted off centre, because centred does not look centred here: MONT is
    # four letters in a narrow arc at the top and TREMBLANT is nine in a wide
    # one at the bottom, so the visual mass below is the greater and a squiggle
    # sitting on the geometric centre reads as sunk. Bbox-centred it left
    # 6.6mm of air above and 1.2mm below; up 2mm that is 4.5/2.6, and the room
    # the lift buys underneath is what lets the fill go back up to 0.80.
    shift=(0.0, 0.12),
)
"""The mark: one continuous pen line, spline-smoothed through its peaks."""

PARAMS = Params(
    obverse=(
        Legend(text="SONNEBORN"),
        # As large as the field allows. Four digits run wide, so what stops it
        # is the rim rather than the legends -- at 0.88 the corners sit 2mm
        # clear of the ramp while SONNEBORN is still 2.7mm off.
        Text(text="2026", fill=0.88),
        # Bigger than the lettering, because a plus sign is two bare strokes
        # with none of a letter's bulk: at the 0.13 the words use it measures
        # 0.33mm through the stroke, under one 0.42mm extrusion. 0.19 puts it
        # at 0.49mm, and three of them read as an ornament at that size anyway.
        Legend(text="+++", at=BOTTOM, height=0.19),
    ),
    reverse=(
        Legend(text="MONT"),
        Legend(text="TREMBLANT", at=BOTTOM),
        SQUIGGLE,
    ),
    diameter=38.0,
    thickness=3.0,
    rim=2.2,
    field_depth=0.7,
    relief=0.6,
    engrave=0.5,
    reeds=80,
    reed_depth=0.4,
)
"""Swap ``obverse`` and ``reverse`` to strike the squiggle and engrave the 26."""


if __name__ == "__main__":
    from tools.coin import main

    raise SystemExit(
        main(["--part", "parts.tremblant_coin", "-o", "out/tremblant_coin"])
    )
