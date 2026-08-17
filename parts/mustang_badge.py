"""MUSTANG badge: bevelled letters on a raised rail, taped to a flat panel.

The whole part is glyph outlines turned into two flared stacks -- the letters to
full thickness, a rail behind them to rail thickness -- and the interesting
decisions are about how a word becomes a single printable object, how a bevel
gets onto letterforms that refuse to be chamfered, and which face ends up
against the paint.

WHY THERE IS A RAIL, AND WHY IT SITS WHERE IT DOES. A word is not a solid. Set
MUSTANG in any normal typeface and you get seven islands with two to five
millimetres of air between them, so "print the badge" means printing seven loose
letters and aligning them on the car by eye, with adhesive that grabs on
contact. The original emblem solves this the same way, with a thin bar running
the length of the word -- and the bar is not under the letters, it crosses them.
The serif feet hang below it. That is the detail that makes the real thing read
as one cast piece rather than as letters glued to a stick, so the rail here is
placed the same way: its underside sits at the height where the serifs have
finished flaring, measured off the glyphs rather than guessed (``serif_top``).

An earlier version hung the rail below the baseline instead, which is easier and
wrong -- it reads as an underline, and it makes the badge 2mm taller than the
lettering for no reason.

HOW THE BEVEL IS BUILT, AND WHY NOT THE OBVIOUS WAY. Chrome emblems have a flat
letter face and a steep bevel down to the base. Every native way of asking for
that fails on this geometry, and it is worth recording which, because they all
look reasonable until you run them:

    chamfer() on the 218 top edges     ValueError: try a smaller length
    extrude(taper=45)                  BRepFill: incoherent intersection
    offset(..., Kind.INTERSECTION)     Null TopoDS_Shape / Unexpected result
    loft(outline -> smaller outline)   the number of holes must be the same

What does work is ``offset`` with ``Kind.ARC``, in either direction. The
dividing line is the join style, not the direction: ARC tucks a fillet into each
convex corner, while INTERSECTION extends both edges until they meet, and it is
the extending-and-intersecting that goes wrong where glyph edges are nearly
parallel or very short. So the bevel is assembled from offsets and straight
prisms, which are the two operations this geometry tolerates.

Given that, the bevel is built as a flare going *down* from the letter face
rather than a taper going up to it. The two describe the same solid, and the
choice is deliberate:

  - the top face is the true typeface outline, untouched, so no thin stroke or
    serif can be eaten by the bevel however large it gets -- shrinking upward
    would thin every stroke by twice the bevel and close up the counter of the A;
  - every layer is smaller than the one below it, so nothing overhangs.

The flare is a staircase of ``LAYER``-high steps rather than a smooth ramp. That
is not an approximation in the finished part: a 45 degree bevel printed at 0.2mm
layers *is* a staircase of 0.2mm steps, so modelling it that way costs nothing
real and avoids lofting between sections whose hole counts differ. Outward
offsets on these glyphs stop working somewhere between 0.8 and 1.0mm, which is
where ``MAX_BEVEL`` comes from -- measured, not chosen.

Because the flare adds material outside the nominal outline, the word is solved
to ``length - 2 * bevel`` so that ``length`` still means what it says: the size
you measure on the car.

WHY IT PRINTS FACE UP. The back is the bond face and it wants to be flat, smooth
and continuous, which is exactly what a first layer on a clean plate is.
Printing letters-down would give prettier letter faces -- plate texture instead
of top surface -- but it puts small unconnected islands of ASA on the plate, and
ASA on small isolated footprints is how you find out what warping looks like.
Face up, the first layer is the whole badge silhouette in one connected piece
(the rail crosses every letter, so the sketch is one region), every layer above
it is the same size or smaller, and nothing overhangs. The letter faces come out
as top surfaces instead, which is what ironing is for.

WHY THE TAPE IS NOT THE PROBLEM HERE. Unlike anything hung under a desk, a badge
on a tailgate is loaded in shear: gravity runs down the panel, not away from it.
Shear is the mode a pressure-sensitive adhesive is strongest in, and the badge
weighs a few tens of grams over a bond area of a couple of thousand square
millimetres. The load case that actually matters is heat and airflow -- a dark
panel in sun runs far above room temperature and the badge sees moving air at
speed -- which is an argument about tape selection and surface prep, not about
bond area. Use a proper automotive attachment tape, not a general-purpose
double-sided, and clean the paint with isopropyl first.

WHY ASA. Direct sun for years. PLA would sag on a hot tailgate and go chalky in
UV within a season; ASA is the material ABS-grade exterior trim is made from and
holds colour and shape outdoors, which is the entire reason it exists.

WHICH FONT, AND HOW SURE. Ford never released the badge lettering as a typeface,
so there is no correct file to go and download; the question is only what comes
closest. Two different answers depending on era, and they are not alike: the
modern (2005-on) wordmark is a custom bold SANS, while the classic 1965-72 pin-on
fender and trunk letters are a SERIF -- a slab in the Clarendon family. That is
worth stating plainly because the intuitive guess is a wide sans for both, and it
is wrong for the classic one.

The best identification available is Clarendon Wide (Medium), from a sign-maker
running font-recognition software against a photo of a real emblem. Independent
suggestions on the same emblem were Times New Roman Bold slightly extended and
Osiris BQ Bold -- different faces, but all serif, all in the same width class,
which is the part worth trusting. macOS ships Superclarendon, a Clarendon, so the
best-supported candidate needs no download and is the default here. Rockwell is
the closest alternative already installed and reads as a more geometric slab.
Treat this as a good match rather than a verified one: the evidence is a font-ID
tool and forum consensus, not a Ford drawing.

Set to 150mm the default lands near 19.5mm caps, 7.7:1 -- which is the other
reason to believe it. A badge is a wide, short object, and a normal-width serif
at badge width falls straight into that proportion where a text sans does not.

PRINTED 2026-08-16, third attempt. The two failures before it were not print
problems at all, and the reason is worth writing down because it is invisible
from the model: the printer's CLUMPING DETECTOR stopped both of them, on layer
10 each time.

That routine fires on layers 3, 10 and 19 only (``wrapping_detection_gcode``,
"P2S 20250822 clumping"). The bar ends at layer 8, so layer 9 is where this
badge stops being one connected shape and becomes seven separate islands --
and layer 10 asks a detector that looks for material clumped on the head to
interpret exactly that transition, two layers after it happens. Both prints
died there, one reported as spaghetti and one as "filament stuck on the print
head", with the part firmly stuck down, the nozzle clear, and the quality
perfect up to the stop.

So: turn the clumping check OFF for this part. Any design that goes from one
island to many partway up will provoke it, which covers raised lettering on a
rail generally. It is not a reason to change the geometry. Note also that the
gcode already carries ``M1015.3 S0 ;disable clog detect`` and the exported job
had ``enable_wrapping_detection = 0`` -- the printer-side setting overrides
both, so it has to be switched off on the printer.

The successful print was a GUI slice: no brim (``auto_brim`` chose none), 2
walls, 15% infill, no ironing, fan off, 23 min, 5.1g. Worth knowing that the
brim this module asks for has still never been printed, and that the part held
to the plate without one three times.

FLAT PANEL, BUT NOT NECESSARILY. ``sag`` cuts a cylindrical hollow into the back
for a panel that crowns across the badge's length. It defaults to zero because
this one is going on a flat tailgate. If the finished badge rocks when you offer
it up, lay a straightedge across the mounting spot, measure the gap under the
middle in millimetres, and put that number in ``sag`` -- the radius is derived
from it. A flat back on a crowned panel touches at the ends only, and tape that
never contacts in the middle lets go at speed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import median

from build123d import (
    Align,
    Axis,
    Cylinder,
    Kind,
    Part,
    Pos,
    Rectangle,
    Rotation,
    Sketch,
    Text,
    extrude,
    offset,
    scale,
)

ASA_DENSITY = 1.07
"""g/cm^3 for Bambu ASA, for the mass estimate that decides whether the tape
choice is even worth thinking about."""

MIN_WALL = 1.2
"""mm. Below this a feature is fewer than three extrusions wide on a 0.4mm
nozzle and stops behaving like a solid."""

REFERENCE_SIZE = 100.0
"""Glyphs are laid out once at this size to measure the word, then rebuilt at the
size that makes the word come out ``length`` long. Fonts do not expose a
width-per-string, so measuring is the only way to solve for it."""

LAYER = 0.2
"""mm. Step height of the bevel staircase, set to the print's layer height so
the model's steps and the print's steps are the same steps."""

MAX_BEVEL = 0.8
"""mm. Outward offsets on these glyph outlines succeed at 0.8 and fail at 1.0
with "Unexpected result type" -- the same ceiling the halo version of this part
ran into. Measured, not chosen: a bigger bevel needs a different construction,
not a bigger number."""

SAMPLE_BAND = 0.15
"""mm. Height of the horizontal slice used to measure how wide the word is at a
given height. Thin enough to resolve a serif bracket, thick enough that the
intersection has area worth dividing."""

SERIF_PLATEAU = 1.5
"""How much wider than the bare stems the word may still be and count as "clear
of the serifs". Serifs on a Clarendon are bracketed, so the width does not drop
off a cliff -- it falls steeply while the slab ends, then creeps as the bracket
blends in. This threshold picks the end of the steep part."""

RAIL_HEIGHT_FRACTION = 0.13
"""Rail height as a fraction of cap height, off the reference emblem: the bar
is a little thicker than the serif slabs and sits in the bottom quarter of the
letters."""


@dataclass
class Params:
    """Everything the badge is. Defaults are a 150mm classic fender emblem."""

    text: str = "MUSTANG"
    length: float = 150.0
    """Overall width of the word, end to end, bevel included. The dimension you
    measure off the car, and the one everything else is solved against."""

    font: str = "Superclarendon"
    """Family name. Superclarendon is macOS's Clarendon, and Clarendon Wide is
    the closest published identification of the classic emblem lettering -- see
    the module docstring for how firm that is. ``Rockwell`` is the alternative
    already installed. Anything genuinely custom goes through ``font_path``."""

    font_path: str | None = None
    """Path to a .ttf/.otf. Preferred over ``font``, because a family name that
    is not installed is silently substituted rather than refused -- see
    ``check_font``."""

    stretch: float = 1.0
    """Horizontal scale applied after sizing. Badge lettering is usually wider
    than a text face at the same cap height; this dials that in without needing
    a second font."""

    thickness: float = 4.0
    """Letter thickness, bond face to letter face."""

    bevel: float = 0.6
    """Width of the flare at the base of the letters -- the chrome-emblem bevel.
    Zero gives slab-sided letters. Capped at ``MAX_BEVEL``. The bar is not
    bevelled; ``_stack`` says why."""

    rail: float = 1.6
    """Thickness of the bar joining the letters. Stands proud of the panel, so it
    is what the badge looks like it is floating on."""

    rail_lift: float | None = None
    """Height of the underside of the rail above the letter baseline. ``None``
    measures it: the rail lands just clear of the serifs, as on the original.
    Set a number to override."""

    rail_height: float | None = None
    """How tall the bar is. ``None`` takes ``RAIL_HEIGHT_FRACTION`` of the cap
    height, so it scales with the badge."""

    sag: float = 0.0
    """Measured hollow needed in the back for a crowned panel. Zero is flat."""

    tape: float = 1.1
    """Automotive attachment tape thickness, for reporting only."""

    def validate(self) -> None:
        if self.length <= 0:
            raise ValueError("length must be positive")
        if not self.text.strip():
            raise ValueError("there is no badge without a word to put on it")
        if self.rail >= self.thickness:
            raise ValueError(
                f"rail ({self.rail}mm) must be thinner than the letters "
                f"({self.thickness}mm), or the letters stop standing proud of it "
                f"and the badge is a flat plaque"
            )
        for name, value in (("thickness", self.thickness), ("rail", self.rail)):
            if value < MIN_WALL:
                raise ValueError(
                    f"{name} is {value}mm, under the {MIN_WALL}mm that prints as a "
                    f"solid on a 0.4mm nozzle"
                )
        if self.bevel < 0:
            raise ValueError("bevel is a width, so it cannot be negative")
        if self.bevel > MAX_BEVEL:
            raise ValueError(
                f"bevel {self.bevel}mm is over the {MAX_BEVEL}mm where outward "
                f"offsets on glyph outlines start failing outright. The limit is "
                f"measured, so raising it just moves the crash into build()"
            )
        if self.rail_lift is not None and self.rail_lift < 0:
            raise ValueError("rail_lift is a height above the baseline")
        if self.rail_height is not None and self.rail_height <= 0:
            raise ValueError("rail_height must be positive")
        if self.sag < 0:
            raise ValueError("sag is a hollow depth, so it cannot be negative")
        if self.sag >= self.rail:
            raise ValueError(
                f"sag ({self.sag}mm) is at least the rail thickness ({self.rail}mm), "
                f"so the hollow would cut the rail in half at the ends. Thicken the "
                f"rail or split the badge into shorter pieces"
            )

    @property
    def radius(self) -> float | None:
        """Panel radius implied by the measured sag, across ``length``.

        The sagitta relation, R = c^2/(8s) + s/2, solved for radius. Reported
        rather than asked for, because nobody can measure a radius on a car and
        everybody can measure a gap under a straightedge.
        """
        if self.sag <= 0:
            return None
        return self.length**2 / (8 * self.sag) + self.sag / 2


def check_font(params: Params) -> None:
    """Refuse a font name the renderer is only pretending to have.

    OCCT does not fail on a missing family. It prints a warning to stderr,
    substitutes Arial and carries on, so the badge comes out in the wrong
    letterforms and looks plausible enough to slice. That is the same failure
    shape as a config the slicer silently ignores, and it gets the same
    treatment: compare what the named font renders against what Arial renders,
    and if they are identical to the micron, it is Arial.
    """
    if params.font_path or "arial" in params.font.lower():
        return
    mine = Text(params.text, REFERENCE_SIZE, font=params.font).bounding_box().size
    arial = Text(params.text, REFERENCE_SIZE, font="Arial").bounding_box().size
    if abs(mine.X - arial.X) < 1e-9 and abs(mine.Y - arial.Y) < 1e-9:
        raise ValueError(
            f"font {params.font!r} is not installed -- the renderer quietly "
            f"substituted Arial, so this badge would print in the wrong "
            f"letterforms. Install it, or set font_path to a .ttf/.otf file."
        )


def _word(params: Params) -> Sketch:
    """The letters at final size, centred on x, sitting on the baseline y=0.

    Solved to ``length`` minus the bevel it will grow on each end, since the
    flare is added outside this outline.
    """
    kwargs = (
        {"font_path": params.font_path} if params.font_path else {"font": params.font}
    )
    probe = Text(params.text, REFERENCE_SIZE, **kwargs)
    measured = probe.bounding_box().size.X * params.stretch
    if measured <= 0:
        raise ValueError(f"{params.text!r} rendered to nothing in this font")

    target = params.length - 2 * params.bevel
    if target <= 0:
        raise ValueError(
            f"a {params.length}mm badge is narrower than the {params.bevel}mm bevel "
            f"on each end leaves room for"
        )
    word = Text(params.text, REFERENCE_SIZE * target / measured, **kwargs)
    if params.stretch != 1.0:
        word = scale(word, (params.stretch, 1.0, 1.0))
    box = word.bounding_box()
    return Pos(-box.center().X, -box.min.Y) * word


def _width_at(word: Sketch, y: float) -> float:
    """How much of a horizontal line at height ``y`` is inside the letters.

    Summed across the whole word, so it counts every stem at once. The serifs
    make this large near the baseline and it settles to the bare stem width
    above them, which is the signal ``serif_top`` reads.
    """
    box = word.bounding_box()
    band = Pos(box.center().X, y + SAMPLE_BAND / 2) * Rectangle(
        box.size.X + 10, SAMPLE_BAND
    )
    strip = word & band
    return strip.area / SAMPLE_BAND if strip.area else 0.0


def serif_top(word: Sketch) -> float:
    """Height above the baseline where the serifs have finished flaring.

    Measured, because it is a property of the font and the whole point of
    ``font_path`` is that the font can change. Takes the bare stem width from
    the middle of the letters, then walks up from the baseline until the word is
    no more than ``SERIF_PLATEAU`` times that -- the point where the serif slabs
    have ended and only the bracket is still blending in.

    Read from the top of the search range downwards, taking the HIGHEST slice
    still wider than the threshold. Scanning up from the baseline and stopping
    at the first narrow slice looks equivalent and is not: the undersides of the
    serifs are slightly rounded, so the slice sitting right on the baseline
    catches only the tips and is the narrowest one on the whole letter. That
    reading puts the rail at the baseline, which is the bug this is written to
    avoid.

    Clamped either side: a font whose profile does not look like this at all
    should give a rail in a sane place rather than one at the baseline or half
    way up the caps.
    """
    cap = word.bounding_box().size.Y
    stems = median(_width_at(word, cap * f) for f in (0.30, 0.40, 0.50, 0.60))
    if stems <= 0:
        return cap * 0.10
    limit = stems * SERIF_PLATEAU
    steps = 40
    rise = cap * 0.25 / steps
    for i in range(steps, -1, -1):
        if _width_at(word, rise * i) > limit:
            return min(max(rise * (i + 1), cap * 0.04), cap * 0.20)
    return cap * 0.10


def _rail(params: Params, word: Sketch) -> Sketch:
    """The bar, crossing the letters above their serifs."""
    box = word.bounding_box()
    lift = params.rail_lift if params.rail_lift is not None else serif_top(word)
    height = params.rail_height or box.size.Y * RAIL_HEIGHT_FRACTION
    bar = Rectangle(box.size.X, height, align=(Align.CENTER, Align.MIN))
    return Pos(box.center().X, lift) * bar


def _grow(sketch: Sketch, amount: float) -> Sketch:
    """Outward offset with arc joins.

    ``Kind.ARC`` is the join style that survives letterforms, and the only one:
    the same call with ``Kind.INTERSECTION`` fails on this word at every amount
    tried, in both directions. See the module docstring.
    """
    if amount <= 1e-9:
        return sketch
    return offset(sketch, amount, kind=Kind.ARC)


def _joined(letters: Sketch, bar: Sketch) -> Sketch:
    """The badge silhouette: letters plus the bar that ties them together.

    Insists on a single face, which is two checks in one. It is the design
    requirement -- a badge that comes off the plate in pieces is not a badge,
    and the counter of an A is an inner wire rather than a second face, so one
    face is exactly right. It also catches a specific OCCT failure: fusing a
    grown glyph sketch to a grown rectangle sometimes returns a partial result,
    seven or twelve faces instead of one, with no error raised. Extruding that
    gives an invalid slab, and fusing the invalid slab to a good one annihilates
    both -- which showed up here as a badge silently missing its bottom
    millimetre, reported by every downstream measurement as if it were fine.
    """
    joined = letters + bar
    if len(joined.faces()) != 1:
        raise ValueError(
            f"the badge silhouette came out as {len(joined.faces())} regions "
            f"instead of one. Either the bar is not reaching every glyph -- check "
            f"rail_lift and rail_height against the cap height -- or OCCT's 2D "
            f"fuse has failed on this font, which it does silently and which is "
            f"why this is checked rather than assumed"
        )
    return joined


def _stack(params: Params, word: Sketch, bar: Sketch) -> Part:
    """The badge as a stack of prisms, letters bevelled, bar square-sided.

    The letters' top face is the true glyph outline and each step below it is
    one ``LAYER`` wider -- see the module docstring for why the bevel is built
    as a downward flare rather than a chamfer, a taper or a loft.

    THE BAR IS DELIBERATELY NOT BEVELLED. Bevelling it would mean its outline
    changing with height, and every height would need its own fuse of a grown
    glyph sketch to a grown rectangle. That fuse is not dependable: on DIN
    Condensed it returns a partial result at two of the four offsets a 0.6mm
    bevel visits, and the failure is not numerical -- nudging the offset by
    microns does not move it, because it depends on where the bar's edge lands
    relative to the font's horizontal features. Any bar whose edge sweeps
    through that band will find it in some font. A square-sided bar needs one
    fuse, at one offset, computed once. It costs a bevel on a 1.6mm bar that
    sits mostly behind the letters, and buys a part that builds in any font.
    """
    bevel = params.bevel
    steps = max(1, round(bevel / LAYER)) if bevel > 0 else 0
    rise = bevel / steps if steps else 0.0
    shoulder = params.thickness - bevel
    if shoulder <= 0:
        raise ValueError(
            f"a {bevel}mm bevel does not fit under letters {params.thickness}mm thick"
        )

    silhouette = _joined(_grow(word, bevel), bar)
    heights = sorted(
        {0.0, params.rail, shoulder} | {shoulder + k * rise for k in range(steps + 1)}
    )

    solid: Part | None = None
    for low, high in zip(heights, heights[1:]):
        if high - low < 1e-6:
            continue
        if high <= params.rail + 1e-9:
            section = silhouette  # the bar is still with us down here
        else:
            # Offset that puts the slab's TOP on the ideal bevel, so the topmost
            # slab carries the untouched outline.
            section = _grow(word, min(bevel, max(0.0, params.thickness - high)))
        slab = Pos(0, 0, low) * extrude(section, amount=high - low)
        solid = slab if solid is None else solid + slab

    assert solid is not None  # heights always spans 0 to thickness
    return solid


def _hollow(params: Params, part: Part) -> Part:
    """Cut a cylindrical valley into the back for a crowned panel.

    Axis across the badge, sized so the arc meets the back plane at the two ends
    and reaches ``sag`` into the material at the middle -- which is exactly the
    gap a straightedge measures.
    """
    if params.sag <= 0:
        return part
    radius = params.radius
    assert radius is not None
    span = part.bounding_box().size.Y * 4
    cutter = Rotation(90, 0, 0) * Cylinder(radius=radius, height=span)
    return part - Pos(0, 0, params.sag - radius) * cutter


def build(params: Params | None = None) -> Part:
    """The badge, bond face down on z=0, letters growing in +Z."""
    params = params or Params()
    params.validate()
    check_font(params)

    word = _word(params)
    badge = _stack(params, word, _rail(params, word))

    # The boolean that assembles the slabs can fail by dropping one instead of
    # by raising, and a badge missing its bottom millimetre measures fine on
    # every other number in report(). So check the thing that would be wrong.
    box = badge.bounding_box()
    if abs(box.min.Z) > 1e-6 or abs(box.max.Z - params.thickness) > 1e-6:
        raise ValueError(
            f"the badge spans z {box.min.Z:.3f}..{box.max.Z:.3f} instead of "
            f"0..{params.thickness}. A slab was lost assembling the stack"
        )

    badge = _hollow(params, badge)
    if len(badge.solids()) != 1 or not badge.is_valid:
        raise ValueError(
            f"the badge came out as {len(badge.solids())} piece(s), valid="
            f"{badge.is_valid}. The rail is not crossing every letter -- check "
            f"rail_lift and rail_height against the cap height, or whether the "
            f"font has a glyph that does not reach down to the rail"
        )
    return badge


def plated(params: Params | None = None, part: Part | None = None) -> Part:
    """Print orientation: bond face on the plate, letters up.

    Already the modelled orientation, so this only guarantees it is sitting on
    z=0 rather than assuming the hollow left it there.
    """
    params = params or Params()
    part = build(params) if part is None else part
    return Pos(0, 0, -part.bounding_box().min.Z) * part


def report(params: Params | None = None, part: Part | None = None) -> dict:
    """Numbers worth knowing before committing filament to it."""
    params = params or Params()
    part = build(params) if part is None else part
    box = part.bounding_box()
    volume = part.volume / 1000.0  # mm^3 -> cm^3
    # A zero-width window here silently returns nothing: the back face lands on
    # z=0 to within a rounding error, not on it, so the band has to have width.
    back = box.min.Z
    bond = sum(
        f.area
        for f in part.faces().filter_by_position(Axis.Z, back - 0.01, back + 0.01)
    )
    # The letters' own cap height, with the flare taken back off top and bottom,
    # because "how tall is the lettering" is a question about the typeface.
    caps = box.size.Y - 2 * params.bevel
    return {
        "length": box.size.X,
        "cap_height": caps,
        "overall_height": box.size.Y,
        "thickness": box.size.Z,
        "aspect": box.size.X / caps if caps else 0.0,
        "volume_cm3": volume,
        "mass_g": volume * ASA_DENSITY,
        "bond_area_mm2": bond,
        "panel_radius": params.radius,
    }


CLASSIC_ASPECT = (6.5, 8.0)
"""Length-to-cap-height range a classic fender emblem falls in -- roughly 20mm
letters on a 150mm badge. Approximate, taken off reference images rather than a
drawing, and useful for one thing only: telling you at a glance whether a
candidate font has badge proportions or text proportions. A normal typeface set
to 150mm wide comes out far too tall, which is the single most obvious way a
printed badge reads as homemade."""


def proportion_note(info: dict) -> str | None:
    """Plain-language warning when the lettering is the wrong shape."""
    low, high = CLASSIC_ASPECT
    aspect = info["aspect"]
    if low <= aspect <= high:
        return None
    direction = "too tall and narrow" if aspect < low else "too wide and squat"
    return (
        f"letterforms look {direction} for a fender emblem: {aspect:.1f}:1 "
        f"length-to-cap against roughly {low:g}-{high:g}:1 on the real thing. "
        f"A different font is the right fix; stretch= is the crude one."
    )


def variant(params: Params | None = None, **changes) -> Params:
    return replace(params or Params(), **changes)
