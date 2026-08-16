"""MUSTANG badge: raised letters on a baseline rail, taped to a flat panel.

The whole part is one sketch extruded twice -- the letters to full thickness,
a thin rail behind them to full rail thickness -- and the interesting decisions
are about how a word becomes a single printable object and which face ends up
against the paint.

WHY THERE IS A RAIL. A word is not a solid. Set MUSTANG in any normal typeface
and you get seven islands with two to five millimetres of air between them, so
"print the badge" means printing seven loose letters and then aligning them on
the car by eye, which nobody wants to do with adhesive that grabs on contact.
The obvious fix is to fatten the letters until they touch -- offset the outline
outward and let the halos merge -- and that is what the first version did. It
does not survive contact with real glyphs: OCCT's 2D offset starts returning
garbage on letterforms somewhere past two and a half millimetres, which is
under the gap this font needs closed. So the letters are joined the way real
emblems join them, with a thin web along the baseline. It reads as a deliberate
underline rather than as a repair, it is a rectangle unioned to a sketch so
there is nothing to go wrong, and it puts the join at the bottom edge where it
is least visible.

WHY IT PRINTS FACE UP. The back is the bond face and it wants to be flat,
smooth and continuous, which is exactly what a first layer on a clean plate is.
Printing letters-down would give prettier letter faces -- plate texture instead
of top-surface -- but it puts seven small unconnected islands of ASA on the
plate for the first two millimetres, and ASA on small isolated footprints is
how you find out what warping looks like. Face up, the first layer is the whole
badge silhouette in one connected piece, every layer above it is the same size
or smaller, and nothing overhangs. The letter faces come out as top surfaces
instead, which is what ironing is for.

WHY THE TAPE IS NOT THE PROBLEM HERE. Unlike anything hung under a desk, a
badge on a tailgate is loaded in shear: gravity runs down the panel, not away
from it. Shear is the mode a pressure-sensitive adhesive is strongest in, and
the badge weighs a few tens of grams over a bond area of a thousand-odd square
millimetres. The load case that actually matters is heat and airflow -- a dark
panel in sun runs far above room temperature and the badge sees moving air at
speed -- which is an argument about tape selection and surface prep, not about
bond area. Use a proper automotive attachment tape, not a general-purpose
double-sided, and clean the paint with isopropyl first.

WHY ASA. Direct sun for years. PLA would sag on a hot tailgate and go chalky in
UV within a season; ASA is the material ABS-grade exterior trim is made from
and holds colour and shape outdoors, which is the entire reason it exists.

WHICH FONT, AND HOW SURE. Ford never released the badge lettering as a
typeface, so there is no correct file to go and download; the question is only
what comes closest. Two different answers depending on era, and they are not
alike: the modern (2005-on) wordmark is a custom bold SANS, while the classic
1965-72 pin-on fender and trunk letters are a SERIF -- a slab in the Clarendon
family. That is worth stating plainly because the intuitive guess is a wide
sans for both, and it is wrong for the classic one.

The best identification available is Clarendon Wide (Medium), from a sign-maker
running font-recognition software against a photo of a real emblem. Independent
suggestions on the same emblem were Times New Roman Bold slightly extended and
Osiris BQ Bold -- different faces, but all serif, all in the same width class,
which is the part worth trusting. macOS ships Superclarendon, a Clarendon, so
the best-supported candidate needs no download and is the default here.
Rockwell is the closest alternative already installed and reads as a more
geometric slab. Treat this as a good match rather than a verified one: the
evidence is a font-ID tool and forum consensus, not a Ford drawing.

Set to 150mm the default lands at 19.7mm caps, 7.6:1 -- which is the other
reason to believe it. A badge is a wide, short object, and a normal-width serif
at badge width falls straight into that proportion where a text sans does not.

FLAT PANEL, BUT NOT NECESSARILY. ``sag`` cuts a cylindrical hollow into the
back for a panel that crowns across the badge's length. It defaults to zero
because this one is going on a flat tailgate. If the finished badge rocks when
you offer it up, lay a straightedge across the mounting spot, measure the gap
under the middle in millimetres, and put that number in ``sag`` -- the radius is
derived from it. A flat back on a crowned panel touches at the ends only, and
tape that never contacts in the middle lets go at speed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from build123d import (
    Align,
    Axis,
    Cylinder,
    Part,
    Plane,
    Pos,
    Rectangle,
    Rotation,
    Sketch,
    Text,
    chamfer,
    extrude,
    scale,
)

ASA_DENSITY = 1.07
"""g/cm^3 for Bambu ASA, for the mass estimate that decides whether the tape
choice is even worth thinking about."""

MIN_WALL = 1.2
"""mm. Below this a feature is fewer than three extrusions wide on a 0.4mm
nozzle and stops behaving like a solid."""

REFERENCE_SIZE = 100.0
"""Glyphs are laid out once at this size to measure the word, then rebuilt at
the size that makes the word come out ``length`` long. Fonts do not expose a
width-per-string, so measuring is the only way to solve for it."""


@dataclass
class Params:
    """Everything the badge is. Defaults are a 150mm classic fender emblem."""

    text: str = "MUSTANG"
    length: float = 150.0
    """Overall width of the word, end to end. The dimension you measure off the
    car, and the one everything else is solved against."""

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
    than a text face at the same cap height; this dials that in without
    needing a second font."""

    thickness: float = 4.0
    """Letter thickness, bond face to letter face."""

    rail: float = 1.6
    """Thickness of the web joining the letters. Sits against the panel, so it
    is what the badge looks like it is floating on."""

    rail_drop: float = 2.5
    """How far the rail hangs below the letter baseline -- the visible part."""

    rail_bite: float = 1.0
    """How far the rail reaches up into the letters. Purely so the union is a
    real overlap and not two shapes touching along a line, which is the
    difference between one solid and a warning."""

    face_chamfer: float = 0.0
    """Bevel on the front edges of the letters, for the faceted look chrome
    emblems have. Off by default: chamfering glyph outlines asks OCCT to
    resolve a lot of tight internal corners and it does not always manage."""

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
        if self.rail_bite <= 0:
            raise ValueError(
                "rail_bite must be positive: a rail that only touches the letters "
                "along a line unions into separate solids, not one badge"
            )
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
    """The letters at final size, centred on the origin, sitting on y=0."""
    kwargs = (
        {"font_path": params.font_path}
        if params.font_path
        else {"font": params.font}
    )
    probe = Text(params.text, REFERENCE_SIZE, **kwargs)
    measured = probe.bounding_box().size.X * params.stretch
    if measured <= 0:
        raise ValueError(f"{params.text!r} rendered to nothing in this font")

    size = REFERENCE_SIZE * params.length / measured
    word = Text(params.text, size, **kwargs)
    if params.stretch != 1.0:
        word = scale(word, (params.stretch, 1.0, 1.0))
    box = word.bounding_box()
    return Pos(-box.center().X, -box.min.Y) * word


def _rail(params: Params, word: Sketch) -> Sketch:
    """A bar along the baseline, overlapping the letters so they fuse."""
    box = word.bounding_box()
    height = params.rail_drop + params.rail_bite
    bar = Rectangle(box.size.X, height, align=(Align.CENTER, Align.MIN))
    return Pos(box.center().X, box.min.Y - params.rail_drop) * bar


def _hollow(params: Params, part: Part) -> Part:
    """Cut a cylindrical valley into the back for a crowned panel.

    Axis across the badge, sized so the arc meets the back plane at the two
    ends and reaches ``sag`` into the material at the middle -- which is
    exactly the gap a straightedge measures.
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
    letters = extrude(word, amount=params.thickness)
    web = extrude(_rail(params, word), amount=params.rail)
    badge = letters + web

    if params.face_chamfer > 0:
        front = badge.faces().filter_by_position(Axis.Z, params.thickness, params.thickness)
        edges = front.edges()
        if edges:
            badge = chamfer(edges, params.face_chamfer)

    badge = _hollow(params, badge)
    if len(badge.solids()) != 1:
        raise ValueError(
            f"the badge came out as {len(badge.solids())} separate pieces. The rail "
            f"is not reaching every letter -- check that rail_drop and rail_bite "
            f"span the baseline, or that the font has no glyph sitting above it"
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
    caps = box.size.Y - params.rail_drop
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
