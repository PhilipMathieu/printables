"""Cameras the window shroud is cut for, and where their light goes.

Two kinds of number live here, and they are kept apart on purpose.

SOURCED numbers come from the maker's own spec page: the lens, the field of
view, the night-vision wavelength and the overall size. For the Tapo C100 that
is TP-Link's US page for hardware v5.60.

ESTIMATED numbers are everything about the head and its face: how big the head
is, where the lens sits, how far its bezel stands proud, where the light sensor
and status light are. TP-Link publish none of it. These were measured off
TP-Link's own three-quarter product photo, with the pixel scale taken from the
spec's 98.9mm overall height and the face panel's foreshortening backed out by
assuming it is square, which a published slim-shell remix of the C100's
internals (a 52.6mm square board) agrees with. Expect them to be within a
couple of millimetres, not tenths. Every one is a field, so measure the real
camera with calipers and pass the numbers in rather than trusting these.

The IR LEDs, light sensor and lower-left port sit on a small board behind the
face panel, with a hole for the lens. MEASURED on a real board with calipers:
the lens hole is 17mm across, the two surface-mount IR LEDs are 23mm apart
centre to centre with their bottom edges level with the top of the hole, and
the light sensor's dome is centred 6.5mm below the bottom of the hole. The LED
package is taken as 3.5mm square, the common PLCC size, which puts their
centres 1.75mm above the hole's top edge. The black grommet low on the left was
not measured; it comes from a photo of the same board, scaled by the measured
hole. All of these assume the lens is centred in the board's hole. The shroud
keeps the LEDs outside its lens hood, which only works while they are further
from the lens than the hood's outside; see parts.window_shroud.

THE FIELD OF VIEW IS A FISHEYE'S. TP-Link give three angles, horizontal,
vertical and diagonal, and for the C100 the diagonal is wider than a rectilinear
lens with those horizontal and vertical angles could have: 117 degrees against
about 103. So the picture's corners look further out than the edges suggest,
and a hood sized off the horizontal and vertical angles alone would clip them.
``direction`` maps a point on the sensor to a ray by interpolating the angle
off-axis through the three published values, which reproduces all three
exactly and needs no lens model beyond "angle grows with distance from the
centre of the picture".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Feature:
    """Something on the face that wants to see out: a sensor, a light, a mic.

    ``x`` and ``y`` are from the lens centre, y up, and x to your right as
    you stand facing the camera -- which is the camera's left, so the shroud
    mirrors them.
    """

    name: str
    x: float
    y: float
    diameter: float


@dataclass(frozen=True)
class Camera:
    """One camera head, face forward, in millimetres and degrees."""

    name: str

    # Sourced.
    fov_h: float
    fov_v: float
    fov_d: float
    aspect: tuple[int, int]
    lens_focal: float
    lens_f_number: float
    ir_nm: int

    # Estimated: the head, as a rounded square seen from the front.
    head_w: float
    head_h: float
    head_corner: float
    head_depth: float
    """Front to back, not counting the stalk. Only matters as an upper bound
    on how deep a sleeve can go."""
    panel_w: float
    panel_h: float
    panel_corner: float
    """The black glossy face. Everything outside it is the white rim, which is
    where the shroud is allowed to bear."""

    lens_x: float
    lens_y: float
    """Lens centre from the head's centre, y up, x as you face the camera."""
    bezel_d: float
    bezel_proud: float
    """How far the lens bezel stands in front of the face panel."""
    pupil_depth: float
    """How far behind the bezel's front the lens's entrance pupil sits: where
    the field of view fans out from."""

    features: tuple[Feature, ...] = ()

    @property
    def pupil_diameter(self) -> float:
        return self.lens_focal / self.lens_f_number

    def off_axis(self, r: np.ndarray) -> np.ndarray:
        """Degrees off the optical axis for normalised sensor radius ``r``.

        ``r`` is measured in the same units as ``aspect`` -- so for 16:9, r = 9
        is the top edge, 16 the side edge and 18.36 the corner -- and the
        angle is interpolated through the published half-angles at those three.
        Linear between them, which for a fisheye is close and for these tests
        errs wide, since the curve is convex and a chord sits above it.
        """
        a, b = self.aspect
        knots = [0.0, b, a, math.hypot(a, b)]
        if b > a:
            knots = [0.0, a, b, math.hypot(a, b)]
            halves = [0.0, self.fov_h / 2, self.fov_v / 2, self.fov_d / 2]
        else:
            halves = [0.0, self.fov_v / 2, self.fov_h / 2, self.fov_d / 2]
        return np.interp(r, knots, halves)

    def direction(self, u: np.ndarray, v: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Unit rays out of the lens for sensor points ``u, v`` in [-1, 1].

        u = +1 is the right edge of the picture as the camera sees it, v = +1
        the top. The camera looks along -Z (towards the glass, which the shroud
        prints on), x to the camera's right and y up, so these come back in
        the shroud's own coordinates. ``margin`` widens every ray by that many
        degrees, for sizing a hood with something in hand.
        """
        a, b = self.aspect
        x, y = np.asarray(u, float) * a, np.asarray(v, float) * b
        theta = np.radians(self.off_axis(np.hypot(x, y)) + margin)
        phi = np.arctan2(y, x)
        s = np.sin(theta)
        return np.stack([s * np.cos(phi), s * np.sin(phi), -np.cos(theta)], axis=-1)

    def border(self, n: int = 24) -> tuple[np.ndarray, np.ndarray]:
        """Sensor coordinates walking the picture's edge, corners included."""
        t = np.linspace(-1, 1, n, endpoint=False)
        one = np.ones_like(t)
        u = np.concatenate([t, one, -t, -one])
        v = np.concatenate([-one, t, one, -t])
        return u, v

    def footprint(self, distance: float, margin: float = 0.0) -> tuple[float, float]:
        """Half-width and half-height on a plane ``distance`` in front of the
        pupil that the whole picture lands inside, pupil included."""
        u, v = self.border(64)
        d = self.direction(u, v, margin)
        reach = distance / -d[:, 2]
        pad = self.pupil_diameter / 2
        return (float(np.max(np.abs(d[:, 0] * reach))) + pad,
                float(np.max(np.abs(d[:, 1] * reach))) + pad)


TAPO_C100 = Camera(
    name="Tapo C100",
    # Sourced: TP-Link US spec page, Tapo C100 v5.60.
    fov_h=98.8,
    fov_v=53.5,
    fov_d=117.0,
    aspect=(16, 9),  # 1920 x 1080
    lens_focal=3.15,
    lens_f_number=2.0,
    ir_nm=850,
    # Estimated off the product photo; see the module note.
    head_w=56.6,
    head_h=56.6,
    head_corner=12.0,
    head_depth=22.0,
    panel_w=52.7,
    panel_h=52.7,
    panel_corner=10.0,
    lens_x=0.0,
    lens_y=0.0,
    bezel_d=17.0,
    bezel_proud=3.0,
    pupil_depth=2.0,
    features=(
        # Measured on the IR board; see the module note. The LEDs' diameter is
        # the 3.5mm square package's diagonal, so the check clears its corners.
        Feature("left IR LED", -11.5, 8.5 + 1.75, 3.5 * math.sqrt(2)),
        Feature("right IR LED", 11.5, 8.5 + 1.75, 3.5 * math.sqrt(2)),
        Feature("light sensor", 0.0, -(8.5 + 6.5), 4.0),
        # Off the board photo, scaled by the measured 17mm hole.
        Feature("lower-left port", -11.8, -13.4, 4.0),
        # Off the product photo: the green light on the face below the sensor.
        Feature("status light", 0.0, -20.0, 2.0),
    ),
)

CATALOGUE: dict[str, Camera] = {"c100": TAPO_C100}


def named(key: str) -> Camera:
    try:
        return CATALOGUE[key.lower()]
    except KeyError:
        raise KeyError(
            f"no camera called {key!r}; there is {', '.join(CATALOGUE)}"
        ) from None
