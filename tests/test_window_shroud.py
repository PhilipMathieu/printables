"""Geometric assertions for the window shroud.

What decides whether it works is optical and none of it shows in a render: that
nothing of the shroud is in the tilted picture, that light the glass throws
back from the LEDs has no way round the partition to the lens, that the face's
sensors look out, and that the head sits on the lip where it was designed to.
Each is measured off the solid -- rays fired through the mesh, probes dropped
into it -- rather than trusting the arithmetic that built it.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from build123d import Circle, Pos, RectangleRounded, extrude
from shapely.geometry import Point

from geom import cameras
from parts.window_shroud import (
    MAX_LEAN,
    Params,
    _face_plane,
    build,
    coupon,
    lens_chamber,
)
from tools.render import mesh

C100 = cameras.TAPO_C100


@pytest.fixture(scope="module")
def shroud():
    return build(Params())


@pytest.fixture(scope="module")
def shroud_mesh(shroud):
    return mesh(shroud, tol=0.02)


def _hits(points, tris, origins, directions, reach):
    """Möller–Trumbore, every ray against every triangle. True where a ray
    meets the mesh within ``reach`` of its origin."""
    tri = points[tris]
    v0, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    hit = np.zeros(len(origins), bool)
    for i, (o, d, r) in enumerate(zip(origins, directions, reach)):
        p = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, p)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1 / np.where(ok, det, 1), 0)
        s = o - v0
        u = np.einsum("ij,ij->i", s, p) * inv
        q = np.cross(s, e1)
        v = (q @ d) * inv
        t = np.einsum("ij,ij->i", e2, q) * inv
        hit[i] = np.any(ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6) & (t < r))
    return hit


def _to_glass(origins, dirs):
    return -origins[:, 2] / dirs[:, 2]


def _overlap(part, probe) -> float:
    both = part & probe
    return 0.0 if both is None else both.volume


def _head(params: Params, back: float):
    """The head as a block on the face plane, its front ``back`` behind it."""
    c = params.camera
    return _face_plane(params, back) * extrude(
        RectangleRounded(c.head_w, c.head_h, c.head_corner), amount=params.sleeve + 5)


def _lean(part) -> float:
    points, tris = mesh(part, tol=0.02)
    tri = points[tris]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    size = np.linalg.norm(n, axis=1)
    keep = size > 1e-9
    n, tri = n[keep] / size[keep, None], tri[keep]
    on_plate = tri[:, :, 2].max(axis=1) < 1e-6
    return float(np.degrees(np.arcsin(np.clip(-n[~on_plate, 2], -1, 1))).max())


# --- the camera ------------------------------------------------------------


def test_the_fov_model_reproduces_all_three_published_angles():
    """The one sourced fact the chamber is sized off, so check it round-trips:
    the edge midpoints and the corner come back at exactly half of TP-Link's
    horizontal, vertical and diagonal angles."""
    for (u, v), full in (((1, 0), C100.fov_h), ((0, 1), C100.fov_v),
                         ((1, 1), C100.fov_d)):
        d = C100.direction(np.array(u, float), np.array(v, float))
        assert math.degrees(math.acos(-d[2])) == pytest.approx(full / 2)


def test_the_c100_is_wider_on_the_diagonal_than_a_rectilinear_lens():
    """Which is why the chamber is sized off rays and not off two angles."""
    rectilinear = 2 * math.degrees(math.atan(math.hypot(
        math.tan(math.radians(C100.fov_h / 2)), math.tan(math.radians(C100.fov_v / 2)))))
    assert C100.fov_d > rectilinear + 10


def test_an_unknown_camera_says_what_there_is():
    with pytest.raises(KeyError, match="c100"):
        cameras.named("c999")


# --- the tilt --------------------------------------------------------------


def test_the_camera_looks_down_by_the_tilt():
    p = Params()
    down = math.degrees(math.atan2(-p.forward[1], -p.forward[2]))
    assert down == pytest.approx(25.0)
    assert np.dot(p.up, p.forward) == pytest.approx(0)


def test_the_head_comes_as_close_to_the_glass_as_its_top_edge_allows():
    """Untilted the lens could sit 4mm off the pane; tilted, the head's top
    edge reaches the glass first, so the sleeve's top edge is exactly the
    edge gap off it and the lens lands back about 13mm."""
    p = Params()
    _, oh, _ = p.outer_size
    top = p.on_face([[0.0, oh / 2]])[0]
    assert top[2] == pytest.approx(p.edge_gap)
    assert 12 < p.pupil_height < 15
    assert Params(tilt=0).pupil_height == pytest.approx(4.0)


# --- the part --------------------------------------------------------------


def test_it_is_one_valid_solid(shroud):
    assert shroud.is_valid
    assert len(shroud.solids()) == 1


@pytest.mark.parametrize("tilt", [25.0, 0.0], ids=["tilted", "square"])
def test_nothing_in_the_part_overhangs_45_degrees(tilt):
    """Printed glass face down with no support, so every downward face but the
    plate itself has to be one FDM holds on its own: the sleeve leans by the
    tilt and the lens chamber's sides by what the footprint asks."""
    part = build(Params()) if tilt == 25.0 else build(Params(tilt=tilt))
    assert _lean(part) < 45
    assert Params(tilt=tilt).chamber_lean <= MAX_LEAN


def test_both_chambers_and_the_skirt_come_off_the_plate(shroud):
    """All three seal against the glass and the plate face is the flat one.
    The picture's centre and the LEDs' patch of glass are open at the plate;
    the skirt's wall and the partition between are plastic."""
    p = Params()
    assert shroud.bounding_box().min.Z == pytest.approx(0, abs=1e-6)
    axis = p.pupil + p.forward * (p.pupil_height / -p.forward[2])
    open_at = [(axis[0], axis[1]), (11.5, 15.0), (-11.5, 15.0)]
    for x, y in open_at:
        assert not shroud.is_inside((x, y, 0.1)), (x, y)
    fp = p.footprint
    lo, hi = fp.bounds[1], fp.bounds[3]
    for x, y in [(0.0, lo - p.wall / 2), (fp.bounds[2] + p.wall / 2, -15.0),
                 (0.0, hi + p.partition / 2)]:
        assert shroud.is_inside((x, y, 0.1)), (x, y)


def test_nothing_of_the_shroud_is_in_the_picture(shroud_mesh):
    """Fire the picture's edge rays, from points round the rim of the lens's
    pupil, out to the glass. None may touch the shroud. Done at the published
    field of view, so the margin is what is in hand and is not relied on."""
    p = Params()
    points, tris = shroud_mesh
    o, d = p.rays(16, margin=0.0)
    assert not _hits(points, tris, o, d, _to_glass(o, d)).any()


def test_the_ray_test_would_catch_the_shroud_in_the_picture():
    """The test above is only worth something if it can fail: size the lens
    chamber for a picture ten degrees narrower than the real one and the real
    one has to see its walls."""
    small = Params(margin=-10.0, pad=0.0)
    points, tris = mesh(build(small), tol=0.05)
    o, d = small.rays(8, margin=0.0)
    assert _hits(points, tris, o, d, _to_glass(o, d)).mean() > 0.3


def test_light_off_the_glass_in_the_led_chamber_cannot_reach_the_lens(shroud_mesh):
    """Anything the pane throws back inside the LED chamber's patch of glass
    has to cross the partition to get to the lens. Fire from a grid over that
    patch at the pupil: every ray meets plastic first."""
    p = Params()
    points, tris = shroud_mesh
    grown = lens_chamber(p, p.partition)
    leds = p.led_face
    targets = []
    for x in np.arange(-24, 25, 4.0):
        for y in np.arange(-5, 25, 2.0):
            if grown.is_inside((x, y, 0.3)):
                continue
            # The LED chamber's glass is straight under its face.
            b = y / p.up[1]
            if leds.contains(Point(x, b)):
                targets.append((x, y, 0.3))
    targets = np.array(targets)
    assert len(targets) > 20
    d = p.pupil - targets
    reach = np.linalg.norm(d, axis=1)
    d /= reach[:, None]
    assert _hits(points, tris, targets, d, reach - 0.5).all()


def test_the_head_sits_on_the_lip(shroud):
    """Against the face plane the head is clear of everything; half a
    millimetre further in, along the tilted axis, and it is in the lip. So it
    can only go in so far, at the angle it was designed for."""
    p = Params()
    assert _overlap(shroud, _head(p, 1e-3)) == pytest.approx(0, abs=1e-4)
    assert _overlap(shroud, _head(p, -0.5)) > 1


def test_the_bezel_stands_clear_and_the_partition_closes_on_the_face(shroud):
    """The bezel is free inside the lens chamber, and just above it the
    partition comes all the way to the face, so the LEDs' chamber is shut off
    from the lens there."""
    p = Params()
    c = p.camera
    lx, ly = p.lens2d
    bezel = _face_plane(p, -c.bezel_proud) * Pos(lx, ly, 0) * extrude(
        Circle(c.bezel_d / 2), amount=c.bezel_proud)
    assert _overlap(shroud, bezel) == pytest.approx(0, abs=1e-4)
    seam = ly + c.bezel_d / 2 + p.hood_clearance + p.partition / 2
    x, y, z = p.on_face([[lx, seam]], -0.3)[0]
    assert shroud.is_inside((x, y, z))


def test_the_ir_leds_clear_the_partition(shroud):
    """Philip's caliper readings put each LED package's near edge just over a
    millimetre above where the partition meets the face. Keep at least 1mm,
    and check the face in front of each LED is open."""
    p = Params()
    walled = p.lens_face_walled
    leds = [f for f in p.features if "IR LED" in f[0]]
    assert len(leds) == 2
    for name, x, y, dia in leds:
        assert walled.exterior.distance(Point(x, y)) - dia / 2 >= 1.0, name
        assert not walled.contains(Point(x, y)), name
        for a in np.linspace(0, 2 * math.pi, 8, endpoint=False):
            r = dia / 2
            q = p.on_face([[x + r * math.cos(a), y + r * math.sin(a)]], -0.5)[0]
            assert not shroud.is_inside(tuple(q)), name


def test_the_top_of_each_led_beam_reaches_the_glass(shroud_mesh):
    """The LEDs' axes reach the glass right at the partition, so the lower
    half of each beam lands on it -- that is the light that would haze the
    picture. The axis tipped 15 degrees up, and everything above it, gets out
    to light the yard."""
    p = Params()
    points, tris = shroud_mesh
    origins, dirs = [], []
    for name, x, y, dia in p.features:
        if "IR LED" not in name:
            continue
        o = p.on_face([[x, y]], -0.2)[0]
        for up in (15.0, 30.0):
            a = math.radians(up)
            origins.append(o)
            dirs.append(p.forward * math.cos(a) + p.up * math.sin(a))
    o, d = np.array(origins), np.array(dirs)
    assert not _hits(points, tris, o, d, _to_glass(o, d)).any()


def test_the_faces_sensors_look_out_through_the_lens_chamber(shroud_mesh):
    """The light sensor, port and status light are below the lens, inside its
    chamber, so straight ahead of each is open glass."""
    p = Params()
    points, tris = shroud_mesh
    lens = p.lens_face
    sensors = [f for f in p.features if "IR LED" not in f[0]]
    assert sensors
    o = np.array([p.on_face([[x, y]], -0.2)[0] for _, x, y, _ in sensors])
    d = np.tile(p.forward, (len(o), 1))
    assert all(lens.contains(Point(x, y).buffer(dia / 2)) for _, x, y, dia in sensors)
    assert not _hits(points, tris, o, d, _to_glass(o, d)).any()


def test_the_coupon_is_the_sleeve_band_square_on_the_plate():
    p = Params()
    band = coupon(p)
    bb = band.bounding_box()
    assert band.is_valid
    assert bb.min.Z == pytest.approx(0)
    assert bb.size.Z == pytest.approx(1.2 + p.sleeve)
    assert bb.size.X == pytest.approx(p.outer_size[0])


# --- what validate refuses -------------------------------------------------


def test_a_tilt_whose_picture_would_overhang_the_chamber_is_refused():
    with pytest.raises(ValueError, match="overhang"):
        Params(tilt=30.0).validate()


def test_a_tilt_that_runs_the_picture_along_the_glass_is_refused():
    with pytest.raises(ValueError, match="run along the pane"):
        Params(tilt=36.0).validate()


def test_a_tilt_past_40_is_refused():
    with pytest.raises(ValueError, match="outside 0 to 40"):
        Params(tilt=45.0).validate()


def test_a_standoff_that_puts_the_head_on_the_glass_is_refused():
    with pytest.raises(ValueError, match="at least"):
        Params(standoff=8.0).validate()


def test_a_partition_over_the_ir_leds_is_refused():
    with pytest.raises(ValueError, match="IR LED"):
        Params(hood_clearance=3.0).validate()


def test_a_single_bead_partition_is_refused():
    with pytest.raises(ValueError, match="extrusions"):
        Params(partition=0.4).validate()


def test_a_sleeve_deeper_than_the_head_is_refused():
    with pytest.raises(ValueError, match="back of"):
        Params(sleeve=C100.head_depth).validate()


def test_a_camera_with_moved_leds_gets_its_own_check():
    """Measurements in, refusals out: put the LEDs 2mm lower and the
    partition is on them."""
    low = replace(C100, features=tuple(
        replace(f, y=f.y - 2) if "IR LED" in f.name else f for f in C100.features))
    with pytest.raises(ValueError, match="IR LED"):
        Params(camera=low).validate()
