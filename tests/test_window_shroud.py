"""Geometric assertions for the window shroud.

What decides whether it works is optical and none of it shows in a render: that
nothing of the shroud is in the picture, that the hood seals round the lens
while the face's sensors look out past it, and that the head sits on the lip
at the standoff it was designed for. Each is measured off the solid -- rays
fired through the mesh, probes dropped into it -- rather than trusting the
arithmetic that built it.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from build123d import Box, Cylinder, Pos, RectangleRounded, extrude

from geom import cameras
from parts.window_shroud import MAX_LEAN, Params, build, coupon
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


def _overlap(part, probe) -> float:
    return (part & probe).volume


def _head(params: Params, z: float):
    c = params.camera
    return Pos(0, 0, z) * extrude(
        RectangleRounded(c.head_w, c.head_h, c.head_corner), amount=params.sleeve + 5)


# --- the camera ------------------------------------------------------------


def test_the_fov_model_reproduces_all_three_published_angles():
    """The one sourced fact the hood is sized off, so check it round-trips:
    the edge midpoints and the corner come back at exactly half of TP-Link's
    horizontal, vertical and diagonal angles."""
    for (u, v), full in (((1, 0), C100.fov_h), ((0, 1), C100.fov_v),
                         ((1, 1), C100.fov_d)):
        d = C100.direction(np.array(u, float), np.array(v, float))
        assert math.degrees(math.acos(-d[2])) == pytest.approx(full / 2)


def test_the_c100_is_wider_on_the_diagonal_than_a_rectilinear_lens():
    """Which is why the hood is sized off rays and not off two angles."""
    rectilinear = 2 * math.degrees(math.atan(math.hypot(
        math.tan(math.radians(C100.fov_h / 2)), math.tan(math.radians(C100.fov_v / 2)))))
    assert C100.fov_d > rectilinear + 10


def test_an_unknown_camera_says_what_there_is():
    with pytest.raises(KeyError, match="c100"):
        cameras.named("c999")


# --- the part --------------------------------------------------------------


def test_it_is_one_valid_solid(shroud):
    assert shroud.is_valid
    assert len(shroud.solids()) == 1


@pytest.mark.parametrize("standoff", [5.0, 8.0], ids=["straight hood", "flared hood"])
def test_nothing_in_the_part_overhangs_more_than_45_degrees(standoff):
    """Printed glass face down with no support, so every downward face but
    the plate itself has to be one FDM bridges on its own. Run on a flared
    hood too, which is the one wall that is allowed to lean."""
    params = Params(standoff=standoff)
    points, tris = mesh(build(params), tol=0.02)
    tri = points[tris]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    on_plate = tri[:, :, 2].max(axis=1) < 1e-6
    lean = np.degrees(np.arcsin(np.clip(-n[~on_plate, 2], -1, 1)))
    assert lean.max() < 45
    assert params.hood_lean <= MAX_LEAN


def test_the_hood_and_the_skirt_both_come_off_the_plate(shroud):
    """Both have to seal against the glass, and the plate face is the flat one.
    Probe a thin slice just above the plate at the hood wall and at the skirt
    wall, on each side, and find plastic in all of them."""
    p = Params()
    lx, ly = p.lens
    r = p.hood_base + p.hood_wall / 2
    sw, _, _ = p.sleeve_size
    for sign in (-1, 1):
        assert _overlap(shroud, Pos(lx + sign * r, ly, 0.05) * Box(0.3, 0.3, 0.1)) > 0
        assert _overlap(shroud, Pos(sign * (sw / 2 + p.wall / 2), 0, 0.05)
                        * Box(0.3, 0.3, 0.1)) > 0
    assert shroud.bounding_box().min.Z == pytest.approx(0, abs=1e-6)


def test_nothing_of_the_shroud_is_in_the_picture(shroud_mesh):
    """Fire the picture's edge rays, from points round the rim of the lens's
    pupil, out to the glass. None may touch the shroud. Done at the published
    field of view, so the margin is what is in hand and is not relied on."""
    p = Params()
    points, tris = shroud_mesh
    lx, ly = p.lens
    u, v = C100.border(16)
    d = C100.direction(u, v)
    rim = C100.pupil_diameter / 2
    origins, dirs, reach = [], [], []
    for a in np.linspace(0, 2 * math.pi, 6, endpoint=False):
        o = np.array([lx + rim * math.cos(a), ly + rim * math.sin(a), p.pupil_z])
        for ray in d:
            origins.append(o)
            dirs.append(ray)
            reach.append(p.pupil_z / -ray[2])
    assert not _hits(points, tris, np.array(origins), np.array(dirs), np.array(reach)).any()


def test_the_ray_test_would_catch_a_hood_in_the_picture():
    """The test above is only worth something if it can fail: narrow the hood
    until it is inside the corners of the picture and it has to see it."""
    narrow = replace(C100, bezel_d=8.0)
    p = Params(camera=narrow, hood_clearance=0.5, standoff=4.0, margin=-20.0)
    points, tris = mesh(build(p), tol=0.02)
    lx, ly = p.lens
    d = narrow.direction(np.array([1.0, -1.0]), np.array([1.0, -1.0]))
    o = np.tile([lx, ly, p.pupil_z], (2, 1))
    assert _hits(points, tris, o, d, p.pupil_z / -d[:, 2]).all()


def test_the_head_sits_on_the_lip_at_the_standoff(shroud):
    """At the standoff the head is clear of everything; half a millimetre
    nearer the glass and it is in the lip. So it can only go in so far."""
    p = Params()
    assert _overlap(shroud, _head(p, p.standoff + 1e-3)) == pytest.approx(0, abs=1e-6)
    assert _overlap(shroud, _head(p, p.standoff - 0.5)) > 1


def test_the_hood_closes_on_the_face_round_the_bezel(shroud):
    """The hood's end is in the plane of the face and the bezel stands clear
    inside it, so light from the face can only reach the lens through the
    hood's mouth."""
    p = Params()
    lx, ly = p.lens
    bezel = Pos(lx, ly, p.standoff - C100.bezel_proud / 2) * Cylinder(
        C100.bezel_d / 2, C100.bezel_proud)
    assert _overlap(shroud, bezel) == pytest.approx(0, abs=1e-6)
    ring = Pos(lx, ly, p.standoff - 0.05) * (
        Cylinder(p.hood_base + p.hood_wall - 0.1, 0.1) - Cylinder(p.hood_base + 0.1, 0.1))
    assert _overlap(shroud, ring) == pytest.approx(ring.volume, rel=0.05)


def test_the_faces_sensors_look_straight_out(shroud):
    """The light sensor, status light and mic sit between hood and skirt with
    nothing between them and the glass -- no hood wall, no rib."""
    p = Params()
    for name, x, y, dia in p.features:
        probe = Pos(x, y, p.standoff / 2) * Cylinder(dia / 2 + 0.2, p.standoff)
        assert _overlap(shroud, probe) == pytest.approx(0, abs=1e-6), name


def test_the_coupon_is_the_sleeve_band_on_the_plate():
    p = Params()
    band = coupon(p)
    bb = band.bounding_box()
    assert band.is_valid
    assert bb.min.Z == pytest.approx(0)
    assert bb.size.Z < p.sleeve + 2
    assert bb.size.X == pytest.approx(p.outer_size[0])


# --- what validate refuses -------------------------------------------------


def test_a_standoff_shorter_than_the_bezel_is_refused():
    with pytest.raises(ValueError, match="bezel"):
        build(Params(standoff=C100.bezel_proud))


def test_a_standoff_long_enough_to_flare_the_hood_past_40_degrees_is_refused():
    with pytest.raises(ValueError, match="closer to the glass"):
        build(Params(standoff=16.0))


def test_a_rib_across_a_sensor_is_refused():
    with pytest.raises(ValueError, match="rib"):
        build(Params(ribs=(270.0,)))


def test_a_single_bead_wall_is_refused():
    with pytest.raises(ValueError, match="extrusions"):
        build(Params(hood_wall=0.4))


def test_a_sleeve_deeper_than_the_head_is_refused():
    with pytest.raises(ValueError, match="back of"):
        build(Params(sleeve=C100.head_depth))


def test_ribs_on_the_diagonals_would_stand_in_front_of_the_ir_leds():
    """The C100's IR LEDs are on the upper diagonals, which is why the ribs
    are not."""
    with pytest.raises(ValueError, match="IR LED"):
        build(Params(ribs=(45.0, 135.0)))


def test_the_ir_leds_clear_the_hood_with_room_to_spare():
    """Philip's caliper readings put the near corner of each LED package about
    13mm from the lens; the hood's outside is under 11mm. Keep at least 1.5mm
    between them, so an LED a little off its measured spot still lights the
    yard and not the inside of the hood."""
    p = Params()
    lx, ly = p.lens
    outside = p.hood_base + p.hood_wall
    leds = [f for f in p.features if "IR LED" in f[0]]
    assert len(leds) == 2
    for name, x, y, dia in leds:
        near = math.hypot(x - lx, y - ly) - dia / 2
        assert near - outside >= 1.5, name
