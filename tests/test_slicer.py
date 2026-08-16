"""Where the model ends up in the 3mf.

Bambu's complaint when a model is off the plate is "One of the plate is empty
or has no object fully inside it", which names neither object nor axis and
sounds like a problem with the plate list. These tests are the cheap version of
finding that out: they read the written coordinates instead of slicing.
"""

from __future__ import annotations

import re
import zipfile

import pytest
from build123d import Box, Pos

from p2s import inventory, profiles, slicer


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """A small box deliberately modelled nowhere near the bed."""
    part = Pos(-400, 250, -12) * Box(30, 20, 10)
    dest = tmp_path_factory.mktemp("slice") / "box.3mf"
    return slicer.write_project(
        part, dest, filament=inventory.default("PLA").preset_for(0.4)
    )


def test_the_model_lands_on_the_bed(project):
    mach = profiles.machine(0.4)
    x0, x1, y0, y1, z0, z1 = slicer.bed_extent(project)
    assert (x0 + x1) / 2 == pytest.approx(mach.bed_x / 2, abs=0.05)
    assert (y0 + y1) / 2 == pytest.approx(mach.bed_y / 2, abs=0.05)
    assert z0 == pytest.approx(0.0, abs=1e-6)


def test_position_is_baked_into_the_coordinates(project):
    """Not into a build-item transform.

    Bambu rejected a model whose transformed bounding box sat well inside the
    bed, so its containment test and its placement disagree about that matrix.
    Baked coordinates remove the disagreement -- and make bed_extent mean
    something, since with a transform it would report the object's local frame.
    """
    model = zipfile.ZipFile(project).read("3D/3dmodel.model").decode()
    assert "transform=" not in model
    assert re.search(r"<item [^>]*/>", model)


def test_a_second_pass_does_not_move_it_again(project):
    """center_on_bed has to be idempotent: write_project already called it, so
    anything that re-centred on top would walk the model off the plate."""
    before = slicer.bed_extent(project)
    slicer.center_on_bed(project)
    assert slicer.bed_extent(project) == pytest.approx(before, abs=1e-6)


def test_verify_accepts_a_centred_model(project):
    slicer.verify_on_bed(project)


def test_verify_rejects_a_model_off_the_plate(project, tmp_path):
    """The check earns its place only if it actually fails on a bad file."""
    off = tmp_path / "off.3mf"
    with zipfile.ZipFile(project) as zin:
        members = {i.filename: zin.read(i.filename) for i in zin.infolist()}
    model = members["3D/3dmodel.model"].decode()
    model = re.sub(
        r'<vertex x="([-0-9.eE]+)"',
        lambda m: f'<vertex x="{float(m.group(1)) + 900:g}"',
        model,
    )
    members["3D/3dmodel.model"] = model.encode()
    with zipfile.ZipFile(off, "w") as zout:
        for name, data in members.items():
            zout.writestr(name, data)

    with pytest.raises(slicer.SliceError, match="off the plate on x"):
        slicer.verify_on_bed(off)
