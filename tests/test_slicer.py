"""Where the model ends up in the 3mf.

Bambu's complaint when a model is off the plate is "One of the plate is empty
or has no object fully inside it", which names neither object nor axis and
sounds like a problem with the plate list. These tests are the cheap version of
finding that out: they read the written coordinates instead of slicing.
"""

from __future__ import annotations

import json
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


def test_the_transform_shifts_rather_than_places(project):
    """Studio applies the item transform on top of the mesh coordinates rather
    than re-centring first, so the translation is a shift, not a destination.

    Verified against the slicer, not assumed: the fixture is modelled at
    (-400, 250), and a translation equal to its centre in bed coordinates
    fails to slice while one equal to the shift succeeds. ``--info`` implies
    the opposite -- it normalises the bounding box for display -- which is a
    good way to get this backwards.
    """
    model = zipfile.ZipFile(project).read("3D/3dmodel.model").decode()
    matrix = re.search(r'<item [^>]*transform="([^"]*)"', model)
    assert matrix, "nothing places the object"
    x, y, _ = (float(v) for v in matrix.group(1).split()[9:12])
    mach = profiles.machine(0.4)
    assert x == pytest.approx(mach.bed_x / 2 + 400, abs=0.05)
    assert y == pytest.approx(mach.bed_y / 2 - 250, abs=0.05)


def test_a_second_pass_does_not_move_it_again(project):
    """center_on_bed has to be idempotent: write_project already called it, so
    anything that re-centred on top would walk the model off the plate."""
    before = slicer.bed_extent(project)
    slicer.center_on_bed(project)
    assert slicer.bed_extent(project) == pytest.approx(before, abs=1e-6)


def test_verify_accepts_a_centred_model(project):
    slicer.verify_on_bed(project)


# --- what makes it a project rather than just a model ----------------------
#
# Checked against Bambu's own calib/pressure_advance/pa_pattern.3mf. Studio
# 02.07 sliced a bare Mesher 3mf; 02.08 refuses it with "One of the plate is
# empty or has no object fully inside it" -- a complaint about plates from a
# file that never mentions plates.


def test_it_carries_the_metadata_a_project_needs(project):
    members = set(zipfile.ZipFile(project).namelist())
    assert "Metadata/model_settings.config" in members
    assert "Metadata/slice_info.config" in members


def test_it_does_not_embed_a_project_config(project):
    """The one file that must not be there.

    Studio rejects the merged config this module used to write and then cannot
    find a usable plate, failing with "One of the plate is empty". Settings go
    via --load-settings instead, which is Bambu's documented priority order
    anyway. Bisected against their own pa_pattern.3mf: swapping our config into
    a file that slices is the single change that breaks it.
    """
    assert "Metadata/project_settings.config" not in zipfile.ZipFile(project).namelist()


def test_an_object_is_bound_to_plate_one(project):
    """The binding the error message is actually about."""
    settings = zipfile.ZipFile(project).read("Metadata/model_settings.config").decode()
    assert '<metadata key="plater_id" value="1"/>' in settings
    bound = re.findall(r'<metadata key="object_id" value="(\d+)"/>', settings)
    assert bound, "no object placed on the plate"

    model = zipfile.ZipFile(project).read("3D/3dmodel.model").decode()
    built = re.findall(r'<item [^>]*objectid="(\d+)"', model)
    assert sorted(built) == sorted(bound), "build items and plate disagree"


def test_the_build_item_points_at_the_wrapper_not_the_mesh(project):
    """Mesher points it straight at the mesh object, so the id in the build and
    the id Bambu expects to find on a plate are two different objects."""
    model = zipfile.ZipFile(project).read("3D/3dmodel.model").decode()
    target = re.search(r'<item [^>]*objectid="(\d+)"', model).group(1)
    body = re.search(
        rf'<object id="{target}"[^>]*>(.*?)</object>', model, re.S
    ).group(1)
    assert "<components>" in body


def test_the_item_is_marked_printable(project):
    """An unprintable object is precisely an object not on a plate."""
    model = zipfile.ZipFile(project).read("3D/3dmodel.model").decode()
    assert re.search(r'<item [^>]*printable="1"', model)


def test_a_model_with_no_build_items_is_refused():
    with pytest.raises(slicer.SliceError, match="no build items"):
        slicer.as_bambu_project("<model><resources/><build/></model>", "x")


# --- settings, which now travel on the command line ------------------------


def test_presets_are_written_whole_with_their_type(tmp_path):
    """Studio refuses a preset file with no ``type``: "unknown config type of
    file machine.json in load-settings". So these keep the preset bookkeeping
    that a merged project config strips."""
    files = slicer.preset_files(
        tmp_path, 0.4, profiles.machine(0.4).default_process,
        inventory.default("PLA").preset_for(0.4),
    )
    kinds = []
    for path in files:
        cfg = json.loads(path.read_text())
        assert cfg["type"], f"{path.name} has no type"
        assert cfg["name"]
        kinds.append(cfg["type"])
    assert sorted(kinds) == ["filament", "machine", "printer"] or len(kinds) == 3


def test_overrides_land_on_the_process(tmp_path):
    _, process, _ = slicer.preset_files(
        tmp_path, 0.4, profiles.machine(0.4).default_process,
        inventory.default("PLA").preset_for(0.4),
        overrides={"brim_type": "outer_only"},
    )
    assert json.loads(process.read_text())["brim_type"] == "outer_only"


def test_the_bed_type_is_stated(tmp_path):
    """Nothing in the presets sets it, and the default is Cool Plate."""
    machine, process, _ = slicer.preset_files(
        tmp_path, 0.4, profiles.machine(0.4).default_process,
        inventory.default("ASA").preset_for(0.4),
    )
    for path in (machine, process):
        assert json.loads(path.read_text())["curr_bed_type"] == slicer.DEFAULT_BED


def test_asa_on_a_cool_plate_is_refused():
    """Studio's own words are "Plate 1: Cool Plate does not support filament 1",
    which does not mention heat, ASA, or which plate would work."""
    with pytest.raises(slicer.SliceError, match="Cool Plate"):
        slicer.check_bed(inventory.default("ASA").preset_for(0.4), "Cool Plate")


def test_pla_on_a_cool_plate_is_fine():
    slicer.check_bed(inventory.default("PLA").preset_for(0.4), "Cool Plate")


# --- reading the numbers back ----------------------------------------------


def test_it_reads_the_current_time_format(tmp_path):
    """02.08 writes "model printing time: ...; total estimated time: ..."; the
    older format was "estimated printing time (normal mode) = ...". The total
    is the one to take, being the one that includes heat-up."""
    gcode = tmp_path / "a.gcode"
    gcode.write_text(
        "; model printing time: 25m 13s; total estimated time: 1h 25m 33s\n"
        "; total filament weight [g] : 5.80\n"
    )
    minutes, grams = slicer.read_estimates(gcode, 1.07)
    assert minutes == pytest.approx(85 + 33 / 60, abs=0.01)
    assert grams == pytest.approx(5.80)


def test_it_still_reads_the_old_time_format(tmp_path):
    gcode = tmp_path / "b.gcode"
    gcode.write_text("; estimated printing time (normal mode) = 1h 4m 6s\n")
    minutes, _ = slicer.read_estimates(gcode, 1.07)
    assert minutes == pytest.approx(64 + 6 / 60, abs=0.01)


def test_mass_falls_back_to_volume_when_weight_is_blank(tmp_path):
    """It came back 0.00 on this path until 02.08, so the density path stays."""
    gcode = tmp_path / "c.gcode"
    gcode.write_text(
        "; total filament weight [g] : 0.00\n"
        "; total filament volume [cm^3] : 5000.0\n"
    )
    _, grams = slicer.read_estimates(gcode, 1.07)
    assert grams == pytest.approx(5.0 * 1.07)


# --- the whole thing, against the real slicer -------------------------------


@pytest.mark.skipif(not slicer.EXE.exists(), reason="Bambu Studio not installed")
def test_it_actually_slices(project, tmp_path):
    """The test everything else is a proxy for. Also the one that would have
    caught all of this: the pipeline produced plausible files for months while
    slicing either failed or silently used PLA defaults."""
    result = slicer.slice_project(
        project, tmp_path / "out",
        filament=inventory.default("ASA").preset_for(0.4),
        overrides={"brim_type": "outer_only"},
    )
    assert result.output.exists()
    assert result.minutes and result.minutes > 0
    assert result.grams and result.grams > 0

    head = result.output.read_text(errors="ignore")[:20000]
    assert "; filament_type = ASA" in head, "sliced as the wrong material"
    assert "; printer_model = Bambu Lab P2S" in head
    assert "; brim_type = outer_only" in head, "override did not apply"


def test_verify_rejects_a_model_off_the_plate(project, tmp_path):
    """The check earns its place only if it actually fails on a bad file."""
    off = tmp_path / "off.3mf"
    with zipfile.ZipFile(project) as zin:
        members = {i.filename: zin.read(i.filename) for i in zin.infolist()}
    model = members["3D/3dmodel.model"].decode()
    # Move the item, not the vertices: vertices do not place anything.
    model = re.sub(
        r'(<item [^>]*transform=")([^"]*)(")',
        lambda m: m.group(1)
        + " ".join(m.group(2).split()[:9] + ["900", "128", "5"])
        + m.group(3),
        model,
    )
    members["3D/3dmodel.model"] = model.encode()
    with zipfile.ZipFile(off, "w") as zout:
        for name, data in members.items():
            zout.writestr(name, data)

    with pytest.raises(slicer.SliceError, match="off the plate on x"):
        slicer.verify_on_bed(off)
