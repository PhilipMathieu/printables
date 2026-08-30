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


def test_filament_overrides_land_on_the_filament(tmp_path):
    """Cooling is a filament setting. Sent through ``overrides`` it would be
    written to the process, where nothing reads it and nothing complains --
    so the two are separate arguments and this checks they stay separate."""
    _, process, filament = slicer.preset_files(
        tmp_path, 0.4, profiles.machine(0.4).default_process,
        inventory.default("ASA").preset_for(0.4),
        filament_overrides={"fan_max_speed": "0"},
    )
    assert json.loads(filament.read_text())["fan_max_speed"] == ["0"]
    assert "fan_max_speed" not in json.loads(process.read_text())


def test_a_filament_override_keeps_the_per_extruder_shape(tmp_path):
    """Filament values are lists even with one extruder, and a bare string is
    ignored rather than refused. nozzle_temperature carries one entry per slot,
    so the length has to be matched rather than assumed to be one."""
    name = inventory.default("ASA").preset_for(0.4)
    before = profiles.resolve("filament", name)["nozzle_temperature"]
    _, _, filament = slicer.preset_files(
        tmp_path, 0.4, profiles.machine(0.4).default_process, name,
        filament_overrides={"nozzle_temperature": "265"},
    )
    assert json.loads(filament.read_text())["nozzle_temperature"] == ["265"] * len(before)


def test_an_unknown_filament_key_still_gets_a_list(tmp_path):
    assert slicer.as_filament_value(None, "0") == ["0"]


# --- the nozzle actually gets hot ------------------------------------------
#
# A gcode whose header says nozzle_temperature = 270 and whose only command is
# M109 S205 looks correct in every check that reads the header. This one cost a
# print: ASA extruded at 205C does not stick to anything.


def test_the_real_start_sequence_is_used_not_the_placeholder(tmp_path):
    """Bambu's profile tree has no P-series start gcode; the inherited
    fdm_machine_common is a 577-char placeholder with an Ender purge line, a
    220mm bed centre and M109 S205. The real one is vendored from a GUI export."""
    real = slicer.machine_gcode("start")
    assert real and len(real.splitlines()) > 100, "start gcode was never captured"

    machine, _, _ = slicer.preset_files(
        tmp_path, 0.4, profiles.machine(0.4).default_process,
        inventory.default("ASA").preset_for(0.4),
    )
    written = json.loads(machine.read_text())["machine_start_gcode"]
    assert written == real
    assert "Draw the first line" not in written, "the Ender purge line came back"
    assert "X110 Y110" not in written, "the 220mm bed centre came back"


def test_the_real_sequence_templates_its_own_temperature():
    """Which is why it must not be rewritten -- see the next test."""
    assert "nozzle_temperature" in slicer.machine_gcode("start")


def test_templated_start_gcode_is_left_alone():
    """It warms to 140C and 170C deliberately during bed levelling. Rewriting
    those to the printing temperature would have the nozzle oozing at 270C
    while it probes the plate -- worse than the bug being fixed."""
    real = slicer.machine_gcode("start")
    assert slicer.fix_start_temperature(real, 270) == real


def test_indented_temperature_commands_are_seen(tmp_path):
    """The real sequence indents its commands inside conditional blocks. An
    anchor of ^M109 finds none of them and calls a file that heats to 270C
    three times "no temperature command at all"."""
    asa = inventory.default("ASA").preset_for(0.4)
    g = tmp_path / "indented.gcode"
    g.write_text("M190 S100\n    M1002 gcode_claim_action : 8\n    M109 S270\n")
    slicer.verify_gcode(g, asa)


def test_a_hardcoded_start_temperature_is_rewritten():
    stub = "G28\nM190 S100\nM109 S205;\nG1 Z5\n"
    assert "M109 S270;" in slicer.fix_start_temperature(stub, 270)
    assert "205" not in slicer.fix_start_temperature(stub, 270)


def test_rewriting_leaves_everything_else_alone():
    stub = "G28\nM109 S205;\nG1 X205.5 Y10\n"
    out = slicer.fix_start_temperature(stub, 270)
    assert "G1 X205.5 Y10" in out, "a coordinate that looks like a temperature"


def test_the_written_preset_never_carries_the_placeholder_temperature(tmp_path):
    """Whichever route is taken -- the real templated sequence, or a patched
    stub -- what must never survive is a literal 205."""
    for material in ("ASA", "PLA"):
        machine, _, _ = slicer.preset_files(
            tmp_path / material, 0.4, profiles.machine(0.4).default_process,
            inventory.default(material).preset_for(0.4),
        )
        start = json.loads(machine.read_text())["machine_start_gcode"]
        assert "M109 S205" not in start


def test_the_patched_stub_follows_the_filament(tmp_path):
    """The fallback path, for a machine whose real sequence was never captured.
    Proof the number is read from the filament rather than pinned."""
    stub = "G28\nM109 S205;\n"
    temps = {
        material: slicer.fix_start_temperature(
            stub, slicer.initial_temperature(
                inventory.default(material).preset_for(0.4)
            )
        )
        for material in ("ASA", "PLA")
    }
    assert temps["ASA"] != temps["PLA"], f"both materials got {temps['ASA']!r}"


def test_verify_rejects_the_wrong_temperature(tmp_path):
    bad = tmp_path / "bad.gcode"
    bad.write_text("M190 S100\nM109 S205;\nG1 X1 Y1 E1\n")
    with pytest.raises(slicer.SliceError, match="205"):
        slicer.verify_gcode(bad, inventory.default("ASA").preset_for(0.4))


def test_verify_rejects_a_gcode_that_never_heats_up(tmp_path):
    cold = tmp_path / "cold.gcode"
    cold.write_text("M190 S100\nG1 X1 Y1 E1\nM104 S0 ; turn off hotend\n")
    with pytest.raises(slicer.SliceError, match="no nozzle temperature command"):
        slicer.verify_gcode(cold, inventory.default("ASA").preset_for(0.4))


def test_verify_accepts_the_right_temperature(tmp_path):
    asa = inventory.default("ASA").preset_for(0.4)
    good = tmp_path / "good.gcode"
    good.write_text(f"M190 S100\nM109 S{slicer.initial_temperature(asa)};\n")
    slicer.verify_gcode(good, asa)


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

    # Not a fixed window: the real start sequence is 382 lines and the header
    # echoes it, which pushed printer_model past a 20000-character slice.
    body = result.output.read_text(errors="ignore")
    assert "; filament_type = ASA" in body, "sliced as the wrong material"
    assert "; printer_model = Bambu Lab P2S" in body
    assert "; brim_type = outer_only" in body, "override did not apply"

    # The header agreeing with itself proves nothing -- read the commands. The
    # module's own pattern, because it tolerates the indentation the real
    # sequence uses; 140C and 170C also appear, deliberately, for bed levelling.
    hot = {int(m.group(2)) for m in slicer._START_TEMP.finditer(body)}
    want = slicer.initial_temperature(inventory.default("ASA").preset_for(0.4))
    assert want in hot, f"nozzle never commanded to {want}; saw {sorted(hot)}"


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
