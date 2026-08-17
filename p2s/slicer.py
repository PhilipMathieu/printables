"""Drive Bambu Studio headlessly: model in, sliced gcode + estimates out.

SETTINGS GO ON THE COMMAND LINE, NOT IN THE FILE. This module used to embed a
merged machine+process+filament config in the 3mf as
``Metadata/project_settings.config``, on the theory that ``--load-settings``
could not work because a preset loaded from an arbitrary path is not registered
in the preset collection. Both halves of that were wrong, and between them they
account for every slicing problem this repo has had.

The embedded config was never honoured. Studio ignored it and sliced with its
own defaults, which is why every time and mass figure recorded here before
2026-08-16 is a PLA-at-200C figure whatever the project said. From 02.08 it
stopped merely ignoring the config and started refusing the file outright --
"One of the plate is empty or has no object fully inside it", a message about
plates from a file whose plates are fine. Bisected by taking Bambu's own
calib/pressure_advance/pa_pattern.3mf, which slices, and swapping our files
into it one at a time: everything survived except our project_settings.config,
and dropping the config entirely failed the same way. Studio rejects it and
falls back, and in 02.08 the fallback no longer has a usable plate.

``--load-settings`` works fine. Resolved presets written to temp files, passed
as ``machine.json;process.json`` with the filament via ``--load-filaments``,
produce gcode that finally says ``filament_type = ASA``, ``nozzle_temperature =
270``, ``hot_plate_temp = 100`` and names all three presets. Bambu's own
documented priority order is command line > --load-settings > 3mf, so this is
the intended mechanism rather than a workaround.

BED TYPE IS NOT OPTIONAL. Nothing in the presets sets ``curr_bed_type``, so it
defaults to Cool Plate, and validation then fails with "Plate 1: Cool Plate
does not support filament 1" for anything hotter than PLA. It has to be stated.

THE BINARY RUNS DIRECTLY. The old note here said Studio had to be launched
through ``open`` because it stalls on hiservices XPC lookups outside the GUI
session, which also made slicing impossible from a sandboxed process. That is
true of operations that touch the GUI -- loading an STL and ``--export-3mf``
both still hang -- but slicing a 3mf does not. Executing the binary works from
a sandbox, so no LaunchServices, no wrapper, and no unsandboxed escape hatch is
needed for the one thing this module actually does.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from xml.etree import ElementTree
from pathlib import Path

from build123d import Mesher, Part

from . import inventory, profiles

EXE = profiles.BAMBU_STUDIO / "Contents/MacOS/BambuStudio"

# Preset-scoped bookkeeping; a merged project config carries none of it.
_DROP = {
    "setting_id", "is_custom_defined", "from", "instantiation", "type", "name",
    "inherits", "compatible_printers", "compatible_printers_condition",
}


class SliceError(RuntimeError):
    pass


DEFAULT_BED = "Textured PEI Plate"
"""The plate that ships with the P2S, and the one to name unless told
otherwise. Cool Plate is Studio's default and ASA is not allowed on it."""

COOL_PLATE_ONLY = {"PLA", "PVA", "TPU"}
"""Materials Cool Plate is rated for. Everything else needs a hot plate, and
picking wrong is a validation failure rather than a warning."""


def as_filament_value(existing, value: str) -> list:
    """Shape a filament override the way filament presets store values.

    Filament settings are per-extruder lists even when there is one extruder:
    ``"fan_max_speed": ["35"]``, and ``nozzle_temperature`` carries one entry
    per slot. Writing a bare string where Studio expects a list gets the key
    ignored, silently, which is the failure mode this whole module exists to
    avoid. So match the length already there, and fall back to a single entry
    for a key the preset does not carry.
    """
    if isinstance(existing, list) and existing:
        return [value] * len(existing)
    return [value]


def preset_files(
    dest: Path,
    nozzle: float,
    process: str,
    filament: str,
    *,
    bed: str = DEFAULT_BED,
    overrides: dict | None = None,
    filament_overrides: dict | None = None,
) -> tuple[Path, Path, Path]:
    """Write resolved machine/process/filament presets for ``--load-settings``.

    Flattened rather than passed by path, because Studio resolves ``inherits``
    against its own preset collection and these are being handed to it from
    outside it. ``name``/``from`` are put back after resolution: they are what
    ends up in the gcode header as ``printer_settings_id`` and friends, and an
    unnamed preset slices to a file that cannot say what made it.

    ``overrides`` land on the process, which is where the per-part settings a
    stock preset knows nothing about belong -- a brim under a part with a thin
    first layer, say. They are last, so they win.

    ``filament_overrides`` are separate because cooling lives on the filament,
    not the process: ``close_fan_the_first_x_layers``, ``fan_max_speed`` and
    friends are all filament keys. Putting them in ``overrides`` writes them to
    the process, where nothing reads them and nothing complains.
    """
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    fan = filament_overrides or {}
    for kind, name, extra in (
        ("machine", profiles.machine(nozzle).preset, {"curr_bed_type": bed}),
        ("process", process, {"curr_bed_type": bed, **(overrides or {})}),
        ("filament", filament, {}),
    ):
        # Everything the preset carries, _DROP included: these are presets
        # being handed back to Studio, not merged into a project, and it reads
        # the bookkeeping. Without "type" it refuses the file outright --
        # "unknown config type of file machine.json in load-settings".
        cfg = dict(profiles.resolve(kind, name))
        cfg["name"] = name
        cfg["from"] = "system"
        cfg["is_custom_defined"] = "0"
        cfg.update(extra)
        if kind == "filament":
            for key, value in fan.items():
                cfg[key] = as_filament_value(cfg.get(key), value)
        path = dest / f"{kind}.json"
        path.write_text(json.dumps(cfg, indent=1))
        written.append(path)
    return tuple(written)  # type: ignore[return-value]


def check_bed(filament: str, bed: str) -> None:
    """Catch the Cool Plate mistake here, where it can be explained."""
    material = profiles.resolve("filament", filament).get("filament_type", [""])[0]
    if bed == "Cool Plate" and material.upper() not in COOL_PLATE_ONLY:
        raise SliceError(
            f"{material} is not allowed on a Cool Plate. Studio reports this as "
            f'"Plate 1: Cool Plate does not support filament 1" and slices '
            f"nothing. Use {DEFAULT_BED!r} or another hot plate."
        )


def write_project(
    shape: Part | Path,
    dest: Path,
    *,
    nozzle: float = profiles.DEFAULT_NOZZLE,
    process: str | None = None,
    filament: str | None = None,
) -> Path:
    """Write a Bambu project 3mf: the mesh plus the P2S config it slices with."""
    mach = profiles.machine(nozzle)
    process = process or mach.default_process
    filament = filament or inventory.default("PLA").preset_for(nozzle)
    for kind, name in (("process", process), ("filament", filament)):
        if name not in profiles.available(kind, nozzle):
            raise SliceError(f"{name!r} is not a valid {kind} for the {nozzle}mm nozzle")

    dest.parent.mkdir(parents=True, exist_ok=True)
    mesh_only = dest.with_suffix(".mesh.3mf")
    if isinstance(shape, Path):
        shutil.copy(shape, mesh_only)
    else:
        mesher = Mesher()
        mesher.add_shape(shape)
        mesher.write(str(mesh_only))

    with zipfile.ZipFile(mesh_only) as zin:
        members = {i.filename: zin.read(i.filename) for i in zin.infolist()}
    mesh_only.unlink()

    model, model_settings = as_bambu_project(
        members["3D/3dmodel.model"].decode(), dest.stem
    )
    members["3D/3dmodel.model"] = model.encode()
    members["Metadata/model_settings.config"] = model_settings.encode()
    members["Metadata/slice_info.config"] = SLICE_INFO.encode()
    # Deliberately no Metadata/project_settings.config. Studio rejects the one
    # this module used to write and then cannot find a usable plate; settings
    # arrive via preset_files() and --load-settings instead. See the module
    # docstring for how that was pinned down.

    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in members.items():
            zout.writestr(name, data)
    return center_on_bed(dest, nozzle)


SLICE_INFO = """<?xml version="1.0" encoding="UTF-8"?>
<config>
  <header>
    <header_item key="X-BBL-Client-Type" value="slicer"/>
    <header_item key="X-BBL-Client-Version" value="01.07.03.04"/>
  </header>
</config>
"""
"""Copied verbatim from Bambu's own calib/pressure_advance/pa_pattern.3mf.

The version is theirs, not the installed build's, and deliberately so: this
file cannot be tested from a sandboxed session, so the fewer characters that
differ from one Studio is known to open, the fewer places a guess can be wrong.
"""


def as_bambu_project(model: str, name: str) -> tuple[str, str]:
    """Turn a bare 3mf model into one Bambu will treat as a project.

    A 3mf that build123d's Mesher writes is a valid 3mf and Bambu still refuses
    to slice it, with "One of the plate is empty or has no object fully inside
    it" -- a message about plates for a file that never mentions plates. There
    is no plate list in it at all, so every plate Studio makes is empty and it
    reports the symptom rather than the cause. Studio 02.07 tolerated this;
    02.08 does not, which is why files sliced in August stopped slicing.

    Bambu's own bundled projects show what is missing. Three differences from
    what Mesher writes, all read off calib/pressure_advance/pa_pattern.3mf:

    1. ``Metadata/model_settings.config``, whose ``<plate>`` binds an object to
       plate 1 by id. This is the one the error message is actually about.
    2. The build ``<item>`` points at the wrapper object -- the one holding
       ``<components>`` -- not at the mesh object the wrapper references.
       Mesher points it straight at the mesh, so the id in the build and the id
       Bambu expects to find in a plate are different objects.
    3. ``printable="1"`` on the item. An unprintable object is exactly an
       object that is not on a plate, which is the reported symptom again.

    Returns the rewritten model XML and the model_settings.config to sit
    beside it.
    """
    # Which wrapper stands in front of which mesh. Mesher emits them in pairs,
    # a mesh object and a components object that references it.
    wrapper_of = {mesh: wrap for wrap, mesh in mesh_of_wrapper(model).items()}

    instances: list[str] = []

    def retarget(match: re.Match) -> str:
        attrs = match.group(1)
        current = re.search(r'objectid="(\d+)"', attrs)
        if not current:
            return match.group(0)
        target = wrapper_of.get(current.group(1), current.group(1))
        attrs = re.sub(r'objectid="\d+"', f'objectid="{target}"', attrs, count=1)
        if "printable=" not in attrs:
            attrs = f'{attrs.rstrip()} printable="1"'
        instances.append(target)
        return f"<item {attrs}/>"

    model = re.sub(r"<item ([^>]*?)\s*/>", retarget, model)
    if not instances:
        raise SliceError("no build items in the 3mf, so nothing can be put on a plate")

    objects = "\n".join(
        f'  <object id="{oid}">\n'
        f'    <metadata key="name" value="{name}"/>\n'
        f'    <metadata key="extruder" value="1"/>\n'
        f'    <part id="1" subtype="normal_part">\n'
        f'      <metadata key="name" value="{name}"/>\n'
        f'      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>\n'
        f"    </part>\n"
        f"  </object>"
        for oid in dict.fromkeys(instances)
    )
    # Every instance goes on plate 1. Nothing here builds multi-plate projects,
    # and an extra empty plate is the very thing the slicer refuses.
    placements = "\n".join(
        f"    <model_instance>\n"
        f'      <metadata key="object_id" value="{oid}"/>\n'
        f'      <metadata key="instance_id" value="{i}"/>\n'
        f"    </model_instance>"
        for i, oid in enumerate(instances)
    )
    settings = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<config>\n"
        f"{objects}\n"
        "  <plate>\n"
        '    <metadata key="plater_id" value="1"/>\n'
        '    <metadata key="plater_name" value=""/>\n'
        '    <metadata key="locked" value="false"/>\n'
        f"{placements}\n"
        "  </plate>\n"
        "</config>\n"
    )
    return model, settings


def center_on_bed(threemf: Path, nozzle: float = profiles.DEFAULT_NOZZLE) -> Path:
    """Move the model to the middle of the plate, in place.

    Needed because auto-arrange has to stay off: a nested assembly is many
    disjoint solids and the arranger spreads them across the bed as separate
    objects, which destroys the nesting. Without it, though, the model keeps
    its modelled coordinates -- centred near the origin, so half of it hangs
    off the plate and the slice fails without writing anything.
    """
    mach = profiles.machine(nozzle)
    with zipfile.ZipFile(threemf) as zin:
        members = {i.filename: zin.read(i.filename) for i in zin.infolist()}
    model = members["3D/3dmodel.model"].decode()

    bounds = object_bounds(model)
    if not bounds:
        raise SliceError(f"no vertices found in {threemf}")
    lows = [min(b[i] for b in bounds.values()) for i in (0, 2, 4)]
    highs = [max(b[i] for b in bounds.values()) for i in (1, 3, 5)]
    # Shift that would put the whole assembly middle-of-bed, sitting on z=0.
    shift = (
        mach.bed_x / 2 - (lows[0] + highs[0]) / 2,
        mach.bed_y / 2 - (lows[1] + highs[1]) / 2,
        -lows[2],
    )

    def place(match: re.Match) -> str:
        """Put the same global shift on every item.

        Studio applies this transform on top of the mesh coordinates; it does
        not re-centre the object first. That is worth stating because ``--info``
        strongly implies otherwise -- it reports a model written at x 53..203 as
        x -75..75 -- but that is a normalised bounding box for display, not
        where the object goes. Tested by modelling a box at (-400, 250) and
        slicing it both ways: a translation equal to the object's centre in bed
        coordinates fails to slice, and one equal to the shift succeeds.

        One shift for all items, not a per-object placement, so a nested
        assembly keeps its relative layout. The transform is replaced rather
        than added to, which is what makes this safe to run twice.
        """
        attrs = match.group(1)
        matrix = " ".join(f"{v:g}" for v in (1, 0, 0, 0, 1, 0, 0, 0, 1, *shift))
        attrs = re.sub(r'\s*transform="[^"]*"', "", attrs).rstrip()
        return f'<item {attrs} transform="{matrix}"/>'

    model = re.sub(r"<item ([^>]*?)\s*/>", place, model)
    # A malformed rewrite here reads as "input model file can not be parsed"
    # from the slicer, with no hint which file or why -- so check it now.
    try:
        ElementTree.fromstring(model)
    except ElementTree.ParseError as exc:
        raise SliceError(f"bed-centring produced invalid 3mf XML: {exc}") from exc
    members["3D/3dmodel.model"] = model.encode()
    with zipfile.ZipFile(threemf, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in members.items():
            zout.writestr(name, data)
    verify_on_bed(threemf, nozzle)
    return threemf


def object_bounds(model: str) -> dict[str, tuple[float, float, float, float, float, float]]:
    """Bounding box per object, in that object's own mesh coordinates."""
    out: dict[str, tuple[float, float, float, float, float, float]] = {}
    for match in re.finditer(r'<object id="(\d+)"[^>]*>(.*?)</object>', model, re.S):
        verts = re.findall(
            r'<vertex x="([-0-9.eE]+)" y="([-0-9.eE]+)" z="([-0-9.eE]+)"',
            match.group(2),
        )
        if not verts:
            continue
        cols = [[float(v[i]) for v in verts] for i in range(3)]
        out[match.group(1)] = (
            min(cols[0]), max(cols[0]),
            min(cols[1]), max(cols[1]),
            min(cols[2]), max(cols[2]),
        )
    return out


def mesh_of_wrapper(model: str) -> dict[str, str]:
    """Wrapper object id -> the mesh object id its component points at."""
    out: dict[str, str] = {}
    for match in re.finditer(r'<object id="(\d+)"[^>]*>(.*?)</object>', model, re.S):
        ref = re.search(r'<component[^>]*\sobjectid="(\d+)"', match.group(2))
        if ref:
            out[match.group(1)] = ref.group(1)
    return out


def bed_extent(threemf: Path) -> tuple[float, float, float, float, float, float]:
    """Where the model will actually sit on the plate.

    Mesh coordinates plus the build item's translation, which is how Studio
    composes it. Reading the vertices alone would describe the modelled frame
    and cheerfully agree that a part modelled 400mm off the origin is on the
    bed; reading the transform alone says nothing about the object's size.
    """
    with zipfile.ZipFile(threemf) as zin:
        model = zin.read("3D/3dmodel.model").decode()
    bounds = object_bounds(model)
    if not bounds:
        raise SliceError(f"no vertices found in {threemf}")
    meshes = mesh_of_wrapper(model)

    boxes: list[tuple[float, ...]] = []
    for item in re.finditer(r"<item ([^>]*?)\s*/>", model):
        attrs = item.group(1)
        found = re.search(r'objectid="(\d+)"', attrs)
        if not found:
            continue
        mesh_id = meshes.get(found.group(1), found.group(1))
        if mesh_id not in bounds:
            continue
        x0, x1, y0, y1, z0, z1 = bounds[mesh_id]
        matrix = re.search(r'transform="([^"]*)"', attrs)
        move = (
            tuple(float(v) for v in matrix.group(1).split()[9:12])
            if matrix
            else (0.0, 0.0, 0.0)
        )
        boxes.append(
            (x0 + move[0], x1 + move[0],
             y0 + move[1], y1 + move[1],
             z0 + move[2], z1 + move[2])
        )
    if not boxes:
        raise SliceError(f"no build items in {threemf}")
    return (
        min(b[0] for b in boxes), max(b[1] for b in boxes),
        min(b[2] for b in boxes), max(b[3] for b in boxes),
        min(b[4] for b in boxes), max(b[5] for b in boxes),
    )


def verify_on_bed(
    threemf: Path, nozzle: float = profiles.DEFAULT_NOZZLE, margin: float = 1.0
) -> None:
    """Fail here rather than letting the slicer fail vaguely later.

    Bambu's way of saying a model is off the plate is "One of the plate is
    empty or has no object fully inside it", which names neither the object nor
    the axis and reads like a problem with the plate list. Checking the written
    coordinates turns that into a sentence with numbers in it.
    """
    mach = profiles.machine(nozzle)
    x0, x1, y0, y1, z0, z1 = bed_extent(threemf)
    for axis, lo, hi, limit in (
        ("x", x0, x1, mach.bed_x),
        ("y", y0, y1, mach.bed_y),
        ("z", z0, z1, mach.height),
    ):
        if lo < -margin or hi > limit + margin:
            raise SliceError(
                f"{threemf.name} is off the plate on {axis}: {lo:.1f}..{hi:.1f}mm "
                f"against a 0..{limit:.0f}mm bed. The slicer would report this as "
                f"an empty plate."
            )


@dataclass(frozen=True)
class SliceResult:
    project: Path
    output: Path
    minutes: float | None
    grams: float | None
    log: str

    def summary(self) -> str:
        t = f"{self.minutes:.0f} min" if self.minutes is not None else "unknown time"
        g = f"{self.grams:.1f} g" if self.grams is not None else "unknown mass"
        return f"{self.project.stem}: {t}, {g}"


# Estimates come from the gcode header rather than stdout, which is a very long
# trace log with the numbers scattered through it.
#
# Two time formats. 02.07 and earlier wrote "estimated printing time
# (normal mode) = 1h 4m 6s"; 02.08 writes "model printing time: 25m 13s; total
# estimated time: 25m 33s". Take the total where it exists, since it is the one
# that includes heat-up and the tool changes.
_TIME = re.compile(
    r"(?:total estimated time|estimated printing time[^=:]*)\s*[=:]\s*"
    r"(?:(\d+)d\s*)?(?:(\d+)h\s*)?(?:(\d+)m\s*)?(?:(\d+)s)?",
    re.I,
)
_WEIGHT = re.compile(r"total filament weight \[g\]\s*:\s*([0-9.]+)", re.I)
_VOLUME = re.compile(r"total filament volume \[cm\^3\]\s*:\s*([0-9.]+)", re.I)


def read_estimates(gcode: Path, density: float) -> tuple[float | None, float | None]:
    """Minutes and grams from a sliced gcode header.

    Weight is read when Studio fills it in and computed from volume when it
    does not. It used to always come back 0.00 on the CLI path, which is why
    density is still a parameter; 02.08 populates it, and its own figure is
    preferable because it knows about flush and purge that the volume line does
    not. Note Bambu labels the volume field cm^3 and writes mm^3.
    """
    head = gcode.read_text(errors="ignore")[:8000]
    minutes = grams = None
    if m := _TIME.search(head):
        d, h, mi, s = (int(g or 0) for g in m.groups())
        minutes = d * 1440 + h * 60 + mi + s / 60
    if w := _WEIGHT.search(head):
        grams = float(w.group(1)) or None
    if grams is None and (v := _VOLUME.search(head)):
        grams = float(v.group(1)) / 1000.0 * density
    return minutes, grams


def slice_project(
    project: Path,
    outdir: Path | None = None,
    timeout: int = 900,
    *,
    nozzle: float = profiles.DEFAULT_NOZZLE,
    process: str | None = None,
    filament: str | None = None,
    bed: str = DEFAULT_BED,
    overrides: dict | None = None,
    filament_overrides: dict | None = None,
) -> SliceResult:
    """Slice a project 3mf with real presets, and read the estimates back."""
    process = process or profiles.machine(nozzle).default_process
    filament = filament or inventory.default("PLA").preset_for(nozzle)
    if not EXE.exists():
        raise SliceError(f"Bambu Studio not found at {profiles.BAMBU_STUDIO}")
    check_bed(filament, bed)
    outdir = Path(outdir or project.parent / "sliced").resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    machine_json, process_json, filament_json = preset_files(
        outdir / "presets", nozzle, process, filament, bed=bed,
        overrides=overrides, filament_overrides=filament_overrides,
    )
    # The binary, not `open`. Slicing a 3mf needs nothing from the GUI session,
    # so this works from a sandbox; only STL loading and --export-3mf still
    # stall on hiservices XPC lookups.
    cmd = [
        str(EXE),
        # A nested assembly is many disjoint solids, and the CLI treats each as
        # an independent object: left to itself it arranges them side by side
        # across the bed and the nesting is gone. Both must be off.
        "--arrange", "0",
        "--orient", "0",
        "--load-settings", f"{machine_json};{process_json}",
        "--load-filaments", str(filament_json),
        "--slice", "0",
        "--outputdir", str(outdir),
        str(project),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise SliceError("Bambu Studio did not finish within the timeout") from exc
    log = proc.stdout + proc.stderr
    gcode = outdir / "plate_1.gcode"
    if proc.returncode != 0 or not gcode.exists():
        # Studio's own error lines are the only useful part of a very long log.
        errors = [ln for ln in log.splitlines() if "[error]" in ln]
        tail = "\n".join(errors[-6:] or log.strip().splitlines()[-12:])
        raise SliceError(f"slice failed (rc={proc.returncode}):\n{tail}")

    density = float(profiles.resolve("filament", filament)["filament_density"][0])
    minutes, grams = read_estimates(gcode, density)
    return SliceResult(
        project=project, output=gcode, minutes=minutes, grams=grams, log=log
    )
