"""Drive Bambu Studio headlessly: model in, sliced 3mf + estimates out.

Settings are delivered the way Bambu's own ``--export-settings`` writes them:
one merged config, ``from: "project"``, embedded in the 3mf as
``Metadata/project_settings.config``. Passing machine/process/filament as
separate ``--load-settings`` presets does not work -- a preset loaded from an
arbitrary path is not registered in the preset collection, so the compatibility
check rejects it even when the names match exactly.

KNOWN ENVIRONMENT LIMIT: Bambu Studio must run in the user's GUI session, so
it is launched through ``open`` rather than by executing the binary. A
sandboxed process cannot reach LaunchServices either, so this function only
works when it is itself running unsandboxed -- otherwise ``open`` fails with
kLSUnknownErr. From a sandboxed session, call ~/bin/p2s-slice as a top-level
command instead; see tools/p2s-slice.sh.
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


def project_config(nozzle: float, process: str, filament: str) -> dict:
    """Merged machine+process+filament config for embedding in a 3mf."""
    cfg: dict = {}
    for kind, name in (
        ("machine", profiles.machine(nozzle).preset),
        ("process", process),
        ("filament", filament),
    ):
        cfg.update({k: v for k, v in profiles.resolve(kind, name).items() if k not in _DROP})
    cfg["name"] = "project_settings"
    cfg["from"] = "project"
    return cfg


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

    cfg = project_config(nozzle, process, filament)
    with zipfile.ZipFile(mesh_only) as zin, zipfile.ZipFile(
        dest, "w", zipfile.ZIP_DEFLATED
    ) as zout:
        for item in zin.infolist():
            zout.writestr(item, zin.read(item.filename))
        zout.writestr("Metadata/project_settings.config", json.dumps(cfg, indent=1))
    mesh_only.unlink()
    return center_on_bed(dest, nozzle)


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

    xs = [float(v) for v in re.findall(r'<vertex x="([-0-9.eE]+)"', model)]
    ys = [float(v) for v in re.findall(r'<vertex[^>]*\sy="([-0-9.eE]+)"', model)]
    zs = [float(v) for v in re.findall(r'<vertex[^>]*\sz="([-0-9.eE]+)"', model)]
    if not xs:
        raise SliceError(f"no vertices found in {threemf}")
    dx = mach.bed_x / 2 - (min(xs) + max(xs)) / 2
    dy = mach.bed_y / 2 - (min(ys) + max(ys)) / 2
    dz = -min(zs)

    # Bake the move into the coordinates rather than into the <item> transform.
    #
    # Writing a translation into the build item is the tidier edit and it is
    # what this did first. Bambu rejected the result with "One of the plate is
    # empty or has no object fully inside it" for a model whose transformed
    # bounding box sits comfortably inside the bed, so its containment test and
    # its placement do not agree about that matrix. Baked coordinates take the
    # disagreement away: the numbers in the file are bed coordinates, nothing
    # has to be composed to know where the object lands, and the result can be
    # checked by reading the vertices back -- which ``verify_on_bed`` does.
    def move(match: re.Match) -> str:
        x, y, z = (float(match.group(i)) for i in (1, 2, 3))
        return f'<vertex x="{x + dx:g}" y="{y + dy:g}" z="{z + dz:g}"/>'

    model = re.sub(
        r'<vertex x="([-0-9.eE]+)" y="([-0-9.eE]+)" z="([-0-9.eE]+)"\s*/>',
        move,
        model,
    )
    # Any transform left over from an earlier pass would now be applied on top
    # of coordinates that already include it, moving the object twice.
    model = re.sub(r'(<item [^>]*?)\s*transform="[^"]*"', r"\1", model)
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


def bed_extent(threemf: Path) -> tuple[float, float, float, float, float, float]:
    """Where the model actually sits, read back out of the written file.

    Only meaningful because ``center_on_bed`` bakes position into the
    coordinates: with a build-item transform these numbers would be the
    object's local frame and say nothing about the bed.
    """
    with zipfile.ZipFile(threemf) as zin:
        model = zin.read("3D/3dmodel.model").decode()
    axes = []
    for pattern in (
        r'<vertex x="([-0-9.eE]+)"',
        r'<vertex[^>]*\sy="([-0-9.eE]+)"',
        r'<vertex[^>]*\sz="([-0-9.eE]+)"',
    ):
        found = [float(v) for v in re.findall(pattern, model)]
        if not found:
            raise SliceError(f"no vertices found in {threemf}")
        axes.append((min(found), max(found)))
    return (*axes[0], *axes[1], *axes[2])


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


# Estimates come out of the gcode header, not stdout: launched through `open`
# the app's output is not ours to capture.
_TIME = re.compile(
    r"estimated printing time.*?=\s*(?:(\d+)d\s*)?(?:(\d+)h\s*)?(?:(\d+)m\s*)?(?:(\d+)s)?",
    re.I,
)
_VOLUME = re.compile(r"total filament volume \[cm\^3\]\s*:\s*([0-9.]+)", re.I)


def read_estimates(gcode: Path, density: float) -> tuple[float | None, float | None]:
    """Minutes and grams from a sliced gcode header.

    Bambu labels the volume field cm^3 but writes mm^3, and leaves the weight
    field at 0.00 on this CLI path, so mass is computed here from the filament
    preset's density rather than read back.
    """
    head = gcode.read_text(errors="ignore")[:8000]
    minutes = grams = None
    if m := _TIME.search(head):
        d, h, mi, s = (int(g or 0) for g in m.groups())
        minutes = d * 1440 + h * 60 + mi + s / 60
    if v := _VOLUME.search(head):
        grams = float(v.group(1)) / 1000.0 * density
    return minutes, grams


def slice_project(
    project: Path,
    outdir: Path | None = None,
    timeout: int = 900,
    *,
    nozzle: float = profiles.DEFAULT_NOZZLE,
    filament: str | None = None,
) -> SliceResult:
    """Slice a project 3mf and read back time and filament estimates."""
    filament = filament or inventory.default("PLA").preset_for(nozzle)
    if not EXE.exists():
        raise SliceError(f"Bambu Studio not found at {profiles.BAMBU_STUDIO}")
    outdir = Path(outdir or project.parent / "sliced").resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    # Launch through LaunchServices, not the binary directly. Run straight from
    # a terminal the app never reaches the user's GUI session: it fails
    # com.apple.hiservices-xpcservice lookups and stalls before slicing, which
    # is what made this look like a sandbox problem for so long. `open -W`
    # starts it in the Aqua session and waits for it to exit.
    cmd = [
        "open", "-W", "-n", "-a", str(profiles.BAMBU_STUDIO), "--args",
        # A nested assembly is many disjoint solids, and the CLI treats each as
        # an independent object: left to itself it arranges them side by side
        # across the bed and the nesting is gone. Both must be off.
        "--arrange", "0",
        "--orient", "0",
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
        tail = "\n".join(log.strip().splitlines()[-20:])
        raise SliceError(f"slice failed (rc={proc.returncode}):\n{tail}")

    density = float(profiles.resolve("filament", filament)["filament_density"][0])
    minutes, grams = read_estimates(gcode, density)
    return SliceResult(
        project=project, output=gcode, minutes=minutes, grams=grams, log=log
    )
