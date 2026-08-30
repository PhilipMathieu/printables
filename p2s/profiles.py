"""Read Bambu Studio's bundled profiles and expose the P2S subset.

Bambu ships every machine/process/filament preset as JSON with an ``inherits``
chain that has to be flattened before the values mean anything: the P2S machine
preset itself carries no ``printable_area``, it picks that up from a shared
base. Everything here works off the installed Bambu Studio so the numbers can
never drift from what actually slices.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

BAMBU_STUDIO = Path("/Applications/BambuStudio.app")
PROFILE_ROOT = BAMBU_STUDIO / "Contents/Resources/profiles/BBL"

PRINTER_MODEL = "Bambu Lab P2S"
NOZZLES = (0.2, 0.4, 0.6, 0.8)
DEFAULT_NOZZLE = 0.4


class ProfileError(RuntimeError):
    pass


def machine_preset(nozzle: float = DEFAULT_NOZZLE) -> str:
    """Preset name Bambu uses to key process/filament compatibility."""
    if nozzle not in NOZZLES:
        raise ProfileError(f"P2S has no {nozzle}mm nozzle; have {NOZZLES}")
    return f"{PRINTER_MODEL} {nozzle} nozzle"


def _read(kind: str, name: str) -> dict:
    path = PROFILE_ROOT / kind / f"{name}.json"
    if not path.exists():
        raise ProfileError(f"no {kind} profile {name!r} at {path}")
    # Explicitly UTF-8: several Bambu presets carry non-ASCII characters in
    # their names, and read_text() otherwise decodes with the platform default,
    # which is ASCII whenever the process runs without a locale set. That makes
    # profile lookup work in a terminal and blow up under a runner.
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def resolve(kind: str, name: str) -> dict:
    """Flatten a preset's ``inherits`` chain, child values winning."""
    data = _read(kind, name)
    parent = data.pop("inherits", None)
    if not parent:
        return data
    merged = dict(resolve(kind, parent))
    merged.update(data)
    return merged


@lru_cache(maxsize=None)
def _presets(kind: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """(preset name, compatible machine presets) for every preset of a kind.

    Compatibility comes from the resolved ``compatible_printers`` list rather
    than the filename: "Bambu PLA Basic @BBL P2S.json" carries no nozzle in its
    name but is only valid on the 0.4mm machine.
    """
    out = []
    for path in sorted((PROFILE_ROOT / kind).glob("*.json")):
        if " template " in path.stem:  # gcode fragments, not presets
            continue
        try:
            data = resolve(kind, path.stem)
        except (ProfileError, json.JSONDecodeError):
            continue
        compat = data.get("compatible_printers") or []
        if isinstance(compat, str):
            compat = [compat]
        out.append((path.stem, tuple(compat)))
    return tuple(out)


def available(kind: str, nozzle: float = DEFAULT_NOZZLE) -> list[str]:
    """Preset names of ``kind`` ('process' or 'filament') valid on this nozzle."""
    want = machine_preset(nozzle)
    return [name for name, compat in _presets(kind) if want in compat]


def filaments(nozzle: float = DEFAULT_NOZZLE, material: str | None = None) -> list[str]:
    names = available("filament", nozzle)
    if material:
        needle = material.lower()
        names = [n for n in names if needle in n.lower()]
    return names


def processes(nozzle: float = DEFAULT_NOZZLE) -> list[str]:
    return available("process", nozzle)


@dataclass(frozen=True)
class Machine:
    """The P2S constraints a model has to respect, read from the real preset."""

    preset: str
    nozzle: float
    bed_x: float
    bed_y: float
    height: float
    clearance_radius: float
    clearance_height_to_rod: float
    default_process: str

    @property
    def line_width(self) -> float:
        """Nominal extrusion width; the floor for any printable feature."""
        return self.nozzle

    def fits(self, size: tuple[float, float, float], margin: float = 5.0) -> bool:
        x, y, z = size
        return (
            x <= self.bed_x - 2 * margin
            and y <= self.bed_y - 2 * margin
            and z <= self.height
        )


def _corners(printable_area: list[str]) -> tuple[float, float]:
    pts = [tuple(float(v) for v in p.split("x")) for p in printable_area]
    return max(p[0] for p in pts), max(p[1] for p in pts)


@lru_cache(maxsize=None)
def machine(nozzle: float = DEFAULT_NOZZLE) -> Machine:
    preset = machine_preset(nozzle)
    cfg = resolve("machine", preset)
    bed_x, bed_y = _corners(cfg["printable_area"])
    return Machine(
        preset=preset,
        nozzle=nozzle,
        bed_x=bed_x,
        bed_y=bed_y,
        height=float(cfg["printable_height"]),
        clearance_radius=float(cfg["extruder_clearance_radius"]),
        clearance_height_to_rod=float(cfg["extruder_clearance_height_to_rod"]),
        default_process=cfg["default_print_profile"],
    )


def preset_path(kind: str, name: str) -> Path:
    """On-disk path, for handing straight to the Bambu Studio CLI."""
    path = PROFILE_ROOT / kind / f"{name}.json"
    if not path.exists():
        raise ProfileError(f"no {kind} preset {name!r}")
    return path
