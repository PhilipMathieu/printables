"""What is actually on the shelf.

No AMS on this printer: one filament per print, colour changes only via a
manual swap pause. So a design that needs two colours needs to be two parts.
"""

from __future__ import annotations

from dataclasses import dataclass

HAS_AMS = False


@dataclass(frozen=True)
class Filament:
    label: str
    material: str
    product: str  # Bambu product name, e.g. "Bambu PLA Basic"
    remaining: float  # rough fraction of roll left

    def preset_for(self, nozzle: float) -> str:
        """Find this product's preset for a nozzle.

        Bambu is inconsistent about the suffix -- "Bambu PLA Basic @BBL P2S"
        covers 0.4mm, but ASA spells out "@BBL P2S 0.4 nozzle" -- so match on
        the product name and let the preset list say which stem exists.
        """
        from . import profiles

        for stem in profiles.filaments(nozzle):
            if stem.split(" @BBL")[0] == self.product:
                return stem
        raise LookupError(f"{self.product!r} has no preset for the {nozzle}mm nozzle")


STOCK = (
    Filament("fenway green PLA", "PLA", "Bambu PLA Basic", 0.85),
    Filament("off-white PLA", "PLA", "Bambu PLA Basic", 1.0),
    Filament("black ASA", "ASA", "Bambu ASA", 1.0),
)


def by_material(material: str) -> list[Filament]:
    return [f for f in STOCK if f.material.lower() == material.lower()]


def default(material: str = "PLA") -> Filament:
    """Prefer the fullest roll of the requested material."""
    options = by_material(material)
    if not options:
        raise LookupError(f"no {material} in stock; have {[f.label for f in STOCK]}")
    return max(options, key=lambda f: f.remaining)
