"""The house rules, made enforceable. See CLAUDE.md for why each one exists.

The rule these carry is the diagram and statics loop: anything that carries load
gets drawn in section, from the model, before it is printed. Two mechanical
errors got through this repository and both were invisible in prose and obvious
in a drawing -- the clamp's nut trap opening the way the load pushed, and the
break arm's lever lying on the surface its own fulcrum was in.

What can actually be tested about a rule like that is narrow, and pretending
otherwise would be worse than admitting it. Three things can:

That every part is *classified*, so a new one cannot arrive without somebody
deciding whether anything bears on it. That the load-bearing ones each have a
drawing that regenerates from the current model rather than a PNG that was true
once. And that the drawing is committed, so a reader has it without running
anything.

The statics itself lives in the part's own tests, where the numbers are.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import pytest

import parts

ROOT = Path(__file__).resolve().parents[1]

CARRIES_LOAD = {
    "bench_dogs": ("tools.bench_dogs", "docs/bench_dogs.png"),
    "dog_deck": ("tools.bench_dogs", "docs/bench_dogs.png"),
    "dog_rig": ("tools.break_test", "docs/break_test.png"),
    "plant_clip": ("tools.plant_clip", "docs/plant_clip.png"),
    "bag_holder": ("tools.bag_holder", "docs/bag_holder.png"),
}
"""Part module -> (its drawing tool, the drawing it commits).

Everything here goes through the loop in CLAUDE.md. ``dog_deck`` shares the dog
set's figure because it is the surface every other part in that set reacts
against, and drawing it alone would show a plate with holes in it and none of
the load path.
"""

DECORATIVE = {
    "coin": "a disc with a relief on it; nothing bears on it but the table",
    "tremblant_coin": "a coin with a different relief; nothing bears on it either",
    "einstein_fidget": "tiles pushed about by hand; the load is a fingertip",
    "lamborghini_badge": "adhesive-mounted trim, carrying its own weight",
    "mustang_badge": "adhesive trim as well, and lighter than the Lamborghini one",
}
"""Part module -> why no statics. A reason, not a checkbox: "decorative" is the
claim that nothing bears on it, and writing the sentence is what makes anyone
check whether that is true."""


def _part_modules() -> set[str]:
    return {
        m.name
        for m in pkgutil.iter_modules(parts.__path__)
        if not m.name.startswith("_")
    }


def test_every_part_is_classified():
    """The rule that catches the next one. Both mechanical errors this repo has
    had were written in the state of not having asked whether the part carried
    anything, so a new part module fails the suite until somebody asks."""
    known = set(CARRIES_LOAD) | set(DECORATIVE)
    missing = _part_modules() - known
    assert not missing, (
        f"unclassified part module(s): {', '.join(sorted(missing))}. Add each to "
        f"CARRIES_LOAD with a drawing tool and a committed figure, or to "
        f"DECORATIVE with a sentence saying what does not bear on it."
    )


def test_the_classification_names_only_real_modules():
    """A registry that has drifted from the tree is a registry that stopped
    being checked."""
    stale = (set(CARRIES_LOAD) | set(DECORATIVE)) - _part_modules()
    assert not stale, f"classified but gone: {', '.join(sorted(stale))}"


def test_nothing_is_classified_twice():
    both = set(CARRIES_LOAD) & set(DECORATIVE)
    assert not both, f"both load-bearing and decorative: {', '.join(sorted(both))}"


@pytest.mark.parametrize("name", sorted(DECORATIVE))
def test_a_decorative_part_says_why(name):
    """One line, in prose, about what does not bear on it -- not the word
    "decorative" on its own."""
    reason = DECORATIVE[name]
    assert len(reason.split()) >= 5, f"{name}: {reason!r} is not a reason"


@pytest.mark.parametrize("name", sorted(CARRIES_LOAD))
def test_a_load_bearing_part_commits_its_drawing(name):
    """So a reader has the section without running anything, and so a change
    that silently alters the load path shows up in the diff as an image."""
    _, image = CARRIES_LOAD[name]
    assert (ROOT / image).is_file(), f"{name} has no committed {image}"


@pytest.mark.parametrize("name", sorted(CARRIES_LOAD))
def test_a_load_bearing_parts_drawing_regenerates_from_the_model(name, tmp_path):
    """The whole value of drawing from ``Params`` rather than by hand: the
    figure cannot quietly go on describing a part that has since changed. Called
    with no arguments on purpose -- a drawing that needs to be set up by hand is
    a drawing that stops being redrawn."""
    tool, image = CARRIES_LOAD[name]
    draw = getattr(importlib.import_module(tool), "draw", None)
    assert callable(draw), f"{tool} has no draw(); see CLAUDE.md"
    made = draw(tmp_path / Path(image).name)
    assert Path(made).is_file() and Path(made).stat().st_size > 0


def test_the_rules_are_written_down_where_they_will_be_read():
    """This module enforces what it can. The reasoning, and the parts of the
    loop no test can reach, live in CLAUDE.md -- so that file existing and
    naming the loop is itself part of the standard."""
    text = (ROOT / "CLAUDE.md").read_text()
    assert "diagram and statics loop" in text.lower()
    for cue in ("free body", "contact", "load path"):
        assert cue in text.lower(), f"CLAUDE.md no longer mentions {cue}"
