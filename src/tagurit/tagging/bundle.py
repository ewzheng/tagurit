"""
Save and load a ready tagger as one directory.

A bundle holds ``tagger.json``, which names the model and records the tile
size, the priority reference, and the precision the reference was
calibrated in, plus whatever the model needs. A PatchCore bundle adds the
memory bank file; a CLIP bundle records its checkpoint id and prompts in
the settings, since its weights come from the Hugging Face Hub. A bundle
is what gets shipped to the scout, and later what the cloudlet would
replace to update it.

The model is imported only when a bundle is loaded, so importing this
module never pulls in torch.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tagurit.tagging.priority import PriorityScale
from tagurit.tagging.tagger import Tagger

if TYPE_CHECKING:
    from tagurit.model.clip import ClipScorer
    from tagurit.model.patchcore import PatchcoreScorer

SETTINGS_FILE = "tagger.json"
PATCHCORE_FILE = "patchcore.pt"


def save_patchcore_bundle(
    directory: Path, scorer: PatchcoreScorer, scale: PriorityScale, tile_size: int
) -> None:
    """
    Write a PatchCore tagger to ``directory``, creating it if needed.

    Existing bundle files in the directory are OVERWRITTEN.

    Parameters:
        - directory (Path): bundle folder
        - scorer (PatchcoreScorer): fitted scorer whose bank is saved
        - scale (PriorityScale): calibrated priority mapping
        - tile_size (int): tile side length the scale was calibrated with

    Return: void
    """
    directory.mkdir(parents=True, exist_ok=True)
    scorer.save(directory / PATCHCORE_FILE)
    _write_settings(
        directory,
        {"model": "patchcore", "model_file": PATCHCORE_FILE},
        scale,
        tile_size,
        scorer.half,
    )


def save_clip_bundle(
    directory: Path, scorer: ClipScorer, scale: PriorityScale, tile_size: int
) -> None:
    """
    Write a CLIP tagger to ``directory``, creating it if needed.

    Only settings are written; the checkpoint is fetched by id on load.
    An existing settings file is OVERWRITTEN.

    Parameters:
        - directory (Path): bundle folder
        - scorer (ClipScorer): scorer whose checkpoint id and prompts are saved
        - scale (PriorityScale): calibrated priority mapping
        - tile_size (int): tile side length the scale was calibrated with

    Return: void
    """
    directory.mkdir(parents=True, exist_ok=True)
    _write_settings(
        directory, {"model": "clip", "clip": asdict(scorer.config)}, scale, tile_size, scorer.half
    )


def load_bundle(directory: Path, device: str | None = None, half: bool | None = None) -> Tagger:
    """
    Load a tagger saved by ``save_patchcore_bundle`` or ``save_clip_bundle``.

    A missing settings file, missing keys, or an unknown model name RAISES.
    Overriding the calibrated precision shifts raw scores slightly, far
    too little to reorder frames.

    Parameters:
        - directory (Path): bundle folder
        - device (str | None): torch device for the model, None to pick one
        - half (bool | None): score in float16; None uses the bundle's setting

    Return: a Tagger ready to score frames
    """
    settings = json.loads((directory / SETTINGS_FILE).read_text())
    if settings["model"] not in ("patchcore", "clip"):
        raise ValueError(f"unknown model {settings['model']!r} in {directory / SETTINGS_FILE}")
    half = settings["half"] if half is None else half
    if settings["model"] == "patchcore":
        from tagurit.model.patchcore import PatchcoreScorer

        scorer: Any = PatchcoreScorer.load(
            directory / settings["model_file"], device=device, half=half
        )
    else:
        from tagurit.model.clip import ClipConfig, ClipScorer

        fields = dict(settings["clip"])
        fields["targets"] = tuple(fields["targets"])
        fields["backgrounds"] = tuple(fields["backgrounds"])
        scorer = ClipScorer.load(ClipConfig(**fields), device=device, half=half)
    return Tagger(
        scorer=scorer,
        scale=PriorityScale(reference=settings["reference"]),
        tile_size=settings["tile_size"],
    )


def _write_settings(
    directory: Path, model: dict[str, Any], scale: PriorityScale, tile_size: int, half: bool
) -> None:
    """
    Write ``tagger.json``: the model's own entries plus the tagging settings.

    Parameters:
        - directory (Path): bundle folder, already created
        - model (dict[str, Any]): the "model" name and its loading details
        - scale (PriorityScale): calibrated priority mapping
        - tile_size (int): tile side length the scale was calibrated with
        - half (bool): whether the reference was calibrated in float16

    Return: void
    """
    settings = {**model, "tile_size": tile_size, "reference": scale.reference, "half": half}
    (directory / SETTINGS_FILE).write_text(json.dumps(settings, indent=2) + "\n")
