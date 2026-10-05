"""
Save and load a fitted tagger as one directory.

A bundle holds the model's file (for PatchCore, its memory bank) and
``tagger.json``, which names the model and records the tile size, the
priority reference, and the precision the reference was calibrated in. It
is what gets shipped to the scout, and later what the cloudlet would
replace to update the scout's notion of normal.

The model is imported only when a bundle is loaded, so importing this
module never pulls in torch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from tagurit.tagging.priority import PriorityScale
from tagurit.tagging.tagger import Tagger

if TYPE_CHECKING:
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
    settings = {
        "model": "patchcore",
        "model_file": PATCHCORE_FILE,
        "tile_size": tile_size,
        "reference": scale.reference,
        "half": scorer.half,
    }
    (directory / SETTINGS_FILE).write_text(json.dumps(settings, indent=2) + "\n")


def load_bundle(directory: Path, device: str | None = None, half: bool | None = None) -> Tagger:
    """
    Load a tagger saved by ``save_patchcore_bundle``.

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
    if settings["model"] != "patchcore":
        raise ValueError(f"unknown model {settings['model']!r} in {directory / SETTINGS_FILE}")

    from tagurit.model.patchcore import PatchcoreScorer

    half = settings["half"] if half is None else half
    scorer = PatchcoreScorer.load(directory / settings["model_file"], device=device, half=half)
    return Tagger(
        scorer=scorer,
        scale=PriorityScale(reference=settings["reference"]),
        tile_size=settings["tile_size"],
    )
