"""
Save and load a ready tagger as one directory.

A bundle holds ``tagger.json``, which names the model and records the tile
size, the priority reference, the precision the reference was calibrated
in, and, when it was calibrated, the priority from which a frame counts as
a detection (``tagging.order``), plus whatever the model needs. A PatchCore bundle adds the
memory bank file. A CLIP bundle records its checkpoint id, query, template
and novelty window in the settings, since its weights come from the
Hugging Face Hub, and adds the embeddings of the target-free tiles its
query is scaled against, so a new query can be scaled without them. A
bundle is what gets shipped to the scout, and later what the cloudlet would
replace to update it.

The model is imported only when a bundle is loaded, so importing this
module never pulls in torch.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from tagurit.tagging.priority import PriorityScale
from tagurit.tagging.tagger import Tagger

if TYPE_CHECKING:
    from tagurit.model.clip import ClipScorer
    from tagurit.model.patchcore import PatchcoreScorer

SETTINGS_FILE = "tagger.json"
PATCHCORE_FILE = "patchcore.pt"
CLIP_REFERENCE_FILE = "clip_reference.npy"


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
    directory: Path,
    scorer: ClipScorer,
    scale: PriorityScale,
    tile_size: int,
    detection: float | None = None,
) -> None:
    """
    Write a CLIP tagger to ``directory``, creating it if needed.

    Writes the settings and the reference tile embeddings; the checkpoint
    is fetched by id on load. Existing bundle files are OVERWRITTEN. A
    scorer that was never calibrated RAISES ValueError.

    Parameters:
        - directory (Path): bundle folder
        - scorer (ClipScorer): calibrated scorer whose settings and reference are saved
        - scale (PriorityScale): calibrated priority mapping
        - tile_size (int): tile side length the scale was calibrated with
        - detection (float | None): priority from which a frame is a detection, if calibrated

    Return: void
    """
    if scorer.reference is None:
        raise ValueError("calibrate the CLIP scorer before saving it")
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / CLIP_REFERENCE_FILE, scorer.reference)
    _write_settings(
        directory,
        {"model": "clip", "clip": asdict(scorer.config), "reference_file": CLIP_REFERENCE_FILE},
        scale,
        tile_size,
        scorer.half,
        detection,
    )


def load_bundle(directory: Path, device: str | None = None, half: bool | None = None) -> Tagger:
    """
    Load a tagger saved by ``save_patchcore_bundle`` or ``save_clip_bundle``.

    A missing settings file, missing keys, or an unknown model name RAISES.
    So does a CLIP bundle from before query-only scoring, which named
    background prompts: rebuild it with ``scripts/calibrate_clip.py``.
    Overriding the calibrated precision shifts raw scores slightly, far
    too little to reorder frames. A bundle without a detection threshold
    loads with ``Tagger.detection`` as None.

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
        if "backgrounds" in fields or "reference_file" not in settings:
            raise ValueError(
                f"{directory} is a CLIP bundle from before query-only scoring; "
                "rebuild it with scripts/calibrate_clip.py"
            )
        fields["targets"] = tuple(fields["targets"])
        reference = np.load(directory / settings["reference_file"])
        scorer = ClipScorer.load(
            ClipConfig(**fields), reference=reference, device=device, half=half
        )
    return Tagger(
        scorer=scorer,
        scale=PriorityScale(reference=settings["reference"]),
        tile_size=settings["tile_size"],
        detection=settings.get("detection"),
    )


def _write_settings(
    directory: Path,
    model: dict[str, Any],
    scale: PriorityScale,
    tile_size: int,
    half: bool,
    detection: float | None = None,
) -> None:
    """
    Write ``tagger.json``: the model's own entries plus the tagging settings.

    Parameters:
        - directory (Path): bundle folder, already created
        - model (dict[str, Any]): the "model" name and its loading details
        - scale (PriorityScale): calibrated priority mapping
        - tile_size (int): tile side length the scale was calibrated with
        - half (bool): whether the reference was calibrated in float16
        - detection (float | None): detection threshold, left out of the file when None

    Return: void
    """
    settings = {**model, "tile_size": tile_size, "reference": scale.reference, "half": half}
    if detection is not None:
        settings["detection"] = detection
    (directory / SETTINGS_FILE).write_text(json.dumps(settings, indent=2) + "\n")
