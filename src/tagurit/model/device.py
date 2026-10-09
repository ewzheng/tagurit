"""
Pick the torch device a model runs on.

torch is imported inside the function so importing this module stays cheap
and works without the ``model`` extra installed.
"""

from __future__ import annotations

SUPPORTED = ("cuda", "mps", "cpu")


def pick_device(requested: str | None = None) -> str:
    """
    Choose a torch device: the one requested, else CUDA, else Apple MPS, else CPU.

    A requested device that is unavailable RAISES ValueError instead of
    falling back, so a run configured for the GPU NEVER silently runs on the
    CPU. Indexed devices such as "cuda:1" are accepted.

    Parameters:
        - requested (str | None): a torch device string, or None to choose

    Return: a torch device string
    """
    import torch

    if requested is None:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    kind = torch.device(requested).type
    if kind not in SUPPORTED:
        raise ValueError(f"unsupported device {requested!r}; use one of {SUPPORTED}")
    if kind == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if kind == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return requested
