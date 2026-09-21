# Style

PEP 8, as ruff enforces it (`make lint`). Line length 100. snake_case for
functions, variables, and modules. CapWords for classes. UPPER_CASE for module
constants.

## Docstrings

Every module, class, and public function has one. Triple double quotes, text
starting on the line after the opening quotes. Three parts:

1. A narrative paragraph: what the unit does and why it exists. Call out
   invariants, mutation of arguments, and anything about ordering or
   concurrency a caller could get wrong. Emphasize with capitals where it
   matters (MUTATES, NEVER, RAISES).
2. `Parameters:` then one bullet per parameter, `- name (Type): description`.
   Omit the section when there are no parameters.
3. `Return:` and what comes back, or `void`. Omit for `__init__` and dataclasses.

```python
def load(root: Path, name: str, fps: float = 30.0) -> Trace:
    """
    Load one sequence as a Trace.

    Lists the JPEGs on disk and parses the annotation file once. Nothing is
    read from the JPEGs here; frames read lazily. A malformed line RAISES
    rather than being skipped, so a bad dataset fails here and not as a
    silently short trace.

    Parameters:
        - root (Path): a split folder, see ``default_root``
        - name (str): sequence name, e.g. "uav0000086_00000_v"
        - fps (float): nominal frame rate used to derive timestamps

    Return: Trace with frames in ascending index order
    """
```

For a dataclass, the `Parameters:` section documents the fields.

## Code

- Type hints on every public signature.
- `pathlib.Path` over `os.path`.
- Frozen dataclasses for records that cross a function boundary.
- Raise on bad input. Never skip, default, or log-and-continue.
- Standard library first. A dependency has to earn its place; add it with `uv add`.
- Heavy imports (torch, ultralytics) stay out of `__init__.py` and behind functions.
- Tests are offline and deterministic: `tmp_path` fixtures, no network, no weights.
