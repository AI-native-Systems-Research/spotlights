"""Text templates kept out of code.

Templates use `string.Template` (`$name` / `${name}` substitution) so the
embedded Python/YAML — which is full of `{}` — needs no brace escaping. Load
with `render_template(name, **vars)`.
"""

from __future__ import annotations

from pathlib import Path
from string import Template

_DIR = Path(__file__).parent


def load_template(name: str) -> str:
    """Return the raw template text for `name` (e.g. `coral_grader.py.tmpl`)."""
    return (_DIR / name).read_text(encoding="utf-8")


def render_template(name: str, /, **values: str) -> str:
    """Render a template with `$name` placeholders.

    Uses `safe_substitute` so a stray `$` in interpolated content does not
    raise; intentional placeholders are always supplied by the caller.
    """
    return Template(load_template(name)).safe_substitute(values)


__all__ = ["load_template", "render_template"]
