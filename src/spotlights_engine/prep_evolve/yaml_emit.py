"""Deterministic YAML emission for generated evolver configs.

Keeps key insertion order (no alphabetical sort), uses literal block scalars
(`|`) for multi-line strings so embedded digests stay readable, and forces
block style throughout.
"""

from __future__ import annotations

from typing import Any

import yaml


class _OrderedDumper(yaml.SafeDumper):
    pass


def _str_representer(dumper: yaml.SafeDumper, data: str) -> yaml.Node:
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_OrderedDumper.add_representer(str, _str_representer)


def dump_yaml(data: Any) -> str:
    """Serialize `data` to YAML, preserving dict insertion order."""
    return yaml.dump(
        data,
        Dumper=_OrderedDumper,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=100,
    )


__all__ = ["dump_yaml"]
