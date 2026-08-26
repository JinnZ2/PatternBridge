"""Pattern output layer — SVG, PDF, and JSON/data export.

The writers are resolved lazily. ``svgwrite`` and ``reportlab`` are optional
extras, so importing them eagerly here would make the whole output layer —
including ``data_export``, which is pure standard library — unusable unless
both were installed. Module-level ``__getattr__`` (PEP 562) keeps
``from pattern_output import SVGWriter`` working while charging each writer's
dependency only to the code that actually asks for that writer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .data_export import (
    piece_to_dict,
    piece_to_json,
    save_json,
    build_manifest,
    save_manifest,
    save_pattern_set,
)

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from .pdf_writer import PDFWriter
    from .svg_writer import SVGWriter

_LAZY = {"SVGWriter": ".svg_writer", "PDFWriter": ".pdf_writer"}

__all__ = [
    "SVGWriter",
    "PDFWriter",
    "piece_to_dict",
    "piece_to_json",
    "save_json",
    "build_manifest",
    "save_manifest",
    "save_pattern_set",
]


def __getattr__(name: str):
    """Import a writer on first use, so its extra is only needed then."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name, __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
