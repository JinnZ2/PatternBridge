"""Packaging invariants.

`pip install patternbridge` is meant to give a working geometry core on numpy
alone, with every file format, web API and GPU dependency behind an extra.
That promise is easy to break by accident — one module-level `import svgwrite`
in the wrong file and a stranger's fresh install raises on import. These tests
pin it from the repository side; CI additionally installs the built package
bare and imports the core, which catches anything these miss.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest import TestCase, skipUnless

try:  # tomllib is 3.11+; this project supports 3.10.
    import tomllib
except ImportError:  # pragma: no cover - depends on interpreter version
    tomllib = None

ROOT = Path(__file__).resolve().parent.parent

# Third-party packages that must never be imported at module scope by the core.
OPTIONAL_PACKAGES = {
    "svgwrite",
    "reportlab",
    "anthropic",
    "openai",
    "torch",
    "torchvision",
    "PIL",
    "flask",
    "requests",
    "fitz",  # pymupdf
}

# Modules a bare `pip install patternbridge` must be able to import.
CORE_MODULES = [
    "pattern_geometry/piece.py",
    "pattern_geometry/boundary.py",
    "pattern_geometry/encoder.py",
    "pattern_geometry/scaler.py",
    "pattern_geometry/geometric_encoder.py",
    "pattern_geometry/octahedral_state.py",
    "pattern_geometry/spatial_grid.py",
    "pattern_geometry/symmetry_detector.py",
    "pattern_output/data_export.py",
    "pattern_output/__init__.py",
    "tools/import_svg_patterns.py",
    "tools/import_garment_patterns.py",
]


def _module_level_imports(path: Path) -> set[str]:
    """Top-level package names imported when this module is first imported.

    Imports nested inside a function or an `if`/`try` body are deferred or
    guarded, so they do not count against a bare install.
    """
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


class TestCoreStaysLight(TestCase):
    def test_core_modules_do_not_import_optional_packages(self):
        for relative in CORE_MODULES:
            with self.subTest(module=relative):
                leaked = _module_level_imports(ROOT / relative) & OPTIONAL_PACKAGES
                self.assertEqual(
                    leaked,
                    set(),
                    f"{relative} imports {sorted(leaked)} at module scope, which "
                    f"would break `pip install patternbridge` with no extras",
                )

    def test_output_package_exposes_writers_without_importing_them(self):
        # PEP 562 __getattr__ is what lets data_export work when neither
        # svgwrite nor reportlab is installed.
        import pattern_output

        self.assertTrue(hasattr(pattern_output, "__getattr__"))
        for name in ("SVGWriter", "PDFWriter"):
            self.assertIn(name, pattern_output.__all__)

    def test_unknown_attribute_still_raises_attribute_error(self):
        # A lazy __getattr__ must not turn typos into ImportErrors.
        import pattern_output

        with self.assertRaises(AttributeError):
            pattern_output.NoSuchWriter


@skipUnless(tomllib is not None, "tomllib requires Python 3.11+")
class TestProjectMetadata(TestCase):
    def setUp(self):
        self.pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.project = self.pyproject["project"]

    def test_numpy_is_the_only_required_dependency(self):
        required = {d.split(">=")[0].split("[")[0] for d in self.project["dependencies"]}
        self.assertEqual(required, {"numpy"})

    def test_every_optional_package_is_reachable_through_some_extra(self):
        extras = self.project["optional-dependencies"]
        declared = {
            dep.split(">=")[0].split("[")[0].lower()
            for deps in extras.values()
            for dep in deps
        }
        # `fitz` is imported from the pymupdf distribution; PIL from Pillow.
        aliases = {"fitz": "pymupdf", "pil": "pillow"}
        for package in OPTIONAL_PACKAGES:
            name = aliases.get(package.lower(), package.lower())
            with self.subTest(package=package):
                self.assertIn(name, declared)

    def test_dev_extra_omits_torch(self):
        # torch is ~800MB and the tests touching it are import-guarded, so it
        # stays out of the default development install.
        self.assertNotIn("torch", " ".join(self.project["optional-dependencies"]["dev"]))

    def test_readme_and_license_are_declared(self):
        self.assertEqual(self.project["readme"], "README.md")
        self.assertEqual(self.project["license"], "MIT")
