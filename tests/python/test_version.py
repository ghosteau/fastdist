import re
from pathlib import Path

import pytest

import fastdist
import fastdist._fastdist as core

CMAKELISTS = Path(__file__).resolve().parents[2] / "CMakeLists.txt"

# What __init__.py reports when importlib.metadata cannot find an installed
# distribution -- i.e. when the suite is running against a source checkout
# rather than an installed wheel.
UNINSTALLED = "0.0.0+unknown"


def _cmake_version() -> str:
    """The single source of truth: the project() call in CMakeLists.txt."""
    match = re.search(
        r"project\s*\(\s*fastdist\s+VERSION\s+(\d+\.\d+\.\d+)",
        CMAKELISTS.read_text(encoding="utf-8"),
    )
    assert match is not None, "no VERSION found in the project() call"
    return match.group(1)


def test_compiled_module_matches_cmake():
    """The C++ constant must match what CMake declared."""
    assert core.__version__ == _cmake_version()


def test_python_package_matches_compiled_module():
    """The installed wheel's metadata must match the C++ constant.

    Only meaningful against an installed distribution. Running from a source
    checkout there is no metadata to read, and a stale egg-info left over from
    an older build reports whatever version it was generated at -- neither says
    anything about the code under test, so both are skipped rather than failed.
    """
    if fastdist.__version__ == UNINSTALLED:
        pytest.skip("fastdist is not installed; no distribution metadata to check")

    assert fastdist.__version__ == core.__version__, (
        f"installed metadata says {fastdist.__version__} but the compiled module "
        f"says {core.__version__}. If a stale python/fastdist.egg-info is present, "
        f"delete it and rebuild -- importlib.metadata will read it in preference "
        f"to nothing, even when it predates the current version."
    )
