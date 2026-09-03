import re
from pathlib import Path

import fastdist
import fastdist._fastdist as core

CMAKELISTS = Path(__file__).resolve().parents[2] / "CMakeLists.txt"


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
    """The wheel metadata must match the C++ constant."""
    assert fastdist.__version__ == core.__version__