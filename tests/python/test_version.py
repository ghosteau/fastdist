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


STUB = Path(__file__).resolve().parents[2] / "python" / "fastdist" / "_fastdist.pyi"


def test_type_stub_matches_cmake():
    """
    The stub carries a literal version string, because a .pyi is static and
    cannot read CMakeLists at runtime. This keeps CMakeLists the single source
    of truth by failing when the stub was not regenerated after a version bump.
    """
    match = re.search(
        r"^__version__: str = '([^']+)'",
        STUB.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert match is not None, "no __version__ literal found in _fastdist.pyi"
    assert match.group(1) == _cmake_version(), (
        "python/fastdist/_fastdist.pyi is stale. Regenerate it -- see the "
        "Type stubs section of CONTRIBUTING.md."
    )
