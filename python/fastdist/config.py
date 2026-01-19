import json
import platform
from pathlib import Path

# -----------------------------
# Creating the directory / file
# -----------------------------

if platform.system() == "Windows":
    CONFIG_FILE = Path.home() / "AppData" / "Local" / "fastdist" / "config.json"
else:
    CONFIG_FILE = Path.home() / ".config" / "fastdist" / "config.json"

CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)


# ----------------------
# Load config
# ----------------------
def _load():
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except json.JSONDecodeError:
            # bad file -> reset
            return {"cuda_threshold": 10000}
    return {"cuda_threshold": 10000}


_config = _load()


# -----------
# Public APIs
# -----------
def get_cuda_threshold() -> int:
    return _config["cuda_threshold"]


def set_threshold(value: int):
    _config["cuda_threshold"] = int(value)
    CONFIG_FILE.write_text(json.dumps(_config, indent=4))
