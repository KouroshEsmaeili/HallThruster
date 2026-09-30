"""Paths and scalar constants shared by the translated package."""

from pathlib import Path
import sysconfig


_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_INSTALLED_DATA_ROOT = Path(sysconfig.get_path("data")) / "share" / "hallthruster"

if (_SOURCE_ROOT / "reactions").is_dir():
    PACKAGE_ROOT = _SOURCE_ROOT
    REACTION_FOLDER = _SOURCE_ROOT / "reactions"
    LANDMARK_FOLDER = _SOURCE_ROOT / "landmark"
else:
    PACKAGE_ROOT = Path(__file__).resolve().parent
    REACTION_FOLDER = _INSTALLED_DATA_ROOT / "reactions"
    LANDMARK_FOLDER = _INSTALLED_DATA_ROOT / "landmark"

LANDMARK_RATES_FILE = LANDMARK_FOLDER / "landmark_rates.csv"
TEST_DIR = _SOURCE_ROOT / "test"
PYTHON_PATH = _SOURCE_ROOT / "python"

MIN_NUMBER_DENSITY = 1e6
