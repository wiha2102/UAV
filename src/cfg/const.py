"""
src/cfg/const.py
----------------
Constants used in the project is collected in this script is simply used for 
flexible customizable and guarantees a consistent usage of values anywhere
in this project.
"""
from typing import Final
from pathlib import Path


# ============================================================
#   Physical Constants
# ============================================================

LIGHT_SPEED         : Final[float]  = 2.99792458e8   # m/s
THERMAL_NOISE       : Final[float]  = -174.0         # dBm/Hz

TERRESTRIAL         : Final[str]    = "Terrestrial"
AERIAL              : Final[str]    = "Aerial"


# ============================================================
#   Filenames and path constants
# ============================================================

PREPROC_FN          : Final[str]    = "preproc.json"
WEIGHTS_FN          : Final[str]    = "model.weights.h5"
LINK_CONFIG_FN      : Final[str]    = "link_config.json"
PATH_CONFIG_FN      : Final[str]    = "path_config.json"
GEN_CONFIG_FN       : Final[str]    = "gen_config.json"
METRICS_FN          : Final[str]    = "training_metrics.json"
STATE_FN            : Final[str]    = "training_state.json"


# ============================================================
#   File-extensions
# ============================================================

PKL_EXT             : Final[str]    = ".pkl"
H5_EXT              : Final[str]    = ".h5"
JSON_EXT            : Final[str]    = ".json"
