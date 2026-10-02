"""Per-tokamak datasets for validating transport codes.

A cluster fit worker runs this file before it pins its BLAS threads,
so nothing imported here may load numpy (see gp_fitting/__init__.py).
"""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).parent

EPISODE_DIM = "shot"
TIME_DIM = "time_idx"
TIME_COORD = "time"
RADIAL_DIM = "rho_tor_norm"
