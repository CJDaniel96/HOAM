"""HOAM: Hybrid Orthogonal Attention Model for SMT component polarity classification."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hoam")
except PackageNotFoundError:  # running from source without an install
    __version__ = "0.0.0+unknown"
