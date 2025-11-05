"""
BLF Python - Binary Logging Format Reader with DBC Support

This package provides tools for reading and decoding BLF (Binary Logging Format) files
using DBC (CAN Database) files. Data is stored as NumPy arrays for efficient analysis.

Main Classes:
    BLF: High-level interface for reading BLF files and accessing decoded CAN messages

Example:
    >>> from blf_python import BLF
    >>> blf = BLF('recording.blf', ['vehicle.dbc'], channel=1)
    >>> timestamps = blf.data['GpsStatus']['Time']
    >>> gps_mode = blf.data['GpsStatus']['GpsPosMode']
"""

from .blf import BLF

__all__ = ["BLF"]
__version__ = "0.1.0"
