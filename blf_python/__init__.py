"""
BLF Python - Binary Logging Format Reader with DBC Support

This package provides tools for reading and decoding BLF (Binary Logging Format) files
using DBC (CAN Database) files. Data is stored as NumPy arrays for efficient analysis.

Main Classes:
    BLF: High-level interface for reading BLF files and accessing decoded CAN messages

Example:
    >>> from blf_python import BLF
    >>> # Single DBC for all channels (wildcard)
    >>> blf = BLF('recording.blf', [(-1, 'vehicle.dbc')])
    >>> # Access signal data
    >>> timestamps = blf.get_signal('GpsStatus', 'Time')
    >>> gps_mode = blf.get_signal('GpsStatus', 'GpsPosMode')
    >>> # Or use dictionary-style access
    >>> velocity = blf['Distance']['Distance']
    >>> # Get metadata
    >>> unit = blf.get_signal_unit('Distance', 'Distance')
"""

from .blf import BLF

__all__ = ["BLF"]
__version__ = "0.1.0"
