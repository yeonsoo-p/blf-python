"""
BLF File Reader and Decoder

This module provides a high-level interface for reading and decoding BLF (Binary Logging Format)
files using DBC (CAN Database) files.

Example:
    >>> from blf_python import BLF
    >>> blf = BLF('recording.blf', ['example/IMU.dbc'], channel=1)
    >>> # Access signal data
    >>> timestamps = blf.data['GpsStatus']['Time']
    >>> gps_mode = blf.data['GpsStatus']['GpsPosMode']
"""

from pathlib import Path
import numpy as np

try:
    from . import blf_python as _blf
except ImportError:
    import blf_python as _blf


class BLF:
    """
    BLF file reader with DBC decoding support.

    This class reads BLF files and decodes CAN messages using provided DBC database files.
    The decoded data is stored as NumPy arrays organized by message name and signal name.

    Attributes:
        filepath (str): Path to the BLF file
        dbc_files (List[str]): List of DBC file paths used for decoding
        channel (int): CAN channel filter (-1 for all channels)
        data (Dict[str, Dict[str, np.ndarray]]): Decoded message and signal data
            Structure: data[message_name][signal_name] = numpy_array
            Special signal 'Time' contains timestamps for each message

    Example:
        >>> blf = BLF('recording.blf', ['vehicle.dbc', 'sensors.dbc'], channel=1)
        >>> print(blf.messages)  # List all decoded messages
        >>> print(blf.get_signals('GpsStatus'))  # List signals in a message
        >>> timestamps = blf.data['GpsStatus']['Time']
        >>> gps_mode = blf.data['GpsStatus']['GpsPosMode']
    """

    def __init__(self, filepath: str | Path, dbc_files: list[str | Path], channel: int = -1):
        """
        Initialize BLF reader and decode the file.

        Args:
            filepath: Path to the BLF file to read (str or Path)
            dbc_files: List of DBC file paths for message decoding (str or Path)
            channel: CAN channel to filter (default: -1 for all channels)

        Raises:
            FileNotFoundError: If BLF file or any DBC file doesn't exist
            ValueError: If DBC files list is empty or files cannot be parsed
            IOError: If BLF file cannot be opened or read

        Note:
            The BLF file is processed immediately upon instantiation.
            Multiple DBC files can be provided - the first matching message
            definition will be used for decoding.
        """
        # Convert to Path for validation
        filepath_obj = Path(filepath)
        if not filepath_obj.exists():
            raise FileNotFoundError(f"BLF file not found: {filepath}")

        if not dbc_files:
            raise ValueError("At least one DBC file must be provided")

        dbc_files_obj = [Path(f) for f in dbc_files]
        for dbc_file in dbc_files_obj:
            if not dbc_file.exists():
                raise FileNotFoundError(f"DBC file not found: {dbc_file}")

        # Store parameters as strings (for C extension compatibility)
        self.filepath = str(filepath_obj)
        self.dbc_files = [str(f) for f in dbc_files_obj]
        self.channel = channel

        # Read and decode the BLF file
        self.data: dict[str, dict[str, np.ndarray]] = _blf.read_blf(
            self.filepath, self.dbc_files, channel
        )

    @property
    def messages(self) -> list[str]:
        """
        Get list of all decoded message names.

        Returns:
            List of message names found in the BLF file

        Example:
            >>> blf = BLF('recording.blf', ['vehicle.dbc'])
            >>> for msg in blf.messages:
            ...     print(msg)
        """
        return list(self.data.keys())

    def get_signals(self, message_name: str) -> list[str]:
        """
        Get list of all signal names for a specific message.

        Args:
            message_name: Name of the message

        Returns:
            List of signal names (including 'Time')

        Raises:
            KeyError: If message name doesn't exist

        Example:
            >>> signals = blf.get_signals('GpsStatus')
            >>> print(signals)  # ['Time', 'GpsNumSats', 'GpsPosMode', ...]
        """
        if message_name not in self.data:
            raise KeyError(f"Message '{message_name}' not found. Available messages: {self.messages}")

        return list(self.data[message_name].keys())

    def get_message_count(self, message_name: str) -> int:
        """
        Get the number of samples for a specific message.

        Args:
            message_name: Name of the message

        Returns:
            Number of message instances found in the BLF file

        Raises:
            KeyError: If message name doesn't exist

        Example:
            >>> count = blf.get_message_count('GpsStatus')
            >>> print(f"Found {count} GpsStatus messages")
        """
        if message_name not in self.data:
            raise KeyError(f"Message '{message_name}' not found")

        return len(self.data[message_name]["Time"])

    def __repr__(self) -> str:
        """String representation of BLF object."""
        return f"BLF(filepath='{self.filepath}', messages={len(self.messages)}, channel={self.channel})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [f"BLF File: {self.filepath}"]
        lines.append(f"Channel: {self.channel if self.channel >= 0 else 'All'}")
        lines.append(f"DBC Files: {', '.join(Path(f).name for f in self.dbc_files)}")
        lines.append(f"\nMessages ({len(self.messages)}):")

        for msg_name in sorted(self.messages):
            count = self.get_message_count(msg_name)
            signal_count = len(self.get_signals(msg_name)) - 1  # Exclude 'Time'
            lines.append(f"  {msg_name}: {count} samples, {signal_count} signals")

        return "\n".join(lines)

    def info(self) -> None:
        """
        Print detailed information about the loaded BLF file.

        Example:
            >>> blf = BLF('recording.blf', ['vehicle.dbc'])
            >>> blf.info()
        """
        print(str(self))

    def get_signal_data(self, message_name: str, signal_name: str) -> np.ndarray:
        """
        Get signal data as NumPy array.

        Args:
            message_name: Name of the message
            signal_name: Name of the signal

        Returns:
            NumPy array containing the signal values

        Raises:
            KeyError: If message or signal doesn't exist

        Example:
            >>> gps_mode = blf.get_signal_data('GpsStatus', 'GpsPosMode')
            >>> print(gps_mode.shape, gps_mode.dtype)
        """
        if message_name not in self.data:
            raise KeyError(f"Message '{message_name}' not found")

        if signal_name not in self.data[message_name]:
            available = self.get_signals(message_name)
            raise KeyError(f"Signal '{signal_name}' not found in message '{message_name}'. Available signals: {available}")

        return self.data[message_name][signal_name]

    def get_time_series(self, message_name: str) -> np.ndarray:
        """
        Get timestamps for a specific message.

        Args:
            message_name: Name of the message

        Returns:
            NumPy array containing timestamps in seconds

        Raises:
            KeyError: If message doesn't exist

        Example:
            >>> timestamps = blf.get_time_series('GpsStatus')
            >>> print(f"Duration: {timestamps[-1] - timestamps[0]:.2f} seconds")
        """
        return self.get_signal_data(message_name, "Time")
