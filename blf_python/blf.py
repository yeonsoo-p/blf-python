"""
BLF File Reader and Decoder

This module provides a high-level interface for reading and decoding BLF (Binary Logging Format)
files using DBC (CAN Database) files.

Example:
    >>> from blf_python import BLF
    >>> blf = BLF('recording.blf', ['example/IMU.dbc'], channel=1)
    >>> # Access signal data
    >>> timestamps = blf.get_signal('GpsStatus', 'Time')
    >>> gps_mode = blf.get_signal('GpsStatus', 'GpsPosMode')
    >>> # Or use dictionary-style access
    >>> velocity = blf['Distance']['Distance']
"""

from pathlib import Path
import numpy as np

try:
    from . import blf_python as _blf
except ImportError:
    import blf_python as _blf


class MessageProxy:
    """Proxy object for dictionary-style access to signals within a message."""

    def __init__(self, blf_obj, message_name: str):
        self._blf = blf_obj
        self._message_name = message_name

    def __getitem__(self, signal_name: str) -> np.ndarray:
        """Get signal data using dictionary-style access."""
        return self._blf.get_signal(self._message_name, signal_name)

    def __contains__(self, signal_name: str) -> bool:
        """Check if a signal exists in this message."""
        try:
            signals = self._blf.get_signals(self._message_name)
            return signal_name in signals
        except KeyError:
            return False

    def __repr__(self) -> str:
        signals = self._blf.get_signals(self._message_name)
        return f"MessageProxy(message='{self._message_name}', signals={len(signals)})"


class BLF:
    """
    BLF file reader with DBC decoding support.

    This class reads BLF files and decodes CAN messages using provided DBC database files.
    The decoded data is accessed via zero-copy NumPy arrays.

    Attributes:
        messages (list[str]): List of all decoded message names

    Example:
        >>> blf = BLF('recording.blf', ['vehicle.dbc', 'sensors.dbc'], channel=1)
        >>> print(blf.messages)  # List all decoded messages
        >>> print(blf.get_signals('GpsStatus'))  # List signals in a message
        >>> # Access signals
        >>> timestamps = blf.get_signal('GpsStatus', 'Time')
        >>> gps_mode = blf.get_signal('GpsStatus', 'GpsPosMode')
        >>> # Or use dictionary-style access
        >>> velocity = blf['Distance']['Distance']
    """

    def __init__(self, filepath: str | Path, channel_dbc_list: list[tuple[int, str | Path]]):
        """
        Initialize BLF reader and decode the file.

        Args:
            filepath: Path to the BLF file to read (str or Path)
            channel_dbc_list: List of (channel, dbc_filepath) tuples
                Each tuple specifies which DBC file to use for which CAN channel

        Raises:
            FileNotFoundError: If BLF file or any DBC file doesn't exist
            ValueError: If channel_dbc_list is empty or contains invalid entries
            IOError: If BLF file cannot be opened or read

        Example:
            >>> # Single channel
            >>> blf = BLF('recording.blf', [(4, 'vehicle.dbc')])
            >>> # Multiple channels with different DBCs
            >>> blf = BLF('recording.blf', [(1, 'powertrain.dbc'), (2, 'chassis.dbc')])

        Note:
            The BLF file is processed immediately upon instantiation.
            Each channel can have its own DBC file for decoding.
        """
        # Convert to Path for validation
        filepath_obj = Path(filepath)
        if not filepath_obj.exists():
            raise FileNotFoundError(f"BLF file not found: {filepath}")

        if not channel_dbc_list:
            raise ValueError("At least one (channel, dbc_filepath) tuple must be provided")

        # Validate and convert channel_dbc_list
        validated_list = []
        dbc_files_for_display = []

        for entry in channel_dbc_list:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise ValueError("Each entry must be a (channel, dbc_filepath) tuple")

            channel, dbc_path = entry

            if not isinstance(channel, int):
                raise ValueError("Channel must be an integer")

            dbc_path_obj = Path(dbc_path)
            if not dbc_path_obj.exists():
                raise FileNotFoundError(f"DBC file not found: {dbc_path}")

            validated_list.append((channel, dbc_path_obj))
            dbc_files_for_display.append(str(dbc_path_obj))

        # Store parameters
        self.filepath = str(filepath_obj)
        self.dbc_files = dbc_files_for_display
        self.channel_dbc_list = validated_list

        # Create C extension BLF object
        self._blf = _blf.BLF(filepath_obj, validated_list)

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
        return self._blf.messages

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
        return self._blf.get_signals(message_name)

    def get_signal(self, message_name: str, signal_name: str) -> np.ndarray:
        """
        Get signal data as zero-copy NumPy array.

        Args:
            message_name: Name of the message
            signal_name: Name of the signal

        Returns:
            NumPy array containing the signal values (zero-copy)

        Raises:
            KeyError: If message or signal doesn't exist

        Example:
            >>> gps_mode = blf.get_signal('GpsStatus', 'GpsPosMode')
            >>> print(gps_mode.shape, gps_mode.dtype)
        """
        return self._blf.get_signal(message_name, signal_name)

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
        return self._blf.get_message_count(message_name)

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
        return self.get_signal(message_name, "Time")

    def __getitem__(self, message_name: str) -> MessageProxy:
        """
        Get message proxy for dictionary-style signal access.

        Args:
            message_name: Name of the message

        Returns:
            MessageProxy object that supports signal access via []

        Example:
            >>> velocity = blf['Distance']['Distance']
            >>> time = blf['GpsStatus']['Time']
        """
        return MessageProxy(self._blf, message_name)

    def __contains__(self, message_name: str) -> bool:
        """
        Check if a message exists in the BLF file.

        Args:
            message_name: Name of the message

        Returns:
            True if message exists, False otherwise

        Example:
            >>> if 'Distance' in blf:
            ...     print("Distance message found")
        """
        return message_name in self.messages

    def __repr__(self) -> str:
        """String representation of BLF object."""
        channels = [ch for ch, _ in self.channel_dbc_list]
        return f"BLF(filepath='{self.filepath}', messages={len(self.messages)}, channels={channels})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [f"BLF File: {self.filepath}"]
        channels_str = ', '.join(str(ch) for ch, _ in self.channel_dbc_list)
        lines.append(f"Channels: {channels_str}")
        lines.append(f"DBC Files: {', '.join(Path(f).name for f in self.dbc_files)}")
        lines.append(f"\nMessages ({len(self.messages)}):")

        for msg_name in sorted(self.messages):
            signals = self.get_signals(msg_name)
            time_array = self.get_signal(msg_name, "Time")
            count = len(time_array)
            signal_count = len(signals) - 1  # Exclude 'Time'
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
