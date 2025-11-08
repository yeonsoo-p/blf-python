"""
BLF File Reader and Decoder

This module provides a high-level interface for reading and decoding BLF (Binary Logging Format)
files using DBC (CAN Database) files.

Example:
    >>> from blf_python import BLF
    >>> blf = BLF('recording.blf', [(-1, 'example/IMU.dbc')])
    >>> # Access signal data
    >>> timestamps = blf.get_signal('GpsStatus', 'Time')
    >>> gps_mode = blf.get_signal('GpsStatus', 'GpsPosMode')
    >>> # Or use dictionary-style access
    >>> velocity = blf['Distance']['Distance']
"""

from pathlib import Path
import numpy as np

from . import blf_python as _blf_c


class MessageProxy:
    """Proxy object for dictionary-style access to signals within a message."""

    def __init__(self, blf, message_name: str):
        self._blf = blf
        self._message_name = message_name
        self._cached_data = None  # Cache for 2D array from get_all_signals()
        self._cached_units = None  # Cache for units dict
        self._cached_factors = None  # Cache for factors dict
        self._cached_offsets = None  # Cache for offsets dict

    def __getitem__(self, signal_name: str) -> np.ndarray:
        """Get signal data using dictionary-style access."""
        return self.get_signal(signal_name)

    def __contains__(self, signal_name: str) -> bool:
        """Check if a signal exists in this message."""
        try:
            signals = self._blf.get_signals(self._message_name)
            return signal_name in signals
        except KeyError:
            return False

    def get_signal(self, signal_name: str) -> np.ndarray:
        """Get signal data for a specific signal."""
        # Just delegate to C++ - it returns zero-copy strided view
        return self._blf.get_signal(self._message_name, signal_name)

    def get_signal_names(self) -> list[str]:
        """Get list of all signal names in this message."""
        return self._blf.get_signals(self._message_name)

    def get_signal_units(self) -> dict[str, str]:
        """Get dictionary of signal name -> unit string."""
        if self._cached_units is None:
            self._cached_units = self._blf.get_signal_units(self._message_name)
        return self._cached_units

    def get_signal_unit(self, signal_name: str) -> str:
        """Get unit string for a specific signal."""
        units = self.get_signal_units()
        if signal_name not in units:
            raise KeyError(f"Signal '{signal_name}' not found in message '{self._message_name}'")
        return units[signal_name]

    def get_signal_factors(self) -> dict[str, float]:
        """Get dictionary of signal name -> scaling factor."""
        if self._cached_factors is None:
            self._cached_factors = self._blf.get_signal_factors(self._message_name)
        return self._cached_factors

    def get_signal_factor(self, signal_name: str) -> float:
        """Get scaling factor for a specific signal."""
        factors = self.get_signal_factors()
        if signal_name not in factors:
            raise KeyError(f"Signal '{signal_name}' not found in message '{self._message_name}'")
        return factors[signal_name]

    def get_signal_offsets(self) -> dict[str, float]:
        """Get dictionary of signal name -> scaling offset."""
        if self._cached_offsets is None:
            self._cached_offsets = self._blf.get_signal_offsets(self._message_name)
        return self._cached_offsets

    def get_signal_offset(self, signal_name: str) -> float:
        """Get scaling offset for a specific signal."""
        offsets = self.get_signal_offsets()
        if signal_name not in offsets:
            raise KeyError(f"Signal '{signal_name}' not found in message '{self._message_name}'")
        return offsets[signal_name]

    def get_all_signals(self) -> np.ndarray:
        """Get all signals as a 2D array (time + all signals)."""
        # Cache the 2D array for repeated access
        if self._cached_data is None:
            self._cached_data = self._blf.get_message_data(self._message_name)
        return self._cached_data

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
        >>> blf = BLF('recording.blf', [(-1, 'vehicle.dbc')])
        >>> print(blf.messages)  # List all decoded messages
        >>> print(blf.get_signals('GpsStatus'))  # List signals in a message
        >>> # Access signals
        >>> timestamps = blf.get_signal('GpsStatus', 'Time')
        >>> gps_mode = blf.get_signal('GpsStatus', 'GpsPosMode')
        >>> # Or use dictionary-style access
        >>> velocity = blf['Distance']['Distance']
        >>> # Get metadata
        >>> unit = blf.get_signal_unit('Distance', 'Distance')
        >>> factor = blf.get_signal_factor('Distance', 'Distance')
    """

    def __init__(self, filepath: str | Path, channel_dbc_list: list[tuple[int, str | Path]]):
        """
        Initialize BLF reader and decode the file.

        Args:
            filepath: Path to the BLF file to read (str or Path)
            channel_dbc_list: List of (channel, dbc_filepath) tuples
                Each tuple specifies which DBC file to use for which CAN channel.
                Use -1 for wildcard (matches all channels)

        Raises:
            FileNotFoundError: If BLF file or any DBC file doesn't exist
            ValueError: If channel_dbc_list is empty or contains invalid entries
            IOError: If BLF file cannot be opened or read

        Example:
            >>> # Single DBC for all channels (wildcard)
            >>> blf = BLF('recording.blf', [(-1, 'vehicle.dbc')])
            >>> # Specific channel
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

            validated_list.append((channel, str(dbc_path_obj)))
            dbc_files_for_display.append(str(dbc_path_obj))

        # Store parameters
        self.filepath = str(filepath_obj)
        self.dbc_files = dbc_files_for_display
        self.channel_dbc_list = validated_list

        # Create C extension BLF object
        self._blf = _blf_c.BLF(str(filepath_obj), validated_list)

        # Cache for MessageProxy objects (to preserve their internal caches)
        self._message_proxies = {}

    def get_message_names(self) -> list[str]:
        """Get list of all decoded message names."""
        return self._blf.get_message_names()

    @property
    def messages(self) -> list[str]:
        """Get list of all decoded message names (property for backward compatibility)."""
        return self.get_message_names()

    def get_signals(self, message_name: str) -> list[str]:
        """Get list of all signal names for a specific message."""
        return self._blf.get_signals(message_name)

    def get_signal(self, message_name: str, signal_name: str) -> np.ndarray:
        """Get signal data as NumPy array with scaling applied."""
        # Use MessageProxy cache for efficiency
        return self[message_name].get_signal(signal_name)

    def get_message(self, message_name: str) -> np.ndarray:
        """
        Get entire message as 2D NumPy array (time + all signals).

        Args:
            message_name: Name of the message

        Returns:
            2D NumPy array with shape (num_samples, num_signals+1)
            Column 0 is timestamps, columns 1+ are signals
        """
        return self[message_name].get_all_signals()

    def get_all_messages(self) -> dict[str, np.ndarray]:
        """
        Get all messages as dictionary of message_name -> 2D array.

        Returns:
            Dictionary mapping message names to 2D NumPy arrays
        """
        return {msg: self.get_message(msg) for msg in self.messages}

    def get_message_count(self, message_name: str) -> int:
        """Get the number of samples for a specific message."""
        return self._blf.get_message_count(message_name)

    def get_time_series(self, message_name: str) -> np.ndarray:
        """Get timestamps for a specific message."""
        return self.get_signal(message_name, "Time")

    def get_signal_unit(self, message_name: str, signal_name: str) -> str:
        """Get unit string for a signal."""
        return self[message_name].get_signal_unit(signal_name)

    def get_signal_factor(self, message_name: str, signal_name: str) -> float:
        """Get scaling factor for a signal."""
        return self[message_name].get_signal_factor(signal_name)

    def get_signal_offset(self, message_name: str, signal_name: str) -> float:
        """Get scaling offset for a signal."""
        return self[message_name].get_signal_offset(signal_name)

    def __getitem__(self, message_name: str) -> MessageProxy:
        """Get message proxy for dictionary-style signal access."""
        # Use cached MessageProxy to preserve internal data cache
        if message_name not in self._message_proxies:
            self._message_proxies[message_name] = MessageProxy(self._blf, message_name)
        return self._message_proxies[message_name]

    def __contains__(self, message_name: str) -> bool:
        """Check if a message exists in the BLF file."""
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
        """Print detailed information about the loaded BLF file."""
        print(str(self))
